# Deploying to the VPS

The agent runs on the VPS as a single Docker container executing
`python -m app.cli.watch` — a poll loop, not a server. It exposes no port and
needs no reverse proxy. Every push to `main` redeploys it automatically
(`.github/workflows/deploy.yml`).

The repository contains **no secrets**. Configuration lives in five files that
are placed on the VPS by hand, once, and are never touched again by any deploy.
Get those five right and the pipeline is the easy part.

---

## 1. One-time VPS bootstrap

### Prerequisites

- Docker Engine with the Compose v2 plugin (`docker compose`, not
  `docker-compose`)
- The shared `postgres18` container running `pgvector/pgvector:pg18`, managed
  by its own compose project at `/root/postgres/`
- The `postgress-net` docker network, with both that container and this one
  attached to it
- SSH access as the user the pipeline will connect as

### The network

The database publishes **no host port** — deliberately. It is reachable only by
containers that join its network, which is why this project's
`docker-compose.yml` declares `postgress-net` as external and attaches the agent
to it. Docker's embedded DNS then resolves `postgres18` by name.

```bash
docker network create postgress-net        # once per machine; harmless if it exists
```

Compose will **not** create an external network for you. On a machine where it
is missing, `docker compose up` fails with *"network postgress-net declared as
external, but could not be found"*.

### Create the database

The agent will not create its own database:

```bash
PW=$(openssl rand -base64 24 | tr -d '/+=' | head -c 28)
docker exec postgres18 psql -U postgres -c "CREATE ROLE triage LOGIN PASSWORD '$PW';"
docker exec postgres18 psql -U postgres -c "CREATE DATABASE triage OWNER triage;"
docker exec postgres18 psql -U postgres -d triage -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo "$PW"   # put this in DOCKER_DATABASE_URL below, then forget it
```

> Migration `0004` also runs `CREATE EXTENSION IF NOT EXISTS vector`, but the
> `triage` role is not a superuser and cannot create extensions. Creating it
> once as `postgres` above makes that migration a no-op instead of a failure.

### Clone the repo

The path is fixed — the deploy script `cd`s to it literally:

```bash
mkdir -p /root/apps
cd /root/apps
git clone https://github.com/hemanttyagi1001/ai-recruiter-email-triage-agent.git
cd ai-recruiter-email-triage-agent
```

### Place the five files

None of these are in git. Copy them from your workstation:

```bash
# from your workstation, in the repo directory
scp .env credentials.json token.json candidate.toml hemant_tyagi_aiml_engineer.pdf \
    root@VPS:/root/apps/ai-recruiter-email-triage-agent/
```

| File | What it is | If missing |
|---|---|---|
| `.env` | DB URL, Azure key, autonomy mode | Container exits at import — `app/config.py` raises on a missing required var |
| `credentials.json` | Google OAuth client identity | `FileNotFoundError` from `load_credentials` |
| `token.json` | Live OAuth refresh token | The agent tries to open a browser it does not have, and hangs |
| `candidate.toml` | Your profile, used in replies | Profile load fails at startup |
| `hemant_tyagi_aiml_engineer.pdf` | Resume, attached on request | Compose creates an empty **directory** at the mount point and every attachment fails |

Then fix ownership and permissions:

```bash
cd /root/apps/ai-recruiter-email-triage-agent
chmod 600 .env credentials.json token.json
chown 10001:10001 token.json
mkdir -p logs && chown -R 10001:10001 logs
```

Both `chown`s matter, for the same reason: a **bind mount carries the host's
ownership into the container**, masking whatever the Dockerfile chowned. The
container runs as UID 10001 and must write to both paths — `token.json` on every
hourly OAuth refresh, `logs/watch.log` on every line it logs. Root-owned either
way and the agent fails. The deploy script re-applies the `logs/` one on every
run; `token.json` it deliberately does not touch.

**The `chown 10001` is not optional.** The container drops to UID 10001
(`Dockerfile`), and `load_credentials` rewrites `token.json` in place every time
the hourly access token expires. A root-owned file gives you an agent that
authenticates perfectly for one hour and then dies with a permission error that
looks nothing like its cause.

### Set the container's database URL

Inside a container `localhost` is the container itself, so `DATABASE_URL` from
`.env` cannot reach the database. Compose reads a separate variable for this. In
the VPS `.env`:

```bash
DOCKER_DATABASE_URL=postgresql+psycopg://triage:THE_PASSWORD@postgres18:5432/triage
```

`postgres18` is a container name, resolved by Docker's embedded DNS because both
containers sit on `postgress-net`. There is no host port and no IP to hardcode.

> Locally the shape is different — `host.docker.internal:5432` against a
> pgvector container that does publish 5432. Both work; `extra_hosts` in
> `docker-compose.yml` exists for the local one. One variable per environment.

### First run, by hand

Prove it works before handing control to CI:

```bash
docker compose build
docker compose run --rm agent alembic upgrade head
docker compose up -d
docker compose logs -f agent
```

You are looking for the scope preflight line. `preflight_scopes` reports at boot
what the token can actually do — if it says a **required** scope is missing, the
token is wrong and no amount of redeploying will fix it.

---

## 2. GitHub secrets

The workflow needs four repository secrets. This repo currently has **none** —
add them before the first push to `main`:

```bash
gh secret set VPS_HOST    # the VPS hostname or IP
gh secret set VPS_USER    # the SSH user, e.g. root
gh secret set VPS_PORT    # the SSH port, e.g. 22
gh secret set VPS_SSH_KEY # the FULL private key, including BEGIN/END lines
```

`VPS_SSH_KEY` is the private half of a keypair whose public half is in the VPS
user's `~/.ssh/authorized_keys`. Paste the whole file — a key missing its
trailing newline or its header line fails with an unhelpful handshake error.

---

## 3. What a deploy does

Push to `main` (or run the workflow manually via **Actions → Deploy → Run
workflow**) and the VPS runs, in order:

1. `git fetch` + `git reset --hard origin/main` — tracked files only
2. `docker compose build` — **before** stopping anything, so a broken build
   costs no downtime
3. `docker compose down` — SIGTERM with 60s grace; `watch.py` finishes the
   message in flight
4. `docker compose run --rm agent alembic upgrade head` — schema migrated with
   no agent running against it
5. `docker compose up -d --force-recreate`
6. `docker image prune -f`, then `compose ps` and 30 lines of log as proof

Deploys are serialised by a concurrency group, so two quick pushes queue rather
than racing each other into a half-stopped stack.

### The one thing never to add to that script

`git clean -fd`. It would delete all five untracked files above, including your
OAuth refresh token and Azure API key, and the agent would come back up unable
to authenticate. `git reset --hard` alone is deliberate.

---

## 4. Operating it

```bash
cd /root/apps/ai-recruiter-email-triage-agent

docker compose logs -f agent            # follow the loop
docker compose ps                       # is it up
docker compose restart agent            # after editing .env

docker exec -it ai-recruiter-email-triage-agent python -m app.cli.halt      # kill switch
docker exec -it ai-recruiter-email-triage-agent python -m app.cli.report    # what it did
docker exec -it ai-recruiter-email-triage-agent python -m app.cli.digest    # daily digest
```

The kill switch stops sending without a deploy and without a restart — reach for
it first, not for `compose down`.

### Changing configuration

`.env` changes need `docker compose restart agent`. `candidate.toml` and the
resume PDF are re-read every cycle and every send respectively, so editing those
on the box takes effect with no restart at all.

---

## 5. Troubleshooting

| Symptom | Cause |
|---|---|
| `PermissionError: '/app/logs/watch.log'`, container restarting forever | `logs/` on the host not owned by UID 10001. The bind mount masks the image's own chown |
| Deploy went green but nothing is working | Check `docker compose ps`. Before the health gate existed, a crash-looping container still produced a successful run |
| Authenticates for an hour, then permission errors | `token.json` not owned by UID 10001 |
| `insufficientPermissions` on one call only | An optional scope is missing; `mark_read` degrades, everything else runs |
| `invalid_scope` on refresh, nothing works | Token consented to fewer scopes than requested — delete `token.json`, re-run the consent flow on your workstation, copy it back (see D69) |
| Deploy fails at `alembic upgrade head` | Agent is already stopped. Fix the migration, push again — step 2 rebuilds before anything stops |
| `network postgress-net declared as external, but could not be found` | Run `docker network create postgress-net` on that machine. Compose never creates an external network itself |
| `could not translate host name "postgres18"` | The agent is not on `postgress-net`, or the database container isn't. Check `docker inspect <name> --format '{{json .NetworkSettings.Networks}}'` on both |
| Migration `0004` fails on `CREATE EXTENSION vector` | The `triage` role is not a superuser. Run the `CREATE EXTENSION` once as `postgres` (§1) |
| `NameResolutionError` on Google or Azure | DNS; compose pins 8.8.8.8 and 1.1.1.1 for this reason (D69) |
| Every attachment fails, resume is empty | The PDF was missing at `up` time, so Compose created a directory at the mount point. Place the file, then `docker compose up -d --force-recreate` |

Re-running the OAuth consent flow always happens **on your workstation**, never
on the VPS — the flow opens a browser and blocks, which on a headless box is an
indefinite hang with no explanation. Generate `token.json` locally, then `scp`
it up and re-`chown` it.
