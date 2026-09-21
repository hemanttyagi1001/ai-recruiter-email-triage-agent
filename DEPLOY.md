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
- A Postgres server with the `pgvector` extension available, publishing
  **5432 on the host**. This is the shared `pgvector` container already running
  on the box for other projects.
- SSH access as the user the pipeline will connect as

### Create the database

The agent will not create its own database. From the VPS:

```bash
docker exec -it pgvector psql -U postgres -c "CREATE DATABASE triage;"
docker exec -it pgvector psql -U postgres -c \
  "CREATE USER triage WITH PASSWORD 'choose-a-real-password';"
docker exec -it pgvector psql -U postgres -c \
  "GRANT ALL PRIVILEGES ON DATABASE triage TO triage;"
```

> The `vector` extension is created by migration `0004`, not by you — but the
> role running migrations must be allowed to `CREATE EXTENSION`. On the
> `pgvector/pgvector` image the `postgres` superuser can; a plain role cannot.
> Either run migrations as a superuser once, or `CREATE EXTENSION vector;`
> inside the `triage` database yourself before the first deploy.

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
```

**The `chown 10001` is not optional.** The container drops to UID 10001
(`Dockerfile`), and `load_credentials` rewrites `token.json` in place every time
the hourly access token expires. A root-owned file gives you an agent that
authenticates perfectly for one hour and then dies with a permission error that
looks nothing like its cause.

### Set the container's database URL

Inside a container `localhost` is the container itself, so `DATABASE_URL` from
`.env` cannot reach the host's Postgres. Compose reads a separate variable for
this. In the VPS `.env`:

```bash
DOCKER_DATABASE_URL=postgresql+psycopg://triage:choose-a-real-password@host.docker.internal:5432/triage
```

`host.docker.internal` resolves on Linux only because `docker-compose.yml`
declares `extra_hosts: host.docker.internal:host-gateway`. Docker Desktop
provides it for free; plain Docker Engine does not.

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
| Authenticates for an hour, then permission errors | `token.json` not owned by UID 10001 |
| `insufficientPermissions` on one call only | An optional scope is missing; `mark_read` degrades, everything else runs |
| `invalid_scope` on refresh, nothing works | Token consented to fewer scopes than requested — delete `token.json`, re-run the consent flow on your workstation, copy it back (see D69) |
| Deploy fails at `alembic upgrade head` | Agent is already stopped. Fix the migration, push again — step 2 rebuilds before anything stops |
| `NameResolutionError` on Google or Azure | DNS; compose pins 8.8.8.8 and 1.1.1.1 for this reason (D69) |
| Every attachment fails, resume is empty | The PDF was missing at `up` time, so Compose created a directory at the mount point. Place the file, then `docker compose up -d --force-recreate` |

Re-running the OAuth consent flow always happens **on your workstation**, never
on the VPS — the flow opens a browser and blocks, which on a headless box is an
indefinite hang with no explanation. Generate `token.json` locally, then `scp`
it up and re-`chown` it.
