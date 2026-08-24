# Container image for the recruiter triage agent.
#
# Runs `python -m app.cli.watch` — the poll loop, not the one-shot CLI. The
# other entrypoints (ingest, report, digest, halt, the API) are all present in
# the image and reachable with `docker exec`, which is how you run a one-off
# report or flip the kill switch without stopping the loop.

# WHY 3.12 and not 3.14 (what the host runs): langchain-core still imports
# pydantic.v1, which warns "not compatible with Python 3.14 or greater" on
# every boot. Harmless today, but a container is the wrong place to be relying
# on a deprecation staying inert. pyproject requires >=3.11; 3.12 is the newest
# version the whole dependency tree is actually happy on.
FROM python:3.12-slim

# WHY these two: PYTHONUNBUFFERED so logs reach `docker logs` immediately
# rather than sitting in a pipe buffer — the exact trap that made an ingest
# look hung earlier in this project's history. PYTHONDONTWRITEBYTECODE because
# .pyc files in a layer are dead weight.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# WHY pyproject.toml alone first: Docker caches layers by input. Copying only
# the dependency manifest means `pip install` is re-run when dependencies
# change, not every time a source file is edited — which is most of the time.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir "."

# WHY a non-root user: this process parses untrusted email content. It needs no
# write access to anything but the log directory, so it should not have any.
# GOTCHA: logs/ is created and chowned BEFORE dropping privileges — a non-root
# user cannot mkdir in a root-owned WORKDIR, and _configure_logging() calls
# log_dir.mkdir() on every start.
RUN useradd --create-home --uid 10001 triage \
    && mkdir -p /app/logs \
    && chown -R triage:triage /app
USER triage

# Credentials, profile, and token are mounted at runtime, never baked in.
# See docker-compose.yml. token.json in particular is a live OAuth refresh
# token — an image containing one is a credential leak waiting to be pushed.

# GOTCHA: no HEALTHCHECK. This process has no port and no readiness endpoint;
# a healthcheck would have to shell out and guess. `docker logs` plus the
# `runs` table are the real health signals — a cycle that completes writes a
# row, and one that fails writes a traceback.

CMD ["python", "-m", "app.cli.watch"]
