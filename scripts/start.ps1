<#
.SYNOPSIS
    Start the recruiter triage agent, bringing up Postgres first if needed.

.DESCRIPTION
    This script exists because a container cannot start a sibling container.
    Doing so would mean mounting the Docker socket into the agent, which grants
    it effective root on the host — a large privilege for a process whose whole
    job is parsing untrusted email. Running the dependency check here, on the
    host, gets the same "database first, then app" ordering with none of that.

    Order of operations:
      1. Start the `pgvector` container if it is not already running.
      2. Wait until Postgres actually accepts connections (running != ready).
      3. Verify the `triage` database exists and migrations are applied.
      4. docker compose up -d the agent.

.EXAMPLE
    .\scripts\start.ps1
    .\scripts\start.ps1 -DbContainer my-postgres -TimeoutSeconds 120
#>

[CmdletBinding()]
param(
    # The standalone Postgres container this project talks to.
    [string]$DbContainer = "pgvector",
    # How long to wait for Postgres to accept connections before giving up.
    [int]$TimeoutSeconds = 60,
    # Postgres superuser inside that container, used only for readiness checks.
    [string]$DbUser = "postgres",
    # The database the agent uses.
    [string]$DbName = "triage"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# --- 0. Is Docker even up? ---------------------------------------------------
Write-Step "Checking Docker"
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not responding. Start Docker Desktop and re-run."
}
Write-Ok "Docker is running"

# --- 1. Start the database container if needed -------------------------------
Write-Step "Checking container '$DbContainer'"
$running = docker ps --filter "name=^/$DbContainer$" --format "{{.Names}}"
if ($running -eq $DbContainer) {
    Write-Ok "Already running"
} else {
    $exists = docker ps -a --filter "name=^/$DbContainer$" --format "{{.Names}}"
    if ($exists -ne $DbContainer) {
        # WHY not create it here: this container is shared with other projects
        # and its credentials and volume were chosen elsewhere. Guessing them
        # would produce a second, empty database that merely looks right.
        throw "No container named '$DbContainer' exists. Create it first, or run ``docker compose --profile bundled-db up -d`` to use the bundled Postgres instead."
    }
    Write-Warn "Stopped - starting it"
    docker start $DbContainer | Out-Null
    Write-Ok "Started"
}

# --- 2. Wait for readiness ---------------------------------------------------
# GOTCHA: `docker start` returns as soon as the container is running, which is
# well before Postgres finishes recovery and opens its socket. Starting the
# agent in that window produces a connection-refused crash loop that looks like
# a config error. pg_isready is the actual readiness signal.
Write-Step "Waiting for Postgres to accept connections"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
    docker exec $DbContainer pg_isready -U $DbUser *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    throw "Postgres in '$DbContainer' did not become ready within $TimeoutSeconds seconds."
}
Write-Ok "Accepting connections"

# --- 3. Sanity-check the database and schema ---------------------------------
Write-Step "Checking database '$DbName'"
$dbExists = docker exec $DbContainer psql -U $DbUser -tAc "SELECT 1 FROM pg_database WHERE datname='$DbName'"
if ($dbExists -ne "1") {
    throw "Database '$DbName' does not exist in '$DbContainer'. Create it, then run ``python -m alembic upgrade head``."
}
$version = docker exec $DbContainer psql -U $DbUser -d $DbName -tAc "SELECT version_num FROM alembic_version" 2>$null
if (-not $version) {
    Write-Warn "No alembic_version table - migrations have never run."
    Write-Warn "Run: python -m alembic upgrade head"
} else {
    Write-Ok "Schema at migration $version"
}

# --- 4. Start the agent ------------------------------------------------------
Write-Step "Starting the agent"
# GOTCHA: `docker compose` writes build and pull PROGRESS to stderr, not just
# errors. Under $ErrorActionPreference = "Stop", Windows PowerShell 5.1 turns
# any native stderr line into a terminating NativeCommandError — so a perfectly
# successful build aborts this script. Exit code is the only trustworthy
# success signal for a native command, so we relax the preference across this
# one call and check $LASTEXITCODE ourselves.
$previousEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    docker compose up -d --build agent 2>&1 | ForEach-Object { Write-Host "    $_" }
    $composeExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousEap
}
if ($composeExit -ne 0) { throw "docker compose up failed (exit $composeExit)." }

Write-Ok "Agent is up"
Write-Host ""
Write-Host "  Logs:        docker logs -f ai-recruiter-email-triage-agent"
Write-Host "  Stop:        docker compose down"
Write-Host "  Halt sends:  docker exec ai-recruiter-email-triage-agent python -m app.cli.halt --on --by ops"
Write-Host "  Digest:      docker exec ai-recruiter-email-triage-agent python -m app.cli.digest"
Write-Host ""
