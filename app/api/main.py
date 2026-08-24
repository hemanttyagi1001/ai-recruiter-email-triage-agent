"""
FastAPI application factory. The lifespan builds the graph + checkpointer
once at startup and tears them down on shutdown. Endpoints reach these
singletons via Depends().

Usage:
    uvicorn app.api.main:app --host 127.0.0.1 --port 8000
    (or: python -m app.api)

CONCEPT: FastAPI's lifespan is an async context manager that runs before
the first request and after the last. Perfect for expensive one-time
setup (LLM client init, PostgresSaver.setup(), profile load) that must
happen exactly once per process.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, pending
from app.candidate import load_profile
from app.config import settings
from app.gmail.client import GmailClient
from app.llm.client import LLMClient
from app.pipeline.checkpointer import open_checkpointer
from app.pipeline.graph import build_graph

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    _configure_logging()
    log.info("API starting; building graph + checkpointer")

    profile = load_profile()
    llm = LLMClient()
    gmail = GmailClient.create()

    # WHY __enter__/__exit__ manually instead of `with`: the checkpointer
    # must stay alive for the full lifespan of the app; it can't be inside
    # a `with` block because that block would exit as soon as we yield.
    cp_cm = open_checkpointer()
    checkpointer = cp_cm.__enter__()

    graph = build_graph(llm, profile, gmail, checkpointer)

    app.state.graph = graph
    app.state.checkpointer = checkpointer
    app.state.profile = profile
    app.state._cp_cm = cp_cm  # keep for shutdown

    log.info("API ready on %s:%d", settings.api_host, settings.api_port)

    try:
        yield
    finally:
        # --- shutdown ---
        log.info("API shutting down; closing checkpointer connection")
        cp_cm.__exit__(None, None, None)


app = FastAPI(
    title="Recruiter Triage — Approval API",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(pending.router)


def _configure_logging() -> None:
    settings.log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_dir / "api.log"),
        ],
    )
