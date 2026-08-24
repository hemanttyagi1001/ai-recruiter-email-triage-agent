"""
PostgresSaver factory. One place to construct the LangGraph checkpointer
so tests and production wire it the same way.

CONCEPT: PostgresSaver takes a raw psycopg (v3) Connection — it does NOT
use SQLAlchemy. That means our DATABASE_URL for psycopg differs from
SQLAlchemy's:
    SQLAlchemy: postgresql+psycopg://user:pw@host/db
    psycopg:    postgresql://user:pw@host/db
The `+psycopg` bit is a SQLAlchemy dialect selector, not part of a real
PostgreSQL URL. We strip it below.

The `.setup()` call is idempotent — it creates the checkpoint tables on
first run and no-ops thereafter. Safe to call at every boot.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

from app.config import settings

log = logging.getLogger(__name__)


def _psycopg_url(sqlalchemy_url: str) -> str:
    # postgresql+psycopg://... → postgresql://...
    if sqlalchemy_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + sqlalchemy_url[len("postgresql+psycopg://"):]
    if sqlalchemy_url.startswith("postgresql+"):
        # any other dialect selector — strip it
        _, rest = sqlalchemy_url.split("+", 1)
        _, tail = rest.split("://", 1)
        return "postgresql://" + tail
    return sqlalchemy_url


@contextmanager
def open_checkpointer():
    """Yield a ready-to-use PostgresSaver with an open psycopg connection.

    Usage:
        with open_checkpointer() as cp:
            graph = build_graph(llm, profile, gmail, cp)
            graph.invoke(...)

    The connection closes cleanly on context exit — important because
    unclosed psycopg connections leak file descriptors.

    GOTCHA: PostgresSaver holds the Connection object. If you close the
    connection while the graph still references the saver, subsequent
    graph.invoke() calls raise `connection closed`. Keep the checkpointer
    alive for the lifetime of the graph.
    """
    url = _psycopg_url(settings.database_url)
    # autocommit=True: LangGraph's checkpointer manages its own transactions
    # via savepoints and expects the underlying connection to be in
    # autocommit mode. Without this, checkpoint writes silently accumulate
    # inside an outer implicit transaction and don't commit until the
    # connection closes — which means "restart" sees no checkpoints.
    conn = Connection.connect(url, autocommit=True)
    try:
        saver = PostgresSaver(conn)
        saver.setup()  # idempotent: creates checkpoints tables on first run
        log.info("PostgresSaver ready (thread_id namespace shared across CLI + API)")
        yield saver
    finally:
        conn.close()
