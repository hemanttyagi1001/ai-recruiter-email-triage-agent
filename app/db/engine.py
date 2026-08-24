"""
SQLAlchemy engine and session factory. Everything that needs a DB session
asks for one via `session_scope()`, which guarantees commit-on-success,
rollback-on-exception, and close-always. Centralising this here means no
node has to remember the try/except/finally boilerplate — get it wrong once
and you leak connections until the pool is exhausted.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

# CONCEPT: `pool_pre_ping=True` sends a cheap SELECT 1 before handing out a
# pooled connection. A batch ingest job sits idle between runs; without
# pre-ping, the first query after an idle period can fail with "server closed
# the connection unexpectedly" if the DB dropped it. Pre-ping trades ~1ms for
# eliminating that class of transient failure.
# ALTERNATIVE: use pool_recycle=<seconds>. Coarser, harder to tune correctly.
engine = create_engine(
    settings.database_url,
    future=True,
    pool_pre_ping=True,
)

# WHY expire_on_commit=False: after commit, ORM objects retain their loaded
# attributes instead of being marked expired. Callers that want to keep
# reading a message after we commit its status change do not eat a second
# SELECT — matters when nodes hand objects to each other across commits.
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for a unit-of-work transaction.

    Usage:
        with session_scope() as s:
            s.add(obj)
            # commit on exit; rollback if the block raises.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
