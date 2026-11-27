"""Engine and session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(database_url: str) -> Engine:
    """`database_url` may be `local` / `local:<db>` for the embedded dev server (see localdb)."""
    from filinglens.db.localdb import resolve

    # pool_pre_ping: Neon suspends idle compute, so stale connections are common.
    return create_engine(resolve(database_url), pool_pre_ping=True, future=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
