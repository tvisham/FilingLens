"""A local, open-source Postgres + pgvector for development, with no install or Docker.

Uses `pgserver` (embedded PostgreSQL 16 with pgvector, `uv sync --extra localdb`). Set
`DATABASE_URL="local"` (or `local:<database>`) and every FilingLens command connects to a
server kept in `data/pgdata`, starting it if needed. The port is chosen at startup, which is
why the URL is resolved at runtime rather than written into `.env`.

For tests, `TEST_DATABASE_URL="local:filinglens_test"` gives a separate database the DB
tests may reset.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

PGDATA = Path("data/pgdata")
DEFAULT_DB = "filinglens"
_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def is_local(url: str | None) -> bool:
    return bool(url) and (url == "local" or str(url).startswith("local:"))


def local_url(database: str = DEFAULT_DB, pgdata: Path = PGDATA) -> str:
    """Start (or reuse) the embedded server, create `database` if missing, return its URL."""
    if not _NAME.match(database):
        raise ValueError(f"invalid local database name: {database!r}")
    try:
        from pgserver.postgres_server import get_server
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("DATABASE_URL=local needs the embedded server: `uv sync --extra localdb`") from exc
    pgdata.parent.mkdir(parents=True, exist_ok=True)
    server = get_server(pgdata.resolve(), cleanup_mode=None)  # keeps running between commands
    parts = urlsplit(server.get_uri())
    user = parts.username or "postgres"
    # psycopg rather than the bundled psql.exe, which fails to start under some Windows
    # process setups (e.g. inside pytest's captured output).
    import psycopg

    with psycopg.connect(
        host=parts.hostname, port=parts.port, user=user, dbname="postgres", autocommit=True
    ) as conn:
        found = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,)).fetchone()
        if found is None:
            conn.execute(f'CREATE DATABASE "{database}"')  # name validated by _NAME above
    return f"postgresql+psycopg://{user}@{parts.hostname}:{parts.port}/{database}"


def resolve(url: str) -> str:
    """`local` / `local:<db>` -> the embedded server's URL; anything else unchanged."""
    if not is_local(url):
        return url
    _, _, database = url.partition(":")
    return local_url(database or DEFAULT_DB)


def stop(pgdata: Path = PGDATA) -> None:
    from pgserver.postgres_server import get_server

    if (pgdata / "postmaster.pid").exists():
        get_server(pgdata.resolve(), cleanup_mode="stop").cleanup()
