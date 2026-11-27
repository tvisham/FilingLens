"""A small on-disk JSON cache keyed by URL, with per-entry TTL."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.root = root
        self._clock = clock

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str, ttl_seconds: float) -> Any | None:
        """Return the cached value if present and younger than `ttl_seconds`."""
        path = self._path(key)
        try:
            with path.open(encoding="utf-8") as fh:
                entry = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if entry.get("key") != key:  # hash collision guard
            return None
        if self._clock() - float(entry["stored_at"]) > ttl_seconds:
            return None
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump({"key": key, "stored_at": self._clock(), "value": value}, fh)
        os.replace(tmp, path)  # atomic on Windows and POSIX
