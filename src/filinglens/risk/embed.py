"""Embeddings behind a small interface, so the model can be swapped (Voyage, Azure OpenAI).

Default: local `BAAI/bge-small-en-v1.5` (384 dims) via sentence-transformers, installed
with the `embeddings` extra (`uv sync --extra embeddings`). Vectors are L2-normalised, so
cosine similarity is a dot product. `HashingEmbedder` is a deterministic, dependency-free
stand-in for tests; it is not semantically meaningful beyond shared words.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

DIMENSIONS = 384
BGE_SMALL = "BAAI/bge-small-en-v1.5"
# bge v1.5 recommends this prefix for short retrieval queries (not for passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeSmallEmbedder:
    name = BGE_SMALL
    dimensions = DIMENSIONS

    def __init__(self, batch_size: int = 32, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError(
                "sentence-transformers is not installed: run `uv sync --extra embeddings`"
            ) from exc
        self._model: Any = SentenceTransformer(BGE_SMALL, device=device)
        self._batch_size = batch_size

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in v] for v in vectors]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._encode([BGE_QUERY_PREFIX + text])[0]


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Bag-of-words feature hashing into `dimensions` buckets, L2-normalised."""

    name = "hashing-bow"

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dimensions
        for token in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            v[h % self.dimensions] += 1.0 if (h >> 63) & 1 else -1.0
        return normalise(v)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def normalise(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else [0.0] * len(v)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two L2-normalised vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    n = len(vectors)
    return normalise([sum(col) / n for col in zip(*vectors, strict=True)])
