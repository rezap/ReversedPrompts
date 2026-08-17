"""Embeddings, and the cache that stops us paying for them twice.

The embedding model is resolved from its **own** environment variable rather
than the shared `$REVPROMPT_MODEL`. A blanket chat-model setting must not
silently become the embedding model, and neither must `--model`: passing a chat
model to the embeddings endpoint fails, and passing it *successfully* would be
worse, because the index would be built from something incomparable to every
other index without saying so.

Caching is content-addressed by `(model, text)`. Re-indexing an unchanged
document costs nothing, which is what makes the test suite free and re-runs
cheap. Changing the model changes the key, so vectors from two models can never
be silently mixed in one index.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
BATCH = 128


def resolve_embedding_model(env: dict[str, str] | None = None) -> str:
    """`$REVPROMPT_EMBEDDING_MODEL`, else the default. Deliberately *not*
    `$REVPROMPT_MODEL` -- see the module docstring."""
    env = os.environ if env is None else env
    return env.get("REVPROMPT_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


@dataclass
class EmbeddingUsage:
    """What we spent. Embeddings are cheap but not free, and a budget that
    ignores them under-reports."""

    calls: int = 0
    texts: int = 0
    cached: int = 0
    tokens: int = 0

    def summary(self) -> str:
        return (f"{self.calls} embedding call(s), {self.texts} text(s), "
                f"{self.cached} from cache, {self.tokens} tokens")


@runtime_checkable
class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder:
    """Real embeddings. Batched, because one call per chunk on a book is
    thousands of round trips for no reason."""

    def __init__(self, model: str | None = None, dimensions: int = 1536):
        from openai import OpenAI          # lazy: tests need no dependency

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI()
        self.model = model or resolve_embedding_model()
        self.dimensions = dimensions
        self.usage = EmbeddingUsage()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            r = self._client.embeddings.create(model=self.model, input=batch)
            self.usage.calls += 1
            self.usage.texts += len(batch)
            self.usage.tokens += getattr(getattr(r, "usage", None),
                                         "total_tokens", 0) or 0
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        if out:
            self.dimensions = len(out[0])
        return out


class HashEmbedder:
    """Offline double. Deterministic, free, and not meaningless.

    Uses the hashing trick: each word lands in a bucket and the vector is the
    L2-normalised bucket count. That gives genuine lexical similarity -- texts
    sharing words really are closer -- so an offline test of "does retrieval
    find the passage mentioning this name" is testing something. What it has no
    notion of is *semantics*: paraphrase without shared words looks unrelated,
    which is exactly the gap the real embedder fills.
    """

    def __init__(self, dimensions: int = 64, model: str = "hash-embedder"):
        self.model = model
        self.dimensions = dimensions
        self.usage = EmbeddingUsage()

    def _vector(self, text: str) -> list[float]:
        import re
        vec = [0.0] * self.dimensions
        for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            bucket = struct.unpack("<Q", digest)[0] % self.dimensions
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.usage.calls += 1
        self.usage.texts += len(texts)
        return [self._vector(t) for t in texts]


@dataclass
class CachedEmbedder:
    """Wraps an embedder with a content-addressed on-disk cache.

    Only uncached texts reach the inner embedder, and they go in one batch, so
    re-indexing a document after changing one paragraph pays for one paragraph.
    """

    inner: Embedder
    directory: pathlib.Path
    hits: int = 0
    misses: int = 0
    _memo: dict[str, list[float]] = field(default_factory=dict, repr=False)

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(self.inner.model.encode())
        h.update(b"\x00")
        h.update(text.encode())
        return h.hexdigest()

    def _path(self, key: str) -> pathlib.Path:
        return self.directory / key[:2] / f"{key}.json"

    def _load(self, key: str) -> list[float] | None:
        if key in self._memo:
            return self._memo[key]
        path = self._path(key)
        if not path.exists():
            return None
        try:
            vec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None                    # a corrupt entry is a miss, not a crash
        self._memo[key] = vec
        return vec

    def _store(self, key: str, vec: list[float]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vec), encoding="utf-8")
        self._memo[key] = vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        found: dict[int, list[float]] = {}
        todo: list[int] = []
        for i, key in enumerate(keys):
            vec = self._load(key)
            if vec is None:
                todo.append(i)
            else:
                found[i] = vec

        self.hits += len(found)
        self.misses += len(todo)
        if todo:
            fresh = self.inner.embed([texts[i] for i in todo])
            for i, vec in zip(todo, fresh):
                self._store(keys[i], vec)
                found[i] = vec
        return [found[i] for i in range(len(texts))]
