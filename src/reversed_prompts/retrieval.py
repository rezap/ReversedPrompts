"""Hybrid retrieval over a document, so long inputs stop being hopeless.

Why hybrid and not one or the other: this project's tasks span both ends. "Name
the main antagonist" hangs on a rare token -- BM25 finds `Antinous` instantly
while an embedding blurs it into a cloud of Greek names. "Summarise this in
three sentences" has no rare token at all and needs semantics. Neither retriever
alone covers the set, so results from both are fused by reciprocal rank.

Two rules this module exists to keep straight, and they run in opposite
directions:

* **Induction-side** retrieval is keyed by the *output* -- given the answer and
  the book, find the passages that could have produced it. That is the Align
  step, and it is allowed to see the gold output because inferring from it is
  the whole job.
* **Execution-side** retrieval is keyed by the *candidate instruction*, and must
  never see the gold output. Retrieving the answer's neighbourhood and then
  scoring the executor on finding the answer measures nothing.

Nothing in this module is wired into the recovery loop yet. It is built,
inspectable through `revprompt index` and `revprompt retrieve`, and tested --
deciding how the loop should use it comes next.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .chunking import Chunk, chunk_text, merge_spans
from .embedding import CachedEmbedder, Embedder, HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = ROOT / "data" / "index"
RRF_K = 60          # the standard constant; damps the top of each ranking


@dataclass(frozen=True)
class Passage:
    """A retrieved span, expanded to its neighbours and merged.

    `start`/`end` are offsets into the source document, so a passage can always
    be checked against the document it claims to come from.
    """

    doc_id: str
    text: str
    start: int
    end: int
    score: float
    ordinals: tuple[int, ...]

    def cite(self) -> str:
        return f"{self.doc_id}[{self.start}:{self.end}]"


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K,
                           weights: list[float] | None = None,
                           ) -> dict[str, float]:
    """Fuse several ranked id lists into one score per id.

    Rank-based rather than score-based on purpose: BM25 scores and cosine
    similarities are not on the same scale and normalising them is a knob that
    needs tuning per corpus. Ranks need no such knob.

    `weights` is the one knob, and it exists because equal weighting is not
    always right. Measured on the full Odyssey with the offline hashing
    embedder, for the query "Antinous the ringleader of the suitors":
    keyword-only scores 5/5 on precision@5, vector-only scores 0/5, and equal-
    weight fusion scores **3/5** -- fusion made the good retriever worse by
    averaging in a bad one. Equal weights remain the default because with a
    real embedding model the vector side is not noise; the knob is here so that
    claim can be checked and corrected rather than assumed.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, ident in enumerate(ranking):
            scores[ident] = scores.get(ident, 0.0) + weight / (k + rank + 1)
    return scores


def fuse(rankings: list[list[str]], k: int = RRF_K,
         tiebreak: dict[str, float] | None = None,
         weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Fused ids, best first, with ties broken by `tiebreak`.

    The tiebreak is not decoration. Rank fusion throws away *how much* one
    retriever preferred a result, and that matters here more than usual: for
    "Antinous ringleader suitors" over a chunked book, keyword search ranks the
    right passage first with a score 7x the runner-up, while the neighbouring
    passage ranks first by vector. Each is rank 1 once and rank 2 once, so pure
    RRF scores them *exactly* equal and the winner is whatever order the dict
    happened to be in. Passing the keyword score as the tiebreak restores the
    magnitude that rank fusion discarded, for the rare-token case this project
    leans on hardest. Ids are the final tiebreak so the order is deterministic.
    """
    scores = reciprocal_rank_fusion(rankings, k, weights)
    tb = tiebreak or {}
    return sorted(scores.items(),
                  key=lambda kv: (-kv[1], -tb.get(kv[0], 0.0), kv[0]))


@runtime_checkable
class Retriever(Protocol):
    def index(self, doc_id: str, text: str) -> int:
        ...

    def search(self, doc_id: str, *, text: str, k: int = 8,
               expand: int = 1) -> list[Passage]:
        ...


class _BaseRetriever:
    """Chunking, storage of the source, and expansion -- shared by backends.

    The source text is kept beside the index because expansion slices from it.
    Concatenating overlapping chunks instead would duplicate every overlapping
    sentence and charge for the copies.
    """

    def __init__(self, directory: pathlib.Path | str = DEFAULT_INDEX_DIR,
                 embedder: Embedder | None = None, *,
                 target_chars: int | None = None):
        self.directory = pathlib.Path(directory)
        inner = embedder or HashEmbedder()
        self.embedder = CachedEmbedder(inner=inner,
                                       directory=self.directory / "embeddings")
        self.target_chars = target_chars
        self._sources: dict[str, str] = {}
        self._chunks: dict[str, list[Chunk]] = {}

    # ------------------------------------------------------------- storage

    def _doc_dir(self, doc_id: str) -> pathlib.Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id)
        return self.directory / "docs" / safe

    def _save_source(self, doc_id: str, text: str, chunks: list[Chunk]) -> None:
        d = self._doc_dir(doc_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.txt").write_text(text, encoding="utf-8")
        (d / "chunks.json").write_text(json.dumps({
            "doc_id": doc_id,
            "embedding_model": self.embedder.model,
            "chunks": [{"ordinal": c.ordinal, "start": c.start, "end": c.end}
                       for c in chunks],
        }, indent=2), encoding="utf-8")
        self._sources[doc_id] = text
        self._chunks[doc_id] = chunks

    def _load_source(self, doc_id: str) -> tuple[str, list[Chunk]]:
        if doc_id in self._sources:
            return self._sources[doc_id], self._chunks[doc_id]
        d = self._doc_dir(doc_id)
        if not (d / "source.txt").exists():
            raise KeyError(f"{doc_id!r} is not indexed; run index() first")
        text = (d / "source.txt").read_text(encoding="utf-8")
        meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(doc_id=doc_id, ordinal=c["ordinal"],
                        text=text[c["start"]:c["end"]],
                        start=c["start"], end=c["end"])
                  for c in meta["chunks"]]
        self._sources[doc_id] = text
        self._chunks[doc_id] = chunks
        return text, chunks

    def is_indexed(self, doc_id: str) -> bool:
        return (self._doc_dir(doc_id) / "source.txt").exists()

    # ------------------------------------------------------------- indexing

    # -------------------------------------------------------------- searching

    def rankings(self, doc_id: str, text: str, depth: int = 30,
                 ) -> tuple[list[str], list[str], dict[str, float]]:
        """The two rankings and the keyword strengths, before fusion.

        Exposed rather than buried inside `search` because the arms have to be
        measurable separately. "Hybrid retrieval works" is not a checkable
        claim; "keyword scored 5/5, vector 0/5, fused 3/5" is, and it was that
        measurement that caught fusion making the strong retriever worse.
        """
        raise NotImplementedError

    def search(self, doc_id: str, *, text: str, k: int = 8, expand: int = 1,
               weights: list[float] | None = None) -> list[Passage]:
        keyword, vector, strength = self.rankings(doc_id, text,
                                                  depth=max(k * 3, 10))
        if not keyword and not vector:
            return []
        ranked = fuse([keyword, vector], tiebreak=strength, weights=weights)
        return self._to_passages(doc_id, [(int(i), s) for i, s in ranked],
                                 k, expand)

    def _prepare(self, doc_id: str, text: str) -> tuple[list[Chunk], list[list[float]]]:
        kwargs = {"target_chars": self.target_chars} if self.target_chars else {}
        chunks = chunk_text(text, doc_id, **kwargs)
        vectors = self.embedder.embed([c.text for c in chunks]) if chunks else []
        self._save_source(doc_id, text, chunks)
        return chunks, vectors

    # ------------------------------------------------------------ expansion

    def _to_passages(self, doc_id: str, ranked: list[tuple[int, float]], k: int,
                     expand: int) -> list[Passage]:
        """Turn ranked chunk ordinals into expanded, merged passages.

        Expansion is what makes retrieval useful here rather than merely
        correct: the sentence containing a name rarely explains why the name
        matters, and the paragraphs around it usually do.

        `ranked` arrives best-first and that order is preserved through
        merging. Sorting the merged passages by score alone would throw the
        tiebreak away again -- expansion merges neighbours, so several distinct
        hits routinely collapse into one span carrying one score.
        """
        text, chunks = self._load_source(doc_id)
        if not chunks:
            return []
        by_ordinal = {c.ordinal: c for c in chunks}

        spans: list[tuple[int, int]] = []
        span_score: dict[tuple[int, int], float] = {}
        span_rank: dict[tuple[int, int], int] = {}
        span_ordinals: dict[tuple[int, int], set[int]] = {}
        for rank, (ordinal, score) in enumerate(ranked[:k]):
            group = [by_ordinal[o] for o in
                     range(ordinal - expand, ordinal + expand + 1)
                     if o in by_ordinal]
            if not group:
                continue
            span = (min(c.start for c in group), max(c.end for c in group))
            spans.append(span)
            span_score[span] = max(span_score.get(span, 0.0), score)
            span_rank[span] = min(span_rank.get(span, rank), rank)
            span_ordinals.setdefault(span, set()).update(c.ordinal for c in group)

        def inside(sp, start, end):
            return sp[0] >= start and sp[1] <= end

        passages = []
        for start, end in merge_spans(spans):
            parts = [sp for sp in span_score if inside(sp, start, end)]
            passages.append((
                min(span_rank[sp] for sp in parts),
                Passage(doc_id=doc_id, text=text[start:end], start=start,
                        end=end,
                        score=max(span_score[sp] for sp in parts),
                        ordinals=tuple(sorted(
                            {o for sp in parts for o in span_ordinals[sp]})))))
        passages.sort(key=lambda pair: pair[0])
        return [p for _, p in passages]


class MemoryRetriever(_BaseRetriever):
    """In-process backend. The offline double, and the reference implementation.

    Keyword scoring here is deliberately crude -- term overlap, not BM25. It is
    a stand-in that makes the interface testable without pulling in a database,
    not a search engine. `LanceRetriever` is the one meant for real use.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectors: dict[str, dict[int, list[float]]] = {}

    def index(self, doc_id: str, text: str) -> int:
        chunks, vectors = self._prepare(doc_id, text)
        self._vectors[doc_id] = {c.ordinal: v for c, v in zip(chunks, vectors)}
        return len(chunks)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())}

    def rankings(self, doc_id: str, text: str, depth: int = 30,
                 ) -> tuple[list[str], list[str], dict[str, float]]:
        _, chunks = self._load_source(doc_id)
        if not chunks:
            return [], [], {}
        if doc_id not in self._vectors:                # loaded from disk
            self._vectors[doc_id] = {
                c.ordinal: v for c, v in
                zip(chunks, self.embedder.embed([c.text for c in chunks]))}

        query_terms = self._terms(text)
        overlap = {str(c.ordinal): float(len(query_terms & self._terms(c.text)))
                   for c in chunks}
        keyword_ranking = sorted((i for i, n in overlap.items() if n),
                                 key=lambda i: -overlap[i])[:depth]

        qvec = self.embedder.embed([text])[0]
        vectors = self._vectors[doc_id]

        def cosine(v: list[float]) -> float:
            return sum(a * b for a, b in zip(qvec, v))

        vector_ranking = [str(c.ordinal) for c in
                          sorted(chunks, key=lambda c: cosine(vectors[c.ordinal]),
                                 reverse=True)][:depth]

        return keyword_ranking, vector_ranking, overlap


class LanceRetriever(_BaseRetriever):
    """LanceDB backend: embedded, file-on-disk, no server and no cloud calls.

    One table per document rather than one table with a `doc_id` filter. Tasks
    here are per-document, and a table boundary makes cross-document leakage
    impossible instead of merely unlikely.

    LanceDB has hybrid search with reciprocal-rank fusion built in, and this
    deliberately does not use it. Its `_relevance_score` is the fused number
    with the component scores already discarded, so the keyword magnitude
    needed to break ties (see `fuse`) is gone by the time we see it. Running
    the two searches separately costs one extra query and gives both backends
    the *same* fusion code, which is what makes their results comparable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._db = None

    def _connect(self):
        if self._db is None:
            import lancedb                # lazy: an optional dependency
            (self.directory / "lance").mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.directory / "lance"))
        return self._db

    @staticmethod
    def _table_name(doc_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", doc_id)

    def index(self, doc_id: str, text: str) -> int:
        chunks, vectors = self._prepare(doc_id, text)
        if not chunks:
            return 0
        from lancedb.index import FTS

        db = self._connect()
        table = db.create_table(
            self._table_name(doc_id),
            data=[{"ordinal": c.ordinal, "text": c.text, "vector": v}
                  for c, v in zip(chunks, vectors)],
            mode="overwrite")
        table.create_index("text", config=FTS(), replace=True)
        return len(chunks)

    def rankings(self, doc_id: str, text: str, depth: int = 30,
                 ) -> tuple[list[str], list[str], dict[str, float]]:
        db = self._connect()
        table = db.open_table(self._table_name(doc_id))

        keyword_rows = (table.search(_fts_safe(text), query_type="fts")
                        .limit(depth).to_list())
        keyword_ranking = [str(r["ordinal"]) for r in keyword_rows]
        # BM25 score, kept for the tiebreak the fused ranking cannot express.
        strength = {str(r["ordinal"]): float(r.get("_score", 0.0))
                    for r in keyword_rows}

        qvec = self.embedder.embed([text])[0]
        vector_rows = (table.search(qvec, query_type="vector")
                       .limit(depth).to_list())
        vector_ranking = [str(r["ordinal"]) for r in vector_rows]

        return keyword_ranking, vector_ranking, strength


def _fts_safe(query: str) -> str:
    """Strip characters the full-text parser treats as operators.

    A query lifted from a gold output can contain quotes, colons and brackets.
    Those are query syntax, not search terms, and letting them through turns a
    retrieval into a parse error at the worst moment.
    """
    cleaned = re.sub(r"[^\w\s]", " ", query)
    return " ".join(cleaned.split()) or query.strip()


def build(backend: str = "memory", directory: pathlib.Path | str = DEFAULT_INDEX_DIR,
          *, embedder: Embedder | None = None, **kwargs) -> Retriever:
    """Pick a backend by name. `memory` needs nothing; `lance` needs lancedb."""
    if backend == "memory":
        return MemoryRetriever(directory, embedder, **kwargs)
    if backend == "lance":
        return LanceRetriever(directory, embedder, **kwargs)
    raise ValueError(f"unknown retrieval backend {backend!r} "
                     f"(expected 'memory' or 'lance')")
