"""Tests for chunking, embedding cache and hybrid retrieval. No key, no spend.

Two things are being tested and they are worth keeping apart:

* the **plumbing** -- offsets survive chunking, the cache stops repeat calls,
  expansion merges rather than duplicates, fusion combines rankings. All of
  that is exactly checkable offline.
* whether retrieval **finds the right passage**. The offline embedder hashes
  words, so it has real lexical similarity but no semantics. Tests here use
  queries that share vocabulary with their target; a paraphrase-only query is
  precisely what the offline double cannot do, and asserting otherwise would be
  asserting something untrue.
"""
from __future__ import annotations

import json

import pytest

from reversed_prompts import retrieval
from reversed_prompts.chunking import chunk_text, merge_spans
from reversed_prompts.embedding import (CachedEmbedder, HashEmbedder,
                                        resolve_embedding_model)
from reversed_prompts.retrieval import (LanceRetriever, MemoryRetriever,
                                        reciprocal_rank_fusion)

try:                                   # optional dependency; the rest runs without it
    import lancedb                     # noqa: F401
    HAS_LANCE = True
except ImportError:                    # pragma: no cover - depends on install
    HAS_LANCE = False

needs_lance = pytest.mark.skipif(not HAS_LANCE,
                                 reason="lancedb not installed: pip install -e '.[rag]'")


BOOK = """\
Sing to me of the man, Muse, the man of twists and turns.

Telemachus sat brooding among the suitors in the great hall, and none of them
gave him honour. They ate his substance and drank his wine.

Antinous was the ringleader of the suitors, the most insolent of them all, and
it was he who first mocked the beggar at the gate.

The ship sailed past the wandering rocks at dawn, and the crew stopped their
ears with clay against the singing.

Penelope wove the shroud by day and unravelled it again by night, and so she
put off the day of her choosing for three long years.
"""


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"


# ------------------------------------------------------------------ chunking

def test_chunk_offsets_slice_back_to_the_original_text():
    """The whole value of an offset is that it can be checked. If this ever
    fails, every citation the system emits is fiction."""
    for c in chunk_text(BOOK, "book", target_chars=200):
        assert BOOK[c.start:c.end] == c.text


def test_chunks_cover_the_document_in_order():
    chunks = chunk_text(BOOK, "book", target_chars=200)
    assert chunks[0].start == 0
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks == sorted(chunks, key=lambda c: c.start)
    assert chunks[-1].end >= len(BOOK.rstrip())


def test_an_oversized_paragraph_is_split_at_sentence_boundaries():
    para = " ".join(f"Sentence number {i} says something." for i in range(60))
    chunks = chunk_text(para, "d", target_chars=200, overlap_chars=0)
    assert len(chunks) > 1
    assert all(c.text.strip().endswith(".") for c in chunks)


def test_a_paragraph_with_no_sentence_breaks_still_terminates():
    """A pathological input must not loop forever, even if it cannot be split
    into pieces under the target."""
    chunks = chunk_text("x" * 5000, "d", target_chars=100)
    assert chunks
    assert "".join(c.text for c in chunks).startswith("x")


def test_empty_and_blank_documents_produce_no_chunks():
    assert chunk_text("", "d") == []
    assert chunk_text("   \n\n  ", "d") == []


def test_merge_spans_collapses_overlaps_and_keeps_gaps():
    assert merge_spans([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]
    assert merge_spans([]) == []


# ----------------------------------------------------------------- embedding

def test_the_embedding_model_has_its_own_env_var():
    """A blanket chat-model setting must not become the embedding model -- an
    index built with the wrong model is incomparable to every other index."""
    from reversed_prompts.embedding import DEFAULT_EMBEDDING_MODEL
    assert resolve_embedding_model(env={"REVPROMPT_MODEL": "gpt-something"}) \
        == DEFAULT_EMBEDDING_MODEL
    assert resolve_embedding_model(
        env={"REVPROMPT_EMBEDDING_MODEL": "other"}) == "other"


def test_hash_embedder_puts_texts_that_share_words_closer_together():
    e = HashEmbedder()
    a, b, c = e.embed(["the suitors ate his substance",
                       "the suitors drank his wine",
                       "the ship sailed past the rocks"])
    def dot(x, y): return sum(p * q for p, q in zip(x, y))
    assert dot(a, b) > dot(a, c)


def test_hash_embedder_is_deterministic_across_instances():
    assert HashEmbedder().embed(["one text"]) == HashEmbedder().embed(["one text"])


def test_cache_only_sends_uncached_texts_to_the_inner_embedder(tmp_path):
    class Counting(HashEmbedder):
        seen: list[list[str]] = []
        def embed(self, texts):
            Counting.seen.append(list(texts))
            return super().embed(texts)

    Counting.seen = []
    cache = CachedEmbedder(inner=Counting(), directory=tmp_path / "emb")
    cache.embed(["alpha", "beta"])
    cache.embed(["beta", "gamma"])
    assert Counting.seen == [["alpha", "beta"], ["gamma"]]
    assert cache.hits == 1 and cache.misses == 3


def test_cache_survives_a_new_process(tmp_path):
    first = CachedEmbedder(inner=HashEmbedder(), directory=tmp_path / "emb")
    want = first.embed(["persisted text"])
    second = CachedEmbedder(inner=HashEmbedder(), directory=tmp_path / "emb")
    assert second.embed(["persisted text"]) == want
    assert second.hits == 1 and second.misses == 0


def test_changing_the_model_changes_the_cache_key(tmp_path):
    """Vectors from two models must never be silently mixed in one index."""
    a = CachedEmbedder(inner=HashEmbedder(model="m1"), directory=tmp_path / "e")
    b = CachedEmbedder(inner=HashEmbedder(model="m2"), directory=tmp_path / "e")
    a.embed(["same text"])
    b.embed(["same text"])
    assert b.misses == 1, "m2 must not read m1's vector"


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    cache = CachedEmbedder(inner=HashEmbedder(), directory=tmp_path / "emb")
    cache.embed(["text"])
    for path in (tmp_path / "emb").rglob("*.json"):
        path.write_text("{ not json", encoding="utf-8")
    cache._memo.clear()
    assert cache.embed(["text"])          # recomputed rather than raising


# -------------------------------------------------------------------- fusion

def test_fusion_rewards_agreement_between_the_two_rankings():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "b"]])
    assert scores["a"] > scores["c"] > scores["b"]


def test_fusion_keeps_an_id_only_one_ranking_found():
    scores = reciprocal_rank_fusion([["a"], ["b"]])
    assert set(scores) == {"a", "b"}


# ----------------------------------------------------------------- retrieval

def test_search_finds_the_passage_containing_a_rare_name(index_dir):
    r = MemoryRetriever(index_dir, target_chars=200)
    r.index("book", BOOK)
    passages = r.search("book", text="Antinous", k=3, expand=0)
    assert "Antinous" in passages[0].text


def test_a_passage_slices_back_to_the_source(index_dir):
    r = MemoryRetriever(index_dir, target_chars=200)
    r.index("book", BOOK)
    for p in r.search("book", text="Penelope shroud", k=3):
        assert BOOK[p.start:p.end] == p.text


def test_expansion_widens_a_hit_without_duplicating_text(index_dir):
    """Overlapping chunks must be merged, not concatenated -- otherwise the
    same sentence is sent to the model several times and billed each time."""
    r = MemoryRetriever(index_dir, target_chars=200)
    r.index("book", BOOK)
    narrow = r.search("book", text="Antinous ringleader", k=1, expand=0)
    wide = r.search("book", text="Antinous ringleader", k=1, expand=2)
    assert len(wide[0].text) > len(narrow[0].text)
    assert wide[0].text.count("Antinous was the ringleader") == 1


def test_searching_an_unindexed_document_says_so(index_dir):
    with pytest.raises(KeyError, match="not indexed"):
        MemoryRetriever(index_dir).search("missing", text="anything")


def test_an_index_can_be_reopened_without_re_embedding(index_dir):
    MemoryRetriever(index_dir, target_chars=200).index("book", BOOK)
    reopened = MemoryRetriever(index_dir, target_chars=200)
    assert reopened.is_indexed("book")
    assert reopened.search("book", text="wandering rocks", k=2)
    assert reopened.embedder.misses <= 1, "only the query should be new"


def test_the_index_records_which_model_built_it(index_dir):
    r = MemoryRetriever(index_dir, target_chars=200)
    r.index("book", BOOK)
    meta = json.loads((r._doc_dir("book") / "chunks.json")
                      .read_text(encoding="utf-8"))
    assert meta["embedding_model"] == r.embedder.model


def test_full_text_queries_are_stripped_of_parser_syntax():
    """Gold outputs contain quotes and colons. Those are query operators, not
    search terms, and letting them through turns retrieval into a parse error."""
    assert retrieval._fts_safe('who is "Antinous": the ringleader?') \
        == "who is Antinous the ringleader"


def test_build_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown retrieval backend"):
        retrieval.build("elasticsearch")


# ------------------------------------------------------- the LanceDB backend

@needs_lance
def test_lance_backend_does_hybrid_search_with_no_server(index_dir):
    r = LanceRetriever(index_dir, target_chars=200)
    assert r.index("book", BOOK) > 0
    passages = r.search("book", text="Antinous ringleader suitors", k=3, expand=0)
    assert passages and "Antinous" in passages[0].text


@needs_lance
def test_lance_and_memory_backends_agree_on_an_easy_query(index_dir):
    """Not a claim that they rank identically -- only that the obvious answer
    is the obvious answer under both, so the interface is really shared."""
    a = MemoryRetriever(index_dir / "mem", target_chars=200)
    b = LanceRetriever(index_dir / "lance", target_chars=200)
    for r in (a, b):
        r.index("book", BOOK)
        top = r.search("book", text="Penelope wove the shroud", k=2, expand=0)
        assert "Penelope" in top[0].text


@needs_lance
def test_lance_passages_also_slice_back_to_the_source(index_dir):
    r = LanceRetriever(index_dir, target_chars=200)
    r.index("book", BOOK)
    for p in r.search("book", text="clay against the singing", k=3):
        assert BOOK[p.start:p.end] == p.text


def test_fuse_breaks_an_exact_tie_with_the_keyword_score():
    """The case that motivated the tiebreak: each candidate is rank 1 in one
    retriever and rank 2 in the other, so pure RRF scores them identically.
    Keyword magnitude is the signal that should decide it."""
    rankings = [["antinous", "telemachus"], ["telemachus", "antinous"]]
    plain = reciprocal_rank_fusion(rankings)
    assert plain["antinous"] == plain["telemachus"], "the tie is real"

    ordered = retrieval.fuse(rankings, tiebreak={"antinous": 3.9,
                                                 "telemachus": 0.5})
    assert [i for i, _ in ordered][0] == "antinous"


def test_fuse_is_deterministic_when_nothing_breaks_the_tie():
    rankings = [["b", "a"], ["a", "b"]]
    assert retrieval.fuse(rankings) == retrieval.fuse(rankings)


def test_weighting_can_stop_a_weak_retriever_dragging_a_strong_one_down():
    """Measured on the full book: equal-weight fusion scored 3/5 on a query
    where keyword alone scored 5/5, because the vector ranking was noise. The
    weight is the lever for that, so it has to actually move the result."""
    keyword = ["right", "also-right"]
    noise = ["wrong", "wrong-too"]

    equal = [i for i, _ in retrieval.fuse([keyword, noise])]
    assert equal.index("wrong") < equal.index("also-right"), "the failure mode"

    trusted = [i for i, _ in retrieval.fuse([keyword, noise], weights=[3.0, 1.0])]
    assert trusted[:2] == ["right", "also-right"]
