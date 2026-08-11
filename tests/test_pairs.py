"""Integrity checks for the corpus and pair set. No network, no API key.

These guard the properties the induction system will rely on: that the pairs
are well-formed, that every output was generated against the corpus currently
in the tree, and that the prompts carrying explicit constraints have outputs
that actually satisfy them.
"""
import hashlib
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "agentic-ai-survey.md"
PAIRS = ROOT / "data" / "pairs" / "agentic-ai-survey.jsonl"

CATEGORIES = {"fetch", "summarize", "reason"}
SHAPES = {"prose", "list", "table"}
REQUIRED = {
    "id", "category", "input_ref", "input_sha256",
    "target_prompt", "output", "output_shape", "gold_sections",
    "prompt_group", "is_negative",
}
OPTIONAL = {"input_span"}          # control pairs read a slice of the corpus


@pytest.fixture(scope="module")
def corpus():
    return CORPUS.read_text()


@pytest.fixture(scope="module")
def pairs():
    return [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]


def by_id(pairs, pair_id):
    return next(p for p in pairs if p["id"] == pair_id)


# --- shape -----------------------------------------------------------------

def test_pair_count(pairs):
    standalone = [p for p in pairs if p["prompt_group"] == p["id"]]
    controls = [p for p in pairs if p["prompt_group"] != p["id"]]
    assert len(standalone) == 20
    assert len(controls) == 9        # three control sets of three
    assert len(pairs) == 29


def test_category_balance(pairs):
    standalone = [p for p in pairs if p["prompt_group"] == p["id"]]
    counts = {c: sum(1 for p in standalone if p["category"] == c) for c in CATEGORIES}
    assert counts == {"fetch": 7, "summarize": 6, "reason": 7}


def test_records_well_formed(pairs):
    for p in pairs:
        assert REQUIRED <= set(p), f"{p.get('id')}: missing fields"
        assert set(p) <= REQUIRED | OPTIONAL, f"{p.get('id')}: unexpected fields"
        assert p["category"] in CATEGORIES
        assert p["output_shape"] in SHAPES
        assert p["target_prompt"].strip()
        assert p["output"].strip()
        # control pairs read a slice, so section labels do not apply to them
        assert p["gold_sections"] or "input_span" in p


def test_ids_unique(pairs):
    ids = [p["id"] for p in pairs]
    assert len(set(ids)) == len(ids)


# --- provenance ------------------------------------------------------------

def test_outputs_match_the_corpus_in_tree(pairs, corpus):
    """Every output was generated against this exact corpus."""
    digest = hashlib.sha256(corpus.encode()).hexdigest()
    stale = [p["id"] for p in pairs if p["input_sha256"] != digest]
    assert not stale, f"corpus changed since these were generated: {stale}"


def test_input_ref_resolves(pairs):
    for p in pairs:
        assert (ROOT / p["input_ref"]).exists()


def test_control_spans_are_in_range(pairs, corpus):
    for p in pairs:
        if "input_span" in p:
            a, b = p["input_span"]
            assert 0 <= a < b <= len(corpus), p["id"]
            assert corpus[a:b].strip(), p["id"]


def test_negative_controls_answer_NA(pairs):
    for p in pairs:
        if p["is_negative"]:
            assert p["output"].strip() == "NA", p["id"]


def test_gold_sections_resolve_to_real_headings(pairs, corpus):
    headings = re.findall(r"^#{1,4}\s+(.+)$", corpus, re.M)
    numbers = {h.split()[0] for h in headings if re.match(r"^\d", h)}
    for p in pairs:
        for section in p["gold_sections"]:
            if section.startswith("Table") or section == "Abstract":
                continue
            assert section.split()[0] in numbers, f"{p['id']}: no such section {section!r}"


def test_jsonl_matches_its_builder():
    """data/pairs/*.jsonl is generated -- fail if it drifted from the source."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_pairs.py"), "--check"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


# --- the constraints the prompts actually state ----------------------------

def test_summ_06_is_exactly_three_sentences(pairs):
    out = by_id(pairs, "summ-06")["output"]
    assert len(re.findall(r"[.!?](?:\s|$)", out)) == 3


def test_summ_01_respects_its_word_budget(pairs):
    assert len(by_id(pairs, "summ-01")["output"].split()) <= 150


def test_summ_05_second_column_stays_under_25_words(pairs):
    rows = by_id(pairs, "summ-05")["output"].splitlines()[2:]
    for row in rows:
        assert len(row.split("|")[2].split()) <= 25, row


def test_fetch_01_lists_every_autonomy_level_in_order(pairs):
    levels = re.findall(r"\|\s*(\d) - ", by_id(pairs, "fetch-01")["output"])
    assert levels == ["0", "1", "2", "3", "4"]


@pytest.mark.parametrize("pair_id,needle", [
    ("fetch-02", "4000"),
    ("fetch-02", "seven items"),
    ("fetch-03", "2029"),
    ("fetch-03", "180 years"),
    ("fetch-04", "Group Relative Policy Optimization"),
    ("fetch-06", "30 surveys"),
    ("fetch-07", "Workarena++"),
])
def test_quoted_facts_appear_in_the_source(corpus, pair_id, needle):
    """Guards against a fetch output asserting something the document never said."""
    assert needle in corpus, f"{pair_id} cites {needle!r}, absent from the corpus"


def test_table_outputs_are_valid_markdown_tables(pairs):
    for p in pairs:
        if p["output_shape"] != "table":
            continue
        lines = [l for l in p["output"].splitlines() if l.startswith("|")]
        assert len(lines) >= 3, p["id"]
        assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}, p["id"]
        width = lines[0].count("|")
        assert all(l.count("|") == width for l in lines), f"{p['id']}: ragged table"
