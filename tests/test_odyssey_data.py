"""Integrity checks for the altered-Odyssey test set. No network, no API key.

The set has an unusual property that ordinary data tests would miss: its value
depends on the gold answers being *different from what Homer says*. A variant
whose answer happens to match the original tests nothing, because a model can
reach it from memory without reading the passage at all.

So alongside the usual well-formedness checks, these assert that the
alterations actually landed and actually changed the answer.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "source" / "odyssey-pg1727.txt"
CORPUS_DIR = ROOT / "data" / "corpus" / "odyssey"
PAIRS = ROOT / "data" / "pairs" / "odyssey.jsonl"

CATEGORIES = {"fetch", "summarize", "reason"}
SHAPES = {"prose", "list", "table"}
REQUIRED = {
    "id", "category", "input_ref", "input_sha256", "target_prompt", "output",
    "output_shape", "gold_sections", "prompt_group", "is_negative",
    "source_book", "alteration",
}

GODS = ["Jove", "Saturn", "Minerva", "Neptune", "Apollo", "Calypso", "Circe",
        "Mercury", "Hyperion", "Juno", "Venus", "Mars", "Vulcan", "Diana"]

# What Homer actually says, for the facts this set alters. Every one of these
# must be absent from the gold answers, or the variant is not testing anything.
HOMERIC = {
    "hero": "Ulysses",
    "father": "Laertes",
    "islands": ["Dulichium", "Same", "Zacynthus"],
    "wood": "olive",
    "jars": 12,
    "talents": 7,
    "men": 12,
    "ear_stuffing": "wax",
    "guards": ["Eurylochus", "Perimedes"],
}


@pytest.fixture(scope="module")
def records():
    return [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def corpus():
    return {f.stem: f.read_text() for f in CORPUS_DIR.glob("*.md")}


def by_group(records, gid):
    return [r for r in records if r["prompt_group"] == gid]


# --- well-formedness --------------------------------------------------------

def test_records_well_formed(records):
    for r in records:
        assert set(r) == REQUIRED, f"{r.get('id')}: field set differs"
        assert r["category"] in CATEGORIES
        assert r["output_shape"] in SHAPES
        assert r["alteration"] in {"minor", "major"}
        assert r["target_prompt"].strip() and r["output"].strip()


def test_ids_unique(records):
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)


def test_every_input_file_exists_and_is_used(records, corpus):
    referenced = set()
    for r in records:
        path = ROOT / r["input_ref"]
        assert path.exists(), r["input_ref"]
        referenced.add(path.stem)
    assert referenced == set(corpus), "some passages are unused or missing"


def test_passages_match_their_recorded_digest(records):
    """Guards against a passage being edited without its answers being redone."""
    import hashlib
    for r in records:
        text = (ROOT / r["input_ref"]).read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        assert digest == r["input_sha256"], f"{r['id']}: passage changed"


def test_generated_files_are_current():
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "build_odyssey.py"),
                        "--check"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_source_is_the_public_domain_gutenberg_text():
    head = SOURCE.read_text(encoding="utf-8")[:2000]
    assert "Project Gutenberg" in head
    assert "Samuel Butler" in head          # the translation the passages come from


# --- the alterations landed -------------------------------------------------

def test_renamed_passages_do_not_leak_the_original_name(corpus):
    for name, text in corpus.items():
        if name.startswith("naming-"):
            assert "Ulysses" not in text, f"{name} still says Ulysses"
            assert "Laertes" not in text, f"{name} still says Laertes"


def test_each_passage_differs_from_the_untouched_source(corpus):
    """Every passage must actually be altered somewhere.

    Compares whole paragraphs, not the opening: several alterations sit in the
    middle of a paragraph, so a prefix check would call them untouched. Only
    one paragraph need differ -- a passage may carry unaltered context around
    the sentence that was changed, and that context is doing useful work.
    """
    raw = " ".join(SOURCE.read_text(encoding="utf-8").split())
    for name, text in corpus.items():
        paras = [" ".join(p.split()) for p in text.split("\n\n")]
        altered = [p for p in paras if p not in raw]
        assert altered, f"{name} is verbatim Homer -- nothing was changed"


def test_major_alterations_exist_and_are_marked(records):
    majors = {r["input_ref"] for r in records if r["alteration"] == "major"}
    assert len(majors) >= 3, "the set needs outcome-level changes, not only details"


# --- the alterations defeat memory ------------------------------------------

def test_no_gold_answer_matches_what_homer_says(records):
    """The point of the set: a model answering from recall must be wrong."""
    checks = {
        "ody-speaker-name": HOMERIC["hero"],
        "ody-speaker-father": HOMERIC["father"],
        "ody-stake-material": HOMERIC["wood"],
    }
    for gid, homeric in checks.items():
        for r in by_group(records, gid):
            assert homeric.lower() not in r["output"].lower(), (
                f"{r['id']}: gold answer matches Homer, so recall alone passes it")


def test_altered_numbers_differ_from_homer(records):
    for r in by_group(records, "ody-gift-quantities"):
        jars = int(re.search(r"jars=(\d+)", r["output"]).group(1))
        talents = int(re.search(r"talents=(\d+)", r["output"]).group(1))
        assert jars != HOMERIC["jars"], r["id"]
        assert talents != HOMERIC["talents"], r["id"]


def test_altered_islands_differ_from_homer(records):
    for r in by_group(records, "ody-neighbour-islands"):
        for isle in HOMERIC["islands"]:
            assert isle not in r["output"], r["id"]


def test_sirens_details_differ_from_homer(records):
    for r in by_group(records, "ody-sirens-precaution"):
        assert HOMERIC["ear_stuffing"] not in r["output"], r["id"]


def test_bow_outcome_contradicts_the_original(records):
    """In Homer, Ulysses strings it and the arrow passes every axe."""
    for r in by_group(records, "ody-bow-outcome"):
        low = r["output"].lower()
        assert "telemachus" in low or "nobody" in low, r["id"]


def test_answers_within_a_group_are_not_all_the_same(records):
    """If every variant shares an answer, the group cannot distinguish reading
    from recall."""
    groups = {r["prompt_group"] for r in records}
    for gid in groups:
        rows = by_group(records, gid)
        positives = [r["output"] for r in rows if not r["is_negative"]]
        if len(positives) > 1:
            assert len(set(positives)) > 1, f"{gid}: every variant answers alike"


# --- the answers are true of their passage ----------------------------------

def test_extracted_names_appear_in_their_passage(records):
    for gid in ("ody-speaker-name", "ody-speaker-father"):
        for r in by_group(records, gid):
            text = (ROOT / r["input_ref"]).read_text()
            assert r["output"] in text, f"{r['id']}: answer absent from passage"


def test_extracted_wood_appears_in_its_passage(records):
    for r in by_group(records, "ody-stake-material"):
        text = (ROOT / r["input_ref"]).read_text()
        assert f"green {r['output']} wood" in text, r["id"]


def test_island_lists_appear_in_order_in_their_passage(records):
    for r in by_group(records, "ody-neighbour-islands"):
        text = (ROOT / r["input_ref"]).read_text()
        names = [l.lstrip("- ").strip() for l in r["output"].splitlines()]
        positions = [text.find(n) for n in names]
        assert all(p >= 0 for p in positions), r["id"]
        assert positions == sorted(positions), f"{r['id']}: order differs"


def test_negative_answers_are_NA_and_their_passage_names_no_god(records):
    negatives = [r for r in records if r["is_negative"]]
    assert negatives, "the set needs at least one negative"
    for r in negatives:
        assert r["output"].strip() == "NA", r["id"]
        text = (ROOT / r["input_ref"]).read_text()
        found = [g for g in GODS if re.search(rf"\b{g}\b", text)]
        assert not found, f"{r['id']} names {found} but answers NA"


def test_positive_god_answers_name_every_god_present(records):
    for r in by_group(records, "ody-named-gods"):
        if r["is_negative"]:
            continue
        text = (ROOT / r["input_ref"]).read_text()
        present = {g for g in GODS if re.search(rf"\b{g}\b", text)}
        listed = {l.lstrip("- ").strip() for l in r["output"].splitlines()}
        assert listed == present, f"{r['id']}: listed {listed}, passage has {present}"


# --- stated output constraints hold -----------------------------------------

def test_two_sentence_answers_have_two_sentences(records):
    for r in by_group(records, "ody-escape-method"):
        assert len(re.findall(r"[.!?](?:\s|$)", r["output"])) == 2, r["id"]


def test_three_sentence_answers_have_three_sentences(records):
    for r in by_group(records, "ody-three-sentence-summary"):
        assert len(re.findall(r"[.!?](?:\s|$)", r["output"])) == 3, r["id"]


def test_briefings_have_exactly_three_bullets(records):
    for r in by_group(records, "ody-captain-briefing"):
        bullets = [l for l in r["output"].splitlines() if l.startswith("- ")]
        assert len(bullets) == 3, r["id"]


def test_table_answers_are_valid_markdown_tables(records):
    for r in records:
        if r["output_shape"] != "table":
            continue
        lines = [l for l in r["output"].splitlines() if l.startswith("|")]
        assert len(lines) >= 3, r["id"]
        assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}, r["id"]
        width = lines[0].count("|")
        assert all(l.count("|") == width for l in lines), f"{r['id']}: ragged table"


def test_fixed_form_answers_match_their_stated_shape(records):
    for r in by_group(records, "ody-gift-quantities"):
        assert re.fullmatch(r"jars=\d+, talents=\d+", r["output"]), r["id"]
    for r in by_group(records, "ody-landing-party"):
        assert re.fullmatch(r"men=\d+, water=\d+", r["output"]), r["id"]
    for r in by_group(records, "ody-sirens-precaution"):
        assert re.fullmatch(r"ears=\w+; bound by=\w+ and \w+", r["output"]), r["id"]
