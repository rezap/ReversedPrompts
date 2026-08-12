"""Tests for prompt recovery. No API key, no network, no spend.

Two things are being tested and they are easy to confuse:

* the machinery -- grouping, measurement, scoring, the propose/critique/revise
  loop. All of that is checkable offline.
* the *quality* of a recovered prompt. That needs a real model. The offline
  double writes a crude instruction by restating measurements, so nothing here
  asserts that recovery produces a good prompt -- only that the path runs and
  the scores mean what they claim.
"""
from __future__ import annotations

import pytest

from reversed_prompts import ingest, prompts, recover
from reversed_prompts.client import BudgetExceeded, ObedientClient, ScriptedClient
from reversed_prompts.features import extract, infer_shape, strip_markup, syllables
from reversed_prompts.ingest import Pair, PromptGroup
from reversed_prompts.metrics import Scale, describe_gap, tier1, worst_features
from reversed_prompts.parsing import parse_clauses
from reversed_prompts.similarity import (PromptScore, contamination, lexical,
                                         score_prompt)


@pytest.fixture(scope="module")
def pairs():
    return ingest.load()


@pytest.fixture(scope="module")
def groups():
    return ingest.load_groups()


@pytest.fixture(scope="module")
def scale(pairs):
    return recover.fit_scale(pairs)


def make_pair(**kw) -> Pair:
    base = dict(id="x", category="fetch", input_text="doc text here",
                output="an output", target_prompt="do the thing")
    return Pair(**{**base, **kw})


# ----------------------------------------------------------------- grouping

def test_every_pair_lands_in_exactly_one_group(pairs, groups):
    assert sum(len(g) for g in groups) == len(pairs)
    ids = [p.id for g in groups for p in g.pairs]
    assert sorted(ids) == sorted(p.id for p in pairs)


def test_every_group_shares_one_gold_prompt(groups):
    for g in groups:
        assert len({p.target_prompt for p in g.pairs}) == 1, g.id


def test_every_group_spans_several_inputs(groups):
    """Each instruction is applied across variants whose answers differ, so a
    prompt that merely describes one answer cannot pass."""
    assert all(g.is_control for g in groups)
    assert len(groups) >= 10


def test_at_least_one_group_has_a_negative(groups):
    """A negative is what catches a prompt that describes the answer it saw."""
    assert sum(g.has_negative for g in groups) >= 1
    for g in groups:
        for p in g.pairs:
            if p.is_negative:
                assert p.output.strip() == "NA", p.id


def test_control_inputs_actually_differ(groups):
    for g in (x for x in groups if x.is_control):
        texts = {p.input_text for p in g.pairs}
        assert len(texts) == len(g.pairs), f"{g.id}: inputs are not distinct"


def test_negative_inputs_really_lack_the_content(groups):
    """If a 'negative' input named a god, NA would be the wrong gold."""
    import re as _re
    gods = ["Jove", "Saturn", "Minerva", "Neptune", "Apollo", "Calypso",
            "Circe", "Mercury", "Hyperion", "Juno", "Venus"]
    g = next(x for x in groups if x.id == "ody-named-gods")
    for p in g.pairs:
        if p.is_negative:
            found = [n for n in gods if _re.search(r"\b%s\b" % n, p.input_text)]
            assert not found, f"{p.id} names {found} but its gold answer is NA"


def test_group_rejects_conflicting_gold_prompts():
    a = make_pair(id="a", prompt_group="g", target_prompt="do X")
    b = make_pair(id="b", prompt_group="g", target_prompt="do Y")
    with pytest.raises(ValueError, match="different gold prompts"):
        ingest.group([a, b])


def test_each_variant_is_its_own_passage(groups):
    """Variants must differ in text, or the group tests nothing."""
    for g in groups:
        texts = {p.input_text for p in g.pairs}
        assert len(texts) == len(g.pairs), f"{g.id}: variants are identical"


# ------------------------------------------------------------------ parsing

def test_parse_clauses_tolerates_what_models_actually_return():
    assert parse_clauses('["a", "b"]') == ["a", "b"]
    assert parse_clauses('Sure!\n```json\n["a"]\n```') == ["a"]
    assert parse_clauses("- first\n- second") == ["first", "second"]
    assert parse_clauses("1. first\n2. second") == ["first", "second"]
    assert parse_clauses("") == []
    assert parse_clauses("   ") == []


@pytest.mark.parametrize("text", ["No changes needed.", "none", "Nothing to change",
                                  "already correct"])
def test_no_changes_parses_as_empty_not_as_a_change(text):
    """Otherwise 'nothing to fix' becomes a bogus instruction and the loop churns."""
    assert parse_clauses(text) == []


# --------------------------------------------------------------- similarity

def test_lexical_similarity_bounds():
    assert lexical("extract the author names", "extract the author names") == 1.0
    assert lexical("extract author names", "compute the checksum") == 0.0
    mid = lexical("extract the author names, NA if none",
                  "pull out author names; say NA when absent")
    assert 0.0 < mid < 1.0


def test_lexical_ignores_stopwords_and_case():
    assert lexical("Extract THE author names", "extract author names") == 1.0


def test_contamination_catches_a_prompt_carrying_the_answer():
    answer = "Ray Kurzweil predicted a fifty percent chance by twenty twenty nine"
    smuggled = ("List the predictions, for example Ray Kurzweil predicted a fifty "
                "percent chance by twenty twenty nine")
    clean = "List every named person and their prediction. NA if none."
    assert contamination(smuggled, answer) > 0.1
    assert contamination(clean, answer) == 0.0


def test_contamination_is_penalised_not_averaged():
    dirty = PromptScore(judged=0.9, lexical=0.9, contamination=0.5)
    clean = PromptScore(judged=0.6, lexical=0.6, contamination=0.0)
    assert clean.combined > dirty.combined
    assert "CONTAMINATED" in dirty.line()


def test_score_prompt_uses_the_worst_output(groups):
    g = next(x for x in groups if x.is_control)
    s = score_prompt(ScriptedClient(default="8"), "some instruction",
                     g.gold_prompt, [p.output for p in g.pairs])
    assert s.judged == pytest.approx(0.8)
    assert 0.0 <= s.contamination <= 1.0


# ------------------------------------------------------- system's own prompts

def test_system_prompts_are_versioned_and_non_empty():
    assert prompts.VERSION
    for name in ("PRODUCER", "CRITIC", "REVISER", "SIMILARITY_JUDGE", "EXECUTOR"):
        assert getattr(prompts, name).strip()


def test_producer_is_told_not_to_smuggle_the_answer():
    """The contamination check catches it; the instruction should prevent it."""
    assert "contains the answer" in prompts.PRODUCER


def test_producer_is_told_to_state_the_rule_not_the_answer():
    assert "NA" in prompts.PRODUCER and "different document" in prompts.PRODUCER


def test_version_is_recorded_on_every_result(groups, scale):
    r = recover.recover(ObedientClient(), groups[0], scale, rounds=0)
    assert r.prompts_version == prompts.VERSION


# ------------------------------------------------------------------ evidence

def test_measure_reports_shape_and_length(groups):
    g = next(x for x in groups if x.id == "ody-three-sentence-summary")
    text = recover.measure(g.pairs)
    assert "sentences" in text and "words" in text and "shape:" in text
    assert text.count("\n") == len(g.pairs) - 1


def test_measure_flags_negative_inputs(groups):
    g = next(x for x in groups if x.has_negative)
    assert "does NOT contain" in recover.measure(g.pairs)


# ------------------------------------------------------------- loop mechanics

def test_execute_puts_the_document_before_the_instruction(groups, scale):
    client = ScriptedClient(default="ok")
    recover.run_candidate(client, recover.Candidate(text="Summarise."),
                          groups[0], scale)
    _, _, user = client.calls[0]
    assert user.index("<document>") < user.index("Summarise.")


def test_fidelity_is_the_weakest_member_not_the_average(groups, scale):
    """A control set is satisfied only when the instruction works on every input."""
    g = next(x for x in groups if x.is_control)
    c = recover.run_candidate(ObedientClient(), recover.Candidate(text="Do it."),
                              g, scale)
    assert len(c.per_pair) == len(g)
    assert c.fidelity == pytest.approx(min(c.per_pair.values()))


def test_critique_targets_the_worst_input(groups, scale):
    g = next(x for x in groups if x.is_control)
    client = ObedientClient()
    cand = recover.run_candidate(client, recover.Candidate(text="Do it."), g, scale)
    worst = min(cand.per_pair, key=cand.per_pair.get)

    spy = ScriptedClient(default='["change something"]')
    recover.critique(spy, cand, g, scale)
    _, _, user = spy.calls[0]
    wanted = next(p for p in g.pairs if p.id == worst).output
    assert wanted[:60] in user


def test_revise_returns_the_original_when_nothing_changes():
    c = recover.Candidate(text="Same text.")
    same = recover.revise(ScriptedClient(default="Same text."), c, ["x"], 1)
    assert same is c


def test_empty_critique_stops_the_loop(groups, scale):
    class NoChanges(ObedientClient):
        def complete(self, system, user, **k):
            if system.startswith("You review"):
                return super().complete(system, user, **k).__class__(
                    text="[]", model="obedient")
            return super().complete(system, user, **k)

    r = recover.recover(NoChanges(), groups[0], scale, rounds=3)
    assert r.stopped_because == "critic reported nothing to change"
    assert r.rounds == 0


def test_recovery_runs_and_records_provenance(groups, scale):
    r = recover.recover(ObedientClient(), groups[0], scale, rounds=1)
    assert r.best.text.strip()
    assert r.naive.text == recover.NAIVE
    assert r.history
    assert 0.0 < r.best.fidelity <= 1.0
    assert r.stopped_because
    assert set(r.best.per_pair) == {p.id for p in groups[0].pairs}


def test_recovered_prompt_never_contains_the_answer(groups, scale):
    """The guardrail that matters most: no smuggling, on every group."""
    client = ObedientClient()
    results = recover.recover_all(client, groups[:6], rounds=1, scale=scale,
                                  judge=True)
    for r in results:
        assert r.best.prompt_score is not None
        assert r.best.prompt_score.contamination == 0.0, (
            f"{r.group.id} smuggled answer content into the prompt")


def test_scoring_against_gold_happens_after_recovery_not_during(groups, scale):
    """The gold prompt must never reach the producer or critic."""
    client = ObedientClient()
    r = recover.recover(client, groups[0], scale, rounds=1)
    gold = groups[0].gold_prompt
    for role, system, user in client.calls:
        assert gold not in user, f"gold prompt leaked into a {role} call"

    recover.score_against_gold(client, r)
    assert r.best.prompt_score is not None


def test_budget_is_enforced():
    class Tiny(ObedientClient):
        budget = 50

        def complete(self, system, user, **k):
            if self.usage.total_tokens > self.budget:
                raise BudgetExceeded("out of budget")
            return super().complete(system, user, **k)

    client = Tiny()
    with pytest.raises(BudgetExceeded):
        for _ in range(100):
            client.complete("s", "u " * 100)


def test_summarise_reports_what_it_claims(groups, scale):
    results = recover.recover_all(ObedientClient(), groups[:3], rounds=0,
                                  scale=scale, judge=True)
    s = recover.summarise(results)
    assert s["groups"] == 3
    assert 0 <= s["beat_naive"] <= 3
    assert 0.0 < s["mean_fidelity"] <= 1.0
    assert "mean_prompt_similarity" in s


# ------------------------------------------------- features and output scoring
# (regression cover for bugs found while building the measurement layer)

def test_strip_markup_keeps_table_cell_text():
    """Regression: deleting whole table rows zeroed every style feature."""
    table = "| Aspect | Detail |\n|---|---|\n| Reasoning | iterative planning |"
    stripped = strip_markup(table)
    assert "Reasoning" in stripped and "iterative planning" in stripped
    assert "|" not in stripped


def test_no_gold_output_measures_as_empty(pairs):
    """Regression: table-shaped gold measured as having no prose at all."""
    for p in pairs:
        f = extract(p.output)
        assert f["word_count"] > 0, p.id
        assert f["sentence_count"] > 0, p.id


def test_syllable_heuristic():
    assert syllables("cat") == 1
    assert syllables("table") == 2
    assert syllables("") == 1          # never zero, so division stays safe


def test_extract_never_raises_on_odd_input():
    for text in ("", "   ", "|", "###", "- ", "***", "\n\n\n", "NA"):
        assert extract(text)["word_count"] >= 0


def test_features_respond_to_the_thing_they_measure():
    short = "Cats sit. Dogs run. Birds fly."
    long = ("The evaluation of retrieval augmented generation systems requires "
            "careful attention to the interaction between chunking strategy and "
            "the downstream faithfulness of generated claims.")
    assert extract(short)["mean_sentence_len"] < extract(long)["mean_sentence_len"]
    assert extract(long)["flesch"] < extract(short)["flesch"]

    hedged = "This may possibly suggest that results could perhaps vary somewhat."
    assert extract(hedged)["hedge_rate"] > extract("This shows results vary.")["hedge_rate"]

    passive = "The document was written and the claims were verified."
    active = "We wrote the document and verified the claims."
    assert extract(passive)["passive_rate"] > extract(active)["passive_rate"]


def test_identical_text_scores_one(pairs, scale):
    score, breakdown = tier1(pairs[0].output, pairs[0].output, scale)
    assert score == pytest.approx(1.0)
    assert breakdown["combined_distance"] == pytest.approx(0.0)


def test_shape_mismatch_is_penalised(pairs, scale):
    gold = next(p for p in pairs if p.output_shape == "table").output
    prose = "This is a plain prose answer with no table structure whatsoever."
    assert tier1(gold, prose, scale)[1]["shape"] > 0


def test_scale_survives_a_constant_feature():
    s = Scale.fit(["one two three.", "four five six."])
    assert all(v > 0 for v in s.spread.values())


def test_gap_description_states_direction(pairs, scale):
    gap = describe_gap(pairs[0].output, "Short.", scale)
    assert "gold-deviations off" in gap
    assert "too high" in gap or "too low" in gap


def test_infer_shape_tracks_the_corpus_labels(pairs):
    """`output_shape` is a label; `infer_shape` measures markdown. They may
    disagree -- fetch-05 is labelled a list but written as numbered prose."""
    agree = sum(infer_shape(p.output) == p.output_shape for p in pairs)
    assert agree / len(pairs) >= 0.9
    for p in pairs:
        if p.output_shape == "table":
            assert infer_shape(p.output) == "table", p.id


# ---------------------------------------------------------- model resolution

def test_default_model_is_used_when_nothing_overrides_it():
    from reversed_prompts.client import DEFAULT_MODEL, resolve_models
    got = resolve_models(env={})
    assert set(got) == {"inducer", "executor", "judge", "features"}
    assert all(m == DEFAULT_MODEL for m in got.values())


def test_env_var_overrides_the_default_for_every_role():
    from reversed_prompts.client import resolve_models
    got = resolve_models(env={"REVPROMPT_MODEL": "some-model"})
    assert all(m == "some-model" for m in got.values())


def test_per_role_env_var_beats_the_blanket_one():
    from reversed_prompts.client import resolve_models
    got = resolve_models(env={"REVPROMPT_MODEL": "blanket",
                              "REVPROMPT_MODEL_JUDGE": "specific"})
    assert got["judge"] == "specific"
    assert got["executor"] == "blanket"


def test_explicit_override_beats_everything():
    """This is what --model sets, so it has to win."""
    from reversed_prompts.client import resolve_models
    got = resolve_models({"executor": "flag"},
                         env={"REVPROMPT_MODEL": "blanket",
                              "REVPROMPT_MODEL_EXECUTOR": "specific"})
    assert got["executor"] == "flag"
    assert got["judge"] == "blanket"


def test_empty_override_is_ignored_rather_than_blanking_the_model():
    from reversed_prompts.client import DEFAULT_MODEL, resolve_models
    got = resolve_models({"executor": None, "judge": ""}, env={})
    assert got["executor"] == DEFAULT_MODEL
    assert got["judge"] == DEFAULT_MODEL


# ----------------------------------------------------------------- encoding

def test_every_file_read_and_write_declares_an_encoding():
    """Windows defaults to the locale codepage, not UTF-8.

    `Path.read_text()` with no encoding uses cp1252 on a default Windows
    install, which cannot decode the curly quotes in the Odyssey passages --
    it dies with "'charmap' codec can't decode byte 0x9d". The passages are
    UTF-8, so every read and write has to say so.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for d in ("src", "tools", "tests"):
        for f in (root / d).rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in ("read_text", "write_text", "open"):
                    if not any(k.arg == "encoding" for k in node.keywords):
                        offenders.append(f"{f.relative_to(root)}:{node.lineno} {name}()")
    assert not offenders, (
        "these will break on Windows; pass encoding=\"utf-8\":\n  "
        + "\n  ".join(offenders))


def test_corpus_loads_under_a_non_utf8_locale(monkeypatch):
    """End-to-end guard: simulate the Windows default and load the real data."""
    import pathlib as _pl

    real = _pl.Path.read_text

    def cp1252(self, *a, **kw):
        if "encoding" not in kw:
            return self.read_bytes().decode("cp1252")
        return real(self, *a, **kw)

    monkeypatch.setattr(_pl.Path, "read_text", cp1252)
    gs = ingest.load_groups()
    assert len(gs) >= 10
    assert any("’" in p.input_text for g in gs for p in g.pairs), (
        "expected curly quotes to survive the round trip")
