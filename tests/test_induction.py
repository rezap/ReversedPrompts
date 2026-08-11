"""Tests for the Phase 0 induction slice. No API key, no network, no spend.

The loop is exercised end to end against `ObedientClient`, a stand-in that
actually follows shape and length instructions. That makes lift observable
offline: a spec whose structure matches gold produces output whose measured
features match gold.
"""
from __future__ import annotations

import pytest

from reversed_prompts import hypothesize, ingest
from reversed_prompts.client import (BudgetExceeded, ObedientClient,
                                     ScriptedClient, Usage)
from reversed_prompts.evaluate import evaluate, execute, judge
from reversed_prompts.features import extract, infer_shape, strip_markup, syllables
from reversed_prompts.induce import (fit_scale, induce, induce_per_pair,
                                     run_baseline, verify_on_holdout)
from reversed_prompts.metrics import Scale, describe_gap, tier1, worst_features
from reversed_prompts.spec import PromptSpec, ScoredSpec, StructureFacet, StyleFacet


@pytest.fixture(scope="module")
def pairs():
    return ingest.load()


@pytest.fixture(scope="module")
def scale(pairs):
    return fit_scale(pairs)


# ------------------------------------------------------------------- features

def test_syllable_heuristic():
    assert syllables("cat") == 1
    assert syllables("table") == 2
    assert syllables("reasoning") == 3
    assert syllables("") == 1          # never zero, so division stays safe


def test_strip_markup_keeps_table_cell_text():
    """Regression: deleting whole table rows zeroed every style feature."""
    table = "| Aspect | Detail |\n|---|---|\n| Reasoning | iterative planning |"
    stripped = strip_markup(table)
    assert "Reasoning" in stripped and "iterative planning" in stripped
    assert "|" not in stripped


def test_table_output_has_nonzero_word_count(pairs):
    """Regression: table-shaped gold measured as having no prose at all."""
    tables = [p for p in pairs if p.output_shape == "table"]
    assert tables, "corpus should contain table pairs"
    for p in tables:
        assert extract(p.output)["word_count"] > 20, p.id


def test_no_gold_output_measures_as_empty(pairs):
    for p in pairs:
        f = extract(p.output)
        assert f["word_count"] > 0, p.id
        assert f["sentence_count"] > 0, p.id


def test_extract_never_raises_on_odd_input():
    for text in ("", "   ", "|", "###", "- ", "***", "\n\n\n"):
        assert extract(text)["word_count"] >= 0


def test_infer_shape_tracks_the_corpus_labels(pairs):
    """`output_shape` is a human label; `infer_shape` measures markdown.

    They are allowed to disagree -- fetch-05 is labelled `list` but written as
    numbered prose paragraphs, so it measures `prose`. What must hold is that
    the measurement agrees with the label in the large majority of cases and
    never confuses a table for anything else, since the shape term carries the
    single largest weight in the Tier-1 score.
    """
    agree = sum(infer_shape(p.output) == p.output_shape for p in pairs)
    assert agree / len(pairs) >= 0.9

    for p in pairs:
        if p.output_shape == "table":
            assert infer_shape(p.output) == "table", p.id


def test_features_respond_to_the_thing_they_measure():
    short = "Cats sit. Dogs run. Birds fly."
    long = ("The evaluation of retrieval augmented generation systems requires "
            "careful attention to the interaction between chunking strategy and "
            "the downstream faithfulness of generated claims.")
    assert extract(short)["mean_sentence_len"] < extract(long)["mean_sentence_len"]
    assert extract(long)["flesch"] < extract(short)["flesch"]

    hedged = "This may possibly suggest that results could perhaps vary somewhat."
    flat = "This shows that results vary."
    assert extract(hedged)["hedge_rate"] > extract(flat)["hedge_rate"]

    passive = "The document was written and the claims were verified."
    active = "We wrote the document and verified the claims."
    assert extract(passive)["passive_rate"] > extract(active)["passive_rate"]


# -------------------------------------------------------------------- metrics

def test_identical_text_scores_one(pairs, scale):
    score, breakdown = tier1(pairs[0].output, pairs[0].output, scale)
    assert score == pytest.approx(1.0)
    assert breakdown["combined_distance"] == pytest.approx(0.0)


def test_closer_text_scores_higher(pairs, scale):
    gold = next(p for p in pairs if p.id == "summ-04").output
    near = " ".join(gold.split()[:70])          # same voice, slightly short
    far = "| a | b |\n|---|---|\n| c | d |"     # wrong shape entirely
    assert tier1(gold, near, scale)[0] > tier1(gold, far, scale)[0]


def test_shape_mismatch_is_penalised(pairs, scale):
    gold = next(p for p in pairs if p.output_shape == "table").output
    prose = "This is a plain prose answer with no table structure whatsoever."
    assert tier1(gold, prose, scale)[1]["shape"] > 0


def test_scale_survives_a_constant_feature():
    """A feature identical across gold must not divide by zero."""
    s = Scale.fit(["one two three.", "four five six."])
    assert all(v > 0 for v in s.spread.values())


def test_worst_features_and_gap_description(pairs, scale):
    gold = pairs[0].output
    got = "Short."
    worst = worst_features(gold, got, scale)
    assert len(worst) == 5 and worst[0][1] >= worst[-1][1]
    gap = describe_gap(gold, got, scale)
    assert "gold-deviations off" in gap


# ----------------------------------------------------------------------- spec

def test_naive_spec_asserts_nothing_about_the_answer():
    """Regression: a naive baseline given the gold shape inflates the bar."""
    rendered = hypothesize.naive_spec().render()
    assert rendered.strip() == hypothesize.NAIVE_TASK
    for leak in ("prose", "bullet", "table", "words", "sentences"):
        assert leak not in rendered.lower()


def test_render_includes_every_facet():
    spec = PromptSpec(
        task="Do the thing.",
        structure=StructureFacet(shape="list", target_words=100),
        style=StyleFacet(person="second person", hedging="low"),
        selection=["Use only section 3."],
        constraints=["No preamble."],
    )
    r = spec.render()
    assert "Do the thing." in r
    assert "bulleted list" in r
    assert "75-125 words" in r
    assert "second person" in r
    assert "Avoid hedging" in r
    assert "Use only section 3." in r
    assert "No preamble." in r


def test_fingerprint_ignores_bookkeeping():
    a = PromptSpec(task="x", generator="direct", round=1)
    b = PromptSpec(task="x", generator="contrastive", round=9, parent="abc")
    assert a.fingerprint() == b.fingerprint()
    assert PromptSpec(task="y").fingerprint() != a.fingerprint()


def test_scored_specs_sort_by_score():
    lo = ScoredSpec(spec=PromptSpec(task="a"), score=0.1)
    hi = ScoredSpec(spec=PromptSpec(task="b"), score=0.9)
    assert max([lo, hi]) is hi


# --------------------------------------------------------------- hypothesize

def test_deterministic_withholds_length_when_gold_scatters():
    """Regression: committing to the mean hurt every short and long example."""
    scattered = ingest.load(category="summarize")     # CV ~0.44
    assert hypothesize.deterministic(scattered).structure.target_words is None

    consistent = ingest.load(category="reason")       # CV ~0.19
    assert hypothesize.deterministic(consistent).structure.target_words is not None


def test_deterministic_withholds_shape_when_gold_disagrees():
    mixed = ingest.load(category="fetch")             # table + list + prose
    assert hypothesize.deterministic(mixed).structure.shape is None

    uniform = [p for p in ingest.load() if p.output_shape == "prose"][:5]
    assert hypothesize.deterministic(uniform).structure.shape == "prose"


def test_generators_are_deduplicated(pairs):
    client = ScriptedClient(default="")            # direct returns nothing useful
    specs = hypothesize.generate(client, pairs[:4], {})
    assert len({s.fingerprint() for s in specs}) == len(specs)


def test_a_failing_generator_does_not_kill_the_others(pairs):
    class Exploding:
        usage = Usage()

        def complete(self, *a, **k):
            raise RuntimeError("upstream is down")

    specs = hypothesize.generate(Exploding(), pairs[:4], {})
    # The failure marker itself gets deduplicated away -- the fallback spec is
    # byte-identical to the deterministic one, and fingerprints ignore
    # `generator`. What matters is that a dead model still yields a usable
    # candidate rather than an exception.
    assert specs, "deterministic must survive a model outage"
    assert specs[0].structure is not None


def test_clause_parsing_tolerates_prose_around_json():
    parse = hypothesize._parse_clauses
    assert parse('["a", "b"]') == ["a", "b"]
    assert parse('Sure! Here you go:\n```json\n["a"]\n```') == ["a"]
    assert parse("- first\n- second") == ["first", "second"]
    assert parse("") == []


# ------------------------------------------------------------------ evaluate

def test_execute_puts_the_document_before_the_instruction(pairs):
    """Prefix stability is what makes prompt caching bite."""
    client = ScriptedClient(default="ok")
    execute(client, PromptSpec(task="Summarise."), pairs[0])
    _, _, user = client.calls[0]
    assert user.index("<document>") < user.index("Summarise.")


def test_judge_parses_and_clamps():
    assert judge(ScriptedClient(default="8"), "g", "c") == pytest.approx(0.8)
    assert judge(ScriptedClient(default="10"), "g", "c") == pytest.approx(1.0)
    assert judge(ScriptedClient(default="no idea"), "g", "c") == pytest.approx(0.5)


def test_evaluate_records_provenance(pairs, scale):
    scored = evaluate(ObedientClient(), PromptSpec(task="x"), pairs[:2], scale)
    assert scored.model == "obedient"
    assert scored.prompt_tokens > 0
    assert set(scored.per_pair) == {p.id for p in pairs[:2]}


def test_judge_is_a_tiebreaker_not_the_arbiter(pairs, scale):
    """Tier 1 must keep majority weight in the combined score (§4.4)."""
    spec = PromptSpec(task="x")
    without = evaluate(ObedientClient(), spec, pairs[:2], scale)
    with_judge = evaluate(ObedientClient(), spec, pairs[:2], scale, use_judge=True)
    assert with_judge.judge_score is not None
    implied = (with_judge.score - 0.7 * without.score) / 0.3
    assert implied == pytest.approx(with_judge.judge_score, abs=1e-6)


# --------------------------------------------------------------------- loop

def test_budget_is_enforced():
    class Tiny(ObedientClient):
        def __init__(self):
            super().__init__()
            self.budget = 50

        def complete(self, system, user, **k):
            if self.usage.total_tokens > self.budget:
                raise BudgetExceeded("out of budget")
            return super().complete(system, user, **k)

    with pytest.raises(BudgetExceeded):
        client = Tiny()
        for _ in range(100):
            client.complete("s", "u " * 100)


def test_split_is_deterministic(pairs):
    a_train, a_held = ingest.split(pairs, holdout=3, seed=7)
    b_train, b_held = ingest.split(pairs, holdout=3, seed=7)
    assert [p.id for p in a_held] == [p.id for p in b_held]
    assert not {p.id for p in a_held} & {p.id for p in a_train}


def test_split_rejects_impossible_holdout(pairs):
    with pytest.raises(ValueError):
        ingest.split(pairs, holdout=len(pairs))


def test_baseline_outputs_feed_the_contrastive_generator(pairs, scale):
    scored, outputs = run_baseline(ObedientClient(), pairs[:3], scale)
    assert set(outputs) == {p.id for p in pairs[:3]}
    assert 0 < scored.score <= 1.0


def test_per_pair_reconstruction_beats_naive_everywhere(pairs, scale):
    """Recovering a spec from one pair reproduces that pair better than naive.

    This is the mode the shipped corpus supports: one prompt per pair, so
    facets are induced from a single example. It measures reconstruction, not
    generalization -- see `induce_per_pair`.
    """
    results = induce_per_pair(ObedientClient(), pairs, scale)
    losses = [(p.id, ind.score, nai.score)
              for p, ind, nai in results if ind.score <= nai.score]
    assert not losses, f"induced failed to beat naive on: {losses}"


@pytest.mark.parametrize("category", ["fetch", "summarize", "reason"])
def test_corpus_level_induction_runs_and_is_scored(category):
    """Corpus-level search completes and produces an auditable result.

    Deliberately *not* asserting that it beats naive. DESIGN.md §1 assumes one
    prompt per corpus; this pair set has one prompt per pair, so pooling facets
    across pairs is fighting the data and the outcome flips with the split.
    Asserting a win here would encode a coin-flip as a requirement. The
    corpus-level exit criterion needs a shared-task corpus to be meaningful.
    """
    pairs = ingest.load(category=category)
    train, held = ingest.split(pairs, holdout=2)
    client = ObedientClient()
    scale = fit_scale(train)

    result = induce(client, train, rounds=1, eval_sample=3, scale=scale)
    best, base = verify_on_holdout(client, result.best, result.baseline.spec,
                                   held, scale)
    assert 0 < best.score <= 1.0 and 0 < base.score <= 1.0
    assert result.best.spec.render()
    assert client.usage.calls > 0


def test_loop_records_history_and_stops_cleanly():
    train, _ = ingest.split(ingest.load(category="reason"), holdout=2)
    result = induce(ObedientClient(), train, rounds=2, eval_sample=2)
    assert result.history
    assert result.best.score >= max(h.score for h in result.history) - 1e-9
    assert result.stopped_because
    assert result.seconds >= 0
