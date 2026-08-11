"""[C] Hypothesize -- propose candidate specs.

Three generators with different failure modes, run together for diversity
(DESIGN.md §4.3):

1. **deterministic** -- reads structure and style straight off the gold
   outputs. No model, so no hallucination is possible. Weak on *task*, exact
   on shape and length.
2. **direct** -- shows the inducer a sample of pairs: "what instruction
   produced these?" Strong on task, vague on style.
3. **contrastive** -- runs a deliberately naive baseline, diffs its output
   against gold, and asks only for *the delta as prompt clauses*. The model
   never has to describe summarization in general, only what makes this
   summarization different. Strongest generator for style.

The naive baseline that generator 3 needs is the same one the exit criterion
measures against, so it is computed once and shared.
"""
from __future__ import annotations

import json
import re
import statistics

from .client import LLMClient
from .features import extract, infer_shape
from .ingest import Pair
from .spec import PromptSpec, StructureFacet, StyleFacet

LENGTH_CV_LIMIT = 0.30    # above this, output length is per-prompt, not per-task
SHAPE_AGREEMENT = 0.70    # below this, output shape is per-prompt, not per-task

NAIVE_TASK = "Answer the question about the document above."

_SYS_DIRECT = (
    "You reverse-engineer prompts. Given inputs and the outputs they produced, "
    "you state the single instruction that would produce those outputs. "
    "Reply with the instruction only -- no preamble, no explanation, no quotes."
)

_SYS_CONTRASTIVE = (
    "You improve prompts by diffing outputs. You are shown what a naive prompt "
    "produced and what was actually wanted. You reply with a JSON array of short "
    "imperative clauses that would close the gap. Each clause must be actionable "
    "and specific. Do not restate the task itself -- only what makes this output "
    "different from the naive one. Reply with the JSON array and nothing else."
)


def naive_spec() -> PromptSpec:
    """The baseline the induced prompt has to beat.

    Deliberately knows nothing: no shape, no length, no style. Anything
    derived from the gold outputs would leak into the baseline and raise the
    bar the exit criterion is measured against, making a real improvement look
    like no improvement.
    """
    return PromptSpec(task=NAIVE_TASK, generator="naive")


# --------------------------------------------------------------- deterministic

def _bucket_sentence_length(mean_len: float) -> str:
    return "short" if mean_len < 15 else "long" if mean_len > 26 else "medium"


def _bucket_hedging(rate: float) -> str:
    return "low" if rate < 0.004 else "high" if rate > 0.012 else "medium"


def deterministic(pairs: list[Pair]) -> PromptSpec:
    """Read the measurable facets off the gold outputs. No model involved."""
    feats = [extract(p.output) for p in pairs]
    shapes = [infer_shape(p.output) for p in pairs]
    # Same rule as length: assert shape only when the golds actually agree.
    # A bare majority means shape varies per prompt, and forcing the plurality
    # onto every example costs more on the minority than it gains elsewhere.
    top = max(set(shapes), key=shapes.count)
    shape = top if shapes.count(top) / len(shapes) >= SHAPE_AGREEMENT else None

    words = [f["word_count"] for f in feats]
    sent_counts = {f["sentence_count"] for f in feats}

    # Only assert a length when length actually looks like a property of the
    # task. If the gold outputs scatter, length was set per-prompt, and
    # committing to the mean makes every short and every long example worse --
    # measurably worse than saying nothing. This is the generality
    # regularizer of DESIGN.md §4.4 applied at induction time: do not claim
    # what the evidence does not support.
    mean_words = statistics.fmean(words)
    cv = statistics.pstdev(words) / mean_words if mean_words and len(words) > 1 else 0.0

    structure = StructureFacet(
        shape=shape,
        target_words=round(mean_words) if cv <= LENGTH_CV_LIMIT else None,
        # an exact sentence count only when every gold agrees and the number is
        # small enough to be a real constraint rather than a coincidence
        max_sentences=(int(sent_counts.pop()) if len(sent_counts) == 1
                       and next(iter(sent_counts)) <= 5 else None),
    )

    mean_first = statistics.fmean([f["first_person_rate"] for f in feats])
    mean_second = statistics.fmean([f["second_person_rate"] for f in feats])
    person = None
    if mean_second > 0.01:
        person = "second person"
    elif mean_first > 0.01:
        person = "first person"
    elif mean_first + mean_second < 0.002:
        person = "third person"

    style = StyleFacet(
        person=person,
        hedging=_bucket_hedging(statistics.fmean([f["hedge_rate"] for f in feats])),
        sentence_length=_bucket_sentence_length(
            statistics.fmean([f["mean_sentence_len"] for f in feats])),
    )
    return PromptSpec(task="Answer using only the document above.",
                      structure=structure, style=style, generator="deterministic")


# ---------------------------------------------------------------------- direct

def _sample(pairs: list[Pair], k: int, budget_chars: int = 6000) -> list[Pair]:
    return pairs[:k]


def direct(client: LLMClient, pairs: list[Pair], k: int = 3) -> PromptSpec:
    """Ask the inducer what instruction would produce these outputs."""
    shown = _sample(pairs, k)
    blocks = []
    for i, p in enumerate(shown, 1):
        blocks.append(f"### Example {i}\nOUTPUT:\n{p.output}")
    user = (
        "All examples below were produced from the same source document by one "
        "instruction. The document is long and is not reproduced here; infer the "
        "instruction from the outputs' content, shape and voice.\n\n"
        + "\n\n".join(blocks)
        + "\n\nWhat single instruction produced these?"
    )
    text = client.complete(_SYS_DIRECT, user, role="inducer").text
    base = deterministic(pairs)
    return PromptSpec(task=text.strip() or base.task,
                      structure=base.structure, style=base.style,
                      generator="direct")


# ----------------------------------------------------------------- contrastive

def _parse_clauses(text: str) -> list[str]:
    """Tolerate the model wrapping JSON in prose or a fence."""
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()][:8]
        except json.JSONDecodeError:
            pass
    # fall back to bullet lines
    return [re.sub(r"^\s*[-*\d.]+\s*", "", l).strip()
            for l in text.splitlines() if l.strip()][:8]


def contrastive(client: LLMClient, pairs: list[Pair],
                baseline_outputs: dict[str, str]) -> PromptSpec:
    """Diff naive output against gold and ask for the delta as clauses."""
    blocks = []
    for p in pairs[:2]:
        naive = baseline_outputs.get(p.id, "")
        if not naive:
            continue
        blocks.append(f"NAIVE PRODUCED:\n{naive}\n\nWANTED:\n{p.output}")
    if not blocks:
        return deterministic(pairs)

    user = ("\n\n---\n\n".join(blocks) +
            "\n\nWhat clauses added to the prompt would turn the naive output "
            "into the wanted one?")
    clauses = _parse_clauses(client.complete(_SYS_CONTRASTIVE, user,
                                             role="inducer").text)
    base = deterministic(pairs)
    return PromptSpec(task=base.task, structure=base.structure, style=base.style,
                      constraints=clauses, generator="contrastive")


def generate(client: LLMClient, pairs: list[Pair],
             baseline_outputs: dict[str, str] | None = None) -> list[PromptSpec]:
    """All three generators. Deterministic first so it always exists."""
    specs = [deterministic(pairs)]
    try:
        specs.append(direct(client, pairs))
    except Exception as e:                       # a generator failing is not fatal
        reason = str(e)[:60]
        specs.append(deterministic(pairs).model_copy(
            update={"generator": f"direct-failed:{reason}"}))
    if baseline_outputs:
        specs.append(contrastive(client, pairs, baseline_outputs))
    # dedup on fingerprint
    seen, out = set(), []
    for s in specs:
        fp = s.fingerprint()
        if fp not in seen:
            seen.add(fp)
            out.append(s)
    return out
