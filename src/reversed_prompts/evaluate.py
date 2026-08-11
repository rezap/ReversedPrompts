"""[D] Evaluate -- run candidates and score them.

Phase 0 implements Tier 1 (free, deterministic) and Tier 3 (judge), skipping
Tier 2. The cascade property still holds and is the point: the judge only ever
sees candidates that already survived free filtering, so judge spend is
proportional to the number of *finalists*, not the number of candidates.

Executing a candidate means resending the whole document, so evaluation is the
cost center. Two levers, both on by default: the eval set is subsampled during
search and only widened for finalists, and the document is placed ahead of the
instruction so the cacheable prefix stays identical across every candidate.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .client import LLMClient
from .ingest import Pair
from .metrics import Scale, tier1
from .spec import PromptSpec, ScoredSpec

_SYS_EXECUTE = "Follow the instruction exactly. Output only what it asks for."

_SYS_JUDGE = (
    "You compare two answers to the same question about a document. "
    "Reply with a single integer from 0 to 10: how closely does the CANDIDATE "
    "match the REFERENCE in substance, structure and voice? "
    "10 means indistinguishable in kind. Reply with the integer only."
)


@dataclass(frozen=True)
class Execution:
    pair_id: str
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


def execute(client: LLMClient, spec: PromptSpec, pair: Pair) -> Execution:
    """Run one candidate against one input.

    Document first, instruction second: the prefix is then byte-identical for
    every candidate on this pair, which is what makes prompt caching bite.
    """
    c = client.complete(
        _SYS_EXECUTE,
        f"<document>\n{pair.input_text}\n</document>\n\n{spec.render()}",
        role="executor",
    )
    return Execution(pair.id, c.text, c.model, c.prompt_tokens,
                     c.completion_tokens, c.cached_tokens)


def judge(client: LLMClient, gold: str, candidate: str) -> float:
    """Tier 3. Returns 0..1, or 0.5 when the judge answers unparseably."""
    c = client.complete(
        _SYS_JUDGE,
        f"REFERENCE:\n{gold}\n\n---\n\nCANDIDATE:\n{candidate}",
        role="judge",
    )
    digits = "".join(ch for ch in c.text if ch.isdigit())[:2]
    if not digits:
        return 0.5
    return min(int(digits), 10) / 10.0


def evaluate(client: LLMClient, spec: PromptSpec, pairs: list[Pair],
             scale: Scale, *, use_judge: bool = False) -> ScoredSpec:
    """Score a candidate over a set of pairs. Tier 1 always; Tier 3 optionally."""
    per_pair: dict[str, float] = {}
    breakdowns: list[dict[str, float]] = []
    judge_scores: list[float] = []
    model = ""
    ptok = ctok = cached = 0

    for pair in pairs:
        ex = execute(client, spec, pair)
        model = model or ex.model
        ptok += ex.prompt_tokens
        ctok += ex.completion_tokens
        cached += ex.cached_tokens

        score, breakdown = tier1(pair.output, ex.text, scale)
        per_pair[pair.id] = score
        breakdowns.append(breakdown)

        if use_judge:
            judge_scores.append(judge(client, pair.output, ex.text))

    tier1_mean = statistics.fmean(per_pair.values()) if per_pair else 0.0
    agg = {k: statistics.fmean([b[k] for b in breakdowns])
           for k in ("shape", "structure", "style", "combined_distance")} if breakdowns else {}

    judge_mean = statistics.fmean(judge_scores) if judge_scores else None
    # Tier 1 stays primary; the judge is a tiebreaker, not the arbiter (§4.4).
    combined = tier1_mean if judge_mean is None else 0.7 * tier1_mean + 0.3 * judge_mean

    return ScoredSpec(spec=spec, score=combined, tier1=agg, judge_score=judge_mean,
                      per_pair=per_pair, model=model, prompt_tokens=ptok,
                      completion_tokens=ctok, cached_tokens=cached)


def cascade(client: LLMClient, specs: list[PromptSpec], pairs: list[Pair],
            scale: Scale, *, finalists: int = 2) -> list[ScoredSpec]:
    """Score every candidate on Tier 1, then re-score only the survivors with
    the judge. Returns all candidates, best first."""
    scored = sorted((evaluate(client, s, pairs, scale) for s in specs), reverse=True)
    if not scored:
        return []
    promoted = [evaluate(client, s.spec, pairs, scale, use_judge=True)
                for s in scored[:finalists]]
    return sorted(promoted + scored[finalists:], reverse=True)
