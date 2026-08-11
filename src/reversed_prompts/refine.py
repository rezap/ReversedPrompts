"""[E] Refine -- failure-driven editing via textual gradients.

Refinement is not resampling. For the pair a candidate did worst on, the
measured feature gap is turned into a critique, and the spec is edited in the
opposite semantic direction (ProTeGi). The optimizer also carries an
OPRO-style trajectory of what it has already tried and scored, so it can see
which edits helped and which did not.

The measured half of the gradient comes from `metrics.describe_gap` rather
than from the model's impression of the output. The model is asked to *act on*
numbers it cannot fudge, not to produce them.
"""
from __future__ import annotations

from .client import LLMClient
from .evaluate import execute
from .ingest import Pair
from .metrics import Scale, describe_gap
from .spec import PromptSpec, ScoredSpec

_SYS_REFINE = (
    "You repair prompts. You are given a prompt, measurements of how its output "
    "missed the target, and the history of what has already been tried. "
    "You reply with a JSON array of short imperative clauses that replace the "
    "prompt's current constraint list. Address the measured gaps directly -- if "
    "output is too long, say how long; if the shape is wrong, name the shape. "
    "Keep clauses specific and few. Reply with the JSON array and nothing else."
)


def trajectory_note(history: list[ScoredSpec], limit: int = 4) -> str:
    """What has been tried and what it scored -- the OPRO memory."""
    if not history:
        return "(nothing tried yet)"
    lines = []
    for h in sorted(history, reverse=True)[:limit]:
        clauses = "; ".join(h.spec.constraints) or "(no extra clauses)"
        lines.append(f"- scored {h.score:.3f} via {h.spec.generator}: {clauses}")
    return "\n".join(lines)


def worst_pair(scored: ScoredSpec, pairs: list[Pair]) -> Pair | None:
    if not scored.per_pair:
        return None
    worst_id = min(scored.per_pair, key=lambda k: scored.per_pair[k])
    return next((p for p in pairs if p.id == worst_id), None)


def refine(client: LLMClient, scored: ScoredSpec, pairs: list[Pair],
           scale: Scale, history: list[ScoredSpec], round_no: int) -> PromptSpec:
    """Produce an edited spec aimed at the current candidate's worst failure."""
    from .hypothesize import _parse_clauses

    target = worst_pair(scored, pairs)
    if target is None:
        return scored.spec

    got = execute(client, scored.spec, target).text
    gap = describe_gap(target.output, got, scale)

    user = (
        f"CURRENT PROMPT:\n{scored.spec.render()}\n\n"
        f"MEASURED GAP on its worst example (a 'gold-deviation' is one typical "
        f"unit of variation for that feature across the reference set):\n{gap}\n\n"
        f"ALREADY TRIED:\n{trajectory_note(history)}\n\n"
        "Give the replacement clauses."
    )
    clauses = _parse_clauses(client.complete(_SYS_REFINE, user, role="inducer").text)
    if not clauses:
        return scored.spec

    return scored.spec.model_copy(update={
        "constraints": clauses,
        "generator": f"refined<-{scored.spec.generator}",
        "round": round_no,
        "parent": scored.spec.fingerprint(),
    })
