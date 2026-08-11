"""The induction loop: A -> B -> C -> D -> E cycle -> F.

Align ([B]) is Phase 2; Phase 0 goes straight from ingest to hypothesize. The
cyclic part -- evaluate, refine, re-evaluate, with a beam over the top-k specs
and a hard budget -- is here, because that cycle is the thing Phase 0 exists to
prove or kill.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from . import hypothesize
from .client import BudgetExceeded, LLMClient
from .evaluate import cascade, evaluate, execute
from .ingest import Pair
from .metrics import Scale
from .refine import refine
from .spec import PromptSpec, ScoredSpec


@dataclass
class Result:
    best: ScoredSpec
    baseline: ScoredSpec
    history: list[ScoredSpec] = field(default_factory=list)
    rounds: int = 0
    seconds: float = 0.0
    stopped_because: str = "rounds exhausted"

    @property
    def beats_baseline(self) -> bool:
        return self.best.score > self.baseline.score

    @property
    def lift(self) -> float:
        if not self.baseline.score:
            return 0.0
        return (self.best.score - self.baseline.score) / self.baseline.score

    def report(self) -> str:
        verdict = "PASS" if self.beats_baseline else "FAIL"
        return "\n".join([
            f"induced  {self.best.score:.4f}  ({self.best.spec.generator})",
            f"baseline {self.baseline.score:.4f}  (naive)",
            f"lift     {self.lift:+.1%}   exit criterion: {verdict}",
            f"rounds   {self.rounds} in {self.seconds:.1f}s, "
            f"stopped: {self.stopped_because}",
        ])


def fit_scale(pairs: list[Pair]) -> Scale:
    return Scale.fit([p.output for p in pairs])


def run_baseline(client: LLMClient, pairs: list[Pair], scale: Scale
                 ) -> tuple[ScoredSpec, dict[str, str]]:
    """Score the naive prompt and keep its outputs.

    Those outputs are used twice: as the bar the exit criterion measures
    against, and as the raw material for the contrastive generator.
    """
    spec = hypothesize.naive_spec()
    outputs = {p.id: execute(client, spec, p).text for p in pairs}

    from .metrics import tier1
    per_pair = {p.id: tier1(p.output, outputs[p.id], scale)[0] for p in pairs}
    scored = ScoredSpec(
        spec=spec,
        score=statistics.fmean(per_pair.values()) if per_pair else 0.0,
        per_pair=per_pair,
    )
    return scored, outputs


def induce(client: LLMClient, train: list[Pair], *, rounds: int = 3,
           beam: int = 2, eval_sample: int = 4, scale: Scale | None = None,
           verbose: bool = False) -> Result:
    """Search for a spec that beats the naive baseline on Tier-1 metrics."""
    started = time.monotonic()
    scale = scale or fit_scale(train)
    sample = train[:eval_sample]
    stopped = "rounds exhausted"

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    baseline, baseline_outputs = run_baseline(client, sample, scale)
    say(f"baseline {baseline.score:.4f}")

    history: list[ScoredSpec] = []
    try:
        candidates = hypothesize.generate(client, train, baseline_outputs)
        say(f"round 0: {len(candidates)} candidates "
            f"({', '.join(c.generator for c in candidates)})")
        frontier = cascade(client, candidates, sample, scale)
        history.extend(frontier)
        say(f"round 0 best {frontier[0].score:.4f} ({frontier[0].spec.generator})")

        completed = 0
        for r in range(1, rounds + 1):
            edited: list[PromptSpec] = []
            for parent in frontier[:beam]:
                child = refine(client, parent, sample, scale, history, r)
                if child.fingerprint() != parent.spec.fingerprint():
                    edited.append(child)
            if not edited:
                stopped = "refinement produced no new candidates"
                break

            scored = cascade(client, edited, sample, scale)
            history.extend(scored)
            frontier = sorted(frontier + scored, reverse=True)[:max(beam, 2)]
            completed = r
            say(f"round {r} best {frontier[0].score:.4f} "
                f"({frontier[0].spec.generator})")
    except BudgetExceeded as e:
        stopped = str(e)
        completed = locals().get("completed", 0)
        if not history:
            raise

    best = max(history) if history else baseline
    return Result(best=best, baseline=baseline, history=history,
                  rounds=locals().get("completed", 0),
                  seconds=time.monotonic() - started, stopped_because=stopped)


def verify_on_holdout(client: LLMClient, best: ScoredSpec, baseline_spec: PromptSpec,
                      holdout: list[Pair], scale: Scale) -> tuple[ScoredSpec, ScoredSpec]:
    """The actual exit criterion: re-score both on pairs never seen in search."""
    return (evaluate(client, best.spec, holdout, scale),
            evaluate(client, baseline_spec, holdout, scale))


def induce_per_pair(client: LLMClient, pairs: list[Pair],
                    scale: Scale | None = None) -> list[tuple[Pair, ScoredSpec, ScoredSpec]]:
    """Recover one spec per pair, from that pair alone.

    `induce` above assumes what DESIGN.md §1 assumes: that the corpus shares a
    single prompt, so facets can be pooled across pairs. The pair set in
    `data/` does not have that property -- it was built with a distinct prompt
    per pair -- and corpus-level induction on it is therefore unstable, winning
    or losing depending on the split.

    This mode matches that data: induce from one pair, score on that pair.
    What it measures is *reconstruction*, not generalization -- there is no
    held-out anything, so a good result here says the facet extraction and
    scoring work, and says nothing about whether a recovered prompt transfers.
    Read the two modes accordingly.
    """
    scale = scale or fit_scale(pairs)
    naive = hypothesize.naive_spec()
    out = []
    for pair in pairs:
        spec = hypothesize.deterministic([pair])
        out.append((pair,
                    evaluate(client, spec, [pair], scale),
                    evaluate(client, naive, [pair], scale)))
    return out
