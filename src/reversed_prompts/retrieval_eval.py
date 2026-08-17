"""Measure retrieval quality, so "hybrid search works" becomes checkable.

The claim that needs testing is not that retrieval returns something. It is
that the passage containing the evidence comes back near the top, and that
fusing two retrievers beats either alone. The second half is not automatic:
measured on the full Odyssey with the offline hashing embedder, fusion scored
*worse* than keyword alone, because averaging in a bad ranking makes a good one
worse. That is the kind of result this module exists to surface.

Ground truth is a regex per probe, matched against chunk text. That is cruder
than human relevance judgements and it is honest about what it measures: "did
retrieval find chunks that actually mention the thing", not "did it find the
best chunk". Good enough to catch a retriever that is broken or an arm that is
dragging the fusion down; not good enough to rank two decent configurations.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from .retrieval import Retriever, fuse

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "source" / "odyssey-pg1727.txt"


@dataclass(frozen=True)
class Probe:
    """A query and a pattern that marks a chunk as relevant to it."""

    name: str
    query: str
    relevant: str          # regex, matched case-insensitively against chunk text
    note: str = ""


# Chosen to span the range the recovery task actually covers. The first two
# hang on a rare proper noun, which is what keyword search is good at and
# embeddings blur. The last two are phrased as descriptions rather than names,
# so they need the vector side -- and are where a keyword-only configuration
# should visibly suffer.
PROBES = (
    Probe("antagonist", "Antinous the ringleader of the suitors", r"Antinous",
          note="rare proper noun"),
    Probe("cyclops", "Polyphemus the Cyclops in his cave", r"Polyphemus",
          note="rare proper noun"),
    Probe("shroud", "the wife who wove a garment by day and undid it at night",
          r"shroud|unravel|unpick", note="described, not named"),
    Probe("sirens", "singers whose voices lure sailors onto the rocks",
          r"Siren", note="described, not named"),
)


@dataclass
class ArmResult:
    arm: str
    hits: int
    total: int

    @property
    def precision(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class ProbeResult:
    probe: Probe
    relevant_chunks: int
    arms: dict[str, ArmResult]


def _precision_at_k(ranking: list[str], truth: set[int], k: int) -> ArmResult:
    top = ranking[:k]
    return ArmResult("", sum(1 for i in top if int(i) in truth), len(top))


def evaluate(retriever: Retriever, doc_id: str, *, probes=PROBES, k: int = 5,
             weights: list[float] | None = None) -> list[ProbeResult]:
    """Score each arm separately and fused, per probe."""
    _, chunks = retriever._load_source(doc_id)
    results = []
    for probe in probes:
        pattern = re.compile(probe.relevant, re.I)
        truth = {c.ordinal for c in chunks if pattern.search(c.text)}

        keyword, vector, strength = retriever.rankings(doc_id, probe.query,
                                                       depth=max(k * 4, 20))
        fused = [i for i, _ in fuse([keyword, vector], tiebreak=strength,
                                    weights=weights)]

        arms = {}
        for name, ranking in (("keyword", keyword), ("vector", vector),
                              ("fused", fused)):
            r = _precision_at_k(ranking, truth, k)
            arms[name] = ArmResult(name, r.hits, r.total)
        results.append(ProbeResult(probe, len(truth), arms))
    return results


def format_report(results: list[ProbeResult], k: int) -> str:
    """A table, plus the verdict that actually matters."""
    width = max((len(r.probe.name) for r in results), default=6)
    lines = [f"{'probe':<{width}}  {'relevant':>8}  {'keyword':>9}  "
             f"{'vector':>9}  {'fused':>9}"]
    lines.append("-" * len(lines[0]))
    for r in results:
        lines.append(
            f"{r.probe.name:<{width}}  {r.relevant_chunks:>8}  "
            + "  ".join(f"{r.arms[a].hits:>4}/{r.arms[a].total:<4}"
                        for a in ("keyword", "vector", "fused")))

    means = {a: sum(r.arms[a].precision for r in results) / len(results)
             for a in ("keyword", "vector", "fused")} if results else {}
    lines.append("-" * len(lines[0]))
    lines.append(f"{'mean p@' + str(k):<{width}}  {'':>8}  "
                 + "  ".join(f"{means[a]:>9.3f}"
                             for a in ("keyword", "vector", "fused")))

    if means:
        best_single = max(means["keyword"], means["vector"])
        if means["fused"] < best_single - 1e-9:
            lines.append(
                f"\nNOTE: fusion ({means['fused']:.3f}) is below the better "
                f"single arm ({best_single:.3f}).\nFusing a weak retriever with "
                f"a strong one drags the strong one down. Consider\n"
                f"--weights, or a keyword-first configuration, before building "
                f"on these results.")
        else:
            lines.append(f"\nFusion ({means['fused']:.3f}) is at least as good "
                         f"as the better single arm ({best_single:.3f}).")
    return "\n".join(lines)
