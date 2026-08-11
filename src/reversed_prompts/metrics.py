"""Tier-1 scoring: deterministic distance between a generated output and gold.

Two problems have to be solved before feature differences mean anything.

**Scale.** `word_count` moves in hundreds, `type_token_ratio` in hundredths.
Averaging raw differences would make the score a word-count metric wearing a
disguise. Each feature is therefore divided by its spread across the *gold*
outputs -- how much that feature naturally varies in this corpus. A miss of
one typical gold-to-gold deviation costs the same everywhere.

**Degenerate spread.** Some features barely vary across gold (`heading_count`
is 0 for most of this corpus). Dividing by ~0 makes a trivial absolute
difference look enormous, so the scale floors at a fraction of the feature's
own magnitude.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .features import ALL_KEYS, STRUCTURE_KEYS, STYLE_KEYS, Features, extract, infer_shape

# A miss larger than this many gold-deviations is capped: once a candidate is
# wildly wrong on a feature, being wronger tells the optimizer nothing useful.
CLIP = 4.0

WEIGHT_SHAPE = 0.30
WEIGHT_STRUCTURE = 0.40
WEIGHT_STYLE = 0.30


@dataclass(frozen=True)
class Scale:
    """Per-feature normalizer, fitted on the gold outputs of a corpus."""

    spread: dict[str, float]

    @classmethod
    def fit(cls, gold_texts: list[str]) -> "Scale":
        feats = [extract(t) for t in gold_texts]
        spread: dict[str, float] = {}
        for key in ALL_KEYS:
            vals = [f[key] for f in feats]
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            # floor: 10% of typical magnitude, then an absolute floor so a
            # feature that is 0 across the whole corpus cannot divide by zero
            magnitude = statistics.fmean([abs(v) for v in vals]) if vals else 0.0
            spread[key] = max(sd, 0.10 * magnitude, 1e-6)
        return cls(spread)

    def normalized_diff(self, a: Features, b: Features, key: str) -> float:
        return min(abs(a[key] - b[key]) / self.spread[key], CLIP)


def feature_distance(gold: str, got: str, scale: Scale) -> dict[str, float]:
    """Per-feature normalized distance. Lower is closer."""
    fg, fo = extract(gold), extract(got)
    return {k: scale.normalized_diff(fg, fo, k) for k in ALL_KEYS}


def tier1(gold: str, got: str, scale: Scale) -> tuple[float, dict[str, float]]:
    """Score one generated output against gold. Returns (score, breakdown).

    Score is in (0, 1]; 1.0 means indistinguishable on every measured feature.
    """
    diffs = feature_distance(gold, got, scale)

    style = statistics.fmean([diffs[k] for k in STYLE_KEYS])
    structure = statistics.fmean([diffs[k] for k in STRUCTURE_KEYS])
    shape = 0.0 if infer_shape(gold) == infer_shape(got) else CLIP

    combined = (WEIGHT_SHAPE * shape
                + WEIGHT_STRUCTURE * structure
                + WEIGHT_STYLE * style)

    breakdown = {
        "shape": shape,
        "structure": structure,
        "style": style,
        "combined_distance": combined,
        **{f"d_{k}": v for k, v in diffs.items()},
    }
    return 1.0 / (1.0 + combined), breakdown


def worst_features(gold: str, got: str, scale: Scale, n: int = 5) -> list[tuple[str, float]]:
    """The features this candidate missed by the most -- input to refinement."""
    diffs = feature_distance(gold, got, scale)
    return sorted(diffs.items(), key=lambda kv: -kv[1])[:n]


def describe_gap(gold: str, got: str, scale: Scale, n: int = 5) -> str:
    """Human/LLM-readable account of where a candidate's output missed.

    This is what gets handed to the refiner as the measured half of the
    textual gradient, so it states direction, not just magnitude.
    """
    fg, fo = extract(gold), extract(got)
    lines = []
    for key, dist in worst_features(gold, got, scale, n):
        direction = "too high" if fo[key] > fg[key] else "too low"
        lines.append(
            f"- {key}: {fo[key]:.2f} vs gold {fg[key]:.2f} ({direction}, "
            f"{dist:.1f} gold-deviations off)"
        )
    if infer_shape(gold) != infer_shape(got):
        lines.insert(0, f"- shape: produced {infer_shape(got)}, gold is {infer_shape(gold)}")
    return "\n".join(lines)
