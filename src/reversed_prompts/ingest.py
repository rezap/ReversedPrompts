"""[A] Ingest -- load pairs and split them.

Phase 0 pairs are already plain text, so ingest is loading and splitting. The
`Pair` shape is what the rest of the loop consumes, so when real ingestion
(docling, layout trees) lands in Phase 2 it produces these and nothing
downstream changes.
"""
from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "data" / "pairs" / "agentic-ai-survey.jsonl"


@dataclass(frozen=True)
class Pair:
    id: str
    category: str
    input_text: str
    output: str
    target_prompt: str          # gold; never shown to the inducer
    output_shape: str = "prose"

    @property
    def compression(self) -> float:
        n = len(self.input_text.split())
        return len(self.output.split()) / n if n else 0.0


def load(path: pathlib.Path | str = DEFAULT_PAIRS,
         category: str | None = None) -> list[Pair]:
    path = pathlib.Path(path)
    root = path.resolve().parents[2] if path.is_absolute() else ROOT
    pairs: list[Pair] = []
    cache: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if category and r["category"] != category:
            continue
        ref = r["input_ref"]
        if ref not in cache:
            cache[ref] = (root / ref).read_text()
        pairs.append(Pair(
            id=r["id"],
            category=r["category"],
            input_text=cache[ref],
            output=r["output"],
            target_prompt=r["target_prompt"],
            output_shape=r.get("output_shape", "prose"),
        ))
    if not pairs:
        raise ValueError(f"no pairs loaded from {path}"
                         + (f" for category {category!r}" if category else ""))
    return pairs


def split(pairs: list[Pair], holdout: int = 2, seed: int = 7
          ) -> tuple[list[Pair], list[Pair]]:
    """Train/held-out split. The exit criterion is measured on held-out only.

    Deterministic given a seed, because a split that moves between runs makes
    round-to-round scores incomparable.
    """
    if holdout >= len(pairs):
        raise ValueError(f"holdout {holdout} needs fewer than {len(pairs)} pairs")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return shuffled[holdout:], shuffled[:holdout]
