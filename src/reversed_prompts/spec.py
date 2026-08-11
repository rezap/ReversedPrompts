"""PromptSpec -- the recovered prompt as a structured object, not a string.

Facets are induced by different evidence and optimized by different methods
(DESIGN.md §2). They only get flattened into text at the very end, by
`PromptSpec.render()`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

Shape = Literal["prose", "list", "table"]


class StructureFacet(BaseModel):
    """Read straight off the outputs. Deterministic, no LLM involved."""

    # None means "unspecified", which is what a genuinely naive prompt says
    # about shape. Defaulting this to "prose" would leak the answer into the
    # baseline and quietly inflate the bar the exit criterion measures against.
    shape: Shape | None = None
    target_words: int | None = None
    tolerance: float = 0.25
    sections: list[str] = Field(default_factory=list)
    max_sentences: int | None = None

    def as_clauses(self) -> list[str]:
        out: list[str] = []
        if self.shape == "table":
            out.append("Format the answer as a markdown table.")
        elif self.shape == "list":
            out.append("Format the answer as a bulleted list.")
        elif self.shape == "prose":
            out.append("Answer in prose. Do not use bullet points.")
        if self.max_sentences:
            out.append(f"Use exactly {self.max_sentences} sentences.")
        elif self.target_words:
            lo = int(self.target_words * (1 - self.tolerance))
            hi = int(self.target_words * (1 + self.tolerance))
            out.append(f"Aim for {lo}-{hi} words.")
        if self.sections:
            out.append("Cover these in order: " + "; ".join(self.sections) + ".")
        return out


class StyleFacet(BaseModel):
    """Register and voice. Induced from the measured feature vector."""

    tone: str | None = None
    person: str | None = None
    hedging: Literal["low", "medium", "high"] | None = None
    sentence_length: Literal["short", "medium", "long"] | None = None
    notes: list[str] = Field(default_factory=list)

    def as_clauses(self) -> list[str]:
        out: list[str] = []
        if self.tone:
            out.append(f"Write in a {self.tone} register.")
        if self.person:
            out.append(f"Use the {self.person}.")
        if self.sentence_length:
            out.append(f"Prefer {self.sentence_length} sentences.")
        if self.hedging == "low":
            out.append("State claims directly. Avoid hedging language.")
        elif self.hedging == "high":
            out.append("Hedge claims that the source does not fully support.")
        out.extend(self.notes)
        return out


class PromptSpec(BaseModel):
    """The artifact the induction loop produces."""

    task: str
    structure: StructureFacet = Field(default_factory=StructureFacet)
    style: StyleFacet = Field(default_factory=StyleFacet)
    selection: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)

    generator: str = "unknown"
    round: int = 0
    parent: str | None = None

    def render(self) -> str:
        """Flatten to the prompt text an executor actually receives."""
        parts = [self.task.strip()]
        for group in (self.selection, self.structure.as_clauses(),
                      self.style.as_clauses(), self.constraints):
            parts.extend(c.strip() for c in group if c.strip())
        return "\n".join(parts)

    def fingerprint(self) -> str:
        """Stable id for dedup and provenance. Ignores bookkeeping fields."""
        payload = self.model_dump(exclude={"generator", "round", "parent"})
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


class ScoredSpec(BaseModel):
    """A spec plus everything needed to reproduce and audit its score."""

    spec: PromptSpec
    score: float
    tier1: dict[str, float] = Field(default_factory=dict)
    judge_score: float | None = None
    per_pair: dict[str, float] = Field(default_factory=dict)

    # provenance -- required from the first commit, per DESIGN.md Phase 0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def __lt__(self, other: "ScoredSpec") -> bool:
        return self.score < other.score
