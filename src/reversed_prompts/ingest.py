"""Load pairs and group them by the prompt that produced them.

A **prompt group** is the unit of recovery: one or more pairs that were
produced by the same instruction. Most groups have a single pair. A group with
several is a *control set* -- the same instruction applied to different inputs,
including at least one where the correct answer is "nothing here". Those are
what catch a recovered prompt that merely describes the answer it saw instead
of stating the rule.
"""
from __future__ import annotations

import json
import pathlib
from collections import OrderedDict
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "data" / "pairs" / "odyssey.jsonl"


@dataclass(frozen=True)
class Pair:
    id: str
    category: str
    input_text: str
    output: str
    # Gold. Never shown to the producer, and often simply unknown: recovering
    # it is the point. Empty means "no gold to judge against" -- fidelity still
    # scores, prompt match cannot, and the CLI says so rather than judging
    # against an empty string and reporting the result as a number.
    target_prompt: str = ""
    output_shape: str = "prose"
    prompt_group: str = ""      # defaults to the pair's own id
    is_negative: bool = False   # the correct answer here is "not present"
    source_book: str = ""       # which book of the Odyssey the passage came from
    alteration: str = ""        # "minor" or "major" -- how far it departs from Homer

    @property
    def group(self) -> str:
        return self.prompt_group or self.id


@dataclass(frozen=True)
class PromptGroup:
    """Pairs sharing one gold instruction."""

    id: str
    pairs: list[Pair] = field(default_factory=list)

    @property
    def gold_prompt(self) -> str:
        return self.pairs[0].target_prompt

    @property
    def has_gold(self) -> bool:
        return bool(self.gold_prompt.strip())

    @property
    def is_control(self) -> bool:
        return len(self.pairs) > 1

    @property
    def has_negative(self) -> bool:
        return any(p.is_negative for p in self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


def _read(ref: str, cache: dict[str, str]) -> str:
    """Read a repo-relative file, once per path.

    Paths are resolved against the repo root rather than the pair file, so a
    pair file can be moved without every reference in it going stale.
    """
    if ref not in cache:
        cache[ref] = (ROOT / ref).read_text(encoding="utf-8")
    return cache[ref]


def _output_of(record: dict, cache: dict[str, str]) -> str:
    """The wanted output, inline or from a file.

    `output_ref` exists because an output is not always a short answer: a
    document-to-document task has an output of thousands of words, and a
    JSONL line carrying it is unreadable and un-diffable. Exactly one of the
    two is required -- accepting both and preferring one silently would let a
    stale inline copy override the file someone actually edited.
    """
    has_inline = "output" in record
    has_ref = bool(record.get("output_ref"))
    if has_inline and has_ref:
        raise ValueError(f"{record.get('id')!r} sets both 'output' and "
                         f"'output_ref'; use one")
    if has_ref:
        return _read(record["output_ref"], cache)
    if not has_inline:
        raise ValueError(f"{record.get('id')!r} has neither 'output' nor "
                         f"'output_ref'")
    return record["output"]


def load(path: pathlib.Path | str = DEFAULT_PAIRS,
         category: str | None = None) -> list[Pair]:
    path = pathlib.Path(path)
    pairs: list[Pair] = []
    cache: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if category and r["category"] != category:
            continue
        text = _read(r["input_ref"], cache)
        if r.get("input_span"):                 # a slice of the source document
            start, end = r["input_span"]
            text = text[start:end]
        pairs.append(Pair(
            id=r["id"],
            category=r["category"],
            input_text=text,
            output=_output_of(r, cache),
            target_prompt=r.get("target_prompt", ""),
            output_shape=r.get("output_shape", "prose"),
            prompt_group=r.get("prompt_group", ""),
            is_negative=r.get("is_negative", False),
            source_book=r.get("source_book", ""),
            alteration=r.get("alteration", ""),
        ))
    if not pairs:
        raise ValueError(f"no pairs loaded from {path}"
                         + (f" for category {category!r}" if category else ""))
    return pairs


def group(pairs: list[Pair]) -> list[PromptGroup]:
    """Bucket pairs by prompt group, preserving file order."""
    buckets: "OrderedDict[str, list[Pair]]" = OrderedDict()
    for p in pairs:
        buckets.setdefault(p.group, []).append(p)

    out = []
    for gid, members in buckets.items():
        golds = {m.target_prompt for m in members}
        if len(golds) > 1:
            raise ValueError(
                f"prompt group {gid!r} has {len(golds)} different gold prompts; "
                "pairs in a group must share one instruction")
        out.append(PromptGroup(id=gid, pairs=members))
    return out


def load_groups(path: pathlib.Path | str = DEFAULT_PAIRS,
                category: str | None = None) -> list[PromptGroup]:
    return group(load(path, category=category))
