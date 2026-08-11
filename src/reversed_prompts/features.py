"""Deterministic style and structure measurement.

This is the free, non-gameable half of the scoring design (DESIGN.md §4.4
Tier 1). Everything here is pure Python on purpose: no spacy, no textstat, no
model weights to download, so the whole metric layer runs in CI in
milliseconds and a contributor can reason about every number.

The heuristics below (passive voice, syllables) are approximations of what a
POS tagger would give you. That is acceptable *because these feed a distance
metric* -- both sides of every comparison are measured by the same
approximation, so a consistent bias cancels. It would not be acceptable if
these numbers were reported as linguistic ground truth. They are not.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENT_SPLIT = re.compile(r"(?<=[.!?])[\s\n]+")
VOWELS = "aeiouy"

HEDGES = frozenset("""
may might could possibly perhaps arguably seems seem appears appear suggests
suggest likely unlikely relatively somewhat fairly rather generally often
typically usually tends tend probably apparently roughly approximately
""".split())

FIRST = frozenset("i we me us my our ours mine myself ourselves".split())
SECOND = frozenset("you your yours yourself yourselves".split())
THIRD = frozenset("he she it they him her them his hers its their theirs".split())

# "was written", "is being considered", "have been observed"
PASSIVE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\b"
    r"(?:\s+\w+ly)?"
    r"\s+\w+(?:ed|en|wn|ne)\b",
    re.I,
)

IRREGULAR_PARTICIPLES = frozenset("""
made held kept led met paid said sold told found built sent spent lost meant
""".split())
PASSIVE_IRREGULAR = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\b(?:\s+\w+ly)?\s+("
    + "|".join(IRREGULAR_PARTICIPLES) + r")\b",
    re.I,
)


def syllables(word: str) -> int:
    """Vowel-group heuristic. Good enough for a readability *difference*."""
    w = word.lower()
    groups = re.findall(rf"[{VOWELS}]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def sentences(text: str) -> list[str]:
    stripped = strip_markup(text)
    return [s.strip() for s in SENT_SPLIT.split(stripped) if s.strip()]


def strip_markup(text: str) -> str:
    """Remove markdown scaffolding so prose stats measure prose.

    Cell text is prose and must survive: deleting whole table rows makes a
    table-shaped output measure as having no words at all, which silently
    zeroes every style feature for it.
    """
    # separator rows carry no content; data rows lose their pipes, keep text
    text = re.sub(r"^\s*\|[\s:|-]*\|\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\|(.*)\|\s*$",
                  lambda m: m.group(1).replace("|", ". ") + ".",
                  text, flags=re.M)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)          # heading marks
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)        # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)        # numbered items
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                # bold
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


@dataclass(frozen=True)
class Features:
    """One output's measurable fingerprint."""

    values: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, k: str) -> float:
        return self.values[k]

    def keys(self):
        return self.values.keys()


# Feature names, fixed order. Anything added here is automatically scored.
STYLE_KEYS = (
    "mean_sentence_len", "sd_sentence_len", "type_token_ratio", "mean_word_len",
    "passive_rate", "hedge_rate", "first_person_rate", "second_person_rate",
    "third_person_rate", "flesch", "comma_rate",
)
STRUCTURE_KEYS = (
    "word_count", "sentence_count", "paragraph_count", "heading_count",
    "bullet_count", "numbered_count", "table_rows", "bold_count",
    "list_ratio", "mean_paragraph_words",
)
ALL_KEYS = STYLE_KEYS + STRUCTURE_KEYS


def extract(text: str) -> Features:
    """Measure one output. Never raises on odd input; degrades to zeros."""
    lines = text.splitlines()
    prose = strip_markup(text)
    sents = sentences(text)
    words = WORD.findall(prose)
    lower = [w.lower() for w in words]
    n_words = len(words)
    n_sents = len(sents)

    sent_lens = [len(WORD.findall(s)) for s in sents] or [0]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    bullets = sum(1 for l in lines if re.match(r"^\s*[-*+]\s+", l))
    numbered = sum(1 for l in lines if re.match(r"^\s*\d+\.\s+", l))
    table_rows = sum(1 for l in lines if re.match(r"^\s*\|.*\|\s*$", l))
    headings = sum(1 for l in lines if re.match(r"^#{1,6}\s", l))

    syl = sum(syllables(w) for w in words)
    passives = len(PASSIVE.findall(prose)) + len(PASSIVE_IRREGULAR.findall(prose))

    def rate(count: int) -> float:
        return count / n_words if n_words else 0.0

    flesch = 0.0
    if n_words and n_sents:
        flesch = (206.835
                  - 1.015 * (n_words / n_sents)
                  - 84.6 * (syl / n_words))

    listish = bullets + numbered + table_rows
    return Features({
        "mean_sentence_len": statistics.fmean(sent_lens),
        "sd_sentence_len": statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0,
        "type_token_ratio": len(set(lower)) / n_words if n_words else 0.0,
        "mean_word_len": statistics.fmean([len(w) for w in words]) if words else 0.0,
        "passive_rate": passives / n_sents if n_sents else 0.0,
        "hedge_rate": rate(sum(1 for w in lower if w in HEDGES)),
        "first_person_rate": rate(sum(1 for w in lower if w in FIRST)),
        "second_person_rate": rate(sum(1 for w in lower if w in SECOND)),
        "third_person_rate": rate(sum(1 for w in lower if w in THIRD)),
        "flesch": flesch,
        "comma_rate": prose.count(",") / n_sents if n_sents else 0.0,
        "word_count": float(n_words),
        "sentence_count": float(n_sents),
        "paragraph_count": float(len(paragraphs)),
        "heading_count": float(headings),
        "bullet_count": float(bullets),
        "numbered_count": float(numbered),
        "table_rows": float(table_rows),
        "bold_count": float(len(re.findall(r"\*\*.+?\*\*", text))),
        "list_ratio": listish / len(lines) if lines else 0.0,
        "mean_paragraph_words": n_words / len(paragraphs) if paragraphs else 0.0,
    })


def infer_shape(text: str) -> str:
    """Classify an output as table, list or prose.

    Two bullets is enough to call it a list -- a two-item answer to "list the
    figures" is still a list, and requiring three misread short gold outputs as
    prose, which then dragged the shape term of the score the wrong way.
    """
    f = extract(text)
    if f["table_rows"] >= 3:
        return "table"
    if f["bullet_count"] + f["numbered_count"] >= 2:
        return "list"
    return "prose"
