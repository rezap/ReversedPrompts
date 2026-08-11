"""Is the recovered prompt the same instruction as the gold one?

Scored two ways, because neither alone is trustworthy:

* **Judged** -- a model decides whether someone following the two instructions
  would behave the same way on a *new* document. This is the measure that
  matches what we actually care about, and it needs a model because paraphrase
  is the norm: "list the authors, NA if none" and "extract author names,
  returning NA when absent" are the same instruction.
* **Lexical** -- token overlap. Cheap, deterministic, and useful mainly as a
  sanity check on the judge and as a tie-breaker. A high lexical score with a
  low judged score usually means the candidate copied gold's wording without
  its behaviour; the reverse means honest paraphrase.

There is also a check that has nothing to do with similarity and matters more
than either: whether the candidate smuggled the answer into the instruction.
A prompt containing the answer scores well on both measures above and is
worthless, so `contamination` is reported separately and never averaged in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .client import LLMClient
from .prompts import SIMILARITY_JUDGE, similarity_user

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

STOP = frozenset("""
a an the this that these those of in on to for from with by as at is are was
were be been being and or but if then than so it its and you your do does
""".split())


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text)
            if w.lower() not in STOP and len(w) > 2}


def lexical(recovered: str, gold: str) -> float:
    """Jaccard over content words. 0..1."""
    a, b = _content_words(recovered), _content_words(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def contamination(recovered: str, output: str, *, min_len: int = 4) -> float:
    """Fraction of the candidate's rare content that came from the answer.

    The optimizer's easiest win is to paste the answer into the instruction,
    and it will find that move if nothing penalises it. Any n-gram of
    `min_len`+ words shared between the instruction and the output it is
    supposed to elicit is treated as smuggled.
    """
    def grams(text: str) -> set[tuple[str, ...]]:
        words = [w.lower() for w in WORD.findall(text)]
        return {tuple(words[i:i + min_len])
                for i in range(len(words) - min_len + 1)}

    cand, ans = grams(recovered), grams(output)
    if not cand:
        return 0.0
    return len(cand & ans) / len(cand)


@dataclass(frozen=True)
class PromptScore:
    judged: float          # 0..1, from the similarity judge
    lexical: float         # 0..1, token overlap
    contamination: float   # 0..1, higher is worse -- never averaged in

    @property
    def combined(self) -> float:
        """Judged similarity, with contamination applied as a hard penalty."""
        return max(0.0, self.judged - self.contamination)

    def line(self) -> str:
        flag = "  CONTAMINATED" if self.contamination > 0.05 else ""
        return (f"judged {self.judged:.2f}  lexical {self.lexical:.2f}"
                f"  contamination {self.contamination:.2f}{flag}")


def judge_similarity(client: LLMClient, recovered: str, gold: str) -> float:
    c = client.complete(SIMILARITY_JUDGE, similarity_user(recovered, gold),
                        role="judge")
    digits = "".join(ch for ch in c.text if ch.isdigit())[:2]
    if not digits:
        return 0.5
    return min(int(digits), 10) / 10.0


def score_prompt(client: LLMClient, recovered: str, gold: str,
                 outputs: list[str]) -> PromptScore:
    worst = max((contamination(recovered, o) for o in outputs), default=0.0)
    return PromptScore(
        judged=judge_similarity(client, recovered, gold),
        lexical=lexical(recovered, gold),
        contamination=worst,
    )
