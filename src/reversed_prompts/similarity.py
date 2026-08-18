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
from .prompts import SIMILARITY_JUDGE, similarity_retry_user, similarity_user

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


# Above this share of the instruction's 4-grams coming from the answer, the
# prompt is treated as having smuggled the answer rather than stated the rule.
CONTAMINATION_LIMIT = 0.05


@dataclass(frozen=True)
class PromptScore:
    judged: float          # 0..1, from the similarity judge
    lexical: float         # 0..1, token overlap
    contamination: float   # 0..1, higher is worse -- never averaged in

    @property
    def contaminated(self) -> bool:
        """Did this instruction contain the answer it was supposed to elicit?

        Reported as a flag and counted, never blended into the similarity
        score. A contaminated prompt scores well on both other measures and is
        worthless; averaging it into them would hide exactly that.
        """
        return self.contamination > CONTAMINATION_LIMIT

    def line(self) -> str:
        flag = "  CONTAMINATED" if self.contaminated else ""
        return (f"judged {self.judged:.2f}  lexical {self.lexical:.2f}"
                f"  contamination {self.contamination:.2f}{flag}")


class JudgeFormatError(RuntimeError):
    """The judge would not answer in the required format.

    Raised rather than defaulted. The previous code returned 0.5 for a reply it
    could not parse, which fabricates a measurement: in the summary a made-up
    0.5 is indistinguishable from a judged 0.5. A run that dies loudly can be
    re-run; a run quietly full of invented numbers cannot be detected at all.
    """


# Exactly two decimal places, and no higher than 1.00. Anything looser lets
# "3/10" through as 3.0, or "0.9" through inconsistently with "0.90"; the whole
# point is that the judge's reply is unambiguous before it becomes a score.
SCORE_FORMAT = re.compile(r"^(?:0\.\d{2}|1\.00)$")
MAX_JUDGE_ATTEMPTS = 3


def parse_score(reply: str) -> float | None:
    """The judge's score, or None if the reply is not in the required format."""
    match = SCORE_FORMAT.match(reply.strip())
    return float(match.group(0)) if match else None


def judge_similarity(client: LLMClient, recovered: str, gold: str, *,
                     max_attempts: int = MAX_JUDGE_ATTEMPTS,
                     context: str = "") -> float:
    """Ask the judge for a 0.00-1.00 score, retrying a malformed reply.

    Raises `JudgeFormatError` once the attempts are exhausted. See that class
    for why this is not a default value.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    seen: list[str] = []
    for attempt in range(max_attempts):
        user = (similarity_user(recovered, gold) if attempt == 0
                else similarity_retry_user(recovered, gold, seen[-1]))
        reply = client.complete(SIMILARITY_JUDGE, user, role="judge").text
        score = parse_score(reply)
        if score is not None:
            return score
        seen.append(reply)

    where = f" for {context}" if context else ""
    attempts = "; ".join(f"{i + 1}: {r.strip()[:80]!r}"
                         for i, r in enumerate(seen))
    raise JudgeFormatError(
        f"the judge did not return a score in x.yz form (0.00-1.00)"
        f"{where} after {max_attempts} attempt(s). Replies were -- {attempts}")


def score_prompt(client: LLMClient, recovered: str, gold: str,
                 outputs: list[str], *,
                 max_attempts: int = MAX_JUDGE_ATTEMPTS,
                 context: str = "") -> PromptScore:
    worst = max((contamination(recovered, o) for o in outputs), default=0.0)
    return PromptScore(
        judged=judge_similarity(client, recovered, gold,
                                max_attempts=max_attempts, context=context),
        lexical=lexical(recovered, gold),
        contamination=worst,
    )
