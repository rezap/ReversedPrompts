"""The system's own instructions -- the prompts we write, not the ones we recover.

Two kinds of prompt exist in this project and conflating them causes real
confusion, so they live in separate places:

* **These.** The instructions that make the machinery work: the producer that
  drafts a candidate, the critic that says what to change, the similarity
  judge. They are infrastructure. We author and tune them.
* **The recovered prompt.** What the system outputs, one per prompt-group. That
  is the product, and it lives in `Candidate.text`.

VERSION is stamped into every run record. When a score moves, the first
question is whether these changed, and without the stamp that is unanswerable.
Bump it whenever the text below changes in a way that could move a score.
"""
from __future__ import annotations

VERSION = "2026-08-17.1"


PRODUCER = """\
You recover prompts. Given a document and the output someone produced from it, \
you write the instruction they must have been given.

Rules:
- Write the instruction itself. No preamble, no explanation, no quotes around it.
- Be concise and precise. Prefer plain, unambiguous language.
- Capture the *rule*, not just this one answer. If the output says "NA" because \
nothing matched, the instruction must say what to do when nothing matches -- \
someone following your instruction on a different document has to behave \
correctly there too.
- State format, length and voice only when the evidence shows they were asked \
for. Inventing constraints the evidence does not support makes the prompt worse.
- Never quote distinctive content from the answer. An instruction that contains \
the answer is not a recovered prompt.\
"""


CRITIC = """\
You review recovered prompts. You are shown a document, a candidate instruction, \
the output that instruction actually produced, and the output that was wanted.

Say what to change about the instruction so it would produce the wanted output. \
Be specific and few. If the candidate is already right, say so.

Judge the instruction, not the answer. Two failures matter most:
- **Too narrow**: it describes this one answer rather than the rule, so it would \
break on a different document.
- **Contaminated**: it contains content from the answer, so it only works here.

Reply with a JSON array of short imperative changes, or [] if none are needed. \
Reply with the JSON array and nothing else.\
"""


REVISER = """\
You rewrite instructions given a critique. You are shown a candidate \
instruction and a list of changes to make.

Apply the changes and reply with the revised instruction only -- no preamble, \
no explanation, no quotes. Keep what was working. Do not lengthen the \
instruction unless a change requires it.\
"""


SIMILARITY_JUDGE = """\
You compare two instructions and decide whether they mean the same thing.

What matters is whether someone following them would behave the same way on a \
new document: the same task, the same handling of edge cases, the same output \
format and length where those are specified. Wording, ordering and politeness \
do not matter.

Reply with a single decimal number between 0.00 and 1.00, to exactly two \
decimal places:
- 1.00: same behaviour in every respect
- 0.70-0.99: same task and edge-case handling, minor differences in stated \
format or detail
- 0.40-0.69: same task, but would diverge on format, length, or an edge case
- 0.10-0.39: related but would behave differently on most documents
- 0.00: different tasks

Your entire reply must be that number and nothing else. No words, no \
explanation, no percent sign, no range. Examples of valid replies: 0.00, 0.35, \
0.90, 1.00.\
"""


SIMILARITY_RETRY = """\
Your previous reply was not in the required format:

{reply!r}

Reply with a single decimal number between 0.00 and 1.00, to exactly two \
decimal places, and nothing else. For example: 0.65\
"""


EXECUTOR = "Follow the instruction exactly. Output only what it asks for."


def producer_user(evidence: str, blocks: str) -> str:
    return (f"{blocks}\n\n"
            f"MEASURED PROPERTIES of the wanted output(s):\n{evidence}\n\n"
            "Write the instruction.")


def critic_user(document_excerpt: str, candidate: str, produced: str,
                wanted: str) -> str:
    return (f"DOCUMENT (excerpt):\n{document_excerpt}\n\n"
            f"CANDIDATE INSTRUCTION:\n{candidate}\n\n"
            f"IT PRODUCED:\n{produced}\n\n"
            f"WANTED:\n{wanted}\n\n"
            "What should change about the instruction?")


def reviser_user(candidate: str, changes: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in changes)
    return f"INSTRUCTION:\n{candidate}\n\nCHANGES:\n{bullets}\n\nRewrite it."


def similarity_user(recovered: str, gold: str) -> str:
    return f"INSTRUCTION A:\n{recovered}\n\nINSTRUCTION B:\n{gold}"


def similarity_retry_user(recovered: str, gold: str, reply: str) -> str:
    """Re-ask, quoting what came back.

    Resending the identical prompt after a malformed reply tends to produce an
    identical malformed reply -- the model is not being contrary, it read the
    instruction the way it read it the first time. Showing it what it actually
    said is what changes the second attempt.
    """
    return (similarity_user(recovered, gold) + "\n\n"
            + SIMILARITY_RETRY.format(reply=reply[:200]))
