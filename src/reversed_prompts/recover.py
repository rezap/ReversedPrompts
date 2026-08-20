"""Recover the instruction that turned an input into an output.

The loop, per prompt group:

    propose ─▶ run it ─▶ critique ─▶ revise ─┐
       ▲                                     │
       └─────────────────────────────────────┘

**Propose** hands the producer the outputs plus the deterministic measurements
of them, and asks for the instruction. **Run it** executes the candidate on
every input in the group and scores the results against the gold outputs.
**Critique** shows the critic what the candidate actually produced next to what
was wanted, and asks what to change about the *instruction*. **Revise** applies
those changes.

Two scores come out, and they answer different questions:

* **prompt score** -- would someone following this instruction behave like
  someone following the gold one? This is the product being measured.
* **output fidelity** -- does running it actually reproduce the wanted output?

Neither is sufficient. A candidate can reproduce the output while stating the
wrong rule (it memorised this answer), or read like the gold instruction while
behaving differently. Both are reported; the loop optimises output fidelity,
because that is the signal available without peeking at the gold prompt.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from . import prompts
from .client import BudgetExceeded, LLMClient
from .features import extract, infer_shape
from .ingest import Pair, PromptGroup
from .metrics import SCORING_VERSION, Scale, describe_gap, tier1
from .similarity import MAX_JUDGE_ATTEMPTS, PromptScore, score_prompt

# Per *input* document, in the producer and critic prompts. Characters, not
# tokens -- roughly 8k tokens of English. Outputs are never truncated: they are
# what the instruction has to be inferred from, and an instruction inferred
# from half an answer describes half a task.
EXCERPT_CHARS = 32_000
NAIVE = "Answer the question about the document above."


@dataclass
class Candidate:
    """A recovered instruction and everything known about it."""

    text: str
    origin: str = "producer"
    round: int = 0
    fidelity: float = 0.0
    per_pair: dict[str, float] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    prompt_score: PromptScore | None = None

    def __lt__(self, other: "Candidate") -> bool:
        return self.fidelity < other.fidelity


@dataclass
class GroupResult:
    group: PromptGroup
    best: Candidate
    naive: Candidate
    history: list[Candidate] = field(default_factory=list)
    rounds: int = 0
    seconds: float = 0.0
    stopped_because: str = "rounds exhausted"
    prompts_version: str = prompts.VERSION
    scoring_version: str = SCORING_VERSION
    # Which model produced this, and which corpus the score was normalised
    # against. Carried on the result rather than reconstructed later, because
    # later there is nothing left to reconstruct it from.
    models: dict[str, str] = field(default_factory=dict)
    scale_fingerprint: str = ""

    @property
    def beats_naive(self) -> bool:
        return self.best.fidelity > self.naive.fidelity


# ------------------------------------------------------------------- evidence

def measure(pairs: list[Pair]) -> str:
    """Deterministic facts about the wanted outputs, for the producer.

    Handed over as *evidence*, not as instructions to copy. The producer
    decides what is worth stating; measurements it cannot justify from the
    outputs should not appear in the recovered prompt.
    """
    lines = []
    for p in pairs:
        f = extract(p.output)
        bits = [f"{int(f['word_count'])} words",
                f"{int(f['sentence_count'])} sentences",
                f"shape: {infer_shape(p.output)}"]
        if f["table_rows"]:
            bits.append(f"{int(f['table_rows'])} table rows")
        if f["bullet_count"] + f["numbered_count"]:
            bits.append(f"{int(f['bullet_count'] + f['numbered_count'])} list items")
        if p.is_negative:
            bits.append("this input does NOT contain what was asked for")
        lines.append(f"- output for input {p.id}: " + ", ".join(bits))
    return "\n".join(lines)


def _blocks(pairs: list[Pair], cap: int = EXCERPT_CHARS) -> str:
    parts = []
    for i, p in enumerate(pairs, 1):
        head = f"### Example {i}"
        if p.is_negative:
            head += "  (the document does not contain the requested content)"
        parts.append(f"{head}\nDOCUMENT (excerpt):\n{p.input_text[:cap]}"
                     f"\n\nOUTPUT:\n{p.output}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class Truncation:
    """One input the producer and critic will only partly see."""

    pair_id: str
    group_id: str
    total: int
    cap: int

    @property
    def lost(self) -> int:
        return self.total - self.cap


def truncations(groups: list[PromptGroup],
                cap: int = EXCERPT_CHARS) -> list[Truncation]:
    """Which inputs the cap will cut, worst first.

    The executor always sees the whole document; only the producer and critic
    are capped. So a cut here means the components doing the *inference* are
    reasoning about an output partly generated from text they cannot see, and
    a low score for that group is ambiguous -- it may mean "could not see the
    evidence" rather than "could not recover the prompt". Worth saying out
    loud before any money is spent, not after.
    """
    found = [Truncation(pair_id=p.id, group_id=g.id, total=len(p.input_text),
                        cap=cap)
             for g in groups for p in g.pairs if len(p.input_text) > cap]
    return sorted(found, key=lambda t: t.lost, reverse=True)


# --------------------------------------------------------------------- steps

def propose(client: LLMClient, group: PromptGroup,
            cap: int = EXCERPT_CHARS) -> Candidate:
    text = client.complete(
        prompts.PRODUCER,
        prompts.producer_user(measure(group.pairs), _blocks(group.pairs, cap)),
        role="inducer",
    ).text.strip()
    return Candidate(text=text or NAIVE, origin="producer")


def run_candidate(client: LLMClient, candidate: Candidate, group: PromptGroup,
                  scale: Scale) -> Candidate:
    """Execute on every input in the group and score against the gold outputs."""
    scores, outputs = {}, {}
    for pair in group.pairs:
        c = client.complete(
            prompts.EXECUTOR,
            f"<document>\n{pair.input_text}\n</document>\n\n{candidate.text}",
            role="executor",
        )
        outputs[pair.id] = c.text
        scores[pair.id] = tier1(pair.output, c.text, scale)[0]

    candidate.per_pair = scores
    candidate.outputs = outputs
    # the weakest member sets the score: a control set is only satisfied when
    # the instruction works on every input, including the negative one
    candidate.fidelity = min(scores.values()) if scores else 0.0
    return candidate


def critique(client: LLMClient, candidate: Candidate, group: PromptGroup,
             scale: Scale, cap: int = EXCERPT_CHARS) -> list[str]:
    """Ask what to change about the instruction, worst pair first."""
    from .parsing import parse_clauses

    worst_id = min(candidate.per_pair, key=candidate.per_pair.get)
    pair = next(p for p in group.pairs if p.id == worst_id)
    produced = candidate.outputs.get(worst_id, "")

    gap = describe_gap(pair.output, produced, scale)
    user = prompts.critic_user(pair.input_text[:cap], candidate.text,
                               produced, pair.output)
    user += f"\n\nMEASURED DIFFERENCES between produced and wanted:\n{gap}"
    return parse_clauses(client.complete(prompts.CRITIC, user, role="inducer").text)


def revise(client: LLMClient, candidate: Candidate, changes: list[str],
           round_no: int) -> Candidate:
    text = client.complete(prompts.REVISER,
                           prompts.reviser_user(candidate.text, changes),
                           role="inducer").text.strip()
    if not text or text == candidate.text:
        return candidate
    return Candidate(text=text, origin=f"revised<-{candidate.origin}",
                     round=round_no)


# ---------------------------------------------------------------------- loop

def recover(client: LLMClient, group: PromptGroup, scale: Scale, *,
            rounds: int = 2, verbose: bool = False,
            excerpt_chars: int = EXCERPT_CHARS) -> GroupResult:
    started = time.monotonic()
    stopped = "rounds exhausted"

    def say(m: str) -> None:
        if verbose:
            print(m, flush=True)

    naive = run_candidate(client, Candidate(text=NAIVE, origin="naive"),
                          group, scale)
    say(f"  naive fidelity {naive.fidelity:.4f}")

    history: list[Candidate] = []
    completed = 0
    try:
        current = run_candidate(client, propose(client, group, excerpt_chars),
                                group, scale)
        history.append(current)
        say(f"  round 0 fidelity {current.fidelity:.4f}")

        for r in range(1, rounds + 1):
            changes = critique(client, current, group, scale, excerpt_chars)
            if not changes:
                stopped = "critic reported nothing to change"
                break
            revised = revise(client, current, changes, r)
            if revised is current:
                stopped = "revision produced no change"
                break
            revised = run_candidate(client, revised, group, scale)
            history.append(revised)
            completed = r
            say(f"  round {r} fidelity {revised.fidelity:.4f}")
            current = revised if revised.fidelity > current.fidelity else current
    except BudgetExceeded as e:
        stopped = str(e)
        if not history:
            raise

    best = max(history) if history else naive
    return GroupResult(group=group, best=best, naive=naive, history=history,
                       rounds=completed, seconds=time.monotonic() - started,
                       stopped_because=stopped,
                       models=dict(getattr(client, "models", {})),
                       scale_fingerprint=scale.fingerprint)


def score_against_gold(client: LLMClient, result: GroupResult, *,
                       judge_attempts: int = MAX_JUDGE_ATTEMPTS) -> GroupResult:
    """Attach the prompt-similarity score. Uses the gold prompt, so it runs
    only after recovery is finished -- never inside the loop.

    A judge that will not answer in the required format raises out of here.
    That aborts the run, which is the intended behaviour: a fabricated score is
    worse than a failed run, because nothing downstream can tell it apart from
    a real one.
    """
    golds = [p.output for p in result.group.pairs]
    result.best.prompt_score = score_prompt(
        client, result.best.text, result.group.gold_prompt, golds,
        max_attempts=judge_attempts,
        context=f"group {result.group.id} (recovered prompt)")
    result.naive.prompt_score = score_prompt(
        client, result.naive.text, result.group.gold_prompt, golds,
        max_attempts=judge_attempts,
        context=f"group {result.group.id} (naive baseline)")
    return result


def fit_scale(pairs: list[Pair]) -> Scale:
    return Scale.fit([p.output for p in pairs])


def recover_all(client: LLMClient, groups: list[PromptGroup], *, rounds: int = 2,
                scale: Scale | None = None, judge: bool = True,
                verbose: bool = False,
                excerpt_chars: int = EXCERPT_CHARS,
                judge_attempts: int = MAX_JUDGE_ATTEMPTS) -> list[GroupResult]:
    all_pairs = [p for g in groups for p in g.pairs]
    scale = scale or fit_scale(all_pairs)
    results = []
    for g in groups:
        if verbose:
            kind = "control set" if g.is_control else "single pair"
            print(f"{g.id} ({kind}, {len(g)} pair{'s' if len(g) > 1 else ''})")
        r = recover(client, g, scale, rounds=rounds, verbose=verbose,
                    excerpt_chars=excerpt_chars)
        if judge:
            r = score_against_gold(client, r, judge_attempts=judge_attempts)
        results.append(r)
    return results


def summarise(results: list[GroupResult]) -> dict[str, float]:
    fid = [r.best.fidelity for r in results]
    beat = sum(r.beats_naive for r in results)
    judged = [r.best.prompt_score.judged for r in results if r.best.prompt_score]
    contam = [r.best.prompt_score.contamination for r in results if r.best.prompt_score]
    out = {
        "groups": float(len(results)),
        "beat_naive": float(beat),
        "mean_fidelity": statistics.fmean(fid) if fid else 0.0,
    }
    if judged:
        out["mean_prompt_similarity"] = statistics.fmean(judged)
        out["max_contamination"] = max(contam)
        # Counted, not averaged in. One contaminated prompt among twelve clean
        # ones barely moves a mean, and is the single most important thing to
        # notice in the run.
        out["contaminated_groups"] = float(sum(
            r.best.prompt_score.contaminated for r in results
            if r.best.prompt_score))
    return out
