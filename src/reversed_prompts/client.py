"""LLMClient -- the seam between the optimizer and any model vendor.

The loop depends on *roles*, not model IDs (DESIGN.md §7). Two things this
buys immediately: swapping a role's model is a config change, and the whole
induction loop can be exercised in tests against a scripted fake with no key
and no spend.

Every completion carries its model ID and token usage back with it. That is a
Phase 0 requirement rather than a nicety -- scores that cannot be attributed to
a model are not comparable across rounds, and retrofitting the provenance after
scores exist means discarding the scores.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class Usage:
    """Running total. The budget guard reads this."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, c: Completion) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += c.prompt_tokens
            self.completion_tokens += c.completion_tokens
            self.cached_tokens += c.cached_tokens
            self.by_model[c.model] = self.by_model.get(c.model, 0) + 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        models = ", ".join(f"{m}×{n}" for m, n in sorted(self.by_model.items()))
        return (f"{self.calls} calls, {self.prompt_tokens} in "
                f"({self.cached_tokens} cached) / {self.completion_tokens} out"
                + (f" [{models}]" if models else ""))


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, role: str = "executor",
                 temperature: float = 0.0) -> Completion:
        ...


class BudgetExceeded(RuntimeError):
    pass


class UnknownModel(RuntimeError):
    pass


DEFAULT_MODEL = "gpt-5.6-terra"
ROLES = ("inducer", "executor", "judge", "features")


def resolve_models(overrides: dict[str, str] | None = None,
                   env: dict[str, str] | None = None) -> dict[str, str]:
    """Work out which model each role uses.

    Precedence, strongest first: an explicit override (what `--model` sets),
    `$REVPROMPT_MODEL_<ROLE>`, `$REVPROMPT_MODEL`, then the default. Kept as a
    free function so the precedence can be tested without an API key.
    """
    env = os.environ if env is None else env
    every = env.get("REVPROMPT_MODEL")
    out = {}
    for role in ROLES:
        out[role] = (env.get(f"REVPROMPT_MODEL_{role.upper()}")
                     or every or DEFAULT_MODEL)
    for role, model in (overrides or {}).items():
        if model:
            out[role] = model
    return out


class OpenAIClient:
    """Real calls. Roles map to model IDs through `models`.

    Sampling parameters are best-effort. `temperature=0` and a fixed `seed`
    are what the eval loop wants, but newer reasoning models reject both --
    they only accept the default temperature. Rather than keep a list of which
    models allow what, the client sends them, notices a refusal, drops that
    parameter and retries. The cost is one wasted call per parameter per run;
    the benefit is that a model shipping tomorrow needs no code change.

    Losing `temperature=0` does mean generations vary more between runs. That
    is a real hit to score stability, which is part of why the design leans on
    deterministic Tier-1 metrics rather than trusting single generations.

    Precedence for a role's model, strongest first: an explicit `models`
    argument (what `--model` sets), then `$REVPROMPT_MODEL_<ROLE>`, then
    `$REVPROMPT_MODEL`, then `DEFAULT_MODELS`. Environment variables mean the
    model can be changed without editing code, which matters because a model
    change re-baselines every score and is therefore something you do
    deliberately and often.
    """

    def __init__(self, models: dict[str, str] | None = None,
                 max_tokens_budget: int | None = None, seed: int = 7,
                 verify_models: bool = True, on_unsupported=None):
        from openai import OpenAI  # imported lazily so tests need no dependency

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI()

        self.models = resolve_models(models)
        self.usage = Usage()
        self.budget = max_tokens_budget
        self.seed = seed
        # Populated the first time the API refuses a sampling parameter.
        self.unsupported: set[str] = set()
        self.on_unsupported = on_unsupported
        if verify_models:
            self._verify_models()

    def available_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list().data)

    def _verify_models(self) -> None:
        """Fail before spending anything if a role points at a model this key
        cannot use. Without this the first call dies mid-run on an opaque
        upstream error, after the run has already been set up."""
        try:
            available = set(self.available_models())
        except Exception:
            return                      # cannot list models; let the call fail
        missing = {r: m for r, m in self.models.items() if m not in available}
        if not missing:
            return
        wanted = sorted(set(missing.values()))
        near = [m for m in sorted(available)
                if any(m.startswith(w.split("-")[0][:4]) for w in wanted)]
        hint = "\n  ".join(near[:15]) or "(none look similar)"
        raise UnknownModel(
            f"model(s) not available on this key: {', '.join(wanted)}\n"
            f"roles affected: {', '.join(sorted(missing))}\n"
            f"closest available:\n  {hint}\n"
            f"Set one with --model, $REVPROMPT_MODEL, or a per-role "
            f"$REVPROMPT_MODEL_EXECUTOR.")

    # Sampling controls newer reasoning models reject. Dropped on first refusal
    # rather than gated on a hardcoded model list, which would need editing
    # every time a model ships.
    OPTIONAL_PARAMS = ("temperature", "seed", "top_p")

    def _refused_param(self, err: Exception) -> str | None:
        """Which optional parameter did the API refuse, if any."""
        msg = str(err).lower()
        if "unsupported" not in msg and "not supported" not in msg:
            return None
        for name in self.OPTIONAL_PARAMS:
            if name in self.unsupported:
                continue
            if f"'{name}'" in msg or f'"{name}"' in msg:
                return name
        return None

    def complete(self, system: str, user: str, *, role: str = "executor",
                 temperature: float = 0.0) -> Completion:
        if self.budget is not None and self.usage.total_tokens >= self.budget:
            raise BudgetExceeded(
                f"token budget {self.budget} exhausted ({self.usage.summary()})")

        model = self.models.get(role, self.models["executor"])
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

        # Retry once per refused parameter, then give up. Each refusal is
        # remembered, so a run pays this at most len(OPTIONAL_PARAMS) times
        # rather than on every call.
        for _ in range(len(self.OPTIONAL_PARAMS) + 1):
            kwargs: dict = {"model": model, "messages": messages}
            if "temperature" not in self.unsupported:
                kwargs["temperature"] = temperature
            if "seed" not in self.unsupported:
                kwargs["seed"] = self.seed
            try:
                r = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                refused = self._refused_param(e)
                if refused is None:
                    raise
                self.unsupported.add(refused)
                if self.on_unsupported:
                    self.on_unsupported(refused, model)
        else:                                   # pragma: no cover - defensive
            raise RuntimeError("exhausted parameter retries")
        u = r.usage
        detail = getattr(u, "prompt_tokens_details", None)
        c = Completion(
            text=(r.choices[0].message.content or "").strip(),
            model=r.model,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            cached_tokens=getattr(detail, "cached_tokens", 0) or 0,
        )
        self.usage.record(c)
        return c


class ScriptedClient:
    """Deterministic stand-in for tests and dry runs.

    `responses` maps a substring of the user message to the text to return, so
    a test can script "when asked to hypothesize, say X; when asked to execute,
    say Y" without matching whole prompts. Falls back to `default`.
    """

    def __init__(self, responses: dict[str, str] | None = None,
                 default: str = "", model: str = "scripted"):
        self.responses = responses or {}
        self.default = default
        self.model = model
        self.usage = Usage()
        self.calls: list[tuple[str, str, str]] = []

    def complete(self, system: str, user: str, *, role: str = "executor",
                 temperature: float = 0.0) -> Completion:
        self.calls.append((role, system, user))
        text = self.default
        for needle, response in self.responses.items():
            if needle in user or needle in system:
                text = response
                break
        c = Completion(text=text, model=self.model,
                       prompt_tokens=len(user.split()),
                       completion_tokens=len(text.split()))
        self.usage.record(c)
        return c


class ObedientClient:
    """An offline stand-in that plays every role the loop needs.

    `ScriptedClient` returns a constant, so every candidate scores identically
    and the loop cannot be observed to do anything. This one behaves like a
    competent, literal-minded model:

    * **executor** -- obeys "markdown table", "bulleted list", "exactly N
      sentences", "N-M words" and "Prefer short/long sentences".
    * **producer** -- reads the measured properties it was handed and writes an
      instruction restating them. A crude stand-in for real recovery, but it
      exercises the whole path and its output is checkable.
    * **critic / reviser** -- returns changes and applies them.
    * **judge** -- scores similarity by word overlap.

    What it cannot do is judge *content*: it never decides whether a document
    contains the thing being asked for, so it always fails the negative half of
    a control set. That is the honest position -- offline runs prove the path
    works, not that recovery works.

    Vocabulary is drawn from the source document. With a fixed word list,
    producing more words drives type-token ratio toward zero, so a *correct*
    length instruction lowers the score and the double reports the opposite of
    the truth.
    """

    FALLBACK = ("alignment retrieval reasoning corpus evidence structure prompt "
                "selection budget candidate judge baseline signal").split()

    def __init__(self, model: str = "obedient"):
        self.usage = Usage()
        self.calls: list[tuple[str, str, str]] = []
        self._pool: list[str] = []

    # ---------------------------------------------------------------- helpers

    def _load_pool(self, user: str) -> None:
        import re as _re
        if self._pool:
            return
        doc = user.split("</document>")[0]
        words = _re.findall(r"[A-Za-z][A-Za-z'-]{2,}", doc)
        self._pool = words[::7][:4000] or list(self.FALLBACK)

    def _words(self, n: int, offset: int = 0) -> str:
        pool = self._pool or self.FALLBACK
        return " ".join(pool[(offset * 97 + i * 13) % len(pool)]
                        for i in range(max(n, 1)))

    # ------------------------------------------------------------------ roles

    def _produce(self, user: str) -> str:
        """Restate the measured properties as an instruction."""
        import re as _re
        block = user.split("MEASURED PROPERTIES")[-1]
        words = [int(m) for m in _re.findall(r"(\d+) words", block)]
        sents = [int(m) for m in _re.findall(r"(\d+) sentences", block)]
        shapes = _re.findall(r"shape: (\w+)", block)
        negative = "does NOT contain" in block

        parts = ["Answer the question using only the document above."]
        if shapes and len(set(shapes)) == 1:
            if shapes[0] == "table":
                parts.append("Format the answer as a markdown table.")
            elif shapes[0] == "list":
                parts.append("Format the answer as a bulleted list.")
            else:
                parts.append("Answer in prose. Do not use bullet points.")
        if sents and len(set(sents)) == 1 and sents[0] <= 5:
            parts.append(f"Use exactly {sents[0]} sentences.")
        elif words:
            mean = sum(words) // len(words)
            parts.append(f"Aim for {int(mean * 0.8)}-{int(mean * 1.2)} words.")
        if negative:
            parts.append("If the document does not contain it, reply with exactly: NA")
        return " ".join(parts)

    def _execute(self, user: str) -> str:
        import re as _re
        self._load_pool(user)
        doc, _, instruction = user.partition("</document>")

        # Deliberately no NA handling. Deciding whether a document contains
        # the requested content is the semantic judgement this double cannot
        # make, and a keyword heuristic standing in for it would report success
        # on negative controls that a real model might well fail. Negative
        # cases are only meaningfully exercised against a real model.

        sent_match = _re.search(r"exactly (\d+) sentences", instruction)
        word_match = _re.search(r"(\d+)-(\d+) words", instruction)
        target = 60
        if word_match:
            target = (int(word_match.group(1)) + int(word_match.group(2))) // 2

        per_sentence = 18
        if "Prefer short sentences" in instruction:
            per_sentence = 10
        elif "Prefer long sentences" in instruction:
            per_sentence = 30

        if "markdown table" in instruction:
            rows = max(target // 12, 2)
            return "\n".join(["| Aspect | Detail |", "|---|---|"]
                             + [f"| {self._words(2, i)} | {self._words(6, i)} |"
                                for i in range(rows)])
        if "bulleted list" in instruction:
            items = max(target // 10, 3)
            return "\n".join(f"- {self._words(10, i)}." for i in range(items))
        if sent_match:
            n = int(sent_match.group(1))
            per = max(target // n, 6)
            return " ".join(f"{self._words(per, i).capitalize()}." for i in range(n))
        n = max(target // per_sentence, 1)
        return " ".join(f"{self._words(per_sentence, i).capitalize()}."
                        for i in range(n))

    @staticmethod
    def _similarity(user: str) -> str:
        """Jaccard overlap, formatted the way the judge is asked to reply.

        The format matters as much as the number: the parser rejects anything
        that is not x.yz, so a double replying "7" would make every offline run
        fail for a reason unrelated to what it is testing.
        """
        import re as _re
        a, _, b = user.partition("INSTRUCTION B:")
        def words(t): return {w.lower() for w in _re.findall(r"[A-Za-z]{4,}", t)}
        wa, wb = words(a), words(b)
        if not wa or not wb:
            return "0.00"
        return f"{len(wa & wb) / len(wa | wb):.2f}"

    # ---------------------------------------------------------------- dispatch

    def complete(self, system: str, user: str, *, role: str = "executor",
                 temperature: float = 0.0) -> Completion:
        self.calls.append((role, system, user))

        if system.startswith("You recover prompts"):
            text = self._produce(user)
        elif system.startswith("You review recovered prompts"):
            text = '["State what to do when the document lacks the requested content."]'
        elif system.startswith("You rewrite instructions"):
            base = user.split("INSTRUCTION:")[-1].split("CHANGES:")[0].strip()
            text = base + " If the document does not contain it, reply with exactly: NA"
        elif system.startswith("You compare two instructions"):
            text = self._similarity(user)
        else:
            text = self._execute(user)

        c = Completion(text=text, model="obedient",
                       prompt_tokens=len(user.split()),
                       completion_tokens=len(text.split()))
        self.usage.record(c)
        return c
