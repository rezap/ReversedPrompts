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


class OpenAIClient:
    """Real calls. Roles map to model IDs through `models`."""

    DEFAULT_MODELS = {
        "inducer": "gpt-4.1",
        "executor": "gpt-4.1-mini",
        "judge": "gpt-4.1",
        "features": "gpt-4.1-mini",
    }

    def __init__(self, models: dict[str, str] | None = None,
                 max_tokens_budget: int | None = None, seed: int = 7):
        from openai import OpenAI  # imported lazily so tests need no dependency

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI()
        self.models = {**self.DEFAULT_MODELS, **(models or {})}
        self.usage = Usage()
        self.budget = max_tokens_budget
        self.seed = seed

    def complete(self, system: str, user: str, *, role: str = "executor",
                 temperature: float = 0.0) -> Completion:
        if self.budget is not None and self.usage.total_tokens >= self.budget:
            raise BudgetExceeded(
                f"token budget {self.budget} exhausted ({self.usage.summary()})")

        model = self.models.get(role, self.models["executor"])
        r = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=self.seed,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
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
    """A test double that actually follows shape and length instructions.

    `ScriptedClient` returns a constant, so every candidate scores identically
    and the loop cannot be observed to improve anything. This one simulates a
    model that does what the prompt says: it honours "markdown table",
    "bulleted list", "Aim for N-M words" and "Use exactly N sentences", and
    ignores everything else.

    That makes it a real test of the optimizer rather than of the plumbing --
    a spec whose structure facet matches gold produces output whose measured
    features match gold, so Tier-1 lift is observable with no key and no spend.
    Inducer-role calls return canned JSON so the generators have something to
    parse.
    """

    FALLBACK = ("alignment retrieval reasoning corpus evidence structure prompt "
                "selection budget candidate judge baseline signal").split()

    def __init__(self, model: str = "obedient"):
        self.usage = Usage()
        self.calls: list[tuple[str, str, str]] = []
        self._pool: list[str] = []

    def _load_pool(self, user: str) -> None:
        """Draw vocabulary from the document, as a real model would.

        With a fixed tiny word list, producing more words drives type-token
        ratio toward zero, so a correct length instruction *lowers* the score
        and the double reports the opposite of the truth. Sampling the source
        document keeps lexical features in a realistic range.
        """
        import re as _re
        if self._pool:
            return
        doc = user.split("</document>")[0]
        words = _re.findall(r"[A-Za-z][A-Za-z'-]{2,}", doc)
        # stride through the document so the pool spans it rather than
        # sampling one section's jargon
        self._pool = words[::7][:4000] or list(self.FALLBACK)

    def _words(self, n: int, offset: int = 0) -> str:
        pool = self._pool or self.FALLBACK
        step = 13                      # coprime-ish stride: avoids short cycles
        return " ".join(pool[(offset * 97 + i * step) % len(pool)]
                        for i in range(max(n, 1)))

    def complete(self, system: str, user: str, *, role: str = "executor",
                 temperature: float = 0.0) -> Completion:
        import re as _re
        self.calls.append((role, system, user))

        if role == "inducer":
            text = '["Keep the answer tightly scoped to the question."]'
        elif role == "judge":
            text = "7"
        else:
            self._load_pool(user)
            instruction = user.split("</document>")[-1]
            sent_match = _re.search(r"exactly (\d+) sentences", instruction)
            word_match = _re.search(r"Aim for (\d+)-(\d+) words", instruction)
            target = 60
            if word_match:
                target = (int(word_match.group(1)) + int(word_match.group(2))) // 2
            # obey sentence-length too, otherwise the style facet is invisible
            # to this double and every style clause scores as a no-op
            per_sentence = 18
            if "Prefer short sentences" in instruction:
                per_sentence = 10
            elif "Prefer long sentences" in instruction:
                per_sentence = 30

            if "markdown table" in instruction:
                rows = max(target // 12, 2)
                text = "\n".join(
                    ["| Aspect | Detail |", "|---|---|"]
                    + [f"| {self._words(2, i)} | {self._words(6, i)} |"
                       for i in range(rows)])
            elif "bulleted list" in instruction:
                items = max(target // 10, 3)
                text = "\n".join(f"- {self._words(10, i)}." for i in range(items))
            elif sent_match:
                n = int(sent_match.group(1))
                per = max(target // n, 6)
                text = " ".join(f"{self._words(per, i).capitalize()}." for i in range(n))
            else:
                n = max(target // per_sentence, 1)
                text = " ".join(f"{self._words(per_sentence, i).capitalize()}."
                                for i in range(n))

        c = Completion(text=text, model="obedient",
                       prompt_tokens=len(user.split()),
                       completion_tokens=len(text.split()))
        self.usage.record(c)
        return c
