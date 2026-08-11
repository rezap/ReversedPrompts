"""Tolerant parsing of model replies.

Models wrap JSON in prose, in fences, or drop it for a bullet list. A parser
that only accepts clean JSON turns a usable reply into an empty result, which
looks like "the critic had no suggestions" and silently stops the loop.
"""
from __future__ import annotations

import json
import re

MAX_CLAUSES = 8


def parse_clauses(text: str) -> list[str]:
    """Pull a list of short instructions out of whatever the model returned."""
    if not text or not text.strip():
        return []

    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()][:MAX_CLAUSES]
        except json.JSONDecodeError:
            pass

    # a model that says "no changes needed" in prose means an empty list
    if re.match(r"^\W*(no changes?|none|nothing|already correct)\b", text.strip(), re.I):
        return []

    return [re.sub(r"^\s*[-*\d.)]+\s*", "", l).strip()
            for l in text.splitlines() if l.strip()][:MAX_CLAUSES]
