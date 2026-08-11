#!/usr/bin/env python3
"""End-to-end smoke test: run one pair's gold prompt against the corpus.

This is the Phase 0 executor step in miniature. It proves the key works, the
model ID is valid, and the corpus/pair plumbing lines up -- and it shows the
model's answer next to the stored gold output so you can eyeball the gap the
induction loop will be trying to close.

    export OPENAI_API_KEY=sk-...
    python tools/smoke_openai.py                    # cheapest pair
    python tools/smoke_openai.py --pair reason-02   # a hard one
    python tools/smoke_openai.py --list-models

Costs money. One run sends the whole ~20k-word document (~27k tokens).
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "pairs" / "agentic-ai-survey.jsonl"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


def load_pairs():
    return [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]


def client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed -- pip install -e '.[dev]'")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    return OpenAI()


def list_models(c):
    ids = sorted(m.id for m in c.models.list().data)
    print(f"{len(ids)} models available on this key:\n")
    for i in ids:
        print(f"  {i}")


def check_model(c, model):
    """Fail early and helpfully rather than mid-run on a bad model ID."""
    ids = {m.id for m in c.models.list().data}
    if model in ids:
        return
    near = sorted(i for i in ids if i.split("-")[0] == model.split("-")[0])
    hint = "\n  ".join(near[:12]) if near else "(run --list-models)"
    sys.exit(f"model {model!r} is not available on this key. Closest:\n  {hint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="fetch-01", help="pair id (default: fetch-01)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"model id (default: {DEFAULT_MODEL}, or $OPENAI_MODEL)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--skip-model-check", action="store_true",
                    help="don't verify the model id against /v1/models first")
    args = ap.parse_args()

    c = client()

    if args.list_models:
        list_models(c)
        return

    pairs = load_pairs()
    try:
        pair = next(p for p in pairs if p["id"] == args.pair)
    except StopIteration:
        sys.exit(f"no pair {args.pair!r}. Available: {', '.join(p['id'] for p in pairs)}")

    if not args.skip_model_check:
        check_model(c, args.model)

    corpus = (ROOT / pair["input_ref"]).read_text()

    # Document first, instruction after -- keeps the cacheable prefix stable
    # across candidates, which is the whole cost story for the real eval loop.
    resp = c.chat.completions.create(
        model=args.model,
        temperature=0,
        messages=[
            {"role": "user", "content": f"<document>\n{corpus}\n</document>"},
            {"role": "user", "content": pair["target_prompt"]},
        ],
    )
    got = resp.choices[0].message.content.strip()
    gold = pair["output"]
    u = resp.usage

    bar = "=" * 78
    print(f"{bar}\npair     {pair['id']}  ({pair['category']}, {pair['output_shape']})")
    print(f"model    {resp.model}")
    print(f"tokens   {u.prompt_tokens} in / {u.completion_tokens} out")
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
    if cached is not None:
        print(f"cached   {cached} of {u.prompt_tokens} prompt tokens")
    print(f"\n{bar}\nPROMPT (the gold target the inducer must recover)\n{bar}")
    print(pair["target_prompt"])
    print(f"\n{bar}\nMODEL OUTPUT ({len(got.split())} words)\n{bar}\n{got}")
    print(f"\n{bar}\nGOLD OUTPUT ({len(gold.split())} words)\n{bar}\n{gold}")

    ratio = len(got.split()) / max(len(gold.split()), 1)
    print(f"\n{bar}\nlength ratio model/gold: {ratio:.2f}")
    print("This is a sanity check, not a score. Real scoring is §4.4 of docs/DESIGN.md.")


if __name__ == "__main__":
    main()
