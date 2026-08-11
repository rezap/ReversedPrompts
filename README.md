# ReversedPrompts

An agentic flow to guess prompts that generate an output from an input set.

Given pairs of `(input files, output artifacts)`, recover the prompt that maps inputs to
outputs — including the style, structure, and content-selection rules that a naive
instruction like *"summarize this document"* would miss — then execute that recovered
prompt against new inputs using agentic RAG.

📄 **[Design & implementation plan](docs/DESIGN.md)**

## Status

Design phase, plus the evaluation data the implementation will be built against. No
induction pipeline yet.

- `data/` — one document and 20 `(prompt, output)` pairs. See [data/README.md](data/README.md).
- `tools/` — the scripts that produced that data, both reproducible.
- `tests/` — integrity checks over the pair set.

## Running the checks

```bash
pip install -e '.[dev]'
pytest
```

That needs no API key and costs nothing. It verifies the pairs are well-formed, that every
stored output was generated against the corpus currently in the tree, and that the prompts
carrying explicit constraints have outputs that satisfy them.

In CI, the same checks run from the **checks** workflow — Actions tab → *checks* → *Run
workflow*. It is manual-only; nothing runs on push.

## Talking to the API

Model access is the **OpenAI API** with a single `OPENAI_API_KEY` (§7 of the design doc).

```bash
export OPENAI_API_KEY=sk-...
python tools/smoke_openai.py                    # cheapest pair
python tools/smoke_openai.py --pair reason-02   # a hard one
python tools/smoke_openai.py --list-models
```

This runs one pair's gold prompt against the corpus and prints the model's answer beside
the stored one — the Phase 0 executor step in miniature. It sends the whole ~20k-word
document, so each run costs real money.

The same script runs in CI as the **api-smoke** job, which is off unless you tick
`run_api_smoke` when starting the workflow.

### Configuring the key in GitHub

The smoke job reads `OPENAI_API_KEY` from an environment named `openai`:

1. **Settings → Environments → New environment**, name it `openai`.
2. **Add secret** → name `OPENAI_API_KEY`, paste the key.
3. Optionally add yourself under **Required reviewers** so the job pauses for approval
   before it can spend anything.

An environment is used rather than a plain repository secret so the spend can be gated. Set
a usage limit on the key itself in the OpenAI dashboard too — repository permissions are
the lock, but the cap is what bounds the damage if a key ever leaks.

Fork PRs do not receive secrets, so the smoke job cannot run from an external
contribution. That is intended.

## Regenerating the data

```bash
pip install -e '.[ingest]'
python tools/pdf_to_markdown.py 2504.18875v1.pdf data/corpus/agentic-ai-survey.md
python tools/build_pairs.py
```

Both are deterministic. `tools/build_pairs.py --check` fails if the committed JSONL has
drifted from its source, and `pytest` runs that check for you.
