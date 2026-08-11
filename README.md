# ReversedPrompts

An agentic flow to guess prompts that generate an output from an input set.

Given pairs of `(input files, output artifacts)`, recover the prompt that maps inputs to
outputs — including the style, structure, and content-selection rules that a naive
instruction like *"summarize this document"* would miss — then execute that recovered
prompt against new inputs using agentic RAG.

📄 **[Design & implementation plan](docs/DESIGN.md)**

## Status

**Phase 0 built.** The induction loop runs end to end: ingest → hypothesize → evaluate →
refine → compile, with the whole thing exercisable offline against a simulated model.
Align (`[B]`) and RAG are still Phase 2.

- `src/reversed_prompts/` — the slice. Spec, features, metrics, client, generators, loop, CLI.
- `data/` — one document and 20 `(prompt, output)` pairs. See [data/README.md](data/README.md).
- `tools/` — the scripts that produced that data, both reproducible.
- `tests/` — 54 checks, none of which need a key.

### What Phase 0 found

The design assumes a corpus shares **one** prompt across all pairs (§1), so facets can be
pooled. **The shipped corpus has a distinct prompt per pair**, so pooling fights the data:
corpus-level induction wins or loses depending on the split, which is a coin flip dressed
as a result. Two consequences, both now encoded in the code rather than in someone's head:

- Per-pair reconstruction — recover a spec from one pair, score it on that pair — beats
  naive **20/20**. That validates feature extraction, scoring and the loop. It measures
  reconstruction, not generalization: there is no held-out anything.
- The corpus-level exit criterion in `docs/DESIGN.md` **is not yet satisfiable**, and won't
  be until there's a pair set where the pairs genuinely share a task. The test suite
  asserts that corpus-level search runs and produces auditable output; it deliberately does
  not assert a win.

Building it also forced three fixes worth naming, because each was silently corrupting
scores:

| Bug | Effect |
|---|---|
| `strip_markup` deleted whole table rows | Table outputs measured as **zero words** — every style feature for them was meaningless |
| The "naive" baseline was handed the gold output shape | Inflated the bar the exit criterion measures against |
| The generator asserted mean length and plurality shape unconditionally | Over-committed on corpora where those vary per prompt, scoring *worse* than saying nothing |

The third produced the general rule now applied at induction time: **do not assert what the
evidence does not support.** Length is claimed only when its coefficient of variation
across gold is under 0.30; shape only when ≥70% of gold agrees.

## Architecture

Two loops. **Induction** runs offline and recovers a `PromptSpec` from evidence;
**execution** runs online and applies that spec to new inputs. What makes either agentic
is the cycle — neither is a pipeline that runs once and stops.

```mermaid
flowchart TB
    pairs["pairs<br/>I,O"]

    subgraph IND["INDUCTION · offline · cyclic · budget-bounded"]
        direction LR
        A["A · Ingest"]
        B["B · Align"]
        C["C · Hypothesize"]
        D{"D · Evaluate"}
        E["E · Refine"]
        F["F · Compile<br/>PromptSpec"]

        A --> B --> C --> D
        D -->|"under threshold,<br/>budget left"| E
        E -->|"beam over top-k"| C
        D -->|"converged"| F
    end

    subgraph EXE["EXECUTION · online · agentic RAG"]
        direction LR
        NEW["new input"] --> PLAN["Plan"]
        PLAN --> RET["Retrieve"]
        RET --> DRAFT["Draft"]
        DRAFT --> SC{"Self-check"}
        SC -->|"fails rubric"| RET
        SC -->|"passes"| OUT["output"]
    end

    pairs --> A
    B -.->|"unattributable<br/>spans"| WARN(["human review"])
    F ==>|"retrieval plan · rubric<br/>style · structure"| PLAN

    classDef done fill:#1a7f37,stroke:#1a7f37,color:#fff
    classDef partial fill:#9a6700,stroke:#9a6700,color:#fff
    classDef todo fill:#57606a,stroke:#57606a,color:#fff,stroke-dasharray:4 3
    classDef guard fill:#8250df,stroke:#8250df,color:#fff

    class pairs done
    class A partial
    class B,C,D,E,F,NEW,PLAN,RET,DRAFT,SC,OUT todo
    class WARN guard
```

<sub>🟩 built · 🟨 partial · ⬜ designed, not built · 🟪 guardrail</sub>

The dotted branch off **Align** is the one that matters most — see below.

### What makes it agentic

**The loop is the point.** `D → E → C` cycles until the candidate converges or the budget
runs out. State persists across rounds — an OPRO-style trajectory of `(spec, score)` pairs
so the optimizer can see what it has already tried and what it cost. That shape, cyclic
and stateful and interruptible, is what argues for a graph orchestrator over a script.

**Self-critique drives the edits.** Refinement isn't resampling. For the worst dev
examples the system produces a *textual gradient* — a critique of why this candidate
missed gold — and edits the spec in the opposite semantic direction.

**Roles, not one model.** Four distinct jobs with different cost/quality profiles: a
frontier reasoner to hypothesize and refine, a mid-tier executor that runs every candidate
over every doc (the cost center), a cheap model for structural feature extraction, and a
judge. The judge must never be the model that proposed — otherwise the loop optimizes
toward the judge's quirks rather than toward quality.

**Spend is a first-class constraint.** The evaluation cascade exists so expensive judgment
is rationed to candidates that already survived free filtering:

```mermaid
flowchart LR
    CAND["candidate<br/>prompts"] --> T1{"Tier 1<br/>deterministic<br/>free"}
    T1 -->|"survivors"| T2{"Tier 2<br/>cheap model<br/>semantic + faithfulness"}
    T1 -.->|"eliminated"| X1["✕"]
    T2 -->|"survivors"| T3{"Tier 3<br/>judge model<br/>rubric vs gold"}
    T2 -.->|"eliminated"| X2["✕"]
    T3 --> SCORE["scored<br/>candidate"]

    SCORE --> G1["contamination penalty<br/>strip prompts that smuggle<br/>answer content"]
    G1 --> G2["length/generality regularizer<br/>prefer the shortest<br/>among equals"]

    classDef free fill:#1a7f37,stroke:#1a7f37,color:#fff
    classDef cheap fill:#9a6700,stroke:#9a6700,color:#fff
    classDef dear fill:#cf222e,stroke:#cf222e,color:#fff
    classDef guard fill:#8250df,stroke:#8250df,color:#fff
    classDef plain fill:#57606a,stroke:#57606a,color:#fff

    class T1 free
    class T2 cheap
    class T3 dear
    class G1,G2 guard
    class CAND,SCORE,X1,X2 plain
```

<sub>🟩 free · 🟨 cheap · 🟥 expensive · 🟪 guardrail</sub>

### The two load-bearing pieces

**Align (B) is the central primitive.** Mapping output spans back to the input spans that
support them is what produces nearly all the evidence: what content survives into the
output and what is systematically dropped (the selection policy), the compression ratio
(length budget), whether the output follows the input's order or reorganizes it (a tell
for map-reduce vs. plan-first generation), and how verbatim the overlap is (extractive vs.
abstractive). It also generalizes past documents — doc→summary, spec→code,
contract→extracted JSON and transcript→minutes are all the same question.

**Unattributable spans are the highest-value guardrail.** Output content with no support
anywhere in the input gets surfaced to a human as *"this can't have come from the input"* —
never passed to the hypothesizer. Without that gate, the hypothesizer's rational move is to
invent an instruction that fabricates the content, and the system compiles hallucination
directly into the prompt it hands you.

Full detail in [docs/DESIGN.md](docs/DESIGN.md).

## Running the checks

```bash
pip install -e '.[dev]'
pytest
```

No API key, no cost. Covers the pair set (well-formed, generated against the corpus in the
tree, stated prompt constraints satisfied) and the induction slice (feature extraction,
scoring, the generators' abstention rules, and the loop run end to end against a simulated
model).

Two commands exercise the pipeline without spending anything:

```bash
rp dry-run --category reason     # the full loop, corpus-level
rp per-pair --simulate           # per-pair reconstruction, all 20
rp features summ-01              # the measured feature vector for one output
```

`rp run` and `rp per-pair` without `--simulate` make real calls. Both take `--budget`, a
hard token ceiling that raises rather than overspending.

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
