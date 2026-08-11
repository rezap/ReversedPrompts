# ReversedPrompts — Design & Implementation Plan

Infer the prompt from the evidence. Given a set of `(input files, output artifacts)`
pairs, recover a prompt that maps inputs to outputs, then execute that prompt against
new inputs using agentic RAG.

---

## 1. Problem statement

**Given:** a corpus of pairs `P = {(I₁, O₁), (I₂, O₂), … (Iₙ, Oₙ)}` where `Iᵢ` is one or
more input documents and `Oᵢ` is the artifact a human (or another system) produced from
them.

**Find:** a prompt `p` such that `LLM(p, I) ≈ O` for unseen `I` drawn from the same
distribution.

**Then:** execute `p` at inference time over new inputs, where the inputs may be too
large to fit in context — hence agentic RAG.

The naive framing ("summarize this document") fails because the observable signal in the
pairs is not just *what task* was performed but *how*: register, structure, length,
what got dropped, what got emphasized, what vocabulary was used. Recovering that is the
actual product.

### Why this is hard (and where the design effort goes)

1. **Underdetermination.** Many prompts produce the same output on a given pair. The one
   you recover may be an accident of that pair rather than the real rule.
2. **Scoring.** You cannot optimize what you cannot measure, and "does this read like the
   gold artifact" is not ROUGE.
3. **Unattributable content.** Real outputs contain facts, opinions, and context that are
   simply *not in the input*. No prompt can recover those, and a naive inducer will
   hallucinate instructions to compensate.
4. **Small N.** Users will show up with four examples, not four hundred.

Everything below is organized around those four.

---

## 2. Core idea: the prompt is not a string

Treat the recovered prompt as a **structured, separately-optimizable object** — a
`PromptSpec`. Facets are induced by different evidence and optimized by different
methods, and only get flattened into text at the very end.

| Facet | What it captures | Induced from | Optimized by |
|---|---|---|---|
| `task` | The transformation verb: summarize / extract / rewrite / draft | Whole-pair shape, compression ratio | LLM proposal, rarely changes |
| `style` | Register, person, tense, sentence length, hedging, jargon, formality | Measurable feature vector of `O` | Contrastive refinement vs. measured deltas |
| `structure` | Section tree, ordering, list-vs-prose, length budget, required fields | Parsed layout of `O` | Deterministic — read it off the outputs |
| `selection` | What content from `I` survives into `O`, and what is always dropped | **Alignment map** (§4.2) | Coverage/precision scoring |
| `constraints` | Negatives — what a naive prompt produces that gold never contains | Contrastive diff vs. baseline (§4.3) | Failure-driven, ProTeGi-style |
| `exemplars` | Which pairs to include as few-shot demos | — | Bootstrap search (DSPy) |
| `retrieval` | Chunking, top-k, hybrid weights, map-reduce vs. single-shot | Derived from `selection` + input size | Compiled, then tuned |

Two payoffs from this decomposition:

- **Structure and style are partly *measurable*,** so you get cheap, deterministic,
  non-gameable signal instead of relying entirely on an LLM judge.
- **`retrieval` falls out of `selection`.** If the alignment map shows outputs draw
  evenly from the whole document, you need exhaustive map-reduce coverage; if they draw
  from two sections, you need precise top-k retrieval. This is the bridge from the
  offline inducer to the online RAG runtime, and it's why the two halves belong in one
  system.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDUCTION (offline)                          │
│                                                                      │
│  pairs ──▶ [A] Ingest ──▶ [B] Align ──▶ [C] Hypothesize ──┐         │
│                              │                             │         │
│                              │          ┌──────────────────┘         │
│                              │          ▼                            │
│                              │    [D] Evaluate ◀──── held-out split  │
│                              │          │                            │
│                              │          ▼                            │
│                              └──▶ [E] Refine ──┐                     │
│                                     ▲          │  beam / budget      │
│                                     └──────────┘                     │
│                                          │                           │
│                                          ▼                           │
│                                   [F] Compile ──▶ PromptSpec v1.2    │
└──────────────────────────────────────────┬──────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                        EXECUTION (online)                            │
│  new input ──▶ Plan ──▶ Retrieve ──▶ Draft ──▶ Self-check ──▶ out   │
│                            ▲                        │                │
│                            └────────  revise  ──────┘                │
│                        (retrieval plan + rubric both from PromptSpec)│
└─────────────────────────────────────────────────────────────────────┘
```

The induction loop is cyclic and stateful with a spend budget — that shape is what
argues for a graph orchestrator rather than a linear script.

---

## 4. The induction pipeline in detail

### 4.1 [A] Ingest — normalize to a common intermediate

Everything (PDF, DOCX, PPTX, HTML, code, CSV) becomes **structured markdown plus a
layout tree**: heading hierarchy, tables preserved, char offsets back to the source.
Downstream stages only ever see this intermediate, which is what lets one codebase span
document types.

Chunk on the **heading tree, not fixed windows.** The alignment step needs semantically
coherent units; 512-token sliding windows destroy exactly the signal being measured.

### 4.2 [B] Align — the central primitive

For each segment of `O`, find the spans of `I` that support it, via hybrid retrieval
(BM25 + embeddings, RRF-fused) followed by an LLM adjudication pass on the top
candidates.

This one step yields most of the evidence the whole system runs on:

- **Coverage** — what fraction of `I` is represented in `O`, and *which* parts are
  systematically dropped across all pairs. That's the selection policy.
- **Compression ratio** — drives length budget.
- **Ordering** — does `O` follow `I`'s order, or reorganize thematically? A strong tell
  for map-reduce vs. plan-first generation.
- **Abstraction level** — verbatim n-gram overlap between aligned spans separates
  extractive from abstractive tasks.
- **Unattributable spans** — output content with *no* support in the input. Critical:
  these are surfaced to the user as "this content can't come from the input" rather than
  being fed to the hypothesizer, which would otherwise invent an instruction to
  fabricate it. **This is the single highest-value guardrail in the design** — it's what
  stops the system from compiling hallucination into the prompt.

The alignment primitive generalizes: doc→summary, spec→code, contract→extracted JSON,
transcript→minutes are all "which parts of the input produced which parts of the output."

### 4.3 [C] Hypothesize — propose candidate prompts

Three complementary generators, run in parallel for diversity:

1. **Direct induction.** Show the reasoning model a sample of pairs plus the alignment
   evidence: *"what instruction would produce these outputs from these inputs?"* (APE's
   reverse mode.)
2. **Contrastive induction.** Run a deliberately naive baseline (`"summarize this
   document"`), then diff baseline output against gold and ask for *the delta as prompt
   clauses.* This directly targets the motivating case — the model isn't asked to
   describe summarization, only what makes *this* summarization different. Consistently
   the strongest generator for style.
3. **Deterministic extraction.** Read `structure` and the style feature vector straight
   off the outputs. No LLM involved, no hallucination possible.

Candidates are merged into `PromptSpec` objects, not strings.

### 4.4 [D] Evaluate — cascade, cheap to expensive

Run candidates on a **held-out** split, compare generated vs. gold. Three tiers, each
gating the next, so LLM-judge spend goes only to survivors:

- **Tier 1 (free, deterministic).** Structural match (heading tree edit distance, section
  count, list/prose ratio, length ratio); style feature distance (mean sentence length,
  type-token ratio, passive rate, readability, hedging markers, person/POV); format
  validity (does the JSON parse, does the schema hold).
- **Tier 2 (cheap model).** Semantic similarity of aligned sections; faithfulness — is
  every claim in the generated output grounded in the input.
- **Tier 3 (reasoning model, judge).** Rubric-scored pairwise comparison against gold,
  with the rubric itself derived from the `structure`/`style` facets.

Two scoring guards that matter more than the metrics themselves:

- **Contamination penalty.** Penalize candidate prompts containing rare n-grams or named
  entities lifted from training documents. Without this, the optimizer's easiest win is
  to smuggle answer content into the prompt, and it *will* find that. Cheap to implement,
  catches the most common failure.
- **Length/generality regularizer (MDL-flavored).** Among candidates within noise of each
  other, prefer the shortest and most general. Directly counters underdetermination.

**Judge independence.** Never let the same model both propose and judge in the same round
— you optimize toward the judge's quirks. Routing through OpenRouter (§7) makes the
strong version of this cheap: use a judge from a **different model family** than the
inducer, which is a slug change rather than a second vendor integration. Keep real weight
on Tier 1 deterministic signal, and calibrate against a small human-labeled set
periodically. Report judge–human agreement as a first-class metric; if it drifts, the
whole optimization is measuring noise.

### 4.5 [E] Refine — failure-driven editing

For the worst dev examples, produce a **textual gradient** (ProTeGi): an LLM critique of
*why* this candidate's output missed gold, then edit the spec in the opposite semantic
direction. Maintain an OPRO-style trajectory of `(spec, score)` pairs in the optimizer's
context so it can see what has and hasn't worked.

Beam search over the top-k specs, hard budget on tokens/wall-clock, checkpoint every
round so a run can be resumed or inspected mid-flight.

### 4.6 [F] Compile — freeze a versioned artifact

Emit an immutable, versioned `PromptSpec` carrying its own provenance: score breakdown
per tier, which pairs trained it, which were held out, the chosen exemplars, the derived
retrieval plan, and a confidence band driven by N and by dev/held-out variance. **With
N=4 the honest output is a prompt plus a loud warning, not a number with three decimals.**

---

## 5. The PromptSpec artifact

```python
class PromptSpec(BaseModel):
    id: str
    version: int
    task: str                          # "Summarize a technical incident report"
    style: StyleSpec                   # register, person, tense, sentence stats, lexicon
    structure: StructureSpec           # section tree, ordering, length budget, format
    selection: SelectionPolicy         # include/exclude rules from alignment evidence
    constraints: list[str]             # negatives: "never include remediation steps"
    exemplars: list[ExemplarRef]       # chosen few-shot pairs
    retrieval: RetrievalPlan           # chunking, top-k, hybrid weights, map-reduce?
    provenance: Provenance             # scores, splits, N, confidence, judge agreement
    unattributable: list[str]          # flagged content the input cannot support

    def render(self, ctx: RenderContext) -> str: ...   # → the actual prompt text
    def rubric(self) -> Rubric: ...                    # → self-check criteria at runtime
```

`render()` and `rubric()` come from the *same* object, so what the runtime is asked to do
and what it's graded on cannot drift apart.

---

## 6. Execution runtime (agentic RAG)

The `PromptSpec` configures a plan-retrieve-draft-check loop:

1. **Plan** — decompose per `structure`; each required section becomes a retrieval goal.
2. **Retrieve** — hybrid search per `retrieval`. If `selection` demanded exhaustive
   coverage, map-reduce every chunk instead of top-k'ing; if it was narrow, retrieve and
   rerank precisely. *This is the whole reason induction and execution share a system.*
3. **Draft** — `render()` the spec with retrieved context.
4. **Self-check** — grade the draft against `rubric()` using the same Tier-1
   deterministic checks from evaluation. Structural and style violations are caught
   *for free* here, without a judge call.
5. **Revise** — bounded retry loop on failed criteria only.

Same Tier-1 code path in both offline eval and online self-check: one implementation,
guaranteed consistent.

---

## 7. Tech stack

Recommendation first, rationale second, honest alternative third.

### Core
| Concern | Choice | Why / alternative |
|---|---|---|
| Language | **Python 3.11+** | Match ecosystem. |
| Packaging | **uv** | Fast, lockfile-native. Alt: Poetry. |
| Schemas | **Pydantic v2** | `PromptSpec` is the system's spine; validation and JSON round-trip are load-bearing. |
| Orchestration | **LangGraph** | Cyclic + stateful + checkpointable + interruptible matches §4 exactly. *Keep the optimizer core framework-free* so you can drop it. Alt: plain asyncio state machine — genuinely viable, less magic. |
| CLI | **Typer** | The loop is batch/offline; CLI-first is correct. |
| Model access | **OpenRouter** via `openai` client or **LiteLLM** | One key, one client, cross-family judge for free, per-generation cost accounting. Caveats in §7 Models — provider pinning is mandatory. |

### Models — via OpenRouter

All model calls go through **OpenRouter**, using the `openai` Python client pointed at
`https://openrouter.ai/api/v1` (the API is OpenAI-compatible), or **LiteLLM** if you want
provider-agnostic retries and cost accounting for free. LiteLLM is already a DSPy
dependency, so it is in the tree either way — `dspy.LM("openrouter/<slug>")` works
directly.

Wrap it in a thin internal `LLMClient` protocol regardless. The optimization loop should
depend on *roles*, not slugs; swapping a role's model is then a config change, and role
assignment is itself something you will tune once you see the cost profile.

| Role | Wants | Rationale |
|---|---|---|
| Inducer / optimizer | Frontier reasoning model | Few calls, hardest reasoning: hypothesize + textual gradients. Quality here dominates outcome; cost is negligible. |
| Executor | Strong mid-tier | Many calls — runs every candidate over every eval doc. **The cost center**; this is the slot where OpenRouter's model shopping pays for itself. |
| Feature extraction / Tier-2 | Cheap + fast, reliable JSON | High-volume, structural. Verify the slug advertises structured-output support. |
| Judge | **Different family from the inducer** | Independence (§4.4) — see below. |

OpenRouter slugs are `provider/model` (e.g. `anthropic/claude-opus-5`,
`openai/gpt-…`, `google/gemini-…`). **Resolve exact slugs at runtime from
`GET /api/v1/models` rather than hardcoding them** — the catalog moves, and the same
endpoint returns each model's `supported_parameters` and pricing, which the client can
use to fail fast when a role is assigned a model that can't do structured output.

**What OpenRouter buys this project specifically:** genuine judge independence. §4.4 wants
the judge to be a different model family from the proposer, which normally means a second
vendor integration; here it is a slug change. Take advantage of it — a cross-family judge
is materially harder to game than a same-family one. Built-in cost accounting per
generation also makes the §9 budget controls easy to enforce for real rather than
estimating from token counts.

#### Provider pinning is mandatory here (not optional tuning)

OpenRouter may route the same slug to different upstream providers with different
quantization, tokenizers, and caching behavior. **For most applications that is a
feature. For this one it is a correctness bug**, because the entire system is a
measurement loop: if round 7 scores worse than round 3, you must be certain that is the
prompt changing and not the serving provider changing underneath you.

- Pin the provider (`provider.order` with `allow_fallbacks: false`) for every model used
  in scoring. Fallbacks are fine for one-off interactive calls, never inside an eval.
- Log the **actually-served** provider (returned on each response) into the run trace
  alongside every score. Treat a provider change as invalidating cross-round comparison.
- Pin `seed` and `temperature=0` on the executor for eval runs. Neither guarantees
  determinism, which is exactly why the scoring design leans on deterministic Tier-1
  metrics and repeated sampling rather than trusting single generations.

#### Caching and batching: adjust expectations

- **Prompt caching still matters enormously** — the same input documents are resent on
  every candidate evaluation, and this is the difference between a run costing $5 and
  $200. But semantics are now provider-dependent: Anthropic-served models need explicit
  `cache_control` breakpoints passed through, OpenAI-served models cache automatically,
  and some providers do not cache at all. Another reason to pin. Verify cache hits are
  real by watching the usage fields, not by assuming.
- **No equivalent of a 50%-discount batch API.** Drop that lever from the cost plan.
  Replace it with bounded concurrency, aggressive Tier-1 prefiltering (§4.4), and
  eval-set subsampling until finalists.

#### Embeddings are *not* covered by OpenRouter

OpenRouter serves chat completions; it does not serve embeddings or reranking. The
retrieval stack below therefore needs its own provider regardless — either a hosted
embedding API or local models. Budget for that as a separate integration rather than
discovering it in Phase 2.

### Prompt optimization
- **DSPy** (`MIPROv2`, `BootstrapFewShot`) for instruction + exemplar search. Do not
  rebuild this. Honest caveat: DSPy's program abstraction fights you when the "program"
  is one prompt — **use it as an optimizer library, not as the application framework.**
- Custom ProTeGi/OPRO layer for the style and constraint facets, where the metric is
  bespoke.

### Ingestion
`docling` (best-in-class layout/table extraction, offsets preserved) · `unstructured` as
fallback for odd formats · `tree-sitter` for code · `pandas` for tabular.

### Retrieval

`LanceDB` (embedded, zero-ops, right for a repo-local tool; move to Qdrant only if you
need a server) · `bm25s` for lexical · RRF fusion.

**Embeddings and reranking need their own provider — OpenRouter does not serve them.**
Two viable routes:

- **Local** (`bge-m3` for embeddings, a `sentence-transformers` cross-encoder for rerank).
  Recommended default: it removes a second API dependency, keeps documents on the box,
  costs nothing per call, and embedding quality is not the bottleneck here — the
  alignment step (§4.2) uses retrieval only to generate *candidates* that an LLM then
  adjudicates.
- **Hosted** (Voyage, Jina, Cohere) if corpora get large enough that local inference
  becomes the slow path.

Either way, keep it behind the same kind of thin protocol as the LLM client so the choice
stays reversible.

### Evaluation & observability
`spacy` (POS, passive voice, sentence stats) · `textstat` (readability) · `rapidfuzz`
(n-gram overlap, contamination check) · `bert-score` · `ragas` (faithfulness) ·
**`langfuse`** for tracing and dataset/score management, self-hostable — the run-level
observability here is not optional, you will be debugging why round 7 scored worse than
round 3. Alt: MLflow if you want a prompt registry in the same tool.

### Serving (later)
`FastAPI` + `SQLModel`/Postgres for the spec registry · `streamlit` for the
human-in-the-loop review UI (approve/edit candidates, label pairs) · `pytest` +
`hypothesis` for the deterministic tiers, which are very testable.

---

## 8. Phasing

Each phase is independently useful and de-risks the next.

**Phase 0 — Text-only vertical slice (~2 weeks).** One task type, plain-text pairs, no
RAG, no UI. Ingest → hypothesize → Tier-1+3 eval → refine → PromptSpec, driven by CLI.
Pairs come from `data/pairs/agentic-ai-survey.jsonl` (20 prose pairs over one document;
see `data/README.md` for provenance and the calibration caveat that follows from it).
Includes the `LLMClient` protocol over OpenRouter with provider pinning and served-provider
logging from the first commit — retrofitting that after scores exist means throwing the
scores away.
*Exit criterion: on a held-out pair, the induced prompt beats the naive baseline on the
Tier-1 style/structure metrics.* This slice proves or kills the core thesis for a
fortnight of work — build it before anything else.

**Phase 1 — Facet decomposition & contrastive refinement.** Full `PromptSpec`,
contrastive generator, Tier-2, contamination + length regularizers, LOO cross-validation
for small N.

**Phase 2 — Alignment & multi-format.** Docling ingestion, the alignment map, coverage
and unattributable-content reporting. Now genuinely multi-type.

**Phase 3 — Agentic RAG runtime.** Retrieval plan compilation, map-reduce for long docs,
self-check/revise loop. The system now runs on new inputs at scale.

**Phase 4 — Product surface.** Registry, API, review UI, drift monitoring, re-induction
when new pairs arrive.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Small N** — can't split 4 pairs | Leave-one-out CV; report confidence from N; refuse to claim generalization below a threshold |
| **Underdetermination** — recovered prompt is an accident | MDL regularizer; test on deliberately out-of-distribution inputs |
| **Judge/optimizer collusion** | Separate models; weight deterministic tiers; track judge–human agreement as a metric |
| **Prompt overfits by embedding answers** | Contamination penalty on rare n-grams and entities from training docs |
| **Unattributable output content** | Alignment detects it, surfaces it, and excludes it from induction (§4.2) |
| **Cost blowup** — N candidates × M docs × long docs | Cascade eval; prompt caching (verify hits per provider); bounded concurrency; subsample eval set early, full set only for finalists. No batch-API discount available via OpenRouter |
| **Style is subjective** | Measurable proxies carry Tier 1; the judge only breaks ties among survivors |
| **Silent provider swap corrupts the score trace** — same slug, different upstream host, different scores | Pin `provider.order` with `allow_fallbacks: false` for all scoring calls; log the served provider next to every score; invalidate cross-round comparison when it changes (§7) |
| **Model deprecation mid-project** — OpenRouter's catalog moves | Resolve slugs from `/api/v1/models` at runtime, not hardcoded; roles are config; re-baseline scores when a role's model changes, since old and new scores are not comparable |

---

## 10. Open questions

1. ~~**Multi-input pairs** — when `Iᵢ` is 30 files, is the task per-file or
   corpus-level?~~ **Resolved:** out of scope. `Iᵢ` is a single content blob. Whether
   it was assembled from one file or thirty is an ingestion concern the system never
   sees, so no per-file/corpus-level flag threads through alignment.
2. **Prompt vs. pipeline** — some input→output mappings are not one prompt but a chain.
   Detect this (e.g. when no single prompt clears threshold) and decompose?
3. **Cross-task transfer** — can a library of induced specs bootstrap induction on a new
   task with N=1?
4. **Human corrections as signal** — when a user edits a recovered prompt, that edit is
   the highest-quality gradient available. Feed it back.
