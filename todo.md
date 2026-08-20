# TODO

Tracking for the fixes coming out of the repo review on 2026-08-17. Three PRs,
each with one thesis. Status is updated as each lands.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## PR A — The judge reports what it was told

**Status: `[x]` done — merged (#16)**

The prompt-match number is the headline metric and it is currently wrong in the
worst direction: verbose judge replies score as perfect matches.

### A1 `[x]` Strict judge format, with retries, failing loudly

Measured on the current code:

| judge reply | scored as | should be |
|---|---|---|
| `9/10` | 1.00 | 0.90 |
| `3/10` | **1.00** | 0.30 |
| `I'd say 7 out of 10.` | **1.00** | 0.70 |
| `8` | 0.80 | 0.80 |

`similarity.py` strips non-digits and takes the first two, so anything
containing `/10` or `out of 10` reads as a perfect score. A judge saying
"3/10, these are quite different" is recorded as flawless recovery.

Decided approach:

- The judge is asked for a decimal in `x.yz` form and nothing else. Validated
  with `^(?:0\.\d{2}|1\.00)$` — two decimal places, capped at 1.00. (Reading
  "y, z" as digits 0-9; the stated 0.00-1.00 range is not expressible
  otherwise.)
- A reply that does not match is **retried**, up to `max_attempts` (default 3,
  parameterised). The retry restates the format requirement rather than
  resending the identical prompt, since an identical prompt tends to produce an
  identical malformed reply.
- After the last attempt, raise `JudgeFormatError` naming the group and quoting
  the reply. No silent fallback: the old code returned 0.5 for an unparseable
  reply, which fabricates a measurement that then averages into the summary
  indistinguishably from a real one.

Consequence, accepted deliberately: one malformed reply aborts the run, after
money has been spent. The CLI turns it into a clean error rather than a
traceback so the cause is obvious. Revisit if it proves noisy in practice.

Landed: `--judge-attempts` on `revprompt run`, `JudgeFormatError` naming the
group and quoting every reply, and the usage so far printed alongside so the
spend is visible when the run dies.

Touches `prompts.SIMILARITY_JUDGE` (the scale changes from an integer 0-10 to a
decimal 0.00-1.00), so `prompts.VERSION` is bumped, and the offline double must
emit the new format.

### A2 `[x]` Delete the dead `PromptScore.combined`

Used nowhere but one test, and it implies contamination is applied as a penalty
when nothing applies it. Replaced with `contaminated_groups`: a count in the
summary and the JSON, using the existing 0.05 threshold. Keeps the design's
"reported separately, never averaged in" while making it visible.

### A3 `[x]` Tests

`judge_similarity` currently has **zero** coverage, which is why A1 survived.

- table over realistic reply formats, including the three that scored 1.00
- a malformed reply is retried, and the retry is not a verbatim resend
- exhausting `max_attempts` raises rather than returning a number
- `max_attempts` is honoured as a parameter
- the summary counts contaminated groups

---

## PR B — A stored result can be trusted across runs

**Status: `[x]` done — merged (#17)**

### B1 `[x]` Fit the feature scale on the whole corpus, always

`fit_scale` normalises by spread across the *selected* pairs, so the same
candidate output scores differently depending on which groups ran:

```
fidelity under the whole corpus : 0.8421
fidelity under this group only  : 0.6095
```

`revprompt run --group X` and `revprompt run` therefore disagree about X. The
CLI should fit on every pair in the file regardless of `--group`,
`--category` or `--controls-only`. `recover_all` keeps its `scale=` parameter
for library use.

### B2 `[x]` Record what produced the numbers

The JSON has no model, though the README claims otherwise and `client.py`'s own
docstring calls attribution a hard requirement. Add `models`, `usage`,
`embedding_model`, `timestamp`, and a `scale_corpus` block carrying the path,
pair count and a fingerprint (hash of the sorted gold outputs) so a later run
can detect that two results were fitted on different corpora.

### B3 `[x]` Add `SCORING_VERSION`

PRs A and B both change what a score means. A version constant stamped on every
result is how a number from last week stays interpretable. Separate from
`prompts.VERSION`, which tracks instruction text.

**Note:** results from before A+B are not comparable with results after.

### B4 `[x]` The offline double had the same bug, one level down

Found by checking end to end rather than trusting the unit tests. After fixing
the scale, `run --group X` and `run` *still* disagreed: `ObedientClient` cached
its vocabulary pool from the first document it ever saw, so a group's simulated
output depended on which groups ran before it. Two independent run-order
dependencies, one hiding behind the other. The pool is now keyed per document,
and a test pins the end-to-end property rather than only its parts.

---

## PR C — Role and filter hygiene

**Status: `[~]` C1 done, C2 and C3 open**

### C1 `[x]` Judge independence: warn, and stop over-claiming

README said *"The judge must never be the model that proposed"*. The default
resolves all four roles to the same model and `--model` overrides all four, so
the default configuration did exactly what the README forbade, silently.

Decided (owner's call): a shared judge is **permitted**. It is preferred that
the two differ, the roles are selectable separately, and a shared judge earns a
warning rather than a refusal.

Landed:

- `client.judge_is_independent(models)` — `True` / `False` / `None`. `None`
  when the client names no model per role (the offline double), because
  "unknown" and "shared" are different claims and recording `False` for the
  double would assert one we cannot support.
- `run` warns once, before spending, when the judge resolves to the inducer's
  model, and names the two ways to change it.
- `judge_independent` written into the result JSON, so the caveat travels with
  the number rather than being re-derived from a model list months later.
- `revprompt models` prints the same note, since that is where the
  configuration is actually inspected.
- New `--judge-model`, applied after `--model`, so "everything on A, judge on
  B" is one flag rather than an env var.
- README and DESIGN.md §4.4 now say what ships: preferred, warned, recorded.

Deliberately **not** making `--model` skip the judge — "use this model" meaning
"use it for three of four roles" is a worse surprise than a warning.

### C2 `[ ]` Remove the phantom `features` role

Nothing calls `role="features"`. Tier-1 features are pure Python and free, so
the role is obsolete rather than unimplemented, yet `revprompt models`
advertises a model for it. Drop it, update the two tests asserting the role
set, and note in DESIGN.md §4.4 that feature extraction needs no model — which
is a strength of that design, currently obscured.

### C3 `[ ]` Filter by category at group level

`load()` filters pairs *before* grouping, so a mixed-category group would
silently become partial. Since a group scores as its weakest member, a partial
group scores *higher*. Latent today — no group spans categories. Group first,
then select whole groups, and raise rather than truncate.

---

## PR D — PDFs in, page citations out

**Status: `[x]` done**

Running the system against real work means PDFs on both sides, inputs possibly
thousands of pages. Landed:

- `pdfdoc.py` — text-layer extraction with the cleaning made explicit and
  reported: running headers and footers stripped and **listed**, words rejoined
  across line breaks, pages with no text layer named, and a concern raised when
  stripping took a large share of the document.
- Two heuristic bugs found by testing against a realistic fixture rather than
  a uniform one, both of which **deleted body text**: repeated *long* lines were
  treated as furniture (a warranty clause repeated at a page top is content),
  and digit-blanking collapsed "Section 4. Obligations under clause 4" across
  pages into one apparent running header. Furniture is now short lines only,
  and digits are blanked only on short lines. Both are pinned by tests.
- `chunking.PageMap` — offset → printed page number, binary-searched because a
  thousand-page document does thousands of lookups. Kept beside the text, never
  injected into it: a `[page 12]` marker would be read by the model *and*
  counted by the scorer.
- Retrieval carries the map through indexing and cites it:
  `contract pp. 412-413 [88301:89740]`. Stored with the index, so citation does
  not depend on the original file still being on disk.
- `output_ref` — outputs live in files, so a document-length answer stays
  readable and diffable.
- `target_prompt` is now optional. When the prompt is what you are looking for
  there is no gold to judge against; `run` refuses to judge rather than scoring
  against an empty string, and says to use `--no-judge`.
- `revprompt check` — the generic pair-file integrity check, in CI.
- `EXCERPT_CHARS` 8092 → 32000. Outputs were already never truncated; a test
  now pins that, since it is the property the whole design leans on.

Still open: `--excerpt-chars` is a prefix, and a prefix of a 1000-page contract
is the first 30 pages. That is what wiring retrieval into the producer fixes,
below.

---

## Retrieval phase B — unblocked; the measurement came back

`run_retrieval_eval` on the full book, `text-embedding-3-small`, 698k chars →
1125 chunks ([run 32025475908][r], 2026-08-17):

```
probe       relevant    keyword     vector      fused
antagonist        44     5/5        5/5        5/5
cyclops            6     2/5        1/5        1/5
shroud             7     0/5        3/5        2/5
sirens             9     1/5        4/5        3/5
mean p@5                  0.400      0.650      0.550
```

[r]: https://github.com/rezap/ReversedPrompts/actions/runs/32025475908

Three things this says, none of which the offline run could:

1. **The vector arm is the strong one** (0.650 vs 0.400), the *opposite* of the
   offline reading — where fusion lost to keyword alone (0.300 vs 0.400)
   because the hashing double has no semantics. The offline number is a
   regression guard, not a verdict, and this is the evidence for saying so.
2. **Equal-weight fusion still drags the strong arm down** (0.550). Left as is,
   phase B would ship a retriever measurably worse than using vectors alone.
3. **The arms disagree per probe** — keyword wins `cyclops` (2/5 vs 1/5),
   vector wins `shroud` (3/5 vs 0/5) and `sirens` (4/5 vs 1/5). So fusion has
   something real to combine; the current weighting just is not it.

Next, in order:

- `[ ]` Sweep `--keyword-weight` / `--vector-weight` and keep fusion only if it
  beats 0.650. If nothing does, ship vector-first with keyword as a fallback
  and record that fusion was tried and rejected.
- `[ ]` Replace or drop the `antagonist` probe: 44 relevant chunks, 5/5 on
  every arm, so it separates nothing. A probe that cannot fail is not evidence.
- `[ ]` Only then wire retrieval into the producer and critic.

---

## Sequencing

A → B → C, merged in order. A and B both move scores and running two
score-shifting changes at once makes either hard to attribute. C is independent
and could go in parallel.
