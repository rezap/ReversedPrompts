# TODO

Tracking for the fixes coming out of the repo review on 2026-08-17. Three PRs,
each with one thesis. Status is updated as each lands.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## PR A — The judge reports what it was told

**Status: `[~]` in progress**

The prompt-match number is the headline metric and it is currently wrong in the
worst direction: verbose judge replies score as perfect matches.

### A1 `[~]` Strict judge format, with retries, failing loudly

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

Touches `prompts.SIMILARITY_JUDGE` (the scale changes from an integer 0-10 to a
decimal 0.00-1.00), so `prompts.VERSION` is bumped, and the offline double must
emit the new format.

### A2 `[ ]` Delete the dead `PromptScore.combined`

Used nowhere but one test, and it implies contamination is applied as a penalty
when nothing applies it. Replaced with `contaminated_groups`: a count in the
summary and the JSON, using the existing 0.05 threshold. Keeps the design's
"reported separately, never averaged in" while making it visible.

### A3 `[ ]` Tests

`judge_similarity` currently has **zero** coverage, which is why A1 survived.

- table over realistic reply formats, including the three that scored 1.00
- a malformed reply is retried, and the retry is not a verbatim resend
- exhausting `max_attempts` raises rather than returning a number
- `max_attempts` is honoured as a parameter
- the summary counts contaminated groups

---

## PR B — A stored result can be trusted across runs

**Status: `[ ]` not started**

### B1 `[ ]` Fit the feature scale on the whole corpus, always

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

### B2 `[ ]` Record what produced the numbers

The JSON has no model, though the README claims otherwise and `client.py`'s own
docstring calls attribution a hard requirement. Add `models`, `usage`,
`embedding_model`, `timestamp`, and a `scale_corpus` block carrying the path,
pair count and a fingerprint (hash of the sorted gold outputs) so a later run
can detect that two results were fitted on different corpora.

### B3 `[ ]` Add `SCORING_VERSION`

PRs A and B both change what a score means. A version constant stamped on every
result is how a number from last week stays interpretable. Separate from
`prompts.VERSION`, which tracks instruction text.

**Note:** results from before A+B are not comparable with results after.

---

## PR C — Role and filter hygiene

**Status: `[ ]` not started**

### C1 `[ ]` Judge independence: warn, and stop over-claiming

README says *"The judge must never be the model that proposed"*. The default
resolves all four roles to the same model and `--model` overrides all four, so
the default configuration does exactly what the README forbids, silently.

Warn once per run when `judge == inducer`, record `judge_independent` in the
JSON so the caveat travels with the result, and soften the README to state
what is true today plus how to fix it (`REVPROMPT_MODEL_JUDGE`).

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

## Blocked on a measurement, not on code

**Retrieval phase B** — wiring retrieval into the producer and critic — waits on
the `run_retrieval_eval` workflow result. Offline, fusion scores *below* keyword
alone (0.300 vs 0.400) because the hashing double has no semantics. Whether the
vector arm earns its place is only answerable with real embeddings.

---

## Sequencing

A → B → C, merged in order. A and B both move scores and running two
score-shifting changes at once makes either hard to attribute. C is independent
and could go in parallel.
