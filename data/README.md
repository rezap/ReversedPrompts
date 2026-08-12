# Test set: the Odyssey, altered

Twenty passages from Homer with facts deliberately changed, and 38 pairs built
over them in 13 prompt groups.

```
data/
  source/odyssey-pg1727.txt        Project Gutenberg #1727, Butler translation
  corpus/odyssey/<variant>.md      the altered passages
  pairs/odyssey.jsonl              38 records
tools/
  build_odyssey.py                 produces both; --check guards drift
```

Regenerate with `python tools/build_odyssey.py`. The build is deterministic, and
`--check` fails if the committed files have drifted from it.

## Why an altered classic

Every model has read the Odyssey. That is normally a reason to avoid a text; here
it is the point.

If a passage says the stake was cut from **cedar** and the model answers
*olive*, it did not read the passage — it recalled the poem. So this set
measures two things at once:

- **prompt recovery**, the system's actual job, and
- **groundedness**, whether the answer came from the text in front of the model
  or from its memory of the original.

Those belong together. A recovered prompt that scores well only because the
model already knew the answer has not been validated at all.

## What was changed

**Minor** — names, materials, counts. Cheap to make and precisely where recall
competes with reading.

| Homer says | The passages say |
|---|---|
| Ulysses son of Laertes | Arkesion son of Ophelestes, Thessander son of Kalliphon, … |
| Dulichium, Same, Zacynthus | Kerynthos/Aigilia/Thrinax, Pelagos/Doriskos/Hyrmine, … |
| green olive wood | green cedar / ash / fig wood |
| twelve jars, seven talents | five jars and nine talents; twenty and three; seven and fifteen |
| wax in the crew's ears | pitch, clay, wool |

**Major** — the outcome itself changes, so a model leaning on memory does not
merely get a detail wrong, it tells a different story.

- **`bow-telemachus`** — Ulysses *cannot* bend the bow; Telemachus strings it,
  and the arrow strikes the last axe and falls short.
- **`bow-snapped`** — the bow cracks from horn to horn and breaks. Nobody
  strings it, no shot is made.
- **`escape-jars`** — the men escape hidden in the giant's empty whey jars,
  carried out by the monster and his brother, instead of slung beneath the
  sheep.

## Prompt groups

Every group applies one instruction across several variants **whose answers
differ**. That is what stops a recovered prompt from passing by describing the
one answer it happened to see.

| Group | Pairs | What it asks for |
|---|---|---|
| `ody-speaker-name` | 4 | The speaker's name |
| `ody-speaker-father` | 4 | The speaker's father |
| `ody-neighbour-islands` | 3 | Three islands, as a list, in order |
| `ody-gift-quantities` | 3 | Two numbers in a fixed form |
| `ody-landing-party` | 3 | Two numbers, one altered and one not |
| `ody-stake-material` | 3 | One word |
| `ody-sirens-precaution` | 3 | Two facts in a fixed form |
| `ody-named-gods` | 4 | A list, **with two negatives** |
| `ody-escape-method` | 2 | Exactly two sentences |
| `ody-bow-outcome` | 2 | One sentence, on a changed outcome |
| `ody-three-sentence-summary` | 3 | Exactly three sentences |
| `ody-character-table` | 2 | A markdown table |
| `ody-captain-briefing` | 2 | Exactly three bullets, in a set order |

`ody-named-gods` is the negative control: the same instruction over passages
that do and do not name a god, where the right answer for the latter is `NA`.
A prompt that says *"list the gods"* passes the positives and fails the
negatives; only one that says what to do when there are none passes all four.

## How the answers stay honest

Three properties are enforced by tests rather than by care:

**Alterations must land.** Every substitution declares how many times it must
match. A pattern that stops matching fails the build instead of quietly
producing a passage whose gold answer is now wrong.

**Answers are derived, not restated.** Where an answer follows from the
alteration — a name, a number, an island list — it is computed from the same
table that produced the passage, so the two cannot drift apart.

**Alterations must actually defeat memory.** Tests assert that no gold answer
matches what Homer says, and that the variants within a group do not all answer
alike. A variant whose answer accidentally agrees with the original is testing
nothing, and fails.

Plus the ordinary ones: every answer is checked to be true of its own passage,
every negative is checked to genuinely lack the content, and the stated output
constraints (sentence counts, bullet counts, fixed forms, table shape) are
verified to hold.

## Provenance and its limits

The source is public domain. The alterations and the gold answers were written
by the same model that the system will be scored against, which has the same
caveat as any self-authored set: the pairs are more tightly aligned than
prompt/output pairs collected from real use, where outputs routinely overshoot
or undershoot what was asked.

What is *not* self-referential here is the groundedness half. The correct answer
is fixed by the altered text and contradicts the model's prior, so agreement
cannot come from the model and the gold sharing an author. That part is a real
test.
