#!/usr/bin/env python3
"""Build the Odyssey test set: altered passages plus the pairs that use them.

Why an altered classic rather than an obscure document: every model has read
the Odyssey. If a passage says the stake was cedar and the model answers
"olive", it answered from memory instead of from the text in front of it. That
makes this set a groundedness probe as well as a prompt-recovery benchmark, and
the two are worth measuring together -- a recovered prompt that only works
because the model already knew the answer has not been validated at all.

Alterations come in two strengths:

* **minor** -- names, materials, counts. Cheap, mechanical, and exactly where
  recall competes with reading.
* **major** -- outcomes change. Telemachus strings the bow; the bow snaps; the
  men escape in wine jars rather than under the sheep. A model leaning on
  memory here does not merely get a detail wrong, it tells a different story.

Every substitution declares how many times it must match. A pattern that stops
matching -- because the source text changed, or a previous rule consumed it --
fails the build rather than silently producing a passage whose gold answer is
wrong.

    python tools/build_odyssey.py            # write corpus + pairs
    python tools/build_odyssey.py --check    # verify committed files are current
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "source" / "odyssey-pg1727.txt"
CORPUS_DIR = ROOT / "data" / "corpus" / "odyssey"
OUT = ROOT / "data" / "pairs" / "odyssey.jsonl"

RSQ = "’"          # the source uses curly apostrophes


# --------------------------------------------------------------------- source

def load_books() -> dict[str, str]:
    s = SOURCE.read_text(encoding="utf-8")
    s = s.split("*** START OF THE PROJECT GUTENBERG EBOOK THE ODYSSEY ***")[1]
    s = s.split("*** END OF THE PROJECT GUTENBERG EBOOK THE ODYSSEY ***")[0]
    parts = re.split(r"\n\s*BOOK ([IVXL]+)\s*\n", s)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def clean(text: str) -> str:
    """Strip footnote markers glued to words, normalise whitespace."""
    text = re.sub(r"(?<=[a-zA-Z”.,;:!?])\d{1,3}\b", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    paras = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(paras).strip()


def paragraphs() -> dict[str, list[str]]:
    return {k: clean(v).split("\n\n") for k, v in load_books().items()}


# --------------------------------------------------------------------- chunks

CHUNKS = {
    "naming":   ("IX", [2]),        # "I am Ulysses son of Laertes", Ithaca, the islands
    "gift":     ("IX", [12]),       # the priest's presents: talents, jars, the twelve men
    "blinding": ("IX", [21, 26]),   # the olive club and the boring of the eye
    "escape":   ("IX", [31, 34]),   # bound under the sheep
    "sirens":   ("XII", [13, 14, 16]),
    "bow":      ("XXI", [39, 40, 41]),
}


def chunk_text(name: str, paras: dict[str, list[str]]) -> str:
    book, idxs = CHUNKS[name]
    return "\n\n".join(paras[book][i] for i in idxs)


# ----------------------------------------------------------------- alterations

class Alteration:
    """One substitution that must match an exact number of times."""

    def __init__(self, pattern: str, replacement: str, count: int = 1,
                 regex: bool = False):
        self.pattern = pattern
        self.replacement = replacement
        self.count = count
        self.regex = regex

    def apply(self, text: str, where: str) -> str:
        pat = self.pattern if self.regex else re.escape(self.pattern)
        new, n = re.subn(pat, self.replacement, text)
        if n != self.count:
            raise SystemExit(
                f"{where}: pattern {self.pattern!r} matched {n} times, "
                f"expected {self.count}. The passage or an earlier rule changed.")
        return new


def sub(pattern: str, replacement: str, count: int = 1) -> Alteration:
    return Alteration(pattern, replacement, count)


def rx(pattern: str, replacement: str, count: int = 1) -> Alteration:
    return Alteration(pattern, replacement, count, regex=True)


def whole(replacement: str) -> Alteration:
    """Replace the entire passage. Used for major rewrites, where patching
    phrases would leave the surrounding narrative contradicting itself."""
    return Alteration(r"(?s)\A.*\Z", replacement.replace("\\", "\\\\"), 1, regex=True)


# ------------------------------------------------------------------- variants
# `facts` records what the alteration made true. Gold outputs are built from it
# rather than typed twice, so a variant and its answer cannot drift apart.

VARIANTS: list[dict] = []


def variant(vid: str, chunk: str, alterations: list[Alteration],
            facts: dict | None = None, strength: str = "minor") -> None:
    VARIANTS.append(dict(id=vid, chunk=chunk, alterations=alterations,
                         facts=facts or {}, strength=strength))


# --- the speaker's name (minor; the classic recall trap) ---------------------

for vid, name, father in [
    ("naming-arkesion", "Arkesion", "Ophelestes"),
    ("naming-thessander", "Thessander", "Kalliphon"),
    ("naming-oromedon", "Oromedon", "Bryseus"),
    ("naming-meliton", "Meliton", "Antiphates"),
]:
    variant(vid, "naming",
            [sub("Ulysses son of Laertes", f"{name} son of {father}")],
            facts=dict(hero=name, father=father))

# --- the neighbouring islands (minor; a list, in order) ---------------------

for vid, isles in [
    ("islands-a", ["Kerynthos", "Aigilia", "Thrinax"]),
    ("islands-b", ["Pelagos", "Doriskos", "Hyrmine"]),
    ("islands-c", ["Melanthos", "Ourania", "Skepsis"]),
]:
    a, b, c = isles
    variant(vid, "naming", [
        sub("Dulichium, Same, and the wooded island of Zacynthus",
            f"{a}, {b}, and the wooded island of {c}"),
    ], facts=dict(islands=isles))

# --- the priest's gift (minor; numbers written as words) --------------------

for vid, men, talents, jars in [
    ("gift-a", "ten", "nine", "five"),
    ("gift-b", "eight", "three", "twenty"),
    ("gift-c", "twenty", "fifteen", "seven"),
]:
    variant(vid, "gift", [
        sub("all but the twelve best among them", f"all but the {men} best among them"),
        sub("seven talents of fine gold", f"{talents} talents of fine gold"),
        sub("twelve jars of sweet wine", f"{jars} jars of sweet wine"),
    ], facts=dict(men=men, talents=talents, jars=jars))

# --- what the stake was made of (minor, but load-bearing for the scene) -----

for vid, wood in [("blinding-cedar", "cedar"), ("blinding-ash", "ash"),
                  ("blinding-fig", "fig")]:
    variant(vid, "blinding", [
        sub("it was of green olive wood", f"it was of green {wood} wood"),
        sub(f"the Cyclops{RSQ} eye hiss round the beam of olive wood",
            f"the Cyclops{RSQ} eye hiss round the beam of {wood} wood"),
    ], facts=dict(wood=wood))

# --- the escape (one minor, one major) --------------------------------------

variant("escape-goats", "escape", [
    sub("I bound them noiselessly in threes together",
        "I bound them noiselessly in pairs together"),
    sub("There was to be a man under the middle sheep, and the two on either "
        "side were to cover him, so that there were three sheep to each man.",
        "There was to be a man under each pair, slung between them where the "
        "fleeces met, so that there were two sheep to each man."),
    sub("there was a ram finer than any of the others",
        "there was a he-goat finer than any of the others"),
    sub(f"he drove the ram outside", "he drove the he-goat outside"),
    sub(f"I first got from under the ram{RSQ}s belly",
        f"I first got from under the he-goat{RSQ}s belly"),
], facts=dict(bound="in pairs", animal="he-goat", per_man="two"))

variant("escape-jars", "escape", [
    whole("""\
“As for myself I kept on puzzling to think how I could best save my own life \
and those of my companions; I schemed and schemed, as one who knows that his \
life depends upon it, for the danger was very great. In the end I deemed that \
this plan would be the best. Against the far wall of the cave stood the great \
empty jars in which the monster stored his whey, each of them tall enough to \
hide a grown man crouching, and stopped with a lid of plaited withies. I bade \
my comrades climb into them one by one, and I drew the lids over them myself, \
and last of all I got into the largest and pulled its lid down after me.

“When the child of morning, rosy-fingered dawn, appeared, the monster groped \
his way to the mouth of the cave and rolled the stone aside, and presently his \
brother came up from the next valley to ask what ailed him. The two of them \
carried the jars out into the yard, meaning to scour them clean in the stream, \
and set them down among the sheep pens; and so we were borne out of that cave \
upon the shoulders of the very creatures who had shut us in it. When they had \
gone back inside I pushed up my lid, freed my comrades one after another, and \
we ran for the ship without waiting to drive off so much as a single sheep."""),
], facts=dict(method="hidden inside the giant's empty whey jars",
              carried_by="the monster and his brother",
              took_sheep=False), strength="major")


# --- the bow (both major: the outcome itself changes) -----------------------

variant("bow-telemachus", "bow", [
    sub("But Ulysses, when he had taken it up and examined it all over, strung "
        "it as easily as a skilled bard strings a new peg of his lyre",
        "But Ulysses, when he had taken it up and examined it all over, could "
        "not bend it, and passed it to Telemachus, who strung it as easily as a "
        "skilled bard strings a new peg of his lyre"),
    sub("and his arrow pierced every one of the handle-holes of the axes from "
        "the first onwards till it had gone right through them, and into the "
        "outer courtyard",
        "and his arrow pierced the handle-holes of the axes from the first "
        "onwards until the last, where it struck the iron and fell short into "
        "the dust"),
    sub("I did not miss what I aimed at, and I was not long in stringing my bow.",
        "I missed the last of them, and it was my son and not I who strung the bow."),
], facts=dict(strung_by="Telemachus", all_axes=False), strength="major")

variant("bow-snapped", "bow", [
    # the whole paragraph goes: with the bow broken there is no favourable
    # omen, and leaving Jove in it would also have made this passage's
    # "name every god" answer of NA a lie
    sub("But Ulysses, when he had taken it up and examined it all over, strung "
        "it as easily as a skilled bard strings a new peg of his lyre and makes "
        "the twisted gut fast at both ends. Then he took it in his right hand to "
        "prove the string, and it sang sweetly under his touch like the "
        "twittering of a swallow. The suitors were dismayed, and turned colour "
        "as they heard it; at that moment, moreover, Jove thundered loudly as a "
        "sign, and the heart of Ulysses rejoiced as he heard the omen that the "
        "son of scheming Saturn had sent him.",
        "But Ulysses, when he had taken it up and examined it all over, bent it "
        "against his knee; and the bow, which had lain unstrung through twenty "
        "winters, cracked from horn to horn and broke in his hands. No man "
        "strung it that day, nor any day after. The suitors were dismayed, and "
        "turned colour at the sound of it; but no thunder came for a sign, and "
        "no omen was sent to him at all."),
    sub("He took an arrow that was lying upon the table—for those which the "
        "Achaeans were so shortly about to taste were all inside the quiver—he "
        "laid it on the centre-piece of the bow, and drew the notch of the arrow "
        "and the string toward him, still seated on his seat. When he had taken "
        "aim he let fly, and his arrow pierced every one of the handle-holes of "
        "the axes from the first onwards till it had gone right through them, "
        "and into the outer courtyard. Then he said to Telemachus:",
        "He laid the two pieces down upon the table where the arrows lay "
        "untouched, and no shot was made at the axes at all. Then he said to "
        "Telemachus:"),
    sub("Your guest has not disgraced you, Telemachus. I did not miss what I "
        "aimed at, and I was not long in stringing my bow. I am still strong, "
        "and not as the suitors twit me with being.",
        "Your guest has disgraced you, Telemachus. The bow is broken and there "
        "will be no trial of the axes."),
], facts=dict(strung_by="nobody", all_axes=False, bow_broke=True),
    strength="major")

# --- the Sirens (minor) ------------------------------------------------------

for vid, stuff, guard_a, guard_b in [
    ("sirens-pitch", "pitch", "Eurylochus", "Perimedes"),
    ("sirens-clay", "clay", "Antiphon", "Kleitos"),
    ("sirens-wool", "wool", "Sthenelos", "Halitherses"),
]:
    alts = [
        sub("I look a large wheel of wax and cut it up small with my sword",
            f"I took a large cake of {stuff} and cut it up small with my sword"),
        sub("Then I kneaded the wax in my strong hands",
            f"Then I kneaded the {stuff} in my strong hands"),
        sub("Then my men took the wax from their ears",
            f"Then my men took the {stuff} from their ears"),
    ]
    if (guard_a, guard_b) != ("Eurylochus", "Perimedes"):
        alts.append(sub("Eurylochus and Perimedes bound me",
                        f"{guard_a} and {guard_b} bound me"))
    variant(vid, "sirens", alts,
            facts=dict(stuff=stuff, guards=[guard_a, guard_b]))


# --------------------------------------------------------------------- render

def build_corpus() -> dict[str, str]:
    paras = paragraphs()
    out = {}
    for v in VARIANTS:
        text = chunk_text(v["chunk"], paras)
        for alt in v["alterations"]:
            text = alt.apply(text, v["id"])
        out[v["id"]] = text
    return out


def facts_of(vid: str) -> dict:
    return next(v for v in VARIANTS if v["id"] == vid)["facts"]


# ---------------------------------------------------------------------- pairs

def groups() -> list[dict]:
    """Prompt groups. `out` is a callable taking the variant's facts, so an
    answer is derived from the alteration rather than restated beside it."""
    G: list[dict] = []

    G.append(dict(
        id="ody-speaker-name", category="fetch", shape="prose",
        prompt=("Give the name of the man who is speaking, exactly as the "
                "passage gives it. Reply with the name only, nothing else."),
        members=[(v["id"], lambda f: f["hero"])
                 for v in VARIANTS if v["chunk"] == "naming" and "hero" in v["facts"]],
    ))

    G.append(dict(
        id="ody-speaker-father", category="fetch", shape="prose",
        prompt=("Who does the speaker say his father is? Reply with the "
                "father's name only."),
        members=[(v["id"], lambda f: f["father"])
                 for v in VARIANTS if "father" in v["facts"]],
    ))

    G.append(dict(
        id="ody-neighbour-islands", category="fetch", shape="list",
        prompt=("The speaker names a group of islands lying near his own. List "
                "them one per line as a bulleted list, in the order the passage "
                "gives them. No other text."),
        members=[(v["id"], lambda f: "\n".join(f"- {i}" for i in f["islands"]))
                 for v in VARIANTS if "islands" in v["facts"]],
    ))

    G.append(dict(
        id="ody-gift-quantities", category="fetch", shape="prose",
        prompt=("How many jars of wine and how many talents of gold did the "
                "priest give? Answer on one line in exactly this form, using "
                "numerals: jars=N, talents=N"),
        members=[(v["id"], lambda f: f"jars={WORD_NUM[f['jars']]}, "
                                     f"talents={WORD_NUM[f['talents']]}")
                 for v in VARIANTS if "jars" in v["facts"]],
    ))

    G.append(dict(
        id="ody-landing-party", category="fetch", shape="prose",
        prompt=("How many of his men did the speaker take with him, and how "
                "many parts of water did the priest mix with one of wine? "
                "Answer as: men=N, water=N"),
        members=[(v["id"], lambda f: f"men={WORD_NUM[f['men']]}, water=20")
                 for v in VARIANTS if "men" in v["facts"]],
    ))

    G.append(dict(
        id="ody-stake-material", category="fetch", shape="prose",
        prompt=("What wood was the stake made from? Reply with the single word "
                "the passage uses."),
        members=[(v["id"], lambda f: f["wood"])
                 for v in VARIANTS if "wood" in v["facts"]],
    ))

    G.append(dict(
        id="ody-escape-method", category="summarize", shape="prose",
        prompt=("In exactly two sentences, explain how the men got out of the "
                "cave. Use only what this passage says."),
        members=[
            ("escape-goats",
             lambda f: ("The men were bound in pairs beneath the monster's "
                        "sheep, one man slung between each pair where the "
                        "fleeces met, while the speaker himself clung face "
                        "upwards to the belly of a fine he-goat. When the "
                        "animals were driven out of the cave the speaker "
                        "dropped from the he-goat, freed his comrades, and "
                        "they drove the flock down to the ship.")),
            ("escape-jars",
             lambda f: ("The men hid inside the great empty whey jars standing "
                        "against the cave wall, each stopped with a lid of "
                        "plaited withies, with the speaker sealing his "
                        "comrades in before climbing into the largest himself. "
                        "The monster and his brother then carried the jars out "
                        "into the yard to scour them, and once they had gone "
                        "back inside the speaker pushed up his lid, freed the "
                        "others, and they ran for the ship.")),
        ],
    ))

    G.append(dict(
        id="ody-bow-outcome", category="reason", shape="prose",
        prompt=("Answer in one sentence: who strung the bow, and did the arrow "
                "pass through every one of the axes? Base your answer only on "
                "this passage."),
        members=[
            ("bow-telemachus",
             lambda f: ("Ulysses could not bend the bow and handed it to "
                        "Telemachus, who strung it; the arrow passed through "
                        "the axes from the first onwards but struck the iron "
                        "of the last and fell short, so it did not pass "
                        "through every one.")),
            ("bow-snapped",
             lambda f: ("Nobody strung the bow -- it cracked from horn to horn "
                        "and broke in Ulysses' hands -- and no shot was made "
                        "at the axes at all.")),
        ],
    ))

    G.append(dict(
        id="ody-sirens-precaution", category="fetch", shape="prose",
        prompt=("What did the speaker put in his crew's ears, and which two men "
                "tightened his bonds? Answer on one line as: "
                "ears=X; bound by=A and B"),
        members=[(v["id"], lambda f: f"ears={f['stuff']}; "
                                     f"bound by={f['guards'][0]} and {f['guards'][1]}")
                 for v in VARIANTS if "stuff" in v["facts"]],
    ))

    # A control set with negatives: the same instruction over passages that do
    # and do not name a god.
    G.append(dict(
        id="ody-named-gods", category="fetch", shape="list",
        prompt=("List every god or goddess named in this passage, one per line "
                "as a bulleted list, in the order they first appear. If the "
                "passage names none, reply with exactly: NA"),
        members=[
            ("naming-arkesion", lambda f: "- Calypso\n- Circe\n- Jove"),
            ("gift-a", lambda f: "- Apollo"),
            ("escape-goats", lambda f: "NA", True),
            ("bow-snapped", lambda f: "NA", True),
        ],
    ))

    G.append(dict(
        id="ody-three-sentence-summary", category="summarize", shape="prose",
        prompt=("Summarise this passage in exactly three sentences. No bullet "
                "points, no heading, no preamble."),
        members=[
            ("naming-thessander",
             lambda f: ("The speaker names himself Thessander son of Kalliphon "
                        "and says his cunning has made his fame reach heaven. "
                        "He describes his home on Ithaca, its wooded mountain "
                        "Neritum and the islands lying near it, and insists "
                        "nothing is dearer to a man than his own country. He "
                        "adds that neither Calypso nor Circe could persuade him "
                        "to stay, and turns to the tale of his return from "
                        "Troy.")),
            ("sirens-clay",
             lambda f: ("The speaker tells his crew what Circe prophesied, "
                        "asking them to bind him upright to the mast so that "
                        "he alone may hear the Sirens, and to tie him tighter "
                        "if he begs to be freed. As the ship reaches the "
                        "island the wind drops, the men row, and he stops "
                        "their ears with kneaded clay before they lash him in "
                        "place. When he signals to be released Antiphon and "
                        "Kleitos bind him more firmly still, until the singing "
                        "is out of earshot and they unstop their ears and free "
                        "him.")),
            ("blinding-ash",
             lambda f: ("The speaker cuts six feet from the monster's great ash "
                        "club, has it sharpened and hardened in the fire, and "
                        "hides it until four men chosen by lot can help him use "
                        "it. After the Cyclops has eaten two more of the crew "
                        "and drunk the wine he is offered, he falls into a "
                        "drunken sleep. They heat the stake and bore it into "
                        "his single eye until the blood boils around it, and "
                        "his screams bring the other Cyclopes to the mouth of "
                        "his cave.")),
        ],
    ))

    G.append(dict(
        id="ody-character-table", category="fetch", shape="table",
        prompt=("Produce a markdown table of every named person or creature in "
                "this passage. Two columns: Name, Role. One row each, in order "
                "of first appearance. No text outside the table."),
        members=[
            ("naming-oromedon",
             lambda f: ("| Name | Role |\n|---|---|\n"
                        "| Oromedon | The speaker |\n"
                        "| Bryseus | The speaker's father |\n"
                        "| Calypso | Goddess who kept him in her cave |\n"
                        "| Circe | Aeaean goddess who also wished to keep him |\n"
                        "| Jove | God by whose will he met his adventures |")),
            ("sirens-wool",
             lambda f: ("| Name | Role |\n|---|---|\n"
                        "| Circe | Made the prophecies the speaker relays |\n"
                        "| Sthenelos | Crewman who tightened the speaker's bonds |\n"
                        "| Halitherses | Crewman who tightened the speaker's bonds |")),
        ],
    ))

    G.append(dict(
        id="ody-captain-briefing", category="summarize", shape="list",
        prompt=("Write a briefing for a ship's captain as exactly three "
                "bullet points, each a single sentence, covering in order: the "
                "danger, the precaution taken, and the outcome."),
        members=[
            ("sirens-pitch",
             lambda f: ("- The Sirens sing from an island in a flat calm, and "
                        "any crew that hears them will stay to listen.\n"
                        "- Stop every crewman's ears with kneaded pitch and "
                        "have yourself lashed upright to the mast, with orders "
                        "to be bound tighter if you beg to be released.\n"
                        "- The ship rowed past unharmed: Eurylochus and "
                        "Perimedes tightened the bonds when the speaker "
                        "signalled, and the crew unstopped their ears once the "
                        "singing was out of earshot.")),
            ("escape-jars",
             lambda f: ("- The men were shut inside a monster's cave behind a "
                        "stone too great to move, and were being eaten.\n"
                        "- They hid in the giant's empty whey jars, each large "
                        "enough for a crouching man and stopped with a lid of "
                        "plaited withies.\n"
                        "- The monster and his brother carried the jars out to "
                        "scour them, and the men climbed free and ran for the "
                        "ship without taking any sheep.")),
        ],
    ))

    return G


WORD_NUM = {
    "three": 3, "five": 5, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twelve": 12, "fifteen": 15, "twenty": 20,
}


# ---------------------------------------------------------------------- emit

def build() -> tuple[dict[str, str], list[dict]]:
    corpus = build_corpus()
    records: list[dict] = []

    for g in groups():
        for member in g["members"]:
            vid, out_fn = member[0], member[1]
            is_negative = len(member) > 2 and member[2]
            # hash exactly what lands on disk, trailing newline included, or
            # the recorded digest never matches the file it describes
            text = corpus[vid] + "\n"
            digest = hashlib.sha256(text.encode()).hexdigest()
            v = next(x for x in VARIANTS if x["id"] == vid)
            records.append({
                "id": f"{g['id']}--{vid}",
                "category": g["category"],
                "input_ref": f"data/corpus/odyssey/{vid}.md",
                "input_sha256": digest,
                "target_prompt": g["prompt"],
                "output": out_fn(facts_of(vid)),
                "output_shape": "prose" if is_negative else g["shape"],
                "gold_sections": [],
                "prompt_group": g["id"],
                "is_negative": bool(is_negative),
                "source_book": CHUNKS[v["chunk"]][0],
                "alteration": v["strength"],
            })
    return corpus, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify committed files are current; do not write")
    args = ap.parse_args()

    corpus, records = build()
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"

    if args.check:
        problems = []
        if not OUT.exists() or OUT.read_text() != body:
            problems.append(str(OUT))
        for vid, text in corpus.items():
            f = CORPUS_DIR / f"{vid}.md"
            if not f.exists() or f.read_text() != text + "\n":
                problems.append(str(f))
        if problems:
            raise SystemExit("stale, rerun tools/build_odyssey.py:\n  "
                             + "\n  ".join(problems))
        print(f"ok: {len(corpus)} passages, {len(records)} pairs, "
              f"{len({r['prompt_group'] for r in records})} groups")
        return

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for vid, text in corpus.items():
        (CORPUS_DIR / f"{vid}.md").write_text(text + "\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body)

    majors = sum(1 for v in VARIANTS if v["strength"] == "major")
    print(f"wrote {len(corpus)} passages ({majors} with major alterations) "
          f"and {len(records)} pairs in "
          f"{len({r['prompt_group'] for r in records})} groups")


if __name__ == "__main__":
    main()
