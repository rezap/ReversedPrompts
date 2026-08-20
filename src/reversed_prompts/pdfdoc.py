"""Turn a text-layer PDF into text this pipeline can actually score.

Extraction is not a neutral step here, which is why it gets its own module and
its own diagnostics. Tier-1 fidelity is computed from the *shape* of text --
word count, sentence count, list items, table rows (`features.py`) -- and the
usual PDF-to-text artefacts corrupt exactly those measurements:

* a running header repeated on 400 pages adds 400 copies of a line nobody
  wrote, inflating every length feature;
* a word hyphenated across a line break becomes two words;
* a page with no text layer contributes nothing and says nothing about it.

Left alone, those make a group score badly for reasons that have nothing to do
with whether the prompt was recovered -- the failure this project exists to
distinguish from real ones. So the cleaning is done deliberately, is reported
rather than silent, and is recorded next to the text it produced.

What this module does *not* do is inject page markers into the text. Page
numbers live in a `PageMap` beside it (see `chunking.PageMap`), so citation
survives without the text the model reads being altered to carry it.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field

from .chunking import PageMap

# How much of a page to consider when looking for running headers and footers.
EDGE_LINES = 3
# A line has to appear on at least this share of pages to count as furniture.
REPEAT_SHARE = 0.6
# Below this many pages there is no repetition to speak of, and a line that
# happens to appear twice in a three-page document is probably content.
MIN_PAGES_FOR_REPEATS = 4
# A page yielding less than this is a text-layer gap worth reporting.
THIN_PAGE_CHARS = 20
# Running headers and footers are short. A repeated 200-character sentence is
# boilerplate the document means -- a definition, a warranty clause -- and
# deleting it loses content, which is the more expensive of the two mistakes
# available here.
MAX_FURNITURE_CHARS = 80
# Only *short* lines have their digits blanked before repeats are counted. "Page
# 4 of 312" and "Page 5 of 312" are the same footer; "Section 4. Obligations of
# the parties under clause 4" and its Section 5 counterpart are two different
# sentences, and blanking digits in both makes them look identical -- which
# deletes the body of any document whose sections start at a page top.
MAX_DIGIT_BLANKED_CHARS = 40

_DIGITS = re.compile(r"\d+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")


class PdfUnavailable(RuntimeError):
    """pypdf is not installed. Raised rather than degraded, because a silent
    fallback would produce an empty corpus that looks like a real one."""


@dataclass(frozen=True)
class Diagnostics:
    """What had to be cleaned, and what could not be.

    Reported because the user cannot see it otherwise: a document that
    extracted badly and a prompt that could not be recovered produce the same
    low score, and only this tells them apart.
    """

    pages: int = 0
    chars: int = 0
    thin_pages: tuple[int, ...] = ()
    furniture: tuple[str, ...] = ()
    lines_removed: int = 0
    words_rejoined: int = 0
    chars_removed: int = 0

    @property
    def concerns(self) -> list[str]:
        """Things worth a human's attention, worst first."""
        out = []
        kept = max(self.chars, 1)
        if self.chars_removed > kept * 0.25:
            out.append(f"stripping headers and footers removed "
                       f"{self.chars_removed} chars against {kept} kept. That "
                       f"is a lot to call furniture -- check the list above "
                       f"for real content, and re-run with "
                       f"--no-strip-furniture if any of it is.")
        if self.pages and len(self.thin_pages) > self.pages * 0.1:
            out.append(f"{len(self.thin_pages)} of {self.pages} pages yielded "
                       f"almost no text -- is this really a text-layer PDF, or "
                       f"are those pages scans? They will be invisible to the "
                       f"system.")
        elif self.thin_pages:
            out.append(f"{len(self.thin_pages)} page(s) yielded almost no "
                       f"text: {_brief(self.thin_pages)}")
        if self.pages and self.chars / max(self.pages, 1) < 200:
            out.append(f"{self.chars // max(self.pages, 1)} chars per page on "
                       f"average is low for prose; check the extraction before "
                       f"trusting a score.")
        return out


@dataclass(frozen=True)
class Extraction:
    """Cleaned text, where its pages are, and what cleaning it took."""

    text: str
    page_map: PageMap
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    source: str = ""
    source_sha256: str = ""

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def sidecar(self) -> dict:
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "text_sha256": self.sha256,
            "chars": len(self.text),
            "page_map": self.page_map.to_json(),
            "diagnostics": {
                "pages": self.diagnostics.pages,
                "chars": self.diagnostics.chars,
                "thin_pages": list(self.diagnostics.thin_pages),
                "furniture": list(self.diagnostics.furniture),
                "lines_removed": self.diagnostics.lines_removed,
                "chars_removed": self.diagnostics.chars_removed,
                "words_rejoined": self.diagnostics.words_rejoined,
            },
        }


def _brief(numbers: tuple[int, ...], limit: int = 8) -> str:
    shown = ", ".join(str(n) for n in numbers[:limit])
    return shown + (f" ... (+{len(numbers) - limit} more)"
                    if len(numbers) > limit else "")


def _normalise(line: str) -> str:
    """A line with its varying parts blanked, for counting repeats.

    "Page 4 of 312" and "Page 5 of 312" are the same piece of furniture, and
    counting them as different lines would find no repetition at all in the one
    place repetition is guaranteed. Blanking is limited to short lines: see
    `MAX_DIGIT_BLANKED_CHARS` for what it costs when it is not.
    """
    text = line.strip().casefold()
    if len(text) <= MAX_DIGIT_BLANKED_CHARS:
        return _DIGITS.sub("#", text)
    return text


def find_furniture(pages: list[str], *, edge_lines: int = EDGE_LINES,
                   share: float = REPEAT_SHARE) -> set[str]:
    """Normalised lines that repeat near the top or bottom of most pages.

    Restricted to the edges on purpose. A phrase repeated throughout the body
    of a contract is a defined term, not a header, and stripping it would
    delete content -- the more expensive mistake of the two.
    """
    if len(pages) < MIN_PAGES_FOR_REPEATS:
        return set()
    seen: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.splitlines() if ln.strip()]
        edges = lines[:edge_lines] + lines[-edge_lines:]
        seen.update({_normalise(ln) for ln in edges
                     if ln.strip() and len(ln.strip()) <= MAX_FURNITURE_CHARS})
    threshold = max(int(len(pages) * share), 2)
    return {line for line, n in seen.items() if n >= threshold}


def read_pages(path: pathlib.Path | str) -> list[str]:
    """The raw text of each page, in order. One string per page, always --
    including empty ones, so page numbering stays aligned with the file."""
    try:
        from pypdf import PdfReader
    except ImportError as e:                       # pragma: no cover
        raise PdfUnavailable(
            "reading PDFs needs pypdf: pip install -e '.[pdf]'") from e
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def assemble(pages: list[str], *, strip_furniture: bool = True,
             dehyphenate: bool = True,
             first_page_number: int = 1) -> tuple[str, PageMap, Diagnostics]:
    """Join page texts into one document, cleaning and recording as it goes.

    Offsets are computed on the *cleaned* text, in one pass, because a page map
    built against the raw text would point at the wrong characters the moment
    anything was removed -- and would do so silently.
    """
    furniture = find_furniture(pages) if strip_furniture else set()

    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    numbers: list[int] = []
    thin: list[int] = []
    removed = 0
    dropped_chars = 0
    rejoined = 0
    cursor = 0

    for index, raw in enumerate(pages):
        number = first_page_number + index
        kept = []
        for line in raw.splitlines():
            if furniture and _normalise(line) in furniture:
                removed += 1
                dropped_chars += len(line)
                continue
            kept.append(line)
        body = "\n".join(kept).strip()

        if dehyphenate:
            body, n = _HYPHEN_BREAK.subn(r"\1", body)
            rejoined += n

        if len(body) < THIN_PAGE_CHARS:
            thin.append(number)

        # Pages are separated by a blank line so paragraph-aware chunking does
        # not weld the last sentence of one page to the first of the next.
        piece = body + "\n\n"
        parts.append(piece)
        spans.append((cursor, cursor + len(piece)))
        numbers.append(number)
        cursor += len(piece)

    text = "".join(parts)
    diagnostics = Diagnostics(
        pages=len(pages), chars=len(text), thin_pages=tuple(thin),
        furniture=tuple(sorted(furniture)), lines_removed=removed,
        words_rejoined=rejoined, chars_removed=dropped_chars)
    return text, PageMap(spans=tuple(spans), numbers=tuple(numbers)), diagnostics


def extract(path: pathlib.Path | str, **kwargs) -> Extraction:
    """Read a PDF and clean it, keeping the page map and the diagnostics."""
    path = pathlib.Path(path)
    raw = path.read_bytes()
    text, page_map, diagnostics = assemble(read_pages(path), **kwargs)
    return Extraction(text=text, page_map=page_map, diagnostics=diagnostics,
                      source=path.name,
                      source_sha256=hashlib.sha256(raw).hexdigest())


# ------------------------------------------------------------ building pairs

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class BuiltPair:
    """One input/output PDF pair, extracted and written out."""

    stem: str
    record: dict
    input_extraction: Extraction
    output_extraction: Extraction

    @property
    def concerns(self) -> list[str]:
        return ([f"input: {c}" for c in self.input_extraction.diagnostics.concerns]
                + [f"output: {c}"
                   for c in self.output_extraction.diagnostics.concerns])


@dataclass
class BuildReport:
    built: list[BuiltPair] = field(default_factory=list)
    unmatched_inputs: list[str] = field(default_factory=list)
    unmatched_outputs: list[str] = field(default_factory=list)


def _repo_relative(path: pathlib.Path) -> str:
    """A pair file's refs are resolved against the repo root, so anything it
    points at has to live inside the tree. Said here, once, rather than as a
    confusing FileNotFoundError at load time."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        raise ValueError(
            f"{path} is outside the repository ({ROOT}). Pair files reference "
            f"documents by repo-relative path, so the corpus has to live "
            f"inside the tree -- try --corpus data/corpus/<name>.") from None


def _write_document(target: pathlib.Path, extraction: Extraction) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(extraction.text, encoding="utf-8")
    sidecar = target.with_suffix(target.suffix + ".pages.json")
    sidecar.write_text(json.dumps(extraction.sidecar(), indent=2),
                       encoding="utf-8")


def build_pairs(inputs: pathlib.Path, outputs: pathlib.Path,
                corpus: pathlib.Path, *, category: str = "fetch",
                manifest: dict | None = None, **extract_kwargs) -> BuildReport:
    """Extract matched input/output PDFs into a corpus and pair records.

    Files are matched by stem: `inputs/acme.pdf` goes with `outputs/acme.pdf`.
    An unmatched file on either side is *reported*, never skipped quietly --
    a missing output means a pair silently absent from the run, which looks
    exactly like a pair that was never added.

    The gold prompt is left empty unless a manifest supplies one. That is the
    normal case here: if the prompt were known, there would be nothing to
    recover.
    """
    from .features import infer_shape

    manifest = manifest or {}
    in_pdfs = {p.stem: p for p in sorted(inputs.glob("*.pdf"))}
    out_pdfs = {p.stem: p for p in sorted(outputs.glob("*.pdf"))}
    report = BuildReport(
        unmatched_inputs=sorted(set(in_pdfs) - set(out_pdfs)),
        unmatched_outputs=sorted(set(out_pdfs) - set(in_pdfs)))

    for stem in sorted(set(in_pdfs) & set(out_pdfs)):
        got_in = extract(in_pdfs[stem], **extract_kwargs)
        # Outputs keep their furniture: they are short enough to fit whole,
        # and a heading repeated across the two pages of a summary is content
        # rather than page decoration.
        got_out = extract(out_pdfs[stem], strip_furniture=False)

        input_path = corpus / f"{stem}.txt"
        output_path = corpus / f"{stem}.output.txt"
        _write_document(input_path, got_in)
        _write_document(output_path, got_out)

        extra = dict(manifest.get(stem, {}))
        record = {
            "id": stem,
            "category": extra.pop("category", category),
            "input_ref": _repo_relative(input_path),
            "input_sha256": got_in.sha256,
            "output_ref": _repo_relative(output_path),
            "output_sha256": got_out.sha256,
            "target_prompt": extra.pop("target_prompt", ""),
            "output_shape": extra.pop("output_shape",
                                      infer_shape(got_out.text)),
            "prompt_group": extra.pop("prompt_group", stem),
            "is_negative": extra.pop("is_negative", False),
            "source_pdf": in_pdfs[stem].name,
            "input_pages": len(got_in.page_map),
            "output_pages": len(got_out.page_map),
        }
        record.update(extra)
        report.built.append(BuiltPair(stem=stem, record=record,
                                      input_extraction=got_in,
                                      output_extraction=got_out))
    return report
