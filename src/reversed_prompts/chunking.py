"""Split a document into retrievable pieces that remember where they came from.

Offsets are the point. A chunk that knows its `(start, end)` in the source can
be expanded into its neighbours, merged with an overlapping chunk, and quoted
back with provenance. A chunk that is only a string can do none of that, and
retrieval without provenance is indistinguishable from the model making it up.

Paragraph boundaries are respected because they carry meaning in prose -- a
fixed-width window that cuts mid-sentence retrieves half a fact. Paragraphs
longer than the target are split at sentence boundaries, and only when there is
no sentence boundary to use does a chunk get cut at an arbitrary character.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHARS = 800
OVERLAP_CHARS = 200

_PARAGRAPH = re.compile(r"[^\n]+(?:\n[^\n]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A span of a document, and where in the document it sits."""

    doc_id: str
    ordinal: int
    text: str
    start: int
    end: int

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.ordinal:04d}"

    def __len__(self) -> int:
        return self.end - self.start


def _units(text: str, limit: int) -> list[tuple[int, int]]:
    """Paragraph spans, with over-long paragraphs split at sentences.

    Returns `(start, end)` pairs into `text`, in order. Every span is non-empty
    and no span is longer than `limit` unless the text offers nowhere to split.
    """
    spans: list[tuple[int, int]] = []
    for para in _PARAGRAPH.finditer(text):
        start, end = para.start(), para.end()
        if end - start <= limit:
            spans.append((start, end))
            continue

        # Too long: cut at sentence ends, accumulating until the limit.
        body = text[start:end]
        cut = start
        pos = start
        for piece in _SENTENCE_END.split(body):
            if not piece:
                continue
            found = text.find(piece, pos, end)
            if found == -1:                      # pragma: no cover - defensive
                continue
            pos = found + len(piece)
            if pos - cut >= limit:
                spans.append((cut, pos))
                cut = pos
        if pos > cut:
            spans.append((cut, min(pos, end)))
    return spans


def chunk_text(text: str, doc_id: str, *, target_chars: int = TARGET_CHARS,
               overlap_chars: int = OVERLAP_CHARS) -> list[Chunk]:
    """Pack paragraphs into chunks of roughly `target_chars`, with overlap.

    Overlap exists so a fact stated across a paragraph boundary is whole in at
    least one chunk. It is applied by repeating the previous chunk's trailing
    unit, never by cutting at a fixed offset -- which is also why overlap is
    approximate rather than exactly `overlap_chars`.
    """
    if not text.strip():
        return []

    units = _units(text, target_chars)
    if not units:                                # pragma: no cover - defensive
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        start = units[i][0]
        end = units[i][1]
        j = i + 1
        while j < len(units) and units[j][1] - start <= target_chars:
            end = units[j][1]
            j += 1

        chunks.append(Chunk(doc_id=doc_id, ordinal=len(chunks),
                            text=text[start:end], start=start, end=end))

        # Step back one unit for overlap, but only when that still makes
        # forward progress -- otherwise a single oversized unit loops forever.
        nxt = j
        if overlap_chars > 0 and j - 1 > i and j < len(units):
            back = j - 1
            if units[back][1] - units[back][0] <= overlap_chars:
                nxt = back
        i = nxt if nxt > i else i + 1

    return chunks


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping or touching spans. Retrieval with overlapping
    chunks otherwise hands the model the same sentence several times and
    charges for each copy."""
    if not spans:
        return []
    ordered = sorted(spans)
    out = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = out[-1]
        if start <= last_end:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out
