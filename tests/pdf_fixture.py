"""Build a minimal text-layer PDF, so PDF tests need no binary fixtures.

A checked-in PDF would be opaque: when a test failed you could not see what
the document said without opening it in another program, and you could not
adjust it without regenerating it somewhere else. Building the bytes here keeps
the fixture readable as source and the test suite free of blobs.

This writes the smallest structurally valid PDF that carries text: one font,
one content stream per page, an xref table, a trailer. It is not a general
PDF writer and does not try to be.
"""
from __future__ import annotations


def make_pdf(pages: list[list[str]]) -> bytes:
    """A PDF of `pages`, each a list of lines rendered one under another."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    content_ids = []
    for lines in pages:
        parts = [b"BT /F1 12 Tf 72 720 Td 14 TL"]
        for line in lines:
            esc = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            parts.append(f"({esc}) Tj T*".encode())
        parts.append(b"ET")
        stream = b"\n".join(parts)
        content_ids.append(add(b"<< /Length %d >>\nstream\n%s\nendstream"
                               % (len(stream), stream)))

    # The page tree object is written after the pages, but each page has to
    # name it as its parent, so its id is worked out in advance.
    pages_id = len(objects) + len(pages) + 1
    page_ids = [add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                    b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                    % (pages_id, font, cid)) for cid in content_ids]
    kids = b" ".join(b"%d 0 R" % p for p in page_ids)
    add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids)))
    root = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, root, start))
    return bytes(out)


def report_pages(count: int = 8, *, header: str = "ACME AGREEMENT",
                 footer: str = "Acme Corp | Confidential") -> list[list[str]]:
    """A plausible report: shared furniture at the edges, distinct bodies.

    The bodies differ only by a number on purpose. That is the case that
    catches an over-eager repeat detector: blank the digits and every section
    heading in the document looks like the same running header.
    """
    return [[header,
             f"Page {n} of {count}",
             f"Section {n}. Obligations of the parties under clause {n} of this",
             f"agreement require delivery of milestone {n} before review.",
             footer]
            for n in range(1, count + 1)]
