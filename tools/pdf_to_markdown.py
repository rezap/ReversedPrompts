#!/usr/bin/env python3
"""Convert a single-column LaTeX PDF to sectioned markdown.

    python tools/pdf_to_markdown.py input.pdf output.md

Tuned for the Springer/LaTeX layout of the agentic-AI survey, whose font
hierarchy is:
  17.2 CMR17   title            14.3 CMBX12  section
  12.0 CMBX12  subsection       10.9 CMBX10  subsubsection
  10.0 CMR10   body             <9.5         captions / table cells / footnotes
Tables are recovered structurally via find_tables() and their source text
blocks are suppressed so cell text is not emitted twice.  Headings are
cross-checked against the PDF outline, which is also used to rebuild a clean
table of contents in place of the dot-leader original.
"""
import re
import sys

import pymupdf

HEADING = {14.3: 2, 12.0: 3, 10.9: 4}
TITLE_SIZE = 17.2
BODY_MIN = 9.5
PAGENUM = re.compile(r"^\d{1,3}$")
CAPTION = re.compile(r"^(Fig\.|Figure|Table)\s*\d+")


def norm(s):
    for a, b in (
        ("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl"),
        ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
        ("–", "-"), ("—", "--"), (" ", " "),
    ):
        s = s.replace(a, b)
    return s


def join_lines(lines):
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and not out.endswith("--"):
            out = out[:-1] + ln
        else:
            out += " " + ln
    return re.sub(r"\s+", " ", out).strip()


def cell(c):
    c = norm(c or "")
    c = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", c)      # heal wrapped hyphenation
    c = re.sub(r"(\w)-\s+(?=[a-z])", r"\1", c)        # ...and the flattened form
    return re.sub(r"\s+", " ", c).strip().replace("|", r"\|")


def table_md(rows):
    rows = [[cell(c) for c in r] for r in rows]
    # a header whose trailing cells are blank is a spanning label, not a header:
    # leave those columns unlabelled rather than inventing names for them
    head, body = rows[0], rows[1:]
    width = max([len(head)] + [len(r) for r in body])
    head = head + [""] * (width - len(head))
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * width) + "|"]
    for r in body:
        r = r + [""] * (width - len(r))
        out.append("| " + " | ".join(r) + " |")
    return out


# a bibliography entry opens with "Surname, A.B.:" or an org name ("Agent.ai:")
REF_START = re.compile(r"^([A-Z][\w'\-]+(,\s*[A-Z]\.[A-Z.]*)+|[A-Z][\w.'\-]{1,24}:\s)")


def merge_references(md):
    """The hanging indent splits each reference across blocks -- rejoin them."""
    try:
        i = md.index("## References")
    except ValueError:
        return md
    head, tail = md[: i + 1], [x for x in md[i + 1:] if x.strip()]
    out, cur = [], ""
    for line in tail:
        if REF_START.match(line) and cur:
            out += [cur, ""]
            cur = line
        elif cur:
            cur = cur[:-1] + line if cur.endswith("-") else cur + " " + line
        else:
            cur = line
    if cur:
        out += [cur, ""]
    return head + [""] + out


def inside(bbox, box):
    x0, y0, x1, y1 = bbox
    X0, Y0, X1, Y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return X0 - 2 <= cx <= X1 + 2 and Y0 - 2 <= cy <= Y1 + 2


def block_lines(blk):
    for l in blk["lines"]:
        txt = "".join(s["text"] for s in l["spans"])
        if not txt.strip():
            continue
        spans = [s for s in l["spans"] if s["text"].strip()]
        if not spans:
            continue
        mx = round(max(s["size"] for s in spans), 1)
        bold = all("BX" in s["font"] or "Bold" in s["font"] for s in spans)
        yield norm(txt), mx, bold


def heading_level(size, bold):
    if not bold:
        return None
    for s, lvl in HEADING.items():
        if abs(size - s) < 0.25:
            return lvl
    return None


def main(src, dst):
    doc = pymupdf.open(src)
    md = []
    pending = []
    in_contents = False
    seen_title = False

    def flush():
        nonlocal pending, md
        if pending:
            t = join_lines(pending)
            if t:
                md += [t, ""]
            pending = []

    # clean table of contents rebuilt from the PDF outline
    toc_md = []
    for lvl, title, page in doc.get_toc():
        toc_md.append("  " * (lvl - 1) + f"- {title} (p. {page})")

    for pno, page in enumerate(doc, start=1):
        tables = page.find_tables().tables
        tboxes = [t.bbox for t in tables]
        emitted = set()

        items = []
        for blk in page.get_text("dict")["blocks"]:
            if blk["type"] == 0:
                items.append((blk["bbox"][1], blk["bbox"][0], "text", blk))
        for i, t in enumerate(tables):
            items.append((t.bbox[1], t.bbox[0], "table", i))
        items.sort(key=lambda x: (round(x[0]), round(x[1])))

        for _, _, kind, payload in items:
            if kind == "table":
                if payload in emitted:
                    continue
                emitted.add(payload)
                flush()
                md += table_md(tables[payload].extract()) + [""]
                continue

            blk = payload
            if any(inside(blk["bbox"], tb) for tb in tboxes):
                continue        # text already captured as table cells

            lines = list(block_lines(blk))
            if not lines:
                continue
            raw = join_lines([t for t, *_ in lines])
            if PAGENUM.match(raw.strip()) or raw.startswith("arXiv:"):
                continue

            top = round(max(s for _, s, _ in lines), 1)
            allbold = all(b for _, _, b in lines)

            # title
            if not seen_title and abs(top - TITLE_SIZE) < 0.3:
                flush()
                md += [f"# {raw}", ""]
                seen_title = True
                continue

            lvl = heading_level(top, allbold)
            if lvl and len(raw) < 120:
                flush()
                txt = re.sub(r"^(\d+(\.\d+)*)\s+", "", raw).strip()
                num = re.match(r"^(\d+(\.\d+)*)\s+", raw)
                label = f"{num.group(1)} " if num else ""
                if txt.lower() == "contents":
                    in_contents = True
                    md += ["## Contents", ""] + toc_md + [""]
                    continue
                in_contents = False
                md += [f"{'#' * lvl} {label}{txt}", ""]
                continue

            if in_contents:
                continue        # drop the dot-leader original

            if top < BODY_MIN:
                flush()
                if CAPTION.match(raw):
                    m = CAPTION.match(raw)
                    md += [f"**{m.group(0)}** {raw[m.end():].strip()}", ""]
                elif raw.strip() in ("Abstract",):
                    md += ["## Abstract", ""]
                else:
                    md += [raw, ""]
                continue

            if lines[0][2] and pending:
                flush()
            pending += [t for t, _, _ in lines]
            flush()

    flush()
    md = merge_references(md)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(md))
    with open(dst, "w") as fh:
        fh.write(out.strip() + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[2].strip())
    main(sys.argv[1], sys.argv[2])
