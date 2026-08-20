"""PDF ingestion, page maps, and the citation that comes out the far end.

The theme of these tests is that extraction is lossy in ways that look like
recovery failures. A stripped clause, a hyphen-split word and a page with no
text layer all lower the fidelity score without saying why, so what is asserted
here is mostly *what survives* and *what gets reported*.
"""
from __future__ import annotations

import json

import pytest

from pdf_fixture import make_pdf, report_pages
from reversed_prompts import ingest, pdfdoc, recover, retrieval
from reversed_prompts.chunking import PageMap
from reversed_prompts.embedding import HashEmbedder

pypdf = pytest.importorskip("pypdf", reason="PDF support is an optional extra")


# --------------------------------------------------------------- the page map

def test_an_offset_maps_to_the_page_it_falls_on():
    pm = PageMap(spans=((0, 100), (100, 250), (250, 400)))
    assert pm.page_at(0) == 1
    assert pm.page_at(99) == 1
    assert pm.page_at(100) == 2
    assert pm.page_at(399) == 3


def test_a_span_ending_on_a_boundary_does_not_claim_the_next_page():
    """`end` is exclusive. Off by one here means every passage in a long
    document claims a page it does not touch."""
    pm = PageMap(spans=((0, 100), (100, 200)))
    assert pm.range_for(0, 100) == (1, 1)
    assert pm.range_for(0, 101) == (1, 2)


def test_printed_page_numbers_are_used_when_they_differ_from_the_index():
    """Front matter is numbered separately; the citation should say what is
    printed on the page, not how far into the file it sits."""
    pm = PageMap(spans=((0, 10), (10, 20), (20, 30)), numbers=(0, 1, 2))
    assert pm.page_at(0) == 0
    assert pm.range_for(10, 30) == (1, 2)


def test_a_page_map_survives_a_json_round_trip():
    pm = PageMap(spans=((0, 10), (10, 20)), numbers=(4, 5))
    assert PageMap.from_json(pm.to_json()) == pm
    assert PageMap.from_json(None) is None
    assert PageMap.from_json({"spans": []}) is None


def test_a_page_map_with_the_wrong_number_of_labels_is_rejected():
    with pytest.raises(ValueError):
        PageMap(spans=((0, 10), (10, 20)), numbers=(1,))


# ------------------------------------------------------------- what gets cut

def test_a_running_header_and_footer_are_recognised_as_furniture():
    found = pdfdoc.find_furniture([p_text(p) for p in report_pages(8)])
    assert "acme agreement" in found
    assert "page # of #" in found
    assert "acme corp | confidential" in found


def test_body_lines_differing_only_by_a_number_are_not_furniture():
    """The failure this guards against deleted the body of the document.

    Section headings repeat their wording on every page and differ only in the
    clause number. Blank the digits to catch "Page 4 of 8" and those headings
    collapse into one string that appears on every page -- indistinguishable
    from a running header, and stripped as one.
    """
    found = pdfdoc.find_furniture([p_text(p) for p in report_pages(8)])
    assert not any("obligations of the parties" in line for line in found)


def test_a_long_repeated_clause_is_kept_even_at_a_page_edge():
    """Real furniture is short. A repeated paragraph is boilerplate the
    document means -- a warranty, a definition -- and losing it loses content."""
    clause = ("The supplier warrants that all deliverables shall conform in "
              "every material respect to the specification agreed in writing.")
    pages = [p_text([clause, f"Body of page {n}, worded differently each time."])
             for n in range(1, 7)]
    assert pdfdoc.find_furniture(pages) == set()


def test_repeats_are_ignored_in_a_document_too_short_to_have_a_pattern():
    pages = [p_text(["Same line", "one"]), p_text(["Same line", "two"])]
    assert pdfdoc.find_furniture(pages) == set()


def test_a_word_split_across_a_line_break_is_rejoined():
    text, _, diag = pdfdoc.assemble(["the representa-\ntive signed it"])
    assert "representative signed it" in text
    assert diag.words_rejoined == 1


def test_a_hyphenated_compound_at_a_line_break_is_left_alone():
    """Rejoining only fires into a lowercase continuation, so a genuine
    compound followed by a capital is not silently welded together."""
    text, _, _ = pdfdoc.assemble(["a well-known\nAuthor wrote it"])
    assert "well-known\nAuthor" in text


def test_pages_with_no_text_layer_are_reported_rather_than_ignored():
    _, _, diag = pdfdoc.assemble(["real text on this page", "", "   "])
    assert diag.thin_pages == (2, 3)
    assert any("almost no text" in c for c in diag.concerns)


def test_heavy_stripping_raises_a_concern():
    """The stripper cannot be perfect, so it has to be loud when it takes a
    lot -- that is what makes over-stripping recoverable instead of silent."""
    pages = [p_text(["Repeated banner line here", f"Body {n}."])
             for n in range(1, 9)]
    _, _, diag = pdfdoc.assemble(pages)
    assert diag.chars_removed > 0
    assert any("furniture" in c for c in diag.concerns)


# ----------------------------------------------------------- page map offsets

def test_page_spans_index_back_into_the_cleaned_text():
    """The map is built on the text that survives cleaning. Built on the raw
    text instead, every offset would be wrong by the amount stripped -- and
    wrong quietly, since the citation would still look plausible."""
    text, pm, _ = pdfdoc.assemble([p_text(p) for p in report_pages(6)])
    assert len(pm) == 6
    for i in range(6):
        start, end = pm.spans[i]
        assert f"Section {i + 1}." in text[start:end]
        assert pm.page_at(start) == i + 1


def test_extraction_reads_a_real_pdf_end_to_end(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(make_pdf(report_pages(5)))
    got = pdfdoc.extract(path)
    assert len(got.page_map) == 5
    assert "Section 3." in got.text
    assert "Acme Corp | Confidential" not in got.text     # furniture stripped
    assert got.sidecar()["text_sha256"] == got.sha256


# --------------------------------------------------------------- building pairs

def _build(tmp_path, stems_in, stems_out, **kw):
    inputs, outputs = tmp_path / "in", tmp_path / "out"
    inputs.mkdir(), outputs.mkdir()
    for stem in stems_in:
        (inputs / f"{stem}.pdf").write_bytes(make_pdf(report_pages(5)))
    for stem in stems_out:
        (outputs / f"{stem}.pdf").write_bytes(make_pdf([["Net 45."]]))
    return pdfdoc.build_pairs(inputs, outputs, tmp_path / "corpus", **kw)


def test_pairs_are_matched_by_filename_stem(monkeypatch, tmp_path):
    monkeypatch.setattr(pdfdoc, "ROOT", tmp_path)
    report = _build(tmp_path, ["a", "b"], ["a", "b"])
    assert [b.stem for b in report.built] == ["a", "b"]
    assert report.built[0].record["input_ref"].startswith("corpus/")
    assert report.built[0].record["output_ref"].endswith("a.output.txt")


def test_an_unmatched_pdf_is_reported_rather_than_skipped(monkeypatch, tmp_path):
    """A missing counterpart means a pair quietly absent from the run, which
    looks exactly like a pair that was never added."""
    monkeypatch.setattr(pdfdoc, "ROOT", tmp_path)
    report = _build(tmp_path, ["a", "orphan"], ["a", "stray"])
    assert report.unmatched_inputs == ["orphan"]
    assert report.unmatched_outputs == ["stray"]
    assert [b.stem for b in report.built] == ["a"]


def test_a_manifest_supplies_the_gold_prompt_and_group(monkeypatch, tmp_path):
    monkeypatch.setattr(pdfdoc, "ROOT", tmp_path)
    report = _build(tmp_path, ["a"], ["a"],
                    manifest={"a": {"target_prompt": "State the terms.",
                                    "prompt_group": "terms"}})
    assert report.built[0].record["target_prompt"] == "State the terms."
    assert report.built[0].record["prompt_group"] == "terms"


def test_the_gold_prompt_is_empty_by_default(monkeypatch, tmp_path):
    """Recovering it is the job. Inventing a placeholder would produce a
    prompt-match score against something nobody wrote."""
    monkeypatch.setattr(pdfdoc, "ROOT", tmp_path)
    assert _build(tmp_path, ["a"], ["a"]).built[0].record["target_prompt"] == ""


def test_a_corpus_outside_the_repository_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(pdfdoc, "ROOT", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    with pytest.raises(ValueError, match="outside the repository"):
        _build(tmp_path, ["a"], ["a"])


# ------------------------------------------------------------------- ingest

def test_an_output_can_live_in_a_file_instead_of_the_pair_line(tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    (tmp_path / "doc.txt").write_text("the input", encoding="utf-8")
    (tmp_path / "ans.txt").write_text("a long answer", encoding="utf-8")
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({
        "id": "x", "category": "fetch", "input_ref": "doc.txt",
        "output_ref": "ans.txt"}) + "\n", encoding="utf-8")

    pair = ingest.load(path)[0]
    assert pair.output == "a long answer"
    assert pair.target_prompt == ""


def test_setting_both_output_and_output_ref_is_refused(tmp_path, monkeypatch):
    """Preferring one silently would let a stale inline copy override the file
    someone actually edited."""
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    (tmp_path / "doc.txt").write_text("in", encoding="utf-8")
    (tmp_path / "ans.txt").write_text("out", encoding="utf-8")
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({
        "id": "x", "category": "fetch", "input_ref": "doc.txt",
        "output": "inline", "output_ref": "ans.txt"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        ingest.load(path)


def test_a_pair_with_no_output_at_all_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    (tmp_path / "doc.txt").write_text("in", encoding="utf-8")
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({
        "id": "x", "category": "fetch", "input_ref": "doc.txt"}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="neither"):
        ingest.load(path)


def test_a_group_without_a_gold_prompt_knows_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    (tmp_path / "doc.txt").write_text("in", encoding="utf-8")
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({
        "id": "x", "category": "fetch", "input_ref": "doc.txt",
        "output": "o"}) + "\n", encoding="utf-8")
    assert ingest.load_groups(path)[0].has_gold is False


# ---------------------------------------------------------------- retrieval

def test_a_retrieved_passage_cites_the_page_it_came_from(tmp_path):
    text, pm, _ = pdfdoc.assemble([p_text(p) for p in report_pages(12)])
    r = retrieval.MemoryRetriever(tmp_path, HashEmbedder(), target_chars=200)
    r.index("doc", text, pm)

    passages = r.search("doc", text="milestone 9 before review", k=1, expand=0)
    assert passages
    assert passages[0].pages is not None
    assert "p. " in passages[0].cite()


def test_page_citation_survives_reopening_the_index(tmp_path):
    """The map is stored with the index, not re-read from the original file.
    An index that can only cite pages while the source happens to still be on
    disk cites nothing in practice."""
    text, pm, _ = pdfdoc.assemble([p_text(p) for p in report_pages(12)])
    retrieval.MemoryRetriever(tmp_path, HashEmbedder(),
                              target_chars=200).index("doc", text, pm)

    fresh = retrieval.MemoryRetriever(tmp_path, HashEmbedder(), target_chars=200)
    assert len(fresh.page_map("doc")) == 12
    assert fresh.search("doc", text="clause 4", k=1, expand=0)[0].pages


def test_a_passage_without_a_page_map_cites_offsets_as_before(tmp_path):
    r = retrieval.MemoryRetriever(tmp_path, HashEmbedder(), target_chars=200)
    r.index("doc", "alpha beta gamma\n\ndelta epsilon zeta")
    passage = r.search("doc", text="delta", k=1, expand=0)[0]
    assert passage.pages is None
    assert passage.cite() == f"doc[{passage.start}:{passage.end}]"


# ----------------------------------------------------------- the excerpt cap

def test_the_excerpt_cap_never_truncates_an_output():
    """Inputs are capped; outputs are not. The instruction has to be inferred
    from the whole answer -- half an answer describes half a task."""
    pair = ingest.Pair(id="p", category="fetch", input_text="i" * 5000,
                       output="o" * 5000, target_prompt="")
    block = recover._blocks([pair], cap=100)
    assert "i" * 101 not in block
    assert "o" * 5000 in block


def p_text(lines: list[str]) -> str:
    return "\n".join(lines)
