"""CLI. The work is batch and offline, so this is the whole interface."""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Optional

import typer

from . import ingest, prompts, recover
from .client import (DEFAULT_MODEL, ObedientClient, OpenAIClient,
                     UnknownModel, judge_is_independent, resolve_models)
from .features import ALL_KEYS, extract
from .metrics import SCORING_VERSION, tier1
from .similarity import MAX_JUDGE_ATTEMPTS, JudgeFormatError

app = typer.Typer(add_completion=False, help=__doc__)


def _warn_dropped(param: str, model: str) -> None:
    note = (f"note: {model} rejected '{param}'; continuing without it.")
    if param in ("temperature", "seed"):
        note += ("  Generations will vary more between runs, so scores are "
                 "less repeatable.")
    typer.echo(note, err=True)


def _warn_truncated(cuts: list[recover.Truncation], total_pairs: int) -> None:
    """Say plainly that the producer and critic are working half-blind.

    Printed before the first API call, because the useful response is to stop
    and re-run with a bigger cap, not to read it afterwards next to a score
    that may mean nothing.
    """
    cap = cuts[0].cap
    need = max(c.total for c in cuts)
    bar = "!" * 78
    typer.echo(f"\n{bar}", err=True)
    typer.echo(f"WARNING: {len(cuts)} of {total_pairs} inputs are longer than "
               f"the {cap}-char excerpt cap.", err=True)
    typer.echo("The producer and critic will not see the end of these "
               "documents. A low score", err=True)
    typer.echo("for the affected groups may mean 'could not see the evidence' "
               "rather than", err=True)
    typer.echo("'could not recover the prompt'.\n", err=True)
    shown = cuts[:10]
    width = max(len(c.pair_id) for c in shown)
    for c in shown:
        typer.echo(f"  {c.pair_id:<{width}}  {c.total:>7} chars   "
                   f"{c.lost:>6} cut   (group {c.group_id})", err=True)
    if len(cuts) > 10:
        typer.echo(f"  ... and {len(cuts) - 10} more", err=True)
    typer.echo(f"\nRe-run with --excerpt-chars {need} to see all of them.",
               err=True)
    typer.echo(f"{bar}\n", err=True)


def _warn_shared_judge(models: dict[str, str]) -> None:
    """Say once that the judge and the inducer are the same model.

    Not fatal. A shared judge is a caveat on the prompt-match number, not a
    broken run -- but it is a caveat that has to travel with the number, so it
    is printed here and recorded in the JSON rather than left to be inferred
    from the model list.
    """
    typer.echo(f"\nnote: judge and inducer are both {models['judge']}.",
               err=True)
    typer.echo("      A judge sharing the inducer's model shares its blind "
               "spots, so the", err=True)
    typer.echo("      prompt-match number is a weaker signal than it looks. "
               "Select the", err=True)
    typer.echo("      judge on its own with --judge-model MODEL or "
               "$REVPROMPT_MODEL_JUDGE.\n", err=True)


def _client(simulate: bool, model: str | None, budget: int | None,
            judge_model: str | None = None):
    if simulate:
        return ObedientClient()
    overrides = {r: model for r in ("executor", "inducer", "judge", "features")} \
        if model else {}
    # After --model, so "use this model, but judge with that one" works. The
    # roles are selected separately by design; --model is the shorthand that
    # sets them together.
    if judge_model:
        overrides["judge"] = judge_model
    try:
        return OpenAIClient(models=overrides or None, max_tokens_budget=budget,
                            on_unsupported=_warn_dropped)
    except UnknownModel as e:
        raise typer.BadParameter(str(e)) from None


def _print_group(r: recover.GroupResult, show_outputs: bool) -> None:
    g = r.group
    kind = f"control set, {len(g)} inputs" if g.is_control else "single pair"
    if g.has_negative:
        kind += ", has negative"
    typer.echo(f"\n{'=' * 78}\n{g.id}  ({kind})\n{'=' * 78}")

    typer.echo(f"\nGOLD PROMPT:\n  {g.gold_prompt}")
    typer.echo(f"\nRECOVERED PROMPT ({r.best.origin}, round {r.best.round}):\n"
               f"  {r.best.text}")

    if r.best.prompt_score:
        typer.echo(f"\nprompt match:    {r.best.prompt_score.line()}")
        typer.echo(f"  naive was:     {r.naive.prompt_score.line()}")
    typer.echo(f"output fidelity: {r.best.fidelity:.4f}   "
               f"naive {r.naive.fidelity:.4f}   "
               f"{'BEATS NAIVE' if r.beats_naive else 'no better than naive'}")
    typer.echo(f"per input:       " + "  ".join(
        f"{k}={v:.3f}" for k, v in r.best.per_pair.items()))
    typer.echo(f"stopped after {r.rounds} round(s): {r.stopped_because}")

    if show_outputs:
        for p in g.pairs:
            typer.echo(f"\n  --- {p.id}{' [negative]' if p.is_negative else ''} ---")
            typer.echo(f"  wanted:   {p.output[:200]}")
            typer.echo(f"  produced: {r.best.outputs.get(p.id, '')[:200]}")


@app.command()
def features(
    pair_id: str = typer.Argument(..., help="pair id, e.g. summ-01"),
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """Print the measured feature vector for one gold output. No API needed."""
    pair = next((p for p in ingest.load(pairs_path) if p.id == pair_id), None)
    if pair is None:
        raise typer.BadParameter(f"no pair {pair_id!r}")
    f = extract(pair.output)
    width = max(len(k) for k in ALL_KEYS)
    for k in ALL_KEYS:
        typer.echo(f"{k:<{width}}  {f[k]:>10.3f}")


@app.command()
def groups(
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """List the prompt groups. No API needed."""
    gs = ingest.load_groups(pairs_path)
    controls = [g for g in gs if g.is_control]
    typer.echo(f"{len(gs)} groups over {sum(len(g) for g in gs)} pairs "
               f"({len(controls)} control sets)\n")
    for g in gs:
        mark = "control" if g.is_control else "single "
        neg = " [has negative]" if g.has_negative else ""
        typer.echo(f"  {mark} {g.id:22s} {len(g)} pair(s){neg}")


@app.command()
def compare(
    pair_id: str = typer.Argument(...),
    candidate_file: pathlib.Path = typer.Argument(..., help="file with a candidate output"),
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """Score a candidate output against gold. Deterministic, no API needed."""
    pairs = ingest.load(pairs_path)
    pair = next((p for p in pairs if p.id == pair_id), None)
    if pair is None:
        raise typer.BadParameter(f"no pair {pair_id!r}")
    scale = recover.fit_scale(pairs)
    score, breakdown = tier1(pair.output, candidate_file.read_text(encoding="utf-8"), scale)
    typer.echo(f"output fidelity {score:.4f}")
    for k in ("shape", "structure", "style", "combined_distance"):
        typer.echo(f"  {k:<20} {breakdown[k]:.3f}")


@app.command()
def run(
    group: Optional[str] = typer.Option(None, help="one group id; default is all"),
    category: Optional[str] = typer.Option(None, help="fetch | summarize | reason"),
    controls_only: bool = typer.Option(False, help="only the multi-input control sets"),
    rounds: int = typer.Option(2),
    simulate: bool = typer.Option(False, help="offline double: no key, no spend"),
    judge: bool = typer.Option(True, help="score the recovered prompt against gold"),
    judge_attempts: int = typer.Option(
        MAX_JUDGE_ATTEMPTS,
        help="how many times to re-ask a judge that replies in the wrong format"),
    show_outputs: bool = typer.Option(False, help="print produced vs wanted text"),
    model: Optional[str] = typer.Option(None, help="override every role"),
    judge_model: Optional[str] = typer.Option(
        None, help="the judge only; beats --model. Prefer a model other than "
                   "the one that proposed"),
    budget: Optional[int] = typer.Option(400_000, help="hard token ceiling"),
    excerpt_chars: int = typer.Option(
        recover.EXCERPT_CHARS,
        help="document chars the producer and critic see, per input"),
    out: Optional[pathlib.Path] = typer.Option(None, help="write results here"),
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """Recover the prompt for each group and score it. Spends money unless --simulate."""
    gs = ingest.load_groups(pairs_path, category=category)
    if group:
        gs = [g for g in gs if g.id == group]
        if not gs:
            raise typer.BadParameter(f"no group {group!r}")
    if controls_only:
        gs = [g for g in gs if g.is_control]

    cuts = recover.truncations(gs, excerpt_chars)
    if cuts:
        _warn_truncated(cuts, sum(len(g) for g in gs))

    # Fitted on every pair in the file, not on the selected ones. Feature
    # distances are divided by the spread of this corpus, so fitting on a
    # subset would make `run --group X` and `run` report different numbers for
    # the same output -- and nothing downstream could tell they were different
    # scales.
    scale = recover.fit_scale(ingest.load(pairs_path))

    client = _client(simulate, model, budget, judge_model)
    typer.echo(f"{len(gs)} group(s), system prompts v{prompts.VERSION}, "
               f"scoring v{SCORING_VERSION}, excerpt cap {excerpt_chars}")
    typer.echo(f"scale fitted on {scale.sample_size} gold output(s) "
               f"[{scale.fingerprint}]\n")

    independent = judge_is_independent(getattr(client, "models", {}))
    if judge and independent is False:
        _warn_shared_judge(client.models)

    try:
        results = recover.recover_all(client, gs, rounds=rounds, judge=judge,
                                      scale=scale, verbose=True,
                                      excerpt_chars=excerpt_chars,
                                      judge_attempts=judge_attempts)
    except JudgeFormatError as e:
        # Deliberately fatal: a fabricated similarity score is indistinguishable
        # from a real one downstream. Shown as an error, not a traceback.
        typer.echo(f"\nJUDGE ERROR: {e}", err=True)
        typer.echo(f"usage so far: {client.usage.summary()}", err=True)
        typer.echo("Nothing was scored. Retry, raise --judge-attempts, or set "
                   "$REVPROMPT_MODEL_JUDGE to a model that follows the format.",
                   err=True)
        raise typer.Exit(1) from None
    for r in results:
        _print_group(r, show_outputs)

    s = recover.summarise(results)
    typer.echo(f"\n{'=' * 78}\nSUMMARY")
    typer.echo(f"  groups                {int(s['groups'])}")
    typer.echo(f"  beat naive            {int(s['beat_naive'])}/{int(s['groups'])}")
    typer.echo(f"  mean output fidelity  {s['mean_fidelity']:.4f}")
    if "mean_prompt_similarity" in s:
        typer.echo(f"  mean prompt match     {s['mean_prompt_similarity']:.4f}")
        typer.echo(f"  worst contamination   {s['max_contamination']:.4f}")
        typer.echo(f"  contaminated groups   {int(s['contaminated_groups'])}"
                   f"/{int(s['groups'])}")
    typer.echo(f"  usage                 {client.usage.summary()}")

    if out:
        # Everything needed to say what produced these numbers. A stored score
        # with no model attached cannot be compared with anything -- and the
        # provenance has to be written when the score is, because retrofitting
        # it afterwards means guessing.
        u = client.usage
        out.write_text(json.dumps({
            "scoring_version": SCORING_VERSION,
            "prompts_version": prompts.VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": getattr(client, "models", {"all": getattr(
                client, "model", client.__class__.__name__)}),
            # null when the client does not map models to roles. The caveat
            # belongs next to the score it qualifies: read months later, the
            # model list alone no longer says whether that mattered.
            "judge_independent": independent,
            "embedding_model": None,     # retrieval is not in the loop yet
            "usage": {
                "calls": u.calls,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "cached_tokens": u.cached_tokens,
                "by_model": u.by_model,
            },
            "scale_corpus": {
                "path": str(pairs_path),
                "gold_outputs": scale.sample_size,
                "fingerprint": scale.fingerprint,
            },
            "excerpt_chars": excerpt_chars,
            "truncated_inputs": [c.pair_id for c in cuts],
            "summary": s,
            "groups": [{
                "id": r.group.id,
                "is_control": r.group.is_control,
                "gold_prompt": r.group.gold_prompt,
                "recovered_prompt": r.best.text,
                "output_fidelity": r.best.fidelity,
                "naive_fidelity": r.naive.fidelity,
                "prompt_similarity": (r.best.prompt_score.judged
                                      if r.best.prompt_score else None),
                "contamination": (r.best.prompt_score.contamination
                                  if r.best.prompt_score else None),
                "per_pair": r.best.per_pair,
            } for r in results],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"\nwrote {out}")


@app.command()
def models(
    available: bool = typer.Option(False, "--available",
                                   help="list what this key can actually use"),
) -> None:
    """Show which model each role will use. --available needs a key."""
    typer.echo(f"default: {DEFAULT_MODEL}\n")
    resolved = resolve_models()
    for role, m in resolved.items():
        typer.echo(f"  {role:<10} {m}")
    typer.echo("\nOverride with --model, $REVPROMPT_MODEL, "
               "or $REVPROMPT_MODEL_<ROLE>.")
    if judge_is_independent(resolved) is False:
        typer.echo("\nnote: the judge and the inducer resolve to the same "
                   "model. That is allowed, but a\n      judge with the "
                   "inducer's blind spots is a weaker check. Prefer "
                   "$REVPROMPT_MODEL_JUDGE\n      or --judge-model.")
    if available:
        try:
            client = OpenAIClient(verify_models=False)
        except RuntimeError as e:          # no key: say so, do not stack-trace
            raise typer.BadParameter(f"{e}. --available needs one.") from None
        typer.echo("\navailable on this key:")
        for m in client.available_models():
            typer.echo(f"  {m}")


def _retriever(backend: str, index_dir: pathlib.Path, real: bool):
    from . import retrieval
    from .embedding import HashEmbedder, OpenAIEmbedder
    try:
        embedder = OpenAIEmbedder() if real else HashEmbedder()
    except RuntimeError as e:
        raise typer.BadParameter(f"{e}. Use --no-real-embeddings to run "
                                 f"offline with the hashing double.") from None
    try:
        return retrieval.build(backend, index_dir, embedder=embedder)
    except ImportError:
        raise typer.BadParameter(
            "the 'lance' backend needs lancedb: pip install -e '.[rag]'"
        ) from None


@app.command()
def index(
    doc_id: str = typer.Argument(..., help="name to file this document under"),
    path: pathlib.Path = typer.Argument(..., help="the document to index"),
    backend: str = typer.Option("lance", help="lance | memory"),
    real_embeddings: bool = typer.Option(
        True, help="use the API; --no-real-embeddings uses the offline double"),
    index_dir: pathlib.Path = typer.Option(None, help="where the index lives"),
) -> None:
    """Build a retrieval index for one document. Spends a little on embeddings."""
    from . import retrieval
    r = _retriever(backend, index_dir or retrieval.DEFAULT_INDEX_DIR,
                   real_embeddings)
    text = path.read_text(encoding="utf-8")
    n = r.index(doc_id, text)
    typer.echo(f"indexed {doc_id}: {len(text)} chars -> {n} chunks "
               f"({r.embedder.model})")
    typer.echo(f"  cache: {r.embedder.hits} hit(s), {r.embedder.misses} miss(es)")


@app.command()
def retrieve(
    doc_id: str = typer.Argument(...),
    query: str = typer.Argument(..., help="what to search for"),
    k: int = typer.Option(5, help="how many chunks to keep before expanding"),
    expand: int = typer.Option(1, help="neighbouring chunks to include per hit"),
    chars: int = typer.Option(400, help="how much of each passage to print"),
    backend: str = typer.Option("lance", help="lance | memory"),
    real_embeddings: bool = typer.Option(True),
    index_dir: pathlib.Path = typer.Option(None),
) -> None:
    """Search an indexed document. For inspecting retrieval quality by hand."""
    from . import retrieval
    r = _retriever(backend, index_dir or retrieval.DEFAULT_INDEX_DIR,
                   real_embeddings)
    try:
        passages = r.search(doc_id, text=query, k=k, expand=expand)
    except KeyError as e:
        raise typer.BadParameter(str(e)) from None
    if not passages:
        typer.echo("no passages matched")
        return
    for p in passages:
        typer.echo(f"\n{'-' * 78}\n{p.cite()}  score {p.score:.4f}  "
                   f"chunks {list(p.ordinals)}\n{'-' * 78}")
        typer.echo(p.text[:chars] + ("..." if len(p.text) > chars else ""))


@app.command()
def eval_retrieval(
    source: Optional[pathlib.Path] = typer.Option(
        None, help="document to index and probe; default is the Odyssey"),
    doc_id: str = typer.Option("odyssey-eval"),
    k: int = typer.Option(5, help="precision@k"),
    keyword_weight: float = typer.Option(1.0, help="fusion weight, keyword arm"),
    vector_weight: float = typer.Option(1.0, help="fusion weight, vector arm"),
    backend: str = typer.Option("lance", help="lance | memory"),
    real_embeddings: bool = typer.Option(
        False, help="use the API; the default is the free offline double"),
    index_dir: pathlib.Path = typer.Option(None),
    fail_under: float = typer.Option(
        0.0, help="exit non-zero if mean fused precision falls below this"),
) -> None:
    """Score keyword, vector and fused retrieval against known-relevant chunks.

    Free by default: the offline embedder costs nothing, and its numbers are a
    regression guard rather than a verdict on retrieval. Pass
    --real-embeddings for the measurement that decides anything.
    """
    from . import retrieval, retrieval_eval

    r = _retriever(backend, index_dir or retrieval.DEFAULT_INDEX_DIR,
                   real_embeddings)
    text = (source or retrieval_eval.DEFAULT_SOURCE).read_text(encoding="utf-8")
    n = r.index(doc_id, text)
    typer.echo(f"{doc_id}: {len(text)} chars, {n} chunks, "
               f"embedder {r.embedder.model}\n")

    weights = [keyword_weight, vector_weight]
    results = retrieval_eval.evaluate(r, doc_id, k=k, weights=weights)
    typer.echo(retrieval_eval.format_report(results, k))

    mean_fused = (sum(x.arms["fused"].precision for x in results) / len(results)
                  if results else 0.0)
    if fail_under and mean_fused < fail_under:
        typer.echo(f"\nmean fused precision {mean_fused:.3f} is below "
                   f"--fail-under {fail_under}", err=True)
        raise typer.Exit(1)


@app.command()
def show_prompts() -> None:
    """Print the system's own instructions -- the ones we author, not recover."""
    typer.echo(f"version {prompts.VERSION}\n")
    for name in ("PRODUCER", "CRITIC", "REVISER", "SIMILARITY_JUDGE", "EXECUTOR"):
        typer.echo(f"{'=' * 78}\n{name}\n{'=' * 78}\n{getattr(prompts, name)}\n")


if __name__ == "__main__":
    app()
