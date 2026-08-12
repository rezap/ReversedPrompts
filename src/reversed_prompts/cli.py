"""CLI. The work is batch and offline, so this is the whole interface."""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import typer

from . import ingest, prompts, recover
from .client import (DEFAULT_MODEL, ObedientClient, OpenAIClient,
                     UnknownModel, resolve_models)
from .features import ALL_KEYS, extract
from .metrics import tier1

app = typer.Typer(add_completion=False, help=__doc__)


def _warn_dropped(param: str, model: str) -> None:
    note = (f"note: {model} rejected '{param}'; continuing without it.")
    if param in ("temperature", "seed"):
        note += ("  Generations will vary more between runs, so scores are "
                 "less repeatable.")
    typer.echo(note, err=True)


def _client(simulate: bool, model: str | None, budget: int | None):
    if simulate:
        return ObedientClient()
    overrides = {r: model for r in ("executor", "inducer", "judge", "features")} \
        if model else None
    try:
        return OpenAIClient(models=overrides, max_tokens_budget=budget,
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
    show_outputs: bool = typer.Option(False, help="print produced vs wanted text"),
    model: Optional[str] = typer.Option(None, help="override every role"),
    budget: Optional[int] = typer.Option(400_000, help="hard token ceiling"),
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

    client = _client(simulate, model, budget)
    typer.echo(f"{len(gs)} group(s), system prompts v{prompts.VERSION}\n")

    results = recover.recover_all(client, gs, rounds=rounds, judge=judge,
                                  verbose=True)
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
    typer.echo(f"  usage                 {client.usage.summary()}")

    if out:
        out.write_text(json.dumps({
            "prompts_version": prompts.VERSION,
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
    for role, m in resolve_models().items():
        typer.echo(f"  {role:<10} {m}")
    typer.echo("\nOverride with --model, $REVPROMPT_MODEL, "
               "or $REVPROMPT_MODEL_<ROLE>.")
    if available:
        try:
            client = OpenAIClient(verify_models=False)
        except RuntimeError as e:          # no key: say so, do not stack-trace
            raise typer.BadParameter(f"{e}. --available needs one.") from None
        typer.echo("\navailable on this key:")
        for m in client.available_models():
            typer.echo(f"  {m}")


@app.command()
def show_prompts() -> None:
    """Print the system's own instructions -- the ones we author, not recover."""
    typer.echo(f"version {prompts.VERSION}\n")
    for name in ("PRODUCER", "CRITIC", "REVISER", "SIMILARITY_JUDGE", "EXECUTOR"):
        typer.echo(f"{'=' * 78}\n{name}\n{'=' * 78}\n{getattr(prompts, name)}\n")


if __name__ == "__main__":
    app()
