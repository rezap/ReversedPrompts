"""CLI. Phase 0 is batch and offline, so the CLI is the whole interface."""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import typer

from . import ingest
from .client import ObedientClient, OpenAIClient
from .features import ALL_KEYS, extract
from .induce import fit_scale, induce, induce_per_pair, verify_on_holdout
from .metrics import tier1

app = typer.Typer(add_completion=False, help=__doc__)


def _client(model: str | None, budget: int | None):
    overrides = {}
    if model:
        overrides = {"executor": model, "inducer": model, "judge": model}
    return OpenAIClient(models=overrides or None, max_tokens_budget=budget)


@app.command()
def features(
    pair_id: str = typer.Argument(..., help="pair id, e.g. summ-01"),
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """Print the measured feature vector for one gold output. No API needed."""
    pairs = ingest.load(pairs_path)
    pair = next((p for p in pairs if p.id == pair_id), None)
    if pair is None:
        raise typer.BadParameter(f"no pair {pair_id!r}")
    f = extract(pair.output)
    width = max(len(k) for k in ALL_KEYS)
    for k in ALL_KEYS:
        typer.echo(f"{k:<{width}}  {f[k]:>10.3f}")


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
    scale = fit_scale(pairs)
    score, breakdown = tier1(pair.output, candidate_file.read_text(), scale)
    typer.echo(f"tier-1 score {score:.4f}")
    for k in ("shape", "structure", "style", "combined_distance"):
        typer.echo(f"  {k:<20} {breakdown[k]:.3f}")


@app.command()
def run(
    category: Optional[str] = typer.Option(None, help="fetch | summarize | reason"),
    rounds: int = typer.Option(3),
    beam: int = typer.Option(2),
    eval_sample: int = typer.Option(4, help="pairs scored per candidate during search"),
    holdout: int = typer.Option(2),
    model: Optional[str] = typer.Option(None, help="override every role"),
    budget: Optional[int] = typer.Option(400_000, help="hard token ceiling"),
    out: Optional[pathlib.Path] = typer.Option(None, help="write the PromptSpec here"),
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """Induce a prompt and check it against the naive baseline. Spends money."""
    pairs = ingest.load(pairs_path, category=category)
    train, held = ingest.split(pairs, holdout=holdout)
    typer.echo(f"{len(train)} train / {len(held)} held out"
               + (f" [{category}]" if category else ""))

    client = _client(model, budget)
    scale = fit_scale(train)
    result = induce(client, train, rounds=rounds, beam=beam,
                    eval_sample=eval_sample, scale=scale, verbose=True)

    typer.echo("\n--- search (train) ---")
    typer.echo(result.report())

    typer.echo("\n--- exit criterion (held out) ---")
    best_h, base_h = verify_on_holdout(client, result.best, result.baseline.spec,
                                       held, scale)
    verdict = "PASS" if best_h.score > base_h.score else "FAIL"
    lift = ((best_h.score - base_h.score) / base_h.score) if base_h.score else 0.0
    typer.echo(f"induced  {best_h.score:.4f}")
    typer.echo(f"baseline {base_h.score:.4f}")
    typer.echo(f"lift     {lift:+.1%}   {verdict}")
    typer.echo(f"\nusage: {client.usage.summary()}")

    typer.echo("\n--- recovered prompt ---")
    typer.echo(result.best.spec.render())

    if out:
        out.write_text(json.dumps({
            "spec": result.best.spec.model_dump(),
            "train_score": result.best.score,
            "holdout_score": best_h.score,
            "holdout_baseline": base_h.score,
            "passes_exit_criterion": best_h.score > base_h.score,
            "usage": client.usage.summary(),
        }, indent=2, ensure_ascii=False))
        typer.echo(f"\nwrote {out}")


@app.command()
def per_pair(
    simulate: bool = typer.Option(False, help="use the offline double, no key, no spend"),
    model: Optional[str] = typer.Option(None),
    budget: Optional[int] = typer.Option(400_000),
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """Recover one spec per pair and score it against naive on that pair.

    This is the mode the shipped corpus supports -- it has a distinct prompt
    per pair, so there is no single corpus-level prompt to recover. Measures
    reconstruction, not generalization.
    """
    pairs = ingest.load(pairs_path)
    client = ObedientClient() if simulate else _client(model, budget)
    scale = fit_scale(pairs)

    wins = 0
    for pair, induced, naive in induce_per_pair(client, pairs, scale):
        ok = induced.score > naive.score
        wins += ok
        typer.echo(f"{pair.id:10s} induced={induced.score:.4f} "
                   f"naive={naive.score:.4f}  {'win' if ok else 'LOSS'}")
    typer.echo(f"\n{wins}/{len(pairs)} beat naive")
    typer.echo(f"usage: {client.usage.summary()}")


@app.command()
def dry_run(
    category: Optional[str] = typer.Option(None),
    pairs_path: pathlib.Path = typer.Option(ingest.DEFAULT_PAIRS, "--pairs"),
) -> None:
    """Exercise the whole loop against a simulated model. No key, no spend."""
    pairs = ingest.load(pairs_path, category=category)
    train, _ = ingest.split(pairs, holdout=2)
    client = ObedientClient()
    result = induce(client, train, rounds=1, eval_sample=2, verbose=True)
    typer.echo(result.report())
    typer.echo(f"usage: {client.usage.summary()}")


if __name__ == "__main__":
    app()
