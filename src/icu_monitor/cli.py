"""The ``icu-monitor`` command line.

One entry point for the four things the project does: build a dataset, train a model, serve
the API, and open the dashboard - plus ``tick`` and ``info``, which exist so a deployment can
be checked without a browser.

Every subcommand calls :func:`configure_logging` before it touches anything. On Windows the
default console encoding is cp1252, and this project's log lines legitimately contain ``≥``,
``SpO₂`` and ``°C``; without the UTF-8 reconfiguration the first NEWS2 message raises
``UnicodeEncodeError`` and takes the process with it.

Heavy imports (pandas, scikit-learn, uvicorn) happen inside the handlers rather than at module
scope, so ``icu-monitor --help`` and ``icu-monitor info`` stay instant.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy.engine import make_url

from icu_monitor import __version__
from icu_monitor.config import get_settings
from icu_monitor.logging_setup import configure_logging

logger = logging.getLogger("icu_monitor.cli")

PROGRAM = "icu-monitor"


def _echo(message: str) -> None:
    print(message, flush=True)


def _safe_database_url(url: str | None) -> str:
    """Render a database URL without exposing credentials in diagnostic output."""
    if not url:
        return "unset"
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        # Do not fall back to the raw value: malformed URLs can still contain a password.
        return "<configured URL could not be parsed>"


def _rule(title: str = "") -> None:
    _echo(f"\n{title}\n{'-' * max(12, len(title))}" if title else "-" * 60)


def _src_root() -> Path:
    """The directory that has to be importable for a non-installed checkout.

    ``icu_monitor/__init__.py`` -> ``icu_monitor`` -> ``src``. Passed to child processes so
    ``streamlit run`` works straight from a clone, with no ``pip install -e .``.
    """
    import icu_monitor

    return Path(icu_monitor.__file__).resolve().parent.parent


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    root = str(_src_root())
    existing = env.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    return env


# --------------------------------------------------------------------------- etl


def cmd_etl(args: argparse.Namespace) -> int:
    from icu_monitor.data import build_dataset, synthesise_cohort

    cfg = get_settings()
    if args.synthetic:
        summary = synthesise_cohort(config=cfg, n_stays=args.stays, progress=_echo)
    else:
        try:
            summary = build_dataset(config=cfg, limit=args.limit, progress=_echo)
        except FileNotFoundError as exc:
            _echo(f"{exc}")
            _echo(f"\nRe-run as `{PROGRAM} etl --synthetic` to generate a stand-in cohort.")
            return 2

    _rule("Dataset")
    _echo(f"source          {summary.source}")
    _echo(f"stays seen      {summary.stays_seen:,}")
    _echo(f"stays labelled  {summary.stays_labelled:,}")
    _echo(f"stays used      {summary.stays_used:,}")
    _echo(f"windows         {summary.windows:,}")
    _echo(f"features        {summary.features:,}")
    for label, count in summary.class_counts.items():
        share = count / summary.windows if summary.windows else 0.0
        _echo(f"  {label:<8} {count:>7,}  {share:>6.1%}")
    for name, path in summary.paths.items():
        _echo(f"wrote {name:<10} {path}")
    return 0


# ------------------------------------------------------------------------- train


def cmd_train(args: argparse.Namespace) -> int:
    from icu_monitor.ml.train import train

    cfg = get_settings()
    candidates = [c.strip() for c in args.candidates.split(",")] if args.candidates else None
    try:
        result = train(
            config=cfg,
            candidates=candidates,
            n_splits=args.splits,
            test_fraction=args.test_fraction,
            with_importances=not args.no_importances,
            progress=_echo,
        )
    except FileNotFoundError as exc:
        _echo(f"{exc}")
        return 2

    metrics = result.report.as_dict()
    _rule("Cross-validation")
    for candidate in sorted(result.candidates, key=lambda c: c.macro_f1_mean, reverse=True):
        mark = "*" if candidate.name == result.winner else " "
        _echo(
            f" {mark} {candidate.name:<26} macro-F1 {candidate.macro_f1_mean:.3f}"
            f" ± {candidate.macro_f1_std:.3f}   bal-acc "
            f"{candidate.balanced_accuracy_mean:.3f}"
        )

    _rule("Held-out performance")
    _echo(f"version           {result.version}")
    _echo(f"source            {result.dataset.get('source', 'unrecorded')}")
    _echo(f"macro F1          {metrics.get('macro_f1', 0):.3f}")
    _echo(f"balanced accuracy {metrics.get('balanced_accuracy', 0):.3f}")
    _echo(f"accuracy          {metrics.get('accuracy', 0):.3f}")
    _echo(f"Cohen's kappa     {metrics.get('cohen_kappa', 0):.3f}")
    roc = metrics.get("roc_auc") or {}
    _echo(f"ROC-AUC (macro)   {roc.get('macro', 0):.3f}")
    for label, scores in (metrics.get("per_class") or {}).items():
        _echo(
            f"  {label:<8} precision {scores.get('precision', 0):.3f}"
            f"  recall {scores.get('recall', 0):.3f}  F1 {scores.get('f1', 0):.3f}"
            f"  n={int(scores.get('support', 0)):,}"
        )
    if result.dataset.get("synthetic"):
        # The synthetic cohort is generated from rules, so a model can learn those rules almost
        # perfectly. Printing "modest by design" under a macro-F1 of 1.000 would be the exact
        # opposite of what a reader needs to be told.
        _echo(
            "\nThis model was trained on SYNTHETIC data. The scores above measure how well it "
            "recovered the simulator's own rules - an easier problem than medicine - and are not "
            "clinical performance. Download the archive and re-run `icu-monitor etl` before "
            "quoting any of them."
        )
    else:
        _echo(
            "\nThese numbers are modest by design of the problem, not by accident: read the "
            "model card before drawing conclusions from any single prediction."
        )
    return 0


# ------------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = get_settings()
    host = args.host or cfg.api_host
    port = args.port or cfg.api_port
    if not cfg.api_key:
        _echo(
            "WARNING: ICU_API_KEY is not set, so every /api/v1 route is unauthenticated. "
            "That is fine on localhost and wrong on a network interface."
        )
    if args.reload:
        uvicorn.run("icu_monitor.api.main:app", host=host, port=port, reload=True)
    else:
        from icu_monitor.api.main import app

        uvicorn.run(app, host=host, port=port, log_config=None)
    return 0


# --------------------------------------------------------------------- dashboard


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Hand off to ``streamlit run``.

    Streamlit has to own the process - it installs its own signal handling, watcher and
    server - so this shells out rather than importing anything. The script it runs is the
    one inside the package, which keeps ``icu-monitor dashboard`` working from an installed
    wheel where the repository-root ``app.py`` shim does not exist.
    """
    script = Path(__file__).resolve().parent / "ui" / "app.py"
    if not script.exists():  # pragma: no cover - only if the package is broken
        _echo(f"Dashboard script missing at {script}")
        return 2
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(args.port),
        "--server.headless",
        "false" if args.browser else "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    _echo(f"$ {' '.join(command[2:])}")
    try:
        return subprocess.call(command, env=_child_env())
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 0


# -------------------------------------------------------------------------- tick


# Field names are not flag names: this command spells `bed_count` as `--beds`, and an error that
# points at `--bed-count` sends the operator looking for an option that does not exist.
TICK_FLAGS = {"bed_count": "--beds"}


def cmd_tick(args: argparse.Namespace) -> int:
    """Run the ward for a few ticks and print it. The no-browser proof that this works."""
    import json

    from icu_monitor.monitoring import build_engine

    overrides: dict[str, Any] = {}
    if args.beds is not None:  # not `if args.beds` - 0 is a value to refuse, not an absence
        overrides["bed_count"] = args.beds
    try:
        cfg = get_settings().with_overrides(**overrides) if overrides else get_settings()
    except ValidationError as exc:
        # `--beds 500` is a typo, not a request. Refusing it with the field's own bound is
        # more useful than either a traceback or a five-hundred-bed ward.
        _echo("Invalid option:")
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"]) or "value"
            _echo(f"  {TICK_FLAGS.get(field, '--' + field.replace('_', '-'))}: {error['msg']}")
        return 2

    engine = build_engine(config=cfg)
    try:
        snapshot = engine.run(max(1, args.ticks), backfill=args.ticks > 1)
        if args.json:
            _echo(json.dumps(snapshot.as_dict(), indent=2, default=str))
            return 0

        _rule(f"Ward at tick {snapshot.tick}  ({snapshot.duration_ms:.0f} ms)")
        _echo(f"source  {snapshot.source_label}")
        _echo(f"model   {snapshot.model_version or 'none (NEWS2 + vision only)'}")
        _echo(f"vision  {snapshot.vision_label}")
        _rule()
        header = f"{'bed':<6} {'patient':<20} {'level':<9} {'score':>6} {'NEWS2':>6}  factors"
        _echo(header)
        for bed in sorted(snapshot.beds, key=lambda b: b.assessment.composite_score, reverse=True):
            total = bed.assessment.news2_total
            top = "; ".join(f.description for f in bed.assessment.top_factors[:2]) or "-"
            _echo(
                f"{bed.patient.bed:<6} {bed.patient.display_name:<20} "
                f"{bed.assessment.level.value:<9} {bed.assessment.composite_score:>6.1f} "
                f"{'-' if total is None else total:>6}  {top[:52]}"
            )
        active = engine.alerts.active
        _rule(f"Active alerts ({len(active)})")
        for alert in active:
            _echo(f"[{alert.severity.value:<8}] {alert.patient_id:<10} {alert.message}")
        if not active:
            _echo("none")
        return 0
    finally:
        engine.close()


# -------------------------------------------------------------------------- info


def cmd_info(_: argparse.Namespace) -> int:
    """What this deployment actually resolved to. The first thing to run when something is off."""
    from icu_monitor.api.deps import AppState

    cfg = get_settings()
    _rule(f"{cfg.app_name} v{__version__}")
    _echo(f"environment      {cfg.environment}")
    _echo(f"python           {sys.version.split()[0]} on {sys.platform}")
    _echo(f"project root     {cfg.project_root}")
    _echo(f"database         {_safe_database_url(cfg.database_url)}")
    _echo(f"artifacts        {cfg.artifacts_dir}")
    _echo(f"beds / tick      {cfg.bed_count} / {cfg.tick_seconds:g} s")
    _echo(f"vitals source    {cfg.vitals_source}")
    _echo(f"frame / detector {cfg.frame_source} / {cfg.detector}")
    _echo(
        f"weights          ml {cfg.weight_ml} news2 {cfg.weight_news2} vision {cfg.weight_vision}"
    )
    _echo(
        f"thresholds       NEWS2 {cfg.news2_medium_threshold}/{cfg.news2_high_threshold} · "
        f"composite {cfg.composite_medium_threshold:g}/{cfg.composite_high_threshold:g}/"
        f"{cfg.composite_critical_threshold:g}"
    )
    _echo(f"api              {cfg.api_base_url}  key {'set' if cfg.api_key else 'NOT SET'}")

    state = AppState(cfg)
    try:
        _rule("Readiness")
        for component in state.readiness():
            flag = "ready" if component["ready"] else "degraded"
            _echo(f"  {component['name']!s:<10} {flag:<9} {component['detail']}")
    finally:
        state.close()
    return 0


# ------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="ICU patient risk monitoring - dataset, model, API and dashboard.",
        epilog=(
            "Typical first run:\n"
            f"  {PROGRAM} etl --synthetic     # no 8 MB download needed\n"
            f"  {PROGRAM} train\n"
            f"  {PROGRAM} dashboard\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    etl = sub.add_parser("etl", help="Build the processed window dataset.")
    etl.add_argument("--limit", type=int, help="Only read this many records (a quick check).")
    etl.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a stand-in cohort with the simulator instead of reading the archive.",
    )
    etl.add_argument("--stays", type=int, default=900, help="Synthetic stays to generate.")
    etl.set_defaults(handler=cmd_etl)

    train = sub.add_parser("train", help="Train, evaluate, and write the model artefact.")
    train.add_argument("--candidates", help="Comma-separated subset of the candidate models.")
    train.add_argument("--splits", type=int, default=5, help="Cross-validation folds.")
    train.add_argument("--test-fraction", type=float, default=0.2, dest="test_fraction")
    train.add_argument(
        "--no-importances",
        action="store_true",
        help="Skip permutation importance, which dominates training time.",
    )
    train.set_defaults(handler=cmd_train)

    serve = sub.add_parser("serve", help="Run the FastAPI service with uvicorn.")
    serve.add_argument("--host", help="Override ICU_API_HOST.")
    serve.add_argument("--port", type=int, help="Override ICU_API_PORT.")
    serve.add_argument("--reload", action="store_true", help="Reload on source changes.")
    serve.set_defaults(handler=cmd_serve)

    dash = sub.add_parser("dashboard", help="Open the Streamlit dashboard.")
    dash.add_argument("--port", type=int, default=8501)
    dash.add_argument("--browser", action="store_true", help="Let Streamlit open a browser.")
    dash.set_defaults(handler=cmd_dashboard)

    tick = sub.add_parser("tick", help="Advance the ward and print it - no browser needed.")
    tick.add_argument("--ticks", type=int, default=12, help="Ticks to run before printing.")
    tick.add_argument("--beds", type=int, help="Override ICU_BED_COUNT.")
    tick.add_argument("--json", action="store_true", help="Emit the raw snapshot as JSON.")
    tick.set_defaults(handler=cmd_tick)

    info = sub.add_parser("info", help="Print resolved settings and component readiness.")
    info.set_defaults(handler=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=logging.DEBUG if args.verbose else None)
    try:
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _echo("\nInterrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "cmd_dashboard",
    "cmd_etl",
    "cmd_info",
    "cmd_serve",
    "cmd_tick",
    "cmd_train",
    "main",
]
