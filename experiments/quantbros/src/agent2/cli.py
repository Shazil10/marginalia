"""Command-line interface for Agent 2."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from agent2.agents.models import AgenticRunFailed, PaperDecision
from agent2.agents.orchestrator import OrchestratorAgent
from agent2.database import connect_sqlite, iter_agent1_inputs
from agent2.ingest import load_agent1_from_db, load_agent1_json
from agent2.logic import embedded_preset_name, extract_logic
from agent2.paper_ingest import ingest_papers
from agent2.pipeline import execute_logic_backtest
from agent2.settings import get_settings


LOGGER = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _add_run_flags(parser: argparse.ArgumentParser, settings) -> None:
    parser.add_argument("--paper-id")
    parser.add_argument("--fixture")
    parser.add_argument("--db-path", default=settings.database_path)
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--artifacts-dir", default=settings.artifacts_dir)
    parser.add_argument("--run-id")
    parser.add_argument("--max-harness-turns", type=int, default=16)
    parser.add_argument("--max-tool-calls", type=int, default=32)
    parser.add_argument("--max-wall-clock-seconds", type=int, default=900)
    parser.add_argument("--max-blueprint-attempts", type=int, default=4)
    parser.add_argument("--max-codegen-attempts", type=int, default=5)
    parser.add_argument("--max-execute-attempts", type=int, default=5)
    parser.add_argument("--max-analyst-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, help="Deprecated alias for --max-harness-turns")
    parser.add_argument(
        "--legacy-engine",
        action="store_true",
        help="Run the deprecated LogicSpec/engine path instead of the default blueprint+code harness.",
    )
    parser.add_argument(
        "--legacy-pipeline",
        action="store_true",
        help="Deprecated alias for --legacy-engine.",
    )
    parser.add_argument(
        "--legacy-fallbacks",
        action="store_true",
        help="Deprecated alias for --legacy-engine.",
    )


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="agent2", description="Agent 2 backtesting runtime")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest-papers", help="Ingest local papers into SQLite + fixtures")
    ingest_parser.add_argument("--papers-dir", default=settings.papers_dir)
    ingest_parser.add_argument("--db-path", default=settings.database_path)
    ingest_parser.add_argument("--fixtures-dir", default=settings.fixtures_dir)
    ingest_parser.add_argument(
        "--legacy-heuristics",
        action="store_true",
        help="Deprecated: use heuristic classification and logic hints during ingest.",
    )

    run_parser = subparsers.add_parser("run", help="Run one paper through the default harness")
    _add_run_flags(run_parser, settings)

    run_agentic_parser = subparsers.add_parser("run-agentic", help="Explicit harness run entrypoint")
    _add_run_flags(run_agentic_parser, settings)

    run_all_parser = subparsers.add_parser("run-all", help="Run all ingested papers from the local SQLite DB")
    run_all_parser.add_argument("--db-path", default=settings.database_path)
    run_all_parser.add_argument("--start", type=_parse_date)
    run_all_parser.add_argument("--end", type=_parse_date)
    run_all_parser.add_argument("--commission-bps", type=float, default=1.0)
    run_all_parser.add_argument("--slippage-bps", type=float, default=2.0)
    run_all_parser.add_argument("--artifacts-dir", default=settings.artifacts_dir)
    run_all_parser.add_argument("--limit", type=int)
    run_all_parser.add_argument("--max-harness-turns", type=int, default=16)
    run_all_parser.add_argument("--max-tool-calls", type=int, default=32)
    run_all_parser.add_argument("--max-wall-clock-seconds", type=int, default=900)
    run_all_parser.add_argument("--max-blueprint-attempts", type=int, default=4)
    run_all_parser.add_argument("--max-codegen-attempts", type=int, default=5)
    run_all_parser.add_argument("--max-execute-attempts", type=int, default=5)
    run_all_parser.add_argument("--max-analyst-turns", type=int, default=3)
    run_all_parser.add_argument("--max-turns", type=int, help="Deprecated alias for --max-harness-turns")
    run_all_parser.add_argument("--legacy-engine", action="store_true")
    run_all_parser.add_argument("--legacy-pipeline", action="store_true")
    run_all_parser.add_argument("--legacy-fallbacks", action="store_true")
    return parser


def _load_one(paper_id: str | None, fixture: str | None, db_path: str):
    if paper_id:
        return load_agent1_from_db(db_path, paper_id)
    if fixture:
        return load_agent1_json(fixture)
    raise SystemExit("Provide --paper-id or --fixture")


def _run_legacy(agent1, args):
    LOGGER.warning("Running deprecated legacy engine path.")
    preset = embedded_preset_name(agent1, allow_legacy_fallbacks=True)
    logic, _ = extract_logic(
        agent1,
        preset=preset,
        use_llm=False,
        allow_legacy_fallbacks=True,
    )
    classification = PaperDecision(
        paper_category=agent1.paper_category,
        asset_universe_type=agent1.asset_universe_type,
        asset_universe_notes=agent1.asset_universe_notes,
        regime_claim=agent1.regime_claim,
        reasoning="Deprecated legacy engine path used stored Agent1 metadata.",
    )
    return execute_logic_backtest(
        agent1=agent1,
        classification=classification,
        logic=logic,
        start=args.start,
        end=args.end,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        artifacts_dir=args.artifacts_dir,
        run_id=getattr(args, "run_id", None),
    )


def _print_agentic_summary(result) -> None:
    print(
        json.dumps(
            {
                "status": "success",
                "run_id": result.run_id,
                "paper_id": result.state.paper_id,
                "artifacts_dir": str(result.output_dir),
                "last_step": str(result.last_step),
                "trace": str(result.output_dir / "trace.jsonl"),
                "tear_sheet": str(result.output_dir / "tear_sheet.json"),
            },
            indent=2,
        )
    )


def _run_harness(agent1, args):
    effective_max_turns = args.max_turns if getattr(args, "max_turns", None) is not None else args.max_harness_turns
    orchestrator = OrchestratorAgent()
    return orchestrator.run(
        agent1,
        start=args.start,
        end=args.end,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        artifacts_dir=args.artifacts_dir,
        run_id=getattr(args, "run_id", None),
        max_turns=effective_max_turns,
        max_tool_calls=args.max_tool_calls,
        max_wall_clock_seconds=args.max_wall_clock_seconds,
        max_blueprint_attempts=args.max_blueprint_attempts,
        max_codegen_attempts=args.max_codegen_attempts,
        max_execute_attempts=args.max_execute_attempts,
        max_analyst_turns=args.max_analyst_turns,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "ingest-papers":
        summary = ingest_papers(
            papers_dir=args.papers_dir,
            db_path=args.db_path,
            fixtures_dir=args.fixtures_dir,
            use_llm_classifier=not args.legacy_heuristics,
            allow_legacy_heuristics=args.legacy_heuristics,
        )
        print(
            json.dumps(
                {
                    "ingested_count": summary.ingested_count,
                    "paper_ids": summary.paper_ids,
                    "db_path": str(summary.db_path),
                    "fixtures_dir": str(summary.fixtures_dir),
                    "mode": "legacy" if args.legacy_heuristics else "agentic",
                },
                indent=2,
            )
        )
        return

    if args.command in {"run", "run-agentic"}:
        agent1 = _load_one(args.paper_id, args.fixture, args.db_path)
        if args.legacy_engine or args.legacy_pipeline or args.legacy_fallbacks:
            result = _run_legacy(agent1, args)
            print(result.tear_sheet.model_dump_json(indent=2))
            return
        try:
            result = _run_harness(agent1, args)
        except AgenticRunFailed as exc:
            print(exc.failure.model_dump_json(indent=2))
            raise SystemExit(1) from exc
        _print_agentic_summary(result)
        return

    if args.command == "run-all":
        conn = connect_sqlite(args.db_path)
        succeeded: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []
        try:
            for idx, agent1 in enumerate(iter_agent1_inputs(conn), start=1):
                if args.limit is not None and idx > args.limit:
                    break
                try:
                    if args.legacy_engine or args.legacy_pipeline or args.legacy_fallbacks:
                        result = _run_legacy(agent1, args)
                        succeeded.append(
                            {
                                "paper_id": agent1.paper_id,
                                "run_id": result.run_id,
                                "output_dir": str(result.output_dir),
                                "mode": "legacy",
                            }
                        )
                        continue
                    agentic = _run_harness(agent1, args)
                    succeeded.append(
                        {
                            "paper_id": agent1.paper_id,
                            "run_id": agentic.run_id,
                            "output_dir": str(agentic.output_dir),
                            "last_step": str(agentic.last_step),
                            "mode": "agentic",
                        }
                    )
                except AgenticRunFailed as exc:
                    failed.append(
                        {
                            "paper_id": agent1.paper_id,
                            "message": exc.failure.message,
                            "trace": exc.failure.trace_path,
                            "failure": exc.failure.artifact_paths.get("failure", ""),
                        }
                    )
        finally:
            conn.close()
        print(json.dumps({"succeeded": succeeded, "failed": failed}, indent=2))
        if failed:
            raise SystemExit(1)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
