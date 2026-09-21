"""Command-line entry point for Marginalia.

Examples
--------
Run a deterministic backtest from a spec JSON file (no LLM, no budget needed):

    python -m marginalia.cli backtest --spec my_spec.json --start 2015-01-01

Run the full pipeline on a paper PDF (requires Nebius budget):

    python -m marginalia.cli paper --pdf papers/2018_dowaward-giordano.pdf

Check Nebius connectivity / budget:

    python -m marginalia.cli ping
"""

from __future__ import annotations

import argparse
import json
import sys


def _print_result(result):
    print(json.dumps(result.metrics, indent=2))
    print(f"\noptimized parameters: {result.optimized_parameters}")
    print(f"grid combinations searched: {result.grid_searched}")
    if result.warnings:
        print("warnings:")
        for w in result.warnings:
            print(f"  - {w}")


def cmd_backtest(args) -> int:
    from marginalia.pipeline import backtest_spec_dict

    with open(args.spec) as f:
        spec_dict = json.load(f)
    result = backtest_spec_dict(
        spec_dict, start=args.start, end=args.end,
        cost_bps=args.cost_bps, optimize=not args.no_optimize,
    )
    _print_result(result)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nfull result -> {args.out}")
    return 0


def cmd_paper(args) -> int:
    from marginalia.pipeline import analyze_paper

    out = analyze_paper(
        args.pdf, start=args.start, end=args.end,
        cost_bps=args.cost_bps, optimize=not args.no_optimize,
    )
    print("Extracted spec:")
    print(json.dumps(json.loads(out.spec.model_dump_json()), indent=2, default=str))
    print("\nBacktest:")
    _print_result(out.backtest)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out.summary(), f, indent=2)
        print(f"\nsummary -> {args.out}")
    return 0


def cmd_ping(args) -> int:
    from marginalia.llm import ping, ModelTier, LLMError

    for tier in (ModelTier.FAST, ModelTier.POWER):
        try:
            print(f"{tier.value}: {ping(tier)!r}")
        except LLMError as e:
            print(f"{tier.value}: ERROR {e}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="marginalia", description="Marginalia quant engine")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", default=None)
    common.add_argument("--end", default=None)
    common.add_argument("--cost-bps", type=float, default=1.0, dest="cost_bps")
    common.add_argument("--no-optimize", action="store_true")
    common.add_argument("--out", default=None)

    bt = sub.add_parser("backtest", parents=[common], help="Backtest a spec JSON file")
    bt.add_argument("--spec", required=True)
    bt.set_defaults(func=cmd_backtest)

    pa = sub.add_parser("paper", parents=[common], help="Run the pipeline on a PDF")
    pa.add_argument("--pdf", required=True)
    pa.set_defaults(func=cmd_paper)

    pi = sub.add_parser("ping", help="Check Nebius connectivity/budget")
    pi.set_defaults(func=cmd_ping)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
