"""
generate_reports.py
--------------------
Retroactively generate strategy_profile.json, report.html, and (if daily_returns.csv
exists) quantstats_tearsheet.html for every strategy folder that already has
extracted_strategy.json + backtest_results.json.

Usage:
    python agents/strategy_extraction/generate_reports.py
"""

import os
import sys
import json

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import generate_tearsheet

OUTPUTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def find_strategy_dirs(base):
    """Walk the outputs tree and collect dirs with both required JSON files."""
    dirs = []
    for root, _, files in os.walk(base):
        if "extracted_strategy.json" in files and "backtest_results.json" in files:
            dirs.append(root)
    return sorted(dirs)


def main():
    if not os.path.isdir(OUTPUTS_BASE):
        print(f"Outputs directory not found: {OUTPUTS_BASE}")
        sys.exit(1)

    strategy_dirs = find_strategy_dirs(OUTPUTS_BASE)
    if not strategy_dirs:
        print("No strategy directories found.")
        sys.exit(0)

    print(f"Found {len(strategy_dirs)} strategy folder(s).\n")

    success = 0
    for d in strategy_dirs:
        strategy_name = os.path.basename(d)
        print(f"{'=' * 60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'=' * 60}")

        try:
            with open(os.path.join(d, "extracted_strategy.json")) as f:
                strategy_json = json.load(f)
        except Exception as e:
            print(f"  Could not load extracted_strategy.json: {e}\n")
            continue

        try:
            with open(os.path.join(d, "backtest_results.json")) as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"  Could not load backtest_results.json: {e}\n")
            continue

        try:
            generate_tearsheet(d, strategy_json, metrics)
            success += 1
        except Exception as e:
            print(f"  Report generation failed: {e}")

        print()

    print(f"Done. Reports generated for {success}/{len(strategy_dirs)} strategies.")
    print(f"\nOpen any strategy's report.html in a browser to view its profile.")
    print(f"If daily_returns.csv is missing for a strategy, re-run the full pipeline")
    print(f"(python agents/strategy_extraction/mvp.py) to get the QuantStats tear sheet too.")


if __name__ == "__main__":
    main()
