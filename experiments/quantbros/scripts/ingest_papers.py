"""Ingest local papers into the synthetic SQLite database and JSON fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from agent2.paper_ingest import ingest_papers
from agent2.settings import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest Papers/ into data/papers.sqlite")
    parser.add_argument("--papers-dir", default=settings.papers_dir)
    parser.add_argument("--db-path", default=settings.database_path)
    parser.add_argument("--fixtures-dir", default=settings.fixtures_dir)
    parser.add_argument(
        "--legacy-heuristics",
        action="store_true",
        help="Deprecated: use heuristic paper classification instead of agentic LLM classification.",
    )
    args = parser.parse_args()

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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
