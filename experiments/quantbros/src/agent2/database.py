"""SQLite persistence for ingested papers and Agent 1 style fixtures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from agent2.schemas import (
    Agent1Sections,
    Agent1ToAgent2Input,
    AssetUniverseType,
    PaperCategory,
    RegimeClaim,
)


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            paper_id TEXT UNIQUE NOT NULL,
            schema_version TEXT NOT NULL,
            title TEXT,
            source_path TEXT,
            created_at TEXT,
            content_hash TEXT,
            raw_text TEXT,
            paper_category TEXT,
            asset_universe_type TEXT,
            asset_universe_notes TEXT,
            regime_claim TEXT,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_sections (
            paper_id TEXT PRIMARY KEY,
            abstract TEXT,
            methodology TEXT,
            introduction TEXT,
            conclusion TEXT,
            notes TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS paper_extractions (
            id INTEGER PRIMARY KEY,
            paper_id TEXT NOT NULL,
            extraction_type TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            model TEXT,
            confidence REAL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_papers_paper_id ON papers(paper_id);
        CREATE INDEX IF NOT EXISTS idx_papers_content_hash ON papers(content_hash);
        CREATE INDEX IF NOT EXISTS idx_papers_source_path ON papers(source_path);
        CREATE INDEX IF NOT EXISTS idx_paper_extractions_paper_id ON paper_extractions(paper_id);
        """
    )
    conn.commit()


def upsert_paper(
    conn: sqlite3.Connection,
    *,
    paper_id: str,
    schema_version: str,
    title: str,
    source_path: str,
    created_at: str,
    content_hash: str,
    raw_text: str,
    paper_category: PaperCategory,
    asset_universe_type: AssetUniverseType,
    asset_universe_notes: str,
    regime_claim: RegimeClaim,
    metadata: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO papers (
            paper_id, schema_version, title, source_path, created_at, content_hash, raw_text,
            paper_category, asset_universe_type, asset_universe_notes, regime_claim, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            schema_version=excluded.schema_version,
            title=excluded.title,
            source_path=excluded.source_path,
            created_at=excluded.created_at,
            content_hash=excluded.content_hash,
            raw_text=excluded.raw_text,
            paper_category=excluded.paper_category,
            asset_universe_type=excluded.asset_universe_type,
            asset_universe_notes=excluded.asset_universe_notes,
            regime_claim=excluded.regime_claim,
            metadata_json=excluded.metadata_json
        """,
        (
            paper_id,
            schema_version,
            title,
            source_path,
            created_at,
            content_hash,
            raw_text,
            paper_category.value,
            asset_universe_type.value,
            asset_universe_notes,
            regime_claim.value,
            json.dumps(metadata, sort_keys=True),
        ),
    )
    conn.commit()


def upsert_sections(conn: sqlite3.Connection, paper_id: str, sections: Agent1Sections) -> None:
    conn.execute(
        """
        INSERT INTO paper_sections (paper_id, abstract, methodology, introduction, conclusion, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            abstract=excluded.abstract,
            methodology=excluded.methodology,
            introduction=excluded.introduction,
            conclusion=excluded.conclusion,
            notes=excluded.notes
        """,
        (
            paper_id,
            sections.abstract,
            sections.methodology,
            sections.introduction,
            sections.conclusion,
            sections.notes,
        ),
    )
    conn.commit()


def insert_extraction(
    conn: sqlite3.Connection,
    *,
    paper_id: str,
    extraction_type: str,
    extracted_at: str,
    model: str | None,
    confidence: float | None,
    payload: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO paper_extractions (paper_id, extraction_type, extracted_at, model, confidence, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            extraction_type,
            extracted_at,
            model,
            confidence,
            json.dumps(payload, sort_keys=True),
        ),
    )
    conn.commit()


def load_agent1_input(conn: sqlite3.Connection, paper_id: str) -> Agent1ToAgent2Input:
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown paper_id: {paper_id}")
    sections_row = conn.execute(
        "SELECT * FROM paper_sections WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    sections = Agent1Sections(
        abstract=sections_row["abstract"] if sections_row else "",
        methodology=sections_row["methodology"] if sections_row else "",
        introduction=sections_row["introduction"] if sections_row else "",
        conclusion=sections_row["conclusion"] if sections_row else "",
        notes=sections_row["notes"] if sections_row else "",
    )
    return Agent1ToAgent2Input(
        schema_version=row["schema_version"],
        paper_id=row["paper_id"],
        title=row["title"] or "",
        source_path=row["source_path"],
        parsed_at=metadata.get("parsed_at"),
        sections=sections,
        extracted_tickers=metadata.get("extracted_tickers", []),
        paper_category=row["paper_category"] or PaperCategory.unknown,
        asset_universe_type=row["asset_universe_type"] or AssetUniverseType.unknown,
        asset_universe_notes=row["asset_universe_notes"] or "",
        regime_claim=row["regime_claim"] or RegimeClaim.unknown,
        raw_metadata=metadata,
    )


def iter_agent1_inputs(conn: sqlite3.Connection) -> Iterable[Agent1ToAgent2Input]:
    rows = conn.execute("SELECT paper_id FROM papers ORDER BY paper_id").fetchall()
    for row in rows:
        yield load_agent1_input(conn, row["paper_id"])
