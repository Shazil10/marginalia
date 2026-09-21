"""Load Agent 1 payloads from JSON fixtures, env vars, or the local SQLite database."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent2.database import connect_sqlite, load_agent1_input
from agent2.schemas import Agent1ToAgent2Input, SCHEMA_VERSION


class Agent1LoadError(ValueError):
    pass


def validate_agent1_dict(data: dict[str, Any]) -> Agent1ToAgent2Input:
    model = Agent1ToAgent2Input.model_validate(data)
    current_major = SCHEMA_VERSION.split(".")[0]
    incoming_major = model.schema_version.split(".")[0]
    if incoming_major != current_major:
        raise Agent1LoadError(
            f"Incompatible schema_version {model.schema_version!r}; expected major {current_major}.x"
        )
    return model


def load_agent1_json(path: str | Path) -> Agent1ToAgent2Input:
    fixture = Path(path)
    if not fixture.is_file():
        raise Agent1LoadError(f"Not a file: {fixture}")
    data = json.loads(fixture.read_text(encoding="utf-8"))
    return validate_agent1_dict(data)


def load_agent1_from_db(db_path: str | Path, paper_id: str) -> Agent1ToAgent2Input:
    conn = connect_sqlite(db_path)
    try:
        return load_agent1_input(conn, paper_id)
    except KeyError as exc:
        raise Agent1LoadError(str(exc)) from exc
    finally:
        conn.close()


def load_agent1_from_env() -> Agent1ToAgent2Input:
    raw = os.environ.get("AGENT1_JSON")
    if raw:
        return validate_agent1_dict(json.loads(raw))
    fixture_path = os.environ.get("AGENT1_FIXTURE_PATH")
    if fixture_path:
        return load_agent1_json(fixture_path)
    raise Agent1LoadError("Set AGENT1_JSON or AGENT1_FIXTURE_PATH")
