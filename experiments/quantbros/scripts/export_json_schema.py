"""Emit JSON Schema for Agent1ToAgent2Input to schemas/agent1_to_agent2.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from agent2.schemas import Agent1ToAgent2Input

OUT = ROOT / "schemas" / "agent1_to_agent2.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    schema = Agent1ToAgent2Input.model_json_schema()
    OUT.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
