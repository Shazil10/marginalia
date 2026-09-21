"""Artifact helpers for the agentic orchestration loop."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent2.agents.models import FailureArtifact


class ArtifactManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "trace.jsonl"
        self._versions: dict[str, int] = defaultdict(int)

    def versioned_path(self, stem: str, suffix: str = ".json") -> Path:
        self._versions[stem] += 1
        return self.output_dir / f"{stem}_v{self._versions[stem]}{suffix}"

    def write_model(self, stem: str, payload: BaseModel, *, versioned: bool = True) -> Path:
        path = self.versioned_path(stem) if versioned else self.output_dir / f"{stem}.json"
        path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def write_json(self, stem: str, payload: dict[str, Any], *, versioned: bool = True) -> Path:
        path = self.versioned_path(stem) if versioned else self.output_dir / f"{stem}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def append_trace(self, entry: BaseModel | dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            if isinstance(entry, BaseModel):
                handle.write(entry.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(entry) + "\n")

    def write_failure(self, failure: FailureArtifact) -> Path:
        path = self.output_dir / "failure.json"
        path.write_text(failure.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path
