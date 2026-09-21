"""Expandable template registry.

The registry is the bridge between *metadata* (what strategy families exist, what
they mean, what parameters they take — stored in ``templates_registry.json`` and
freely expandable) and *execution* (the vetted Python in ``templates.py``).

Design rule that keeps us out of the old "LLM writes runtime code" trap:

* Metadata can grow freely (a router agent or human can append entries).
* Execution code only runs if a registered template id maps to a real, tested
  Python function. An entry with ``implemented: false`` (or with no matching
  function) is catalogued and discoverable, but cannot be backtested until a
  human implements + tests it.

So the catalog is what the router/LLM sees; the executable set is the subset that
is actually safe to run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from marginalia.engine.spec import StrategyTemplate
from marginalia.engine.templates import TEMPLATE_FUNCS

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY_PATH = os.path.join(_THIS_DIR, "templates_registry.json")


@dataclass(frozen=True)
class TemplateInfo:
    id: str
    category: str
    display_name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    params: Dict[str, dict] = field(default_factory=dict)
    example_papers: List[str] = field(default_factory=list)
    declared_implemented: bool = True

    @property
    def has_execution(self) -> bool:
        """True only if this id maps to a real Python execution function."""
        try:
            tmpl = StrategyTemplate(self.id)
        except ValueError:
            return False
        return tmpl in TEMPLATE_FUNCS

    @property
    def is_runnable(self) -> bool:
        return self.declared_implemented and self.has_execution

    def as_catalog_entry(self) -> dict:
        """Compact dict shown to the router LLM (no internal flags)."""
        return {
            "id": self.id,
            "category": self.category,
            "display_name": self.display_name,
            "description": self.description,
            "keywords": list(self.keywords),
            "params": {k: v.get("description", "") for k, v in self.params.items()},
        }


class TemplateRegistry:
    """Loads template metadata and exposes catalog / runnable views."""

    def __init__(self, templates: List[TemplateInfo], version: int = 1):
        self._templates: Dict[str, TemplateInfo] = {t.id: t for t in templates}
        self.version = version

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, path: str = DEFAULT_REGISTRY_PATH) -> "TemplateRegistry":
        with open(path) as f:
            data = json.load(f)
        templates = [
            TemplateInfo(
                id=str(entry["id"]),
                category=str(entry.get("category", "other")),
                display_name=str(entry.get("display_name", entry["id"])),
                description=str(entry.get("description", "")),
                keywords=list(entry.get("keywords", [])),
                params=dict(entry.get("params", {})),
                example_papers=list(entry.get("example_papers", [])),
                declared_implemented=bool(entry.get("implemented", True)),
            )
            for entry in data.get("templates", [])
        ]
        return cls(templates, version=int(data.get("version", 1)))

    # -- queries -------------------------------------------------------------

    def get(self, template_id: str) -> Optional[TemplateInfo]:
        return self._templates.get(template_id)

    def all(self) -> List[TemplateInfo]:
        return list(self._templates.values())

    def ids(self) -> List[str]:
        return list(self._templates.keys())

    def categories(self) -> List[str]:
        return sorted({t.category for t in self._templates.values()})

    def runnable(self) -> List[TemplateInfo]:
        return [t for t in self._templates.values() if t.is_runnable]

    def runnable_ids(self) -> List[str]:
        return [t.id for t in self.runnable()]

    def is_runnable(self, template_id: str) -> bool:
        t = self.get(template_id)
        return bool(t and t.is_runnable)

    def catalog(self, runnable_only: bool = True) -> List[dict]:
        """Catalog entries for the router LLM."""
        items = self.runnable() if runnable_only else self.all()
        return [t.as_catalog_entry() for t in items]

    def catalog_text(self, runnable_only: bool = True) -> str:
        """Human/LLM-readable catalog block for prompts."""
        lines = []
        for t in (self.runnable() if runnable_only else self.all()):
            param_str = ", ".join(t.params.keys()) if t.params else "none"
            lines.append(
                f"- {t.id} [{t.category}] — {t.display_name}: {t.description} "
                f"(params: {param_str})"
            )
        return "\n".join(lines)

    def consistency_report(self) -> Dict[str, List[str]]:
        """Diagnose mismatches between metadata and execution code."""
        declared_but_no_code = [
            t.id for t in self._templates.values()
            if t.declared_implemented and not t.has_execution
        ]
        code_but_not_registered = [
            tmpl.value for tmpl in TEMPLATE_FUNCS
            if tmpl.value not in self._templates
        ]
        return {
            "declared_implemented_but_missing_code": declared_but_no_code,
            "has_code_but_not_in_registry": code_but_not_registered,
        }


# Module-level singleton (lazy) so callers share one parsed registry.
_default_registry: Optional[TemplateRegistry] = None


def get_registry(path: str = DEFAULT_REGISTRY_PATH, *, reload: bool = False) -> TemplateRegistry:
    global _default_registry
    if _default_registry is None or reload:
        _default_registry = TemplateRegistry.load(path)
    return _default_registry
