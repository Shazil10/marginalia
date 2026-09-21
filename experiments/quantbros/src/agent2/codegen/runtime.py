"""Sandboxed execution helpers for LLM-generated strategy modules."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from agent2.agents.models import StaticScanResult, StrategyExecutionResult, StructuredFinding
from agent2.schemas import DataPlan


ALLOWLISTED_IMPORTS = {"math", "numpy", "pandas", "statistics"}
DENIED_NAMES = {
    "eval",
    "exec",
    "open",
    "__import__",
    "compile",
    "input",
}
DENIED_IMPORT_PREFIXES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "pathlib",
    "importlib",
    "shutil",
}
RUNNER_SCRIPT = """
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _as_series(obj, name):
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].rename(name)
        raise ValueError(f"{name} must be a Series or single-column DataFrame")
    if isinstance(obj, pd.Series):
        return obj.rename(name)
    raise TypeError(f"{name} must be a pandas Series or DataFrame")


def _as_weights(obj):
    if isinstance(obj, pd.Series):
        return obj.to_frame(name="weight")
    if isinstance(obj, pd.DataFrame):
        return obj
    raise TypeError("weights must be a pandas Series or DataFrame")


def main():
    strategy_path = Path(sys.argv[1])
    prices_path = Path(sys.argv[2])
    cfg_path = Path(sys.argv[3])
    output_dir = Path(sys.argv[4])
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    spec = importlib.util.spec_from_file_location("generated_strategy", strategy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load generated strategy module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_strategy"):
        raise AttributeError("Generated strategy module does not define run_strategy")

    raw = module.run_strategy(prices, cfg)
    if not isinstance(raw, dict):
        raise TypeError("run_strategy must return a dict")
    if "weights" not in raw or "gross_returns" not in raw:
        raise KeyError("run_strategy output must include weights and gross_returns")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights = _as_weights(raw["weights"])
    gross_returns = _as_series(raw["gross_returns"], "gross_returns")
    weights_path = output_dir / "weights.preview.csv"
    gross_path = output_dir / "gross_returns.preview.csv"
    weights.to_csv(weights_path)
    gross_returns.to_csv(gross_path, header=True)

    result = {
        "weights_path": str(weights_path),
        "gross_returns_path": str(gross_path),
        "net_returns_path": None,
        "turnover_path": None,
        "diagnostics_path": None,
    }

    if "net_returns" in raw and raw["net_returns"] is not None:
        net_returns = _as_series(raw["net_returns"], "net_returns")
        net_path = output_dir / "net_returns.preview.csv"
        net_returns.to_csv(net_path, header=True)
        result["net_returns_path"] = str(net_path)

    if "turnover" in raw and raw["turnover"] is not None:
        turnover = _as_series(raw["turnover"], "turnover")
        turnover_path = output_dir / "turnover.preview.csv"
        turnover.to_csv(turnover_path, header=True)
        result["turnover_path"] = str(turnover_path)

    if "diagnostics" in raw and raw["diagnostics"] is not None:
        diagnostics_path = output_dir / "diagnostics.preview.json"
        diagnostics_path.write_text(json.dumps(raw["diagnostics"], indent=2) + "\\n", encoding="utf-8")
        result["diagnostics_path"] = str(diagnostics_path)

    summary_path = output_dir / "sandbox_result.preview.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps({"status": "ok", **result}))


if __name__ == "__main__":
    main()
"""


def static_scan_strategy_code(code: str) -> StaticScanResult:
    findings: list[StructuredFinding] = []
    allowlisted: list[str] = []
    denied: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        findings.append(
            StructuredFinding(
                code="syntax_error",
                severity="error",
                message=str(exc),
                location=f"line {exc.lineno}" if exc.lineno else "",
                requires_repair=True,
            )
        )
        return StaticScanResult(
            passed=False,
            findings=findings,
            summary="Static scan failed due to syntax error.",
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in ALLOWLISTED_IMPORTS:
                    allowlisted.append(alias.name)
                else:
                    denied.append(alias.name)
                    findings.append(
                        StructuredFinding(
                            code="restricted_import",
                            severity="error",
                            message=f"Import {alias.name!r} is not allowlisted.",
                            location=f"line {getattr(node, 'lineno', 0)}",
                            requires_repair=True,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in ALLOWLISTED_IMPORTS:
                allowlisted.append(node.module or "")
            else:
                denied.append(node.module or "")
                findings.append(
                    StructuredFinding(
                        code="restricted_import",
                        severity="error",
                        message=f"Import from {node.module!r} is not allowlisted.",
                        location=f"line {getattr(node, 'lineno', 0)}",
                        requires_repair=True,
                    )
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DENIED_NAMES:
            findings.append(
                StructuredFinding(
                    code="restricted_builtin",
                    severity="error",
                    message=f"Call to {node.func.id!r} is not allowed.",
                    location=f"line {getattr(node, 'lineno', 0)}",
                    requires_repair=True,
                )
            )
        elif isinstance(node, ast.Name) and node.id in DENIED_IMPORT_PREFIXES:
            findings.append(
                StructuredFinding(
                    code="restricted_name",
                    severity="error",
                    message=f"Reference to restricted module {node.id!r}.",
                    location=f"line {getattr(node, 'lineno', 0)}",
                    requires_repair=True,
                )
            )
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in DENIED_IMPORT_PREFIXES:
            findings.append(
                StructuredFinding(
                    code="restricted_attribute",
                    severity="error",
                    message=f"Restricted attribute access via {node.value.id!r}.",
                    location=f"line {getattr(node, 'lineno', 0)}",
                    requires_repair=True,
                )
            )

    passed = not any(item.severity == "error" for item in findings)
    summary = "Static scan passed." if passed else "Static scan found restricted code patterns."
    return StaticScanResult(
        passed=passed,
        findings=findings,
        allowlisted_imports=sorted(set(allowlisted)),
        denied_imports=sorted(set(denied)),
        summary=summary,
    )


def _load_series(path: str | None, name: str) -> pd.Series | None:
    if not path:
        return None
    series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    series.name = name
    return series


def _load_weights(path: str) -> pd.DataFrame:
    weights = pd.read_csv(path, index_col=0, parse_dates=True)
    weights.index = pd.to_datetime(weights.index)
    return weights


def _validate_outputs(
    *,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    gross_returns: pd.Series,
    net_returns: pd.Series | None,
    turnover: pd.Series | None,
) -> list[StructuredFinding]:
    findings: list[StructuredFinding] = []
    if not prices.index.is_monotonic_increasing:
        findings.append(
            StructuredFinding(
                code="prices_index_invalid",
                severity="error",
                message="Input prices index is not monotonic increasing.",
                requires_repair=True,
            )
        )
    if not gross_returns.index.equals(prices.index):
        findings.append(
            StructuredFinding(
                code="gross_returns_alignment",
                severity="error",
                message="gross_returns index does not align to input prices index.",
                requires_repair=True,
            )
        )
    if not weights.index.equals(prices.index):
        findings.append(
            StructuredFinding(
                code="weights_alignment",
                severity="error",
                message="weights index does not align to input prices index.",
                requires_repair=True,
            )
        )
    if gross_returns.dropna().empty:
        findings.append(
            StructuredFinding(
                code="gross_returns_empty",
                severity="error",
                message="gross_returns is all NaN or empty.",
                requires_repair=True,
            )
        )
    if not gross_returns.dropna().empty and not pd.api.types.is_numeric_dtype(gross_returns):
        findings.append(
            StructuredFinding(
                code="gross_returns_dtype",
                severity="error",
                message="gross_returns must be numeric.",
                requires_repair=True,
            )
        )
    if gross_returns.dropna().size and not pd.Series(pd.to_numeric(gross_returns.dropna(), errors="coerce")).notna().all():
        findings.append(
            StructuredFinding(
                code="gross_returns_non_numeric",
                severity="error",
                message="gross_returns contains non-numeric values.",
                requires_repair=True,
            )
        )
    if not gross_returns.dropna().empty and not pd.Series(gross_returns.dropna()).map(lambda x: abs(float(x)) < 10_000).all():
        findings.append(
            StructuredFinding(
                code="gross_returns_absurd",
                severity="error",
                message="gross_returns contains absurd values.",
                requires_repair=True,
            )
        )
    if net_returns is not None and not net_returns.index.equals(prices.index):
        findings.append(
            StructuredFinding(
                code="net_returns_alignment",
                severity="error",
                message="net_returns index does not align to input prices index.",
                requires_repair=True,
            )
        )
    if turnover is not None and not turnover.index.equals(prices.index):
        findings.append(
            StructuredFinding(
                code="turnover_alignment",
                severity="error",
                message="turnover index does not align to input prices index.",
                requires_repair=True,
            )
        )
    return findings


def execute_strategy_code(
    *,
    code_path: Path,
    prices: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    data_plan: DataPlan,
    timeout_seconds: int = 20,
) -> StrategyExecutionResult:
    output_dir = output_dir.resolve()
    code_path = code_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prices_path = output_dir / "prices_input.preview.csv"
    cfg_path = output_dir / "strategy_config.preview.json"
    runner_path = output_dir / "_sandbox_runner.py"
    prices.to_csv(prices_path)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    runner_path.write_text(RUNNER_SCRIPT, encoding="utf-8")

    env = {"PYTHONIOENCODING": "utf-8"}

    completed = subprocess.run(
        [sys.executable, "-I", str(runner_path), str(code_path), str(prices_path), str(cfg_path), str(output_dir)],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        env=env,
    )

    result = StrategyExecutionResult(
        code_path=str(code_path),
        weights_path=str(output_dir / "weights.preview.csv"),
        gross_returns_path=str(output_dir / "gross_returns.preview.csv"),
        net_returns_path=str(output_dir / "net_returns.preview.csv") if (output_dir / "net_returns.preview.csv").exists() else None,
        turnover_path=str(output_dir / "turnover.preview.csv") if (output_dir / "turnover.preview.csv").exists() else None,
        diagnostics_path=str(output_dir / "diagnostics.preview.json") if (output_dir / "diagnostics.preview.json").exists() else None,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        passed=False,
        artifact_paths={
            "prices_input": str(prices_path),
            "strategy_config": str(cfg_path),
            "sandbox_runner": str(runner_path),
        },
        data_plan=data_plan,
        benchmark_used=data_plan.benchmark or "",
    )

    if completed.returncode != 0:
        result.sanity_checks = [
            StructuredFinding(
                code="sandbox_runtime_error",
                severity="error",
                message=f"Sandbox process exited with code {completed.returncode}.",
                requires_repair=True,
            )
        ]
        return result

    try:
        weights = _load_weights(result.weights_path)
        gross_returns = _load_series(result.gross_returns_path, "gross_returns")
        net_returns = _load_series(result.net_returns_path, "net_returns")
        turnover = _load_series(result.turnover_path, "turnover")
        if gross_returns is None:
            raise ValueError("gross_returns output missing after sandbox success")
    except Exception as exc:
        result.sanity_checks = [
            StructuredFinding(
                code="sandbox_output_load_error",
                severity="error",
                message=str(exc),
                requires_repair=True,
            )
        ]
        return result

    result.sanity_checks = _validate_outputs(
        prices=prices,
        weights=weights,
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
    )
    result.artifact_paths.update(
        {
            "weights_preview": result.weights_path,
            "gross_returns_preview": result.gross_returns_path,
        }
    )
    if result.net_returns_path:
        result.artifact_paths["net_returns_preview"] = result.net_returns_path
    if result.turnover_path:
        result.artifact_paths["turnover_preview"] = result.turnover_path
    if result.diagnostics_path:
        result.artifact_paths["diagnostics_preview"] = result.diagnostics_path
    result.passed = not any(item.severity == "error" for item in result.sanity_checks)
    return result
