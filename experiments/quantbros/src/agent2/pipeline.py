"""Deterministic helpers for legacy execution and generated-strategy evaluation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agent2.agents.models import (
    BacktestObservation,
    FinalTearSheet,
    GeneratedStrategyArtifact,
    MetricBundle,
    NarrativeDraft,
    PaperDecision,
    StrategyBlueprint,
    StrategyExecutionResult,
)
from agent2.data import DEFAULT_START, build_data_plan, build_data_plan_from_blueprint, fetch_adj_close
from agent2.engine import run_backtest_from_logic
from agent2.frictions import apply_turnover_costs
from agent2.metrics import build_metric_block
from agent2.plotting import save_equity_curve
from agent2.robustness import evaluate_robustness
from agent2.schemas import Agent1ToAgent2Input, DataPlan, FrictionSpec, LogicSpec, MetricBlock, RobustnessBlock, TearSheet


@dataclass
class RunArtifacts:
    run_id: str
    output_dir: Path
    tear_sheet: TearSheet
    backtest: BacktestObservation


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    clean = returns.fillna(0.0).astype(float)
    return (1.0 + clean).cumprod()


def _write_json(path: Path, payload: dict[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_series(path: str | None, name: str) -> pd.Series | None:
    if not path:
        return None
    series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    series.index = pd.to_datetime(series.index)
    series.name = name
    return series.astype(float)


def _read_weights(path: str) -> pd.DataFrame:
    weights = pd.read_csv(path, index_col=0, parse_dates=True)
    weights.index = pd.to_datetime(weights.index)
    return weights.astype(float)


def _derive_turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.fillna(0.0).diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))


def _derive_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.fillna(0.0).abs().sum(axis=1)


def load_execution_outputs(execution: StrategyExecutionResult) -> dict[str, pd.Series | pd.DataFrame | dict[str, Any] | None]:
    weights = _read_weights(execution.weights_path)
    gross_returns = _read_series(execution.gross_returns_path, "gross_returns")
    net_returns = _read_series(execution.net_returns_path, "net_returns")
    turnover = _read_series(execution.turnover_path, "turnover")
    diagnostics = None
    if execution.diagnostics_path:
        diagnostics = json.loads(Path(execution.diagnostics_path).read_text(encoding="utf-8"))
    return {
        "weights": weights,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "diagnostics": diagnostics,
    }


def build_metric_bundle_from_execution(
    *,
    execution: StrategyExecutionResult,
    output_dir: Path,
    title: str,
    start: date,
    end: date,
) -> MetricBundle:
    outputs = load_execution_outputs(execution)
    weights = outputs["weights"]
    gross_returns = outputs["gross_returns"]
    net_returns = outputs["net_returns"]
    turnover = outputs["turnover"]
    diagnostics = outputs["diagnostics"] or {}
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(gross_returns, pd.Series)

    exposure = _derive_exposure(weights)
    warnings: list[str] = []
    if turnover is None:
        turnover = _derive_turnover(weights)
        warnings.append("turnover_missing_derived_from_weight_changes")
    assert isinstance(turnover, pd.Series)
    if net_returns is None:
        net_returns = gross_returns.copy()
        warnings.append("net_returns_missing_used_gross_returns")
    assert isinstance(net_returns, pd.Series)

    gross_equity = _equity_from_returns(gross_returns)
    net_equity = _equity_from_returns(net_returns)
    gross_metrics = build_metric_block(
        returns=gross_returns,
        equity=gross_equity,
        turnover=turnover,
        exposure=exposure,
    )
    net_metrics = build_metric_block(
        returns=net_returns,
        equity=net_equity,
        turnover=turnover,
        exposure=exposure,
    )
    robustness = build_generic_robustness(
        returns=net_returns,
        equity=net_equity,
        turnover=turnover,
        exposure=exposure,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
    )

    equity_csv = output_dir / "equity_curve.preview.csv"
    equity_frame = pd.DataFrame(
        {
            "gross_equity": gross_equity,
            "net_equity": net_equity,
            "gross_returns": gross_returns,
            "net_returns": net_returns,
            "turnover": turnover,
            "exposure": exposure,
        }
    )
    benchmark_equity = None
    prices_path = execution.artifact_paths.get("prices_input")
    if prices_path and execution.data_plan and execution.data_plan.benchmark:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        benchmark = execution.data_plan.benchmark
        if benchmark in prices.columns:
            benchmark_returns = prices[benchmark].pct_change().fillna(0.0)
            benchmark_equity = _equity_from_returns(benchmark_returns)
            equity_frame["benchmark_equity"] = benchmark_equity
    equity_frame.to_csv(equity_csv)
    png_path = save_equity_curve(
        path=output_dir / "equity_curve.preview.png",
        gross_equity=gross_equity,
        net_equity=net_equity,
        benchmark_equity=benchmark_equity,
        title=title,
    )

    metrics = MetricBundle(
        gross=gross_metrics,
        net=net_metrics,
        robustness=robustness,
        data_plan=execution.data_plan or DataPlan(tickers=[], start=start, end=end),
        friction_note=str(diagnostics.get("friction_handling", "")) if isinstance(diagnostics, dict) else "",
        warnings=warnings,
        artifact_paths={
            "equity_curve_preview_csv": str(equity_csv),
            "equity_curve_preview_png": str(png_path),
        },
        benchmark_used=execution.benchmark_used,
        start=start,
        end=end,
    )
    _write_json(output_dir / "metrics.preview.json", metrics.model_dump(mode="json"))
    _write_json(output_dir / "robustness.preview.json", robustness.model_dump(mode="json"))
    return metrics


def build_generic_robustness(
    *,
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    exposure: pd.Series,
    diagnostics: dict[str, Any] | None = None,
) -> RobustnessBlock:
    diagnostics = diagnostics or {}
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return RobustnessBlock(suspect_flags=["empty_returns"])
    split_idx = max(1, len(clean_returns) // 2)
    split_date = clean_returns.index[split_idx - 1].date()
    train_returns = clean_returns.iloc[:split_idx]
    test_returns = clean_returns.iloc[split_idx:]
    train_equity = _equity_from_returns(train_returns)
    test_equity = _equity_from_returns(test_returns) if not test_returns.empty else pd.Series(dtype=float)
    train_metrics = build_metric_block(
        returns=train_returns,
        equity=train_equity,
        turnover=turnover.reindex(train_returns.index).fillna(0.0),
        exposure=exposure.reindex(train_returns.index).fillna(0.0),
    )
    if test_returns.empty:
        test_metrics = MetricBlock()
    else:
        test_metrics = build_metric_block(
            returns=test_returns,
            equity=test_equity,
            turnover=turnover.reindex(test_returns.index).fillna(0.0),
            exposure=exposure.reindex(test_returns.index).fillna(0.0),
        )
    suspect_flags: list[str] = []
    if (train_metrics.sharpe or 0.0) > 5.0 or (test_metrics.sharpe or 0.0) > 5.0:
        suspect_flags.append("absurd_sharpe")
    if (train_metrics.total_return or 0.0) > 10.0:
        suspect_flags.append("extreme_train_total_return")
    if float(turnover.fillna(0.0).mean()) > 2.0:
        suspect_flags.append("high_turnover")
    if float(exposure.fillna(0.0).mean()) > 1.5:
        suspect_flags.append("high_exposure")
    if isinstance(diagnostics.get("suspect_flags"), list):
        suspect_flags.extend(str(item) for item in diagnostics["suspect_flags"])
    return RobustnessBlock(
        train_sharpe=train_metrics.sharpe,
        test_sharpe=test_metrics.sharpe,
        train_cagr=train_metrics.cagr,
        test_cagr=test_metrics.cagr,
        split_date=split_date,
        walk_forward_windows=1,
        parameter_sensitivity={},
        suspect_flags=list(dict.fromkeys(suspect_flags)),
    )


def build_strategy_cfg(
    *,
    blueprint: StrategyBlueprint,
    data_plan: DataPlan,
    commission_bps: float,
    slippage_bps: float,
    start: date,
    end: date,
) -> dict[str, Any]:
    return {
        "paper_id": blueprint.paper_id,
        "title": blueprint.title,
        "benchmark": blueprint.benchmark or blueprint.universe.benchmark,
        "required_parameters": blueprint.required_parameters,
        "rebalance_schedule": blueprint.rebalance_schedule,
        "holding_period": blueprint.holding_period,
        "risk_controls": blueprint.risk_controls,
        "leverage_rules": blueprint.leverage_rules,
        "timing_and_delay_rules": blueprint.timing_and_delay_rules,
        "implementation_notes": blueprint.implementation_notes,
        "frictions": {
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
        },
        "data_plan": data_plan.model_dump(mode="json"),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
    }


def publish_strategy_outputs(
    *,
    agent1: Agent1ToAgent2Input,
    run_id: str,
    output_dir: Path,
    blueprint: StrategyBlueprint,
    strategy_artifact: GeneratedStrategyArtifact,
    execution: StrategyExecutionResult,
    metrics: MetricBundle,
    narrative: NarrativeDraft,
) -> FinalTearSheet:
    canonical_paths = {
        "strategy_blueprint": output_dir / "strategy_blueprint.json",
        "strategy_code": output_dir / "strategy.py",
        "tear_sheet": output_dir / "tear_sheet.json",
        "equity_curve_csv": output_dir / "equity_curve.csv",
        "equity_curve_png": output_dir / "equity_curve.png",
    }
    _write_json(canonical_paths["strategy_blueprint"], blueprint.model_dump(mode="json"))
    shutil.copyfile(strategy_artifact.path, canonical_paths["strategy_code"])
    if "equity_curve_preview_csv" in metrics.artifact_paths:
        shutil.copyfile(metrics.artifact_paths["equity_curve_preview_csv"], canonical_paths["equity_curve_csv"])
    if "equity_curve_preview_png" in metrics.artifact_paths:
        shutil.copyfile(metrics.artifact_paths["equity_curve_preview_png"], canonical_paths["equity_curve_png"])

    tear_sheet = FinalTearSheet(
        run_id=run_id,
        paper_id=agent1.paper_id,
        title=agent1.title,
        paper_category=agent1.paper_category,
        asset_universe_type=agent1.asset_universe_type,
        asset_universe_notes=agent1.asset_universe_notes,
        regime_claim=agent1.regime_claim,
        blueprint=blueprint,
        strategy_artifact=strategy_artifact,
        execution=execution,
        metrics=metrics,
        narrative=narrative,
        artifacts={},
    )
    artifacts = {
        **execution.artifact_paths,
        **metrics.artifact_paths,
        "strategy_blueprint": str(canonical_paths["strategy_blueprint"]),
        "strategy_code": str(canonical_paths["strategy_code"]),
        "tear_sheet": str(canonical_paths["tear_sheet"]),
        "equity_curve_csv": str(canonical_paths["equity_curve_csv"]),
        "equity_curve_png": str(canonical_paths["equity_curve_png"]),
    }
    tear_sheet = tear_sheet.model_copy(update={"artifacts": artifacts})
    _write_json(canonical_paths["tear_sheet"], tear_sheet.model_dump(mode="json"))
    return tear_sheet


def prepare_blueprint_prices(
    *,
    agent1: Agent1ToAgent2Input,
    blueprint: StrategyBlueprint,
    start: date | None,
    end: date | None,
) -> tuple[DataPlan, pd.DataFrame]:
    start = start or DEFAULT_START
    end = end or date.today()
    data_plan = build_data_plan_from_blueprint(
        agent1=agent1,
        blueprint=blueprint,
        start=start,
        end=end,
    )
    prices = fetch_adj_close(data_plan.tickers, start=start, end=end)
    return data_plan, prices


def execute_logic_backtest(
    *,
    agent1: Agent1ToAgent2Input,
    classification: PaperDecision,
    logic: LogicSpec,
    start: date | None = None,
    end: date | None = None,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    artifacts_dir: str | Path = "artifacts",
    run_id: str | None = None,
    analysis_summary: str = "",
    analysis_risks: list[str] | None = None,
    finalize_outputs: bool = True,
) -> RunArtifacts:
    start = start or DEFAULT_START
    end = end or date.today()
    run_id = run_id or f"{agent1.paper_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output_dir = Path(artifacts_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    data_plan = build_data_plan(
        agent1=agent1,
        logic=logic,
        start=start,
        end=end,
        paper_category=classification.paper_category.value,
        asset_universe_notes=classification.asset_universe_notes,
        regime_claim=classification.regime_claim.value,
    )
    tickers = list(dict.fromkeys(logic.universe.tickers + ([logic.universe.benchmark] if logic.universe.benchmark else [])))
    prices = fetch_adj_close(tickers, start=start, end=end)
    result = run_backtest_from_logic(logic, prices)

    frictions = FrictionSpec(commission_bps=commission_bps, slippage_bps=slippage_bps)
    net_returns = apply_turnover_costs(result.gross_returns, result.turnover, frictions)
    net_equity = _equity_from_returns(net_returns)
    gross_metrics = build_metric_block(
        returns=result.gross_returns,
        equity=result.equity,
        turnover=result.turnover,
        exposure=result.exposure,
    )
    net_metrics = build_metric_block(
        returns=net_returns,
        equity=net_equity,
        turnover=result.turnover,
        exposure=result.exposure,
    )
    robustness = evaluate_robustness(
        prices=prices,
        logic=logic,
        frictions=frictions,
        net_returns=net_returns,
        net_equity=net_equity,
        exposure=result.exposure,
    )

    logic_name = "logic.json" if finalize_outputs else "logic.preview.json"
    data_plan_name = "data_plan.json" if finalize_outputs else "data_plan.preview.json"
    tear_sheet_name = "tear_sheet.json" if finalize_outputs else "tear_sheet.preview.json"
    equity_csv_name = "equity_curve.csv" if finalize_outputs else "equity_curve.preview.csv"
    equity_png_name = "equity_curve.png" if finalize_outputs else "equity_curve.preview.png"

    equity_csv = output_dir / equity_csv_name
    equity_frame = pd.DataFrame(
        {
            "gross_equity": result.equity,
            "net_equity": net_equity,
            "gross_returns": result.gross_returns,
            "net_returns": net_returns,
            "turnover": result.turnover,
            "exposure": result.exposure,
        }
    )
    if result.benchmark_equity is not None:
        equity_frame["benchmark_equity"] = result.benchmark_equity
    equity_frame.to_csv(equity_csv)

    png_path = save_equity_curve(
        path=output_dir / equity_png_name,
        gross_equity=result.equity,
        net_equity=net_equity,
        benchmark_equity=result.benchmark_equity,
        title=f"{agent1.paper_id}: {logic.strategy.value}",
    )

    artifact_paths = {
        "logic": str(output_dir / logic_name),
        "data_plan": str(output_dir / data_plan_name),
        "tear_sheet": str(output_dir / tear_sheet_name),
        "equity_curve_png": str(png_path),
        "equity_curve_csv": str(equity_csv),
    }

    backtest_observation = BacktestObservation(
        data_plan=data_plan,
        frictions=frictions,
        gross=gross_metrics,
        net=net_metrics,
        robustness=robustness,
        artifact_paths=artifact_paths,
        benchmark_used=logic.universe.benchmark or "",
        start=start,
        end=end,
    )

    tear_sheet = TearSheet(
        run_id=run_id,
        paper_id=agent1.paper_id,
        title=agent1.title,
        paper_category=classification.paper_category,
        asset_universe_type=classification.asset_universe_type,
        asset_universe_notes=classification.asset_universe_notes,
        regime_claim=classification.regime_claim,
        logic=logic,
        data_plan=data_plan,
        frictions=frictions,
        gross=gross_metrics,
        net=net_metrics,
        robustness=robustness,
        artifacts=artifact_paths,
        analysis_summary=analysis_summary,
        analysis_risks=analysis_risks or [],
    )

    _write_json(output_dir / logic_name, logic.model_dump(mode="json"))
    _write_json(output_dir / data_plan_name, data_plan.model_dump(mode="json"))
    _write_json(output_dir / tear_sheet_name, tear_sheet.model_dump(mode="json"))

    return RunArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        tear_sheet=tear_sheet,
        backtest=backtest_observation,
    )
