from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent2.codegen.runtime import execute_strategy_code, static_scan_strategy_code
from agent2.schemas import DataPlan


def test_static_scan_rejects_banned_imports() -> None:
    code = "import os\nimport pandas as pd\n\ndef run_strategy(prices, cfg):\n    return {'weights': prices[['SPY']]*0.0, 'gross_returns': prices['SPY']*0.0}\n"
    scan = static_scan_strategy_code(code)
    assert not scan.passed
    assert any(item.code == "restricted_import" for item in scan.findings)


def test_execute_strategy_code_captures_syntax_error(tmp_path: Path) -> None:
    code_path = tmp_path / "strategy_bad.py"
    code_path.write_text("import pandas as pd\n\ndef run_strategy(prices, cfg):\n    return {'weights': prices[['SPY']], 'gross_returns': }\n", encoding="utf-8")
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=pd.date_range("2024-01-01", periods=3, freq="B"))
    data_plan = DataPlan(tickers=["SPY"], benchmark="SPY", start=prices.index[0].date(), end=prices.index[-1].date())
    result = execute_strategy_code(
        code_path=code_path,
        prices=prices,
        cfg={"foo": "bar"},
        output_dir=tmp_path,
        data_plan=data_plan,
        timeout_seconds=10,
    )
    assert not result.passed
    assert result.exit_code != 0
    assert any(item.code == "sandbox_runtime_error" for item in result.sanity_checks)


def test_execute_strategy_code_flags_bad_output_shape(tmp_path: Path) -> None:
    code_path = tmp_path / "strategy_bad_shape.py"
    code_path.write_text(
        "\n".join(
            [
                "import pandas as pd",
                "",
                "def run_strategy(prices: pd.DataFrame, cfg: dict) -> dict:",
                "    weights = prices[['SPY']].iloc[1:].copy()",
                "    gross_returns = prices['SPY'].pct_change().fillna(0.0)",
                "    return {'weights': weights, 'gross_returns': gross_returns}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 103.0]}, index=pd.date_range("2024-01-01", periods=4, freq="B"))
    data_plan = DataPlan(tickers=["SPY"], benchmark="SPY", start=prices.index[0].date(), end=prices.index[-1].date())
    result = execute_strategy_code(
        code_path=code_path,
        prices=prices,
        cfg={"foo": "bar"},
        output_dir=tmp_path,
        data_plan=data_plan,
        timeout_seconds=10,
    )
    assert not result.passed
    assert any(item.code == "weights_alignment" for item in result.sanity_checks)
    assert Path(result.weights_path).exists()
