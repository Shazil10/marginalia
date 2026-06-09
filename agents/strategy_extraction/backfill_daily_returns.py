"""
backfill_daily_returns.py
-------------------------
Re-runs each existing strategy script with strategy-specific patches so it
exports daily_returns, then saves daily_returns.csv. After this, run
generate_reports.py to refresh report.html with full tear sheets.

Usage (from project root):
    python agents/strategy_extraction/backfill_daily_returns.py
"""

import os
import re
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from agents.strategy_extraction.backtest import execute_backtest

OUTPUTS_BASE = os.path.join(SCRIPT_DIR, "outputs")


def find_strategy_dirs():
    """Strategy dirs that have a generated script but no daily_returns.csv yet."""
    dirs = []
    for root, _, files in os.walk(OUTPUTS_BASE):
        if "backtest_results.json" not in files:
            continue
        if "daily_returns.csv" in files:
            continue
        candidates = glob.glob(os.path.join(root, "generated_strategy_v*.py"))
        if not candidates:
            continue
        code_path = max(candidates, key=lambda p: os.path.getmtime(p))
        dirs.append((root, code_path))
    return sorted(dirs, key=lambda x: x[0])


def patch_giordano(content):
    """2018_dowaward-giordano: backtest_strategy(params, close, high, low) has daily_rets."""
    main = content.split("if __name__")[0]
    rest = content[len(main):]

    # 1) In backtest_strategy return, add 'daily_rets': daily_rets
    old_ret = """        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return
        }"""
    new_ret = """        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return,
            'daily_rets': daily_rets
        }"""
    if new_ret in main:
        return content
    if old_ret not in main:
        return None
    main = main.replace(old_ret, new_ret, 1)

    # 2) In the loop, store best_daily_rets when we update best_sharpe
    old_loop = """                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_params = params.copy()
                    best_metrics = metrics.copy()"""
    new_loop = """                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    best_daily_rets = metrics.get('daily_rets')"""
    if "best_daily_rets = metrics.get" in main:
        pass  # already patched
    elif old_loop not in main:
        return None
    else:
        main = main.replace(old_loop, new_loop, 1)

    # 3) Initialize best_daily_rets before the loop
    main = main.replace(
        "    best_sharpe = -np.inf\n    best_params = default_params\n    best_metrics = None",
        "    best_sharpe = -np.inf\n    best_params = default_params\n    best_metrics = None\n    best_daily_rets = None",
        1
    )

    # 3b) In fallback block, set best_daily_rets when we get metrics from backtest_strategy(default_params, ...)
    old_fallback = "    if best_metrics is None:\n        best_metrics = backtest_strategy(default_params, close, high, low)\n        best_params = default_params\n        if best_metrics is None:"
    new_fallback = "    if best_metrics is None:\n        best_metrics = backtest_strategy(default_params, close, high, low)\n        best_params = default_params\n        if best_metrics is not None:\n            best_daily_rets = best_metrics.get('daily_rets')\n        if best_metrics is None:"
    if new_fallback not in main and old_fallback in main:
        main = main.replace(old_fallback, new_fallback, 1)

    # 4) Before final return, add daily_returns_list
    block = """
    # [BACKFILL] Export daily returns for reporting
    if best_daily_rets is not None and len(best_daily_rets) > 0:
        best_daily_rets = best_daily_rets.sort_index().dropna()
        daily_returns_list = [[str(best_daily_rets.index[i].date()), float(best_daily_rets.iloc[i])] for i in range(len(best_daily_rets))]
        start_date = str(best_daily_rets.index.min().date())
        end_date = str(best_daily_rets.index.max().date())
    else:
        daily_returns_list = []
        start_date = ''
        end_date = ''
"""
    old_final = """    return {
        'cagr': float(best_metrics['cagr']),
        'sharpe': float(best_metrics['sharpe']),
        'max_drawdown': float(best_metrics['max_drawdown']),
        'calmar': float(best_metrics['calmar']),
        'win_rate': float(best_metrics['win_rate']),
        'total_return': float(best_metrics['total_return']),
        'optimized_parameters': best_params
    }"""
    new_final = block + """    return {
        'cagr': float(best_metrics['cagr']),
        'sharpe': float(best_metrics['sharpe']),
        'max_drawdown': float(best_metrics['max_drawdown']),
        'calmar': float(best_metrics['calmar']),
        'win_rate': float(best_metrics['win_rate']),
        'total_return': float(best_metrics['total_return']),
        'optimized_parameters': best_params,
        'daily_returns': daily_returns_list,
        'start_date': start_date,
        'end_date': end_date
    }"""
    if "daily_returns': daily_returns_list" in main:
        return main + rest
    if old_final not in main:
        return None
    main = main.replace(old_final, new_final, 1)
    return main + rest


def patch_leverage(content):
    """leverage_long_run: backtest_strategy returns dict; add port_ret as daily_rets."""
    main = content.split("if __name__")[0]
    rest = content[len(main):]

    old_ret = """        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return
        }

    # Parameter ranges"""
    new_ret = """        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return,
            'daily_rets': port_ret
        }

    # Parameter ranges"""
    if "daily_rets': port_ret" in main:
        return content
    if old_ret not in main:
        return None
    main = main.replace(old_ret, new_ret, 1)

    old_loop = """            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = result
                best_params = {"""
    new_loop = """            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = result
                best_daily_rets = result.get('daily_rets')
                best_params = {"""
    if "best_daily_rets = result.get" in main:
        pass
    elif old_loop not in main:
        return None
    else:
        main = main.replace(old_loop, new_loop, 1)

    main = main.replace(
        "    best_result = None\n    best_params = None\n    best_sharpe = -np.inf",
        "    best_result = None\n    best_params = None\n    best_sharpe = -np.inf\n    best_daily_rets = None",
        1
    )

    block = """
    # [BACKFILL] Export daily returns for reporting
    if best_daily_rets is not None and len(best_daily_rets) > 0:
        best_daily_rets = best_daily_rets.sort_index().dropna()
        daily_returns_list = [[str(best_daily_rets.index[i].date()), float(best_daily_rets.iloc[i])] for i in range(len(best_daily_rets))]
        start_date = str(best_daily_rets.index.min().date())
        end_date = str(best_daily_rets.index.max().date())
    else:
        daily_returns_list = []
        start_date = ''
        end_date = ''
"""
    old_final = """    return {
        'cagr': best_result['cagr'],
        'sharpe': best_result['sharpe'],
        'max_drawdown': best_result['max_drawdown'],
        'calmar': best_result['calmar'],
        'win_rate': best_result['win_rate'],
        'total_return': best_result['total_return'],
        'optimized_parameters': best_params
    }"""
    new_final = block + """    return {
        'cagr': best_result['cagr'],
        'sharpe': best_result['sharpe'],
        'max_drawdown': best_result['max_drawdown'],
        'calmar': best_result['calmar'],
        'win_rate': best_result['win_rate'],
        'total_return': best_result['total_return'],
        'optimized_parameters': best_params,
        'daily_returns': daily_returns_list,
        'start_date': start_date,
        'end_date': end_date
    }"""
    if "daily_returns': daily_returns_list" in main:
        return main + rest
    if old_final not in main:
        return None
    main = main.replace(old_final, new_final, 1)
    return main + rest


def patch_short_term(content):
    """short_term: compute_strategy returns dict; add daily_returns as daily_rets."""
    main = content.split("if __name__")[0]
    rest = content[len(main):]

    # Match exact spacing (4 spaces before }, then blank line, then "    # Parameter ranges")
    old_ret = """        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return,
            'n_trades': len(trades_df)
        }
    
    # Parameter ranges"""
    new_ret = """        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return,
            'n_trades': len(trades_df),
            'daily_rets': daily_returns
        }
    
    # Parameter ranges"""
    if "daily_rets': daily_returns" in main:
        return content
    if old_ret not in main:
        return None
    main = main.replace(old_ret, new_ret, 1)

    old_loop = """            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = result
                best_params = params"""
    new_loop = """            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = result
                best_daily_rets = result.get('daily_rets')
                best_params = params"""
    if "best_daily_rets = result.get" in main:
        pass
    elif old_loop not in main:
        return None
    else:
        main = main.replace(old_loop, new_loop, 1)

    main = main.replace(
        "    best_result = None\n    best_params = None\n    best_sharpe = -np.inf",
        "    best_result = None\n    best_params = None\n    best_sharpe = -np.inf\n    best_daily_rets = None",
        1
    )

    # When fallback runs, capture daily_rets from best_result
    old_fallback_st = "        best_result = compute_strategy(df, **default_params)\n        best_params = default_params\n        \n        if best_result is None:"
    new_fallback_st = "        best_result = compute_strategy(df, **default_params)\n        best_params = default_params\n        if best_result is not None:\n            best_daily_rets = best_result.get('daily_rets')\n        \n        if best_result is None:"
    if new_fallback_st not in main and old_fallback_st in main:
        main = main.replace(old_fallback_st, new_fallback_st, 1)

    block = """
    # [BACKFILL] Export daily returns for reporting
    if best_daily_rets is not None and len(best_daily_rets) > 0:
        best_daily_rets = best_daily_rets.sort_index().dropna()
        daily_returns_list = [[str(best_daily_rets.index[i].date()), float(best_daily_rets.iloc[i])] for i in range(len(best_daily_rets))]
        start_date = str(best_daily_rets.index.min().date())
        end_date = str(best_daily_rets.index.max().date())
    else:
        daily_returns_list = []
        start_date = ''
        end_date = ''
"""
    old_final = """    return {
        'cagr': float(best_result['cagr']),
        'sharpe': float(best_result['sharpe']),
        'max_drawdown': float(best_result['max_drawdown']),
        'calmar': float(best_result['calmar']),
        'win_rate': float(best_result['win_rate']),
        'total_return': float(best_result['total_return']),
        'optimized_parameters': best_params
    }"""
    new_final = block + """    return {
        'cagr': float(best_result['cagr']),
        'sharpe': float(best_result['sharpe']),
        'max_drawdown': float(best_result['max_drawdown']),
        'calmar': float(best_result['calmar']),
        'win_rate': float(best_result['win_rate']),
        'total_return': float(best_result['total_return']),
        'optimized_parameters': best_params,
        'daily_returns': daily_returns_list,
        'start_date': start_date,
        'end_date': end_date
    }"""
    if "daily_returns': daily_returns_list" in main:
        return main + rest
    if old_final not in main:
        return None
    main = main.replace(old_final, new_final, 1)
    return main + rest


def patch_code(output_dir, code_path, content):
    """Apply strategy-specific patch based on folder name."""
    name = os.path.basename(output_dir)
    if "2018_dowaward" in name or "giordano" in name:
        return patch_giordano(content)
    if "leverage" in name:
        return patch_leverage(content)
    if "short_term" in name:
        return patch_short_term(content)
    # Fallback: generic patch (quantmentals-style run_strategy(close_prices, best_params))
    return patch_generic(content)


def patch_generic(content):
    """Generic patch for strategies that use run_strategy(close_prices, best_params)."""
    main = content.split("if __name__")[0]
    rest = content[len(main):]
    old_tail = "'optimized_parameters': best_params"
    new_tail = "'optimized_parameters': best_params,\n        'daily_returns': daily_returns_list,\n        'start_date': start_date,\n        'end_date': end_date"
    if new_tail in main:
        return content
    if old_tail not in main:
        return None
    main = main.replace(old_tail, new_tail, 1)
    block = '''
    # [BACKFILL] Export daily returns for reporting
    try:
        _dr = run_strategy(close_prices, best_params)
    except NameError:
        try:
            _dr = run_strategy(prices, best_params)
        except NameError:
            _dr = None
    if _dr is not None and len(_dr) > 0:
        _dr = _dr.sort_index().dropna()
        daily_returns_list = [[str(_dr.index[i].date()), float(_dr.iloc[i])] for i in range(len(_dr))]
        start_date = str(_dr.index.min().date())
        end_date = str(_dr.index.max().date())
    else:
        daily_returns_list = []
        start_date = ''
        end_date = ''
'''
    return_match = list(re.finditer(r"\n    return \{", main))
    if not return_match:
        return None
    insert_pos = return_match[-1].start()
    main = main[:insert_pos] + block + main[insert_pos:]
    return main + rest


def main():
    if not os.path.isdir(OUTPUTS_BASE):
        print(f"Outputs directory not found: {OUTPUTS_BASE}")
        sys.exit(1)

    strategy_dirs = find_strategy_dirs()
    if not strategy_dirs:
        print("No strategy folders missing daily_returns.csv. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(strategy_dirs)} strategy folder(s) without daily_returns.csv.\n")

    success = 0
    for output_dir, code_path in strategy_dirs:
        name = os.path.basename(output_dir)
        print(f"{'='*60}")
        print(f"Backfilling: {name}")
        print(f"{'='*60}")

        with open(code_path, "r") as f:
            original = f.read()

        patched = patch_code(output_dir, code_path, original)
        if patched is None:
            print("  Could not patch this strategy file. Skipping.\n")
            continue

        try:
            with open(code_path, "w") as f:
                f.write(patched)
            metrics, err = execute_backtest(code_path, output_dir)
            if err:
                print(f"  Backtest failed: {err}\n")
            elif metrics:
                print("  daily_returns.csv saved.")
                success += 1
        finally:
            with open(code_path, "w") as f:
                f.write(original)

        print()

    print(f"Done. Exported daily returns for {success}/{len(strategy_dirs)} strategies.")
    print("\nNext: run the following to refresh reports with full tear sheets:")
    print("  python agents/strategy_extraction/generate_reports.py")
    print("\nThen open each strategy's report.html in a browser.")


if __name__ == "__main__":
    main()
