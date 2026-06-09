import yfinance as yf
import numpy as np
import pandas as pd
import random
from itertools import product

def run_backtest():
    # Download data
    tickers = ['SPY', 'UPRO', 'SSO', 'BIL', 'SHV']
    raw = yf.download(tickers, start='2019-01-01', end='2024-12-31', auto_adjust=True)

    # Handle MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw[['Close']]

    close = close.ffill().dropna()

    # Ensure all tickers present; fallback if some missing
    available = close.columns.tolist()

    spy = close['SPY'] if 'SPY' in available else None
    if spy is None:
        raise ValueError("SPY data not available")

    # Use UPRO as 3x, SSO as 2x proxy; BIL as T-bill proxy
    upro = close['UPRO'] if 'UPRO' in available else None
    sso = close['SSO'] if 'SSO' in available else None
    bil = close['BIL'] if 'BIL' in available else None
    shv = close['SHV'] if 'SHV' in available else None

    # T-bill proxy: prefer BIL, fallback SHV
    tbill = bil if bil is not None else shv

    if tbill is None:
        raise ValueError("No T-bill proxy available")

    # Transaction cost in bps
    transaction_cost_bps = 10
    tc = transaction_cost_bps / 10000.0

    def compute_ma(series, period, ma_type='simple'):
        if ma_type == 'simple':
            return series.rolling(window=period).mean()
        else:
            return series.ewm(span=period, adjust=False).mean()

    def get_leveraged_returns(spy_returns, leverage):
        """Simulate leveraged returns from SPY daily returns with margin cost."""
        annual_borrow_rate = 0.02
        daily_borrow_cost = annual_borrow_rate / 252.0
        # Leveraged return = leverage * spy_return - (leverage - 1) * daily_borrow_cost
        lev_ret = leverage * spy_returns - (leverage - 1) * daily_borrow_cost
        return lev_ret

    def backtest_strategy(spy_prices, tbill_prices, ma_period, leverage, ma_type):
        # Compute moving average signal on SPY
        ma = compute_ma(spy_prices, ma_period, ma_type)

        # Align indices
        common_idx = spy_prices.index.intersection(tbill_prices.index)
        spy_p = spy_prices.loc[common_idx]
        tbill_p = tbill_prices.loc[common_idx]
        ma_aligned = ma.loc[common_idx]

        # Drop NaN from MA warmup
        valid_idx = ma_aligned.dropna().index
        spy_p = spy_p.loc[valid_idx]
        tbill_p = tbill_p.loc[valid_idx]
        ma_aligned = ma_aligned.loc[valid_idx]

        # Daily returns
        spy_ret = spy_p.pct_change().fillna(0)
        tbill_ret = tbill_p.pct_change().fillna(0)

        # Signal: 1 = above MA (leveraged), 0 = below MA (T-bills)
        # Use previous day's signal to avoid look-ahead (execution lag = 1 day)
        signal = (spy_p > ma_aligned).astype(int).shift(1).fillna(0)

        # Detect signal changes for transaction costs
        signal_change = signal.diff().abs().fillna(0)

        # Leveraged returns when in equity
        lev_ret = get_leveraged_returns(spy_ret, leverage)

        # Portfolio return
        port_ret = signal * lev_ret + (1 - signal) * tbill_ret

        # Apply transaction costs on signal changes
        port_ret = port_ret - signal_change * tc

        # Compute cumulative returns
        cum_ret = (1 + port_ret).cumprod()

        # Performance metrics
        total_return = cum_ret.iloc[-1] - 1

        n_years = len(port_ret) / 252.0
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        excess_ret = port_ret - (tbill_ret / 252.0 if tbill_ret.mean() > 0.001 else 0)
        sharpe = (port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0

        # Max drawdown
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate (daily)
        win_rate = (port_ret > 0).sum() / len(port_ret) if len(port_ret) > 0 else 0

        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return
        }

    # Parameter ranges
    ma_periods = [50, 100, 150, 200, 250, 300]
    leverages = [1.5, 2.0, 2.5, 3.0]
    ma_types = ['simple', 'exponential']

    all_combinations = list(product(ma_periods, leverages, ma_types))

    # Randomly sample up to 8 combinations
    random.seed(42)
    sampled = random.sample(all_combinations, min(8, len(all_combinations)))

    best_result = None
    best_params = None
    best_sharpe = -np.inf

    for (ma_period, leverage, ma_type) in sampled:
        try:
            result = backtest_strategy(spy, tbill, ma_period, leverage, ma_type)
            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = result
                best_params = {
                    'moving_average_period': ma_period,
                    'leverage_multiplier': leverage,
                    'moving_average_type': ma_type
                }
        except Exception as e:
            continue

    if best_result is None:
        # Fallback with defaults
        best_result = backtest_strategy(spy, tbill, 200, 2.0, 'simple')
        best_params = {
            'moving_average_period': 200,
            'leverage_multiplier': 2.0,
            'moving_average_type': 'simple'
        }

    return {
        'cagr': best_result['cagr'],
        'sharpe': best_result['sharpe'],
        'max_drawdown': best_result['max_drawdown'],
        'calmar': best_result['calmar'],
        'win_rate': best_result['win_rate'],
        'total_return': best_result['total_return'],
        'optimized_parameters': best_params
    }
if __name__ == '__main__':
    import json
    import traceback
    try:
        res = run_backtest()
        print('BACKTEST_RESULT:' + json.dumps(res))
    except Exception as e:
        print('BACKTEST_ERROR:' + traceback.format_exc())
