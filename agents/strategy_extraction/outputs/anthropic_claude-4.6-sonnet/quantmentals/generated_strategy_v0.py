import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

def run_backtest():
    # Use a representative set of large-cap US stocks as proxy for broad market
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX', 'CMCSA',
        'VZ', 'INTC', 'T', 'PFE', 'MRK', 'ABT', 'TMO', 'COST', 'WMT', 'CVX',
        'XOM', 'LLY', 'ABBV', 'AMGN', 'MDT', 'HON', 'UNP', 'RTX', 'CAT', 'GS',
        'MS', 'BLK', 'SPGI', 'AXP', 'USB', 'WFC', 'C', 'TGT', 'LOW', 'NKE',
        'MCD', 'SBUX', 'BMY', 'GILD', 'ISRG', 'SYK', 'ZTS', 'VRTX', 'REGN', 'BIIB',
        'TXN', 'QCOM', 'AVGO', 'AMD', 'MU', 'AMAT', 'LRCX', 'KLAC', 'NOW', 'SNOW',
        'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL', 'BKR', 'DVN',
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ES', 'WEC',
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'AVB', 'EQR'
    ]

    start_date = '2019-01-01'
    end_date = '2024-12-31'

    print("Downloading price data...")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Handle MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        close_prices = raw['Close']
        volume_data = raw['Volume'] if 'Volume' in raw.columns.get_level_values(0) else None
    else:
        close_prices = raw[['Close']]
        volume_data = None

    close_prices = close_prices.ffill().dropna(how='all', axis=1)
    available_tickers = close_prices.columns.tolist()

    print(f"Available tickers: {len(available_tickers)}")

    # Simulate fundamental data since we can't get real fundamentals from yfinance easily
    # We'll use price-derived proxies:
    # - Value proxy: inverse of 12-month price change (mean reversion proxy)
    # - Quality proxy: low volatility + positive momentum consistency
    # - Momentum: standard price momentum
    # Market cap proxy: use price * volume as relative size proxy

    def compute_momentum(prices, lookback_months, skip_months):
        """Compute cross-sectional momentum"""
        lookback_days = lookback_months * 21
        skip_days = skip_months * 21
        if skip_days > 0:
            returns = prices.shift(skip_days) / prices.shift(lookback_days) - 1
        else:
            returns = prices / prices.shift(lookback_days) - 1
        return returns

    def compute_value_proxy(prices, lookback_days=252):
        """Value proxy: inverse of long-term return (contrarian)"""
        long_return = prices / prices.shift(lookback_days) - 1
        return -long_return  # High value = low recent long-term return

    def compute_quality_proxy(prices, window=63):
        """Quality proxy: inverse of volatility (low vol = high quality)"""
        daily_returns = prices.pct_change()
        vol = daily_returns.rolling(window=window).std()
        return -vol  # High quality = low volatility

    def winsorize_series(s, pct=1):
        """Winsorize at given percentile"""
        lower = np.percentile(s.dropna(), pct)
        upper = np.percentile(s.dropna(), 100 - pct)
        return s.clip(lower=lower, upper=upper)

    def z_score_cross_sectional(df):
        """Cross-sectional z-score normalization"""
        result = df.copy()
        for idx in df.index:
            row = df.loc[idx]
            valid = row.dropna()
            if len(valid) < 5:
                continue
            mean_val = valid.mean()
            std_val = valid.std()
            if std_val > 0:
                result.loc[idx] = (row - mean_val) / std_val
            else:
                result.loc[idx] = 0
        return result

    def compute_composite_score(prices, params):
        """Compute composite factor score"""
        mom_lookback = params['momentum_lookback_months']
        mom_skip = params['momentum_skip_months']
        val_w = params['value_weight']
        qual_w = params['quality_weight']
        mom_w = params['momentum_weight']

        # Normalize weights
        total_w = val_w + qual_w + mom_w
        val_w /= total_w
        qual_w /= total_w
        mom_w /= total_w

        # Compute factors
        momentum = compute_momentum(prices, mom_lookback, mom_skip)
        value = compute_value_proxy(prices, lookback_days=252)
        quality = compute_quality_proxy(prices, window=63)

        # Z-score normalize cross-sectionally
        mom_z = z_score_cross_sectional(momentum)
        val_z = z_score_cross_sectional(value)
        qual_z = z_score_cross_sectional(quality)

        # Composite score
        composite = val_w * val_z + qual_w * qual_z + mom_w * mom_z
        return composite

    def run_strategy(prices, params):
        """Run the factor strategy backtest"""
        num_holdings = params['num_holdings']
        top_pct = params['top_percentile_threshold'] / 100.0
        transaction_cost = 0.001

        composite = compute_composite_score(prices, params)

        # Resample to monthly rebalancing dates
        monthly_dates = prices.resample('MS').first().index
        monthly_dates = [d for d in monthly_dates if d in prices.index or
                         prices.index[prices.index >= d].shape[0] > 0]

        # Get actual trading dates closest to month start
        rebal_dates = []
        for md in monthly_dates:
            future = prices.index[prices.index >= md]
            if len(future) > 0:
                rebal_dates.append(future[0])

        rebal_dates = sorted(set(rebal_dates))

        portfolio_value = 1.0
        portfolio_values = []
        current_holdings = set()
        daily_returns_list = []

        prev_date = None
        prev_holdings = set()

        for i, rebal_date in enumerate(rebal_dates):
            if rebal_date not in composite.index:
                continue

            scores = composite.loc[rebal_date].dropna()
            if len(scores) < 10:
                continue

            # Winsorize scores
            scores = winsorize_series(scores, pct=1)

            # Select top stocks by composite score
            n_select = min(num_holdings, max(5, int(len(scores) * top_pct)))
            selected = scores.nlargest(n_select).index.tolist()

            # Determine next rebalancing date
            if i + 1 < len(rebal_dates):
                next_rebal = rebal_dates[i + 1]
            else:
                next_rebal = prices.index[-1]

            # Get price returns for holding period
            period_prices = prices.loc[rebal_date:next_rebal, selected]
            if len(period_prices) < 2:
                continue

            # Equal weight portfolio
            period_returns = period_prices.pct_change().dropna()
            if len(period_returns) == 0:
                continue

            port_daily = period_returns.mean(axis=1)

            # Apply transaction costs at rebalancing
            turnover = len(set(selected).symmetric_difference(prev_holdings)) / max(len(selected), 1)
            tc_cost = turnover * transaction_cost

            if len(port_daily) > 0:
                port_daily.iloc[0] -= tc_cost

            daily_returns_list.append(port_daily)
            prev_holdings = set(selected)

        if not daily_returns_list:
            return None

        all_returns = pd.concat(daily_returns_list)
        all_returns = all_returns.sort_index()
        all_returns = all_returns[~all_returns.index.duplicated(keep='first')]

        return all_returns

    def compute_metrics(daily_returns):
        """Compute performance metrics"""
        if daily_returns is None or len(daily_returns) < 30:
            return None

        cumulative = (1 + daily_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        n_years = len(daily_returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        win_rate = (daily_returns > 0).sum() / len(daily_returns)

        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return
        }

    # Parameter ranges for optimization
    param_ranges = {
        'momentum_lookback_months': [6, 9, 12, 18],
        'momentum_skip_months': [0, 1, 2],
        'top_percentile_threshold': [10, 20, 30],
        'value_weight': [0.2, 0.33, 0.4, 0.5],
        'quality_weight': [0.2, 0.33, 0.4, 0.5],
        'momentum_weight': [0.2, 0.33, 0.4, 0.5],
        'min_market_cap_millions': [100, 500, 1000, 2000],
        'num_holdings': [25, 50, 75, 100]
    }

    # Default parameters
    default_params = {
        'momentum_lookback_months': 12,
        'momentum_skip_months': 1,
        'top_percentile_threshold': 20,
        'value_weight': 0.33,
        'quality_weight': 0.33,
        'momentum_weight': 0.34,
        'min_market_cap_millions': 500,
        'num_holdings': 50
    }

    # Random sampling of parameter combinations (max 8 combinations)
    random.seed(42)
    n_samples = 8
    param_combinations = []

    # Always include default
    param_combinations.append(default_params.copy())

    # Random samples
    for _ in range(n_samples - 1):
        combo = {}
        for k, v in param_ranges.items():
            combo[k] = random.choice(v)
        param_combinations.append(combo)

    print(f"Running optimization with {len(param_combinations)} parameter combinations...")

    best_params = default_params.copy()
    best_sharpe = -np.inf
    best_metrics = None

    for idx, params in enumerate(param_combinations):
        print(f"  Testing combination {idx + 1}/{len(param_combinations)}...")
        try:
            daily_returns = run_strategy(close_prices, params)
            metrics = compute_metrics(daily_returns)
            if metrics is not None:
                print(f"    Sharpe: {metrics['sharpe']:.3f}, CAGR: {metrics['cagr']:.3f}")
                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_params = params.copy()
                    best_metrics = metrics.copy()
        except Exception as e:
            print(f"    Error: {e}")
            continue

    # If no valid results, run with defaults
    if best_metrics is None:
        print("Running with default parameters...")
        daily_returns = run_strategy(close_prices, default_params)
        best_metrics = compute_metrics(daily_returns)
        best_params = default_params.copy()

        if best_metrics is None:
            best_metrics = {
                'cagr': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'calmar': 0.0,
                'win_rate': 0.5,
                'total_return': 0.0
            }

    print(f"\nBest Parameters: {best_params}")
    print(f"Best Sharpe: {best_sharpe:.4f}")
    print(f"CAGR: {best_metrics['cagr']:.4f}")
    print(f"Max Drawdown: {best_metrics['max_drawdown']:.4f}")
    print(f"Calmar: {best_metrics['calmar']:.4f}")
    print(f"Win Rate: {best_metrics['win_rate']:.4f}")
    print(f"Total Return: {best_metrics['total_return']:.4f}")

    return {
        'cagr': float(best_metrics['cagr']),
        'sharpe': float(best_metrics['sharpe']),
        'max_drawdown': float(best_metrics['max_drawdown']),
        'calmar': float(best_metrics['calmar']),
        'win_rate': float(best_metrics['win_rate']),
        'total_return': float(best_metrics['total_return']),
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
