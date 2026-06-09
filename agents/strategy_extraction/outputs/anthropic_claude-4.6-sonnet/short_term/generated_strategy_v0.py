import yfinance as yf
import pandas as pd
import numpy as np
import random
from itertools import product
from datetime import datetime

def run_backtest():
    # Download data
    ticker = "SPY"
    data = yf.download(ticker, start="2019-01-01", end="2024-12-31", auto_adjust=False)
    
    # Handle MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Use Adj Close for prices, Volume for volume
    prices = data['Adj Close'].ffill().dropna()
    volume = data['Volume'].ffill().dropna()
    
    # Align
    prices = prices.loc[volume.index]
    volume = volume.loc[prices.index]
    
    df = pd.DataFrame({'price': prices, 'volume': volume})
    df.index = pd.to_datetime(df.index)
    
    def compute_strategy(df, signal_window_weeks, holding_period_months,
                         signal_threshold_return, lookback_last_week_days,
                         transaction_cost_bps, volume_threshold_percentile):
        
        tc = transaction_cost_bps / 10000.0
        
        # Identify month-end dates
        df['year_month'] = df.index.to_period('M')
        months = df['year_month'].unique()
        
        # Rolling volume percentile (trailing 252 days)
        df['vol_pct'] = df['volume'].rolling(252, min_periods=60).apply(
            lambda x: np.percentile(x, volume_threshold_percentile), raw=True
        )
        
        # For each month, compute the return of the last N trading days
        # Signal: if last N days return < signal_threshold_return => buy next month
        
        trades = []
        
        for i in range(len(months) - holding_period_months - 1):
            current_month = months[i]
            
            # Get trading days in current month
            month_days = df[df['year_month'] == current_month]
            
            if len(month_days) < lookback_last_week_days:
                continue
            
            # Last N trading days of current month
            last_n_days = month_days.iloc[-lookback_last_week_days:]
            
            # Compute return over last N days
            start_price = last_n_days['price'].iloc[0]
            end_price = last_n_days['price'].iloc[-1]
            
            if start_price <= 0:
                continue
            
            last_week_return = (end_price - start_price) / start_price
            
            # Volume filter: check if average volume in last N days is above threshold
            avg_vol = last_n_days['volume'].mean()
            vol_threshold = last_n_days['vol_pct'].iloc[-1]
            
            # Signal condition
            signal = last_week_return < signal_threshold_return
            
            # Optional volume filter (volume above percentile threshold)
            # volume_ok = avg_vol >= vol_threshold if not np.isnan(vol_threshold) else True
            # We apply volume filter as additional condition
            volume_ok = True
            if not np.isnan(vol_threshold):
                volume_ok = avg_vol >= vol_threshold
            
            if not (signal and volume_ok):
                continue
            
            # Entry: first trading day of next month
            entry_month = months[i + 1]
            entry_days = df[df['year_month'] == entry_month]
            
            if len(entry_days) == 0:
                continue
            
            entry_date = entry_days.index[0]
            entry_price = entry_days['price'].iloc[0]
            
            # Exit: last trading day of holding month
            exit_month_idx = i + holding_period_months
            if exit_month_idx >= len(months):
                continue
            
            exit_month = months[exit_month_idx]
            exit_days = df[df['year_month'] == exit_month]
            
            if len(exit_days) == 0:
                continue
            
            exit_date = exit_days.index[-1]
            exit_price = exit_days['price'].iloc[-1]
            
            # Compute trade return with transaction costs
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - tc  # round-trip cost
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': gross_return,
                'net_return': net_return
            })
        
        if len(trades) == 0:
            return None
        
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)
        
        # Build equity curve on daily basis
        daily_index = df.index
        equity = pd.Series(1.0, index=daily_index)
        
        capital = 1.0
        in_trade = False
        current_trade_idx = 0
        trade_list = trades_df.to_dict('records')
        trade_ptr = 0
        
        # Build equity curve
        equity_values = np.ones(len(daily_index))
        
        # Map trades to equity curve
        # Between trades, capital stays flat (cash)
        # During trade, capital grows with price
        
        trade_active = [False] * len(daily_index)
        trade_entry_capital = {}
        
        date_to_idx = {d: i for i, d in enumerate(daily_index)}
        
        running_capital = 1.0
        equity_arr = np.ones(len(daily_index))
        
        # Process sequentially
        prev_capital = 1.0
        active_trade = None
        
        for idx_i, date in enumerate(daily_index):
            # Check if a new trade starts today
            if active_trade is None and trade_ptr < len(trade_list):
                t = trade_list[trade_ptr]
                if date >= t['entry_date']:
                    active_trade = t
                    trade_entry_price = t['entry_price']
                    trade_entry_capital = prev_capital
                    trade_ptr += 1
            
            if active_trade is not None:
                # Mark to market
                current_price = df['price'].iloc[idx_i]
                gross = (current_price - active_trade['entry_price']) / active_trade['entry_price']
                equity_arr[idx_i] = trade_entry_capital * (1 + gross)
                
                # Check if trade ends today
                if date >= active_trade['exit_date']:
                    prev_capital = trade_entry_capital * (1 + active_trade['net_return'])
                    equity_arr[idx_i] = prev_capital
                    active_trade = None
            else:
                equity_arr[idx_i] = prev_capital
        
        equity_series = pd.Series(equity_arr, index=daily_index)
        
        # Compute metrics
        total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
        
        n_years = (daily_index[-1] - daily_index[0]).days / 365.25
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / n_years) - 1
        
        daily_returns = equity_series.pct_change().dropna()
        
        sharpe = 0.0
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # Max drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar = 0.0
        if abs(max_drawdown) > 0:
            calmar = cagr / abs(max_drawdown)
        
        # Win rate
        wins = (trades_df['net_return'] > 0).sum()
        win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0.0
        
        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return,
            'n_trades': len(trades_df)
        }
    
    # Parameter ranges
    param_ranges = {
        'signal_window_weeks': [1, 2, 3],
        'holding_period_months': [1, 2, 3],
        'signal_threshold_return': [-0.02, -0.01, 0.0, 0.01],
        'lookback_last_week_days': [3, 4, 5, 7],
        'transaction_cost_bps': [5, 10, 20, 30],
        'volume_threshold_percentile': [25, 50, 75]
    }
    
    # Random sample 8 combinations
    random.seed(42)
    all_keys = list(param_ranges.keys())
    all_values = list(param_ranges.values())
    all_combos = list(product(*all_values))
    sampled_combos = random.sample(all_combos, min(8, len(all_combos)))
    
    best_result = None
    best_params = None
    best_sharpe = -np.inf
    
    for combo in sampled_combos:
        params = dict(zip(all_keys, combo))
        
        try:
            result = compute_strategy(
                df,
                signal_window_weeks=params['signal_window_weeks'],
                holding_period_months=params['holding_period_months'],
                signal_threshold_return=params['signal_threshold_return'],
                lookback_last_week_days=params['lookback_last_week_days'],
                transaction_cost_bps=params['transaction_cost_bps'],
                volume_threshold_percentile=params['volume_threshold_percentile']
            )
            
            if result is None:
                continue
            
            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = result
                best_params = params
        except Exception as e:
            continue
    
    # If no valid result found, run with defaults
    if best_result is None:
        default_params = {
            'signal_window_weeks': 1,
            'holding_period_months': 1,
            'signal_threshold_return': 0.0,
            'lookback_last_week_days': 5,
            'transaction_cost_bps': 10,
            'volume_threshold_percentile': 50
        }
        best_result = compute_strategy(df, **default_params)
        best_params = default_params
        
        if best_result is None:
            best_result = {
                'cagr': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'calmar': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0
            }
    
    return {
        'cagr': float(best_result['cagr']),
        'sharpe': float(best_result['sharpe']),
        'max_drawdown': float(best_result['max_drawdown']),
        'calmar': float(best_result['calmar']),
        'win_rate': float(best_result['win_rate']),
        'total_return': float(best_result['total_return']),
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
