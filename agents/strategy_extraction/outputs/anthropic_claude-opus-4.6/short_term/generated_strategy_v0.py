import numpy as np
import pandas as pd
import yfinance as yf
import random
import itertools
from datetime import datetime


def run_backtest():
    # Download data
    data = yf.download("SPY", start="2019-01-01", end="2024-12-31", auto_adjust=False)
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']['SPY'].copy()
    else:
        close = data['Close'].copy()
    
    close = close.ffill().dropna()
    close.index = pd.to_datetime(close.index)
    
    # Precompute daily returns
    daily_rets = close.pct_change().fillna(0.0)
    
    # Identify month-end trading days
    close_df = close.to_frame(name='Close')
    close_df['year_month'] = close_df.index.to_period('M')
    
    # Get the last trading day of each month
    month_end_dates = close_df.groupby('year_month').apply(lambda x: x.index[-1])
    month_end_dates = month_end_dates.sort_values()
    
    # All trading dates as a sorted list for indexing
    all_dates = close.index.sort_values()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    # Define parameter grid
    param_grid = list(itertools.product(
        [3, 4, 5, 6, 7],       # lookback_window_days
        [1],                     # holding_period_months
        [-0.02, -0.01, 0.0, 0.01]  # threshold_return
    ))
    
    # Randomly sample up to 10 combinations
    random.seed(42)
    if len(param_grid) > 10:
        sampled_params = random.sample(param_grid, 10)
    else:
        sampled_params = param_grid
    
    transaction_cost = 0.001
    
    best_sharpe = -np.inf
    best_result = None
    best_params = None
    best_daily_series = None
    
    for lookback, holding, threshold in sampled_params:
        # Build strategy returns
        # For each month-end, check if the return over the last `lookback` trading days is < threshold
        # If so, hold SPY for the next `holding` month(s)
        
        strategy_daily_returns = pd.Series(0.0, index=all_dates)
        
        for i in range(len(month_end_dates) - holding):
            me_date = month_end_dates.iloc[i]
            
            # Get the index of this month-end date
            if me_date not in date_to_idx:
                continue
            me_idx = date_to_idx[me_date]
            
            # Compute return over last `lookback` trading days ending at me_date
            start_idx = max(0, me_idx - lookback)
            if start_idx == me_idx:
                continue
            
            lookback_start_date = all_dates[start_idx]
            last_week_return = close.loc[me_date] / close.loc[lookback_start_date] - 1.0
            
            if last_week_return < threshold:
                # Hold for the next month(s)
                # Next month's trading days: from the day after me_date to the month-end of the next month
                next_me_date = month_end_dates.iloc[i + holding]
                
                # Get all trading days from the day after me_date to next_me_date (inclusive)
                holding_mask = (all_dates > me_date) & (all_dates <= next_me_date)
                holding_dates = all_dates[holding_mask]
                
                if len(holding_dates) == 0:
                    continue
                
                # Apply daily returns for holding period
                holding_daily_rets = daily_rets.loc[holding_dates].copy()
                
                # Apply transaction costs on entry and exit
                # Entry cost on first day, exit cost on last day
                holding_daily_rets.iloc[0] -= transaction_cost
                holding_daily_rets.iloc[-1] -= transaction_cost
                
                # Add to strategy returns (if overlapping, last assignment wins - but with monthly, no overlap)
                strategy_daily_returns.loc[holding_dates] = holding_daily_rets.values
        
        # Compute performance metrics
        cumulative = (1 + strategy_daily_returns).cumprod()
        total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1.0
        
        n_years = (all_dates[-1] - all_dates[0]).days / 365.25
        cagr = (cumulative.iloc[-1] / cumulative.iloc[0]) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
        
        ann_std = strategy_daily_returns.std() * np.sqrt(252)
        sharpe = cagr / ann_std if ann_std > 0 else 0.0
        
        # Max drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0
        
        # Win rate (on days with positions)
        active_days = strategy_daily_returns[strategy_daily_returns != 0.0]
        win_rate = (active_days > 0).sum() / len(active_days) if len(active_days) > 0 else 0.0
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {
                'lookback_window_days': lookback,
                'holding_period_months': holding,
                'threshold_return': threshold
            }
            best_result = {
                'cagr': float(cagr),
                'sharpe': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'calmar': float(calmar),
                'win_rate': float(win_rate),
                'total_return': float(total_return),
            }
            best_daily_series = strategy_daily_returns.copy()
    
    # Build daily_returns list
    daily_returns_list = []
    for date, ret in best_daily_series.items():
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        daily_returns_list.append([date_str, float(ret)])
    
    best_result['optimized_parameters'] = best_params
    best_result['start_date'] = all_dates[0].strftime('%Y-%m-%d')
    best_result['end_date'] = all_dates[-1].strftime('%Y-%m-%d')
    best_result['daily_returns'] = daily_returns_list
    
    return best_result
if __name__ == '__main__':
    import json
    import traceback
    try:
        res = run_backtest()
        # Extract and persist daily returns without bloating stdout
        daily_returns_data = res.pop('daily_returns', None)
        if daily_returns_data:
            import pandas as pd
            returns_s = pd.Series(
                [float(r) for _, r in daily_returns_data],
                index=pd.to_datetime([d for d, _ in daily_returns_data]),
                name='daily_return',
            )
            returns_s.to_csv(r'/Users/shazilfarukh/Desktop/agentic-quant/agents/strategy_extraction/outputs/anthropic_claude-opus-4.6/short_term/daily_returns.csv', header=True)
        print('BACKTEST_RESULT:' + json.dumps(res))
    except Exception as e:
        print('BACKTEST_ERROR:' + traceback.format_exc())
