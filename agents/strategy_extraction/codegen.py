import re

def run_codegen_agent(client, model, strategy_json, previous_code=None, error_msg=None):
    """Generates backtesting Python code from strategy JSON, handling self-healing loops."""
    print("Running Codegen Agent...")
    system_prompt = """
    Translate the provided strategy specification JSON into a complete, runnable Python backtesting script.
    - Use yfinance to download daily price data for at least 5 years (e.g., 2019 to 2024).
    - CRITICAL YFINANCE HANDLING: yf.download() returns a DataFrame with MultiIndex columns when multiple tickers are passed. Do NOT simply use `data['Adj Close']` or `data['Close']`. You must handle MultiIndex properly (e.g. `data['Close']` extracts just the Close prices if the level 0 index is 'Close'). 
    - CRITICAL PANDAS HANDLING: Be careful using `.apply()` with sliding window functions like `.rolling()`. An `.apply()` function evaluates one row or column at a time (often as a numpy float), which does *not* have a `.rolling()` method. Instead, call `.rolling()` directly on the DataFrame or Series (e.g. `df.rolling(window=x).mean()`).
    - Handle all missing values (e.g., using .ffill() or .dropna()).
    - IMPORTANT: Include an optimization loop. The strategy will have `parameter_ranges`. 
      - **CRITICAL RESTRICTION ON GRID SEARCH**: Grid searching multiple nested for-loops across years of daily data takes HOURS. You MUST randomly sample only a MAXIMUM of 5 to 10 random combinations from the parameter grid, OR set the ranges to just 1 or 2 variations, so the script finishes executing in under 30 seconds.
    - Wrap everything in a single function called `run_backtest()` that takes no arguments.
    - If a package dependency like 'arch' requires advanced syntax you are not 100% confident in, simplify the logic to use standard pandas/numpy equivalents (e.g. rolling std dev instead of GARCH) to prioritize making the code executable.
    - Return a dict with exactly these keys:
        cagr, sharpe, max_drawdown, calmar, win_rate, total_return,
        optimized_parameters (dict of best parameters found),
        start_date (string 'YYYY-MM-DD' matching the actual backtest start),
        end_date (string 'YYYY-MM-DD' matching the actual backtest end),
        daily_returns (list of [date_string, float] pairs for every day of the BEST strategy's return series,
            e.g. [['2019-01-02', 0.0012], ['2019-01-03', -0.0023], ...]).
      The daily_returns list must cover the entire backtest window, not just the optimisation subset.
    - Output only raw Python code. No markdown fences. No explanation.
    - DO NOT include an `if __name__ == '__main__':` block. Just define the functions.
    """
    
    user_prompt = f"Strategy spec:\n{strategy_json}"
    if previous_code and error_msg:
        user_prompt += f"\n\nWe tried executing this code:\n```python\n{previous_code}\n```\nIt failed with the following error:\n{error_msg}\n\nPlease FIX the code based on the error. Return ONLY the corrected raw python code."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()
        
        # Clean reasoning blocks if using models like DeepSeek-R1
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Extract python code if wrapped in markdown
        match = re.search(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            content = match.group(1).strip()
            
        print("Model output received.")
        return content
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return previous_code # Fallback to previous code to avoid crashing the loop