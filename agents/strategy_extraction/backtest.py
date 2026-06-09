import os
import json
import re
import subprocess
import traceback
from datetime import datetime

def execute_backtest(code_path, output_dir):
    """Runs the generated python script securely, catches errors, and extracts metrics."""
    print(f"\n--- Running Backtester on {code_path} ---")
    
    # Strip any existing '__main__' block the LLM might have falsely injected, and attach our secure runner
    with open(code_path, "r") as f:
        code = f.read()
    
    if "if __name__" in code:
        code = code.split("if __name__")[0]

    # The runner saves daily_returns to CSV inside the subprocess (avoids huge stdout),
    # then prints only the summary metrics as BACKTEST_RESULT.
    returns_csv_path = os.path.join(output_dir, 'daily_returns.csv').replace('\\', '/')
    runner_code = f"""
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
            returns_s.to_csv(r'{returns_csv_path}', header=True)
        print('BACKTEST_RESULT:' + json.dumps(res))
    except Exception as e:
        print('BACKTEST_ERROR:' + traceback.format_exc())
"""
    with open(code_path, "w") as f:
        f.write(code + runner_code)
            
    # Execute the file as a subprocess
    try:
        print("Executing script (this might take a minute as it downloads data)...")
        result = subprocess.run(
            ["python", code_path], 
            capture_output=True, 
            text=True, 
            timeout=600  # Some strategies (e.g. sector rotation with signal precomputation) need several minutes
        )
        
        if result.returncode != 0:
            print("Execution failed! Stderr:")
            print(result.stderr)
            return None, result.stderr
            
        print("Execution finished. Parsing results...")
        
        stdout_text = result.stdout.strip()
        
        # Look for our special token line to ignore yfinance logs
        match = re.search(r'BACKTEST_RESULT:(.*)', stdout_text)
        if match:
            json_str = match.group(1).strip()
            try:
                metrics = json.loads(json_str)

                # Compute backtest_years from start/end dates if provided by the strategy
                start_date = metrics.get('start_date')
                end_date = metrics.get('end_date')
                if start_date and end_date:
                    try:
                        d1 = datetime.strptime(start_date, '%Y-%m-%d')
                        d2 = datetime.strptime(end_date, '%Y-%m-%d')
                        metrics['backtest_years'] = round((d2 - d1).days / 365.25, 2)
                    except ValueError:
                        pass

                print(json.dumps(metrics, indent=2))
                
                results_path = os.path.join(output_dir, "backtest_results.json")
                with open(results_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Results saved to {results_path}")    
                return metrics, None
            except json.JSONDecodeError:
                print("Failed to parse JSON output. Raw string found:", json_str)
                return None, f"JSON Parse Error. Raw: {json_str}"
        else:
            error_match = re.search(r'BACKTEST_ERROR:(.*)', stdout_text)
            if error_match:
                print(f"Algorithm raised an error during backtesting:\n{error_match.group(1)}")
                return None, error_match.group(1)
            else:
                print("Could not find result token in stdout. Raw output:")
                print(stdout_text)
                return None, stdout_text
            
    except subprocess.TimeoutExpired:
        print("Backtest timed out.")
        return None, "TimeoutExpired: The script took too long to execute."
    except Exception as e:
        print(f"Failed to run backtest subprocess: {e}")
        return None, str(e)
