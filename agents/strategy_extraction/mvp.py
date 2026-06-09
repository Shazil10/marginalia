import os
import sys
import json

# Add the root 'agentic-quant' directory to the Python path so it can find 'utils' and 'agents'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.llm_client import get_llm_client
from extraction import extract_text_from_pdf, run_extraction_agent
from codegen import run_codegen_agent
from backtest import execute_backtest
from reporting import generate_tearsheet

PAPER_PATH = os.path.join(os.path.dirname(__file__), "short_term.pdf")

def main():
    # 1. Initialize Client
    try:
        client, model = get_llm_client()
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Using model: {model}")
    
    # 2. Extract Text from PDF
    print(f"Extracting text from {PAPER_PATH}...")
    paper_text = extract_text_from_pdf(PAPER_PATH)
    
    if not paper_text:
        print("Failed to extract text. Exiting.")
        return

    # Prepare outputs directory
    paper_name = os.path.splitext(os.path.basename(PAPER_PATH))[0]
    safe_model_name = model.replace("/", "_")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", safe_model_name, paper_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Layer 1: Extraction Agent
    strategy_json = run_extraction_agent(client, model, paper_text)
    print("\n--- Extracted Strategy ---")
    print(strategy_json)
    
    # Save the generated JSON and parse it for later use
    json_path = os.path.join(output_dir, "extracted_strategy.json")
    with open(json_path, "w") as f:
        f.write(strategy_json)

    try:
        strategy_dict = json.loads(strategy_json)
    except Exception:
        strategy_dict = {}

    # 4. Layer 2 & 3: Code Gen & Execution (Self-Healing Loop)
    code = None
    error_msg = None
    MAX_RETRIES = 15
    attempt = 0
    metrics = None
    
    while attempt < MAX_RETRIES:
        print(f"\n--- Code Generation Attempt {attempt + 1}/{MAX_RETRIES} ---")
        code = run_codegen_agent(client, model, strategy_json, previous_code=code, error_msg=error_msg)
        
        # Clean up code output
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.strip().endswith("```"):
            code = code.strip()[:-3]
            
        code_path = os.path.join(output_dir, f"generated_strategy_v{attempt}.py")
        print(f"Generated Code saved to {code_path}")
        with open(code_path, "w") as f:
            f.write(code.strip())

        # Execute Backtest
        metrics, error_msg = execute_backtest(code_path, output_dir)
        
        if metrics:
            print("\n✅ Backtest completed successfully!")
            print("\n--- Generating Reports ---")
            generate_tearsheet(output_dir, strategy_dict, metrics)
            break
        else:
            print(f"❌ Backtest failed on attempt {attempt + 1}. Attempting to self-heal...")
            attempt += 1
            
    if not metrics:
        print(f"\n❌ Reached {MAX_RETRIES} retries. Could not auto-fix the script completely.")

if __name__ == "__main__":
    main()
