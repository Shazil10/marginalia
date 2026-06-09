import os
import sys
import json
import re

# Add the root 'agentic-quant' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.llm_client import get_llm_client

INTAKE_SYSTEM_PROMPT = """You are an expert financial intake specialist for an autonomous quant fund.
Your job is to convert freeform natural language from a user into a structured risk profile JSON.

Extract the following fields from the user's message:
- `max_drawdown_tolerance`: (float 0.0 to 1.0) The maximum acceptable loss.
- `investment_horizon_years`: (integer)
- `capital`: (float) The amount in dollars.
- `target_annual_return`: (float 0.0 to 1.0)
- `risk_class`: (string) Must be one of: "conservative", "moderate-conservative", "moderate", "aggressive".
- `excluded_sectors`: (list of strings) e.g., ["oil", "weapons"]
- `benchmark`: (string) Default is "SPY".
- `clarification_needed`: (string or null) Set to a clarifying question ONLY if the user's input is completely ambiguous. Otherwise, null.

APPLY THIS STRICT MAPPING LOGIC:
- Language suggesting fear of loss or crash -> "conservative" (Drawdown: 0.10, Target Return: 0.07)
- "Some risk is okay" -> "moderate-conservative" (Drawdown: 0.20, Target Return: 0.10)
- Wanting growth and tolerating swings -> "moderate" (Drawdown: 0.30, Target Return: 0.13)
- Wanting maximum returns regardless of loss -> "aggressive" (Drawdown: 0.50, Target Return: 0.18)

DEFAULTS:
- If capital is not mentioned: 10000
- If horizon is not mentioned: 10
- Infer target return and max drawdown from the risk class mapping if not explicitly requested.

Return ONLY valid JSON block. No preamble, no explanation, no markdown fences outside the JSON.
Format:
{
  "capital": 10000,
  "investment_horizon_years": 10,
  "risk_class": "moderate",
  "max_drawdown_tolerance": 0.30,
  "target_annual_return": 0.13,
  "excluded_sectors": [],
  "benchmark": "SPY",
  "clarification_needed": null
}
"""

def parse_json_response(response_text: str) -> dict:
    """Safely parse the LLM output into a dictionary."""
    # Strip markdown if present
    text = response_text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    
    # Simple regex to find the json block if there's surrounding text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
        
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM: {str(e)}\nRaw Response:\n{response_text}")

def run_intake_agent(client, model: str, user_input: str) -> dict:
    """
    Takes natural language user input and returns a structured risk profile dict.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1, # Keep it low for structured data extraction
    )
    
    raw_response = response.choices[0].message.content
    return parse_json_response(raw_response)

if __name__ == "__main__":
    # Test script for the Intake Agent
    try:
        client, model = get_llm_client()
        
        test_inputs = [
            "I have $25k saved up for a house in 5 years. I want to grow it steadily but I really can't afford to lose more than 10 percent of it. No oil companies.",
            "I only have two grand right now. Put it all into the most aggressive thing you can find. Let it ride, I don't care if I lose it. No ESG stuff.",
            "Make me some money please."
        ]
        
        for i, user_text in enumerate(test_inputs):
            print(f"\n--- Test Case {i+1} ---")
            print(f"User Input: \"{user_text}\"")
            result = run_intake_agent(client, model, user_text)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error testing Intake Agent: {e}")
