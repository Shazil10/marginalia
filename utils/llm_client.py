import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm_client():
    """
    Returns a configured OpenAI-compatible client and the optimal model name.
    Uses OpenRouter for access to Claude models as the global SOTA for this task.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    client = OpenAI(
        api_key=openrouter_key, 
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Claude 4.6 Opus for complex document extraction and Python coding.
    model = "anthropic/claude-opus-4.6"
    
    return client, model
