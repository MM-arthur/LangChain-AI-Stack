# Core LLM utilities - centralized LLM initialization + retry logic

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_community.chat_models.moonshot import MoonshotChat

load_dotenv()

OUTPUT_TOKEN_INFO = {
    "moonshot-v1-8k": {"max_tokens": 8000},
    "moonshot-v1-32k": {"max_tokens": 32000},
    "moonshot-v1-128k": {"max_tokens": 128000},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
}


def init_llm(model_name: str = None) -> MoonshotChat:
    """
    Initialize Moonshot LLM with model config.
    No retry here — retry is handled at call sites via with_retry().
    """
    if model_name is None:
        model_name = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")

    return MoonshotChat(
        model=model_name,
        temperature=0.1,
        max_tokens=OUTPUT_TOKEN_INFO.get(model_name, {}).get("max_tokens", 8000),
        api_key=os.getenv("MOONSHOT_API_KEY"),
    )


def get_default_model() -> str:
    return os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")