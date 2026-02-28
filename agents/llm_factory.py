"""
LLM factory — returns a LangChain-compatible chat model for the given provider.

Supported providers:
  openai   — requires OPENAI_API_KEY
  bedrock  — requires AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
             or an IAM role) and langchain-aws installed

Set LLM_PROVIDER env var to switch the default, or pass provider explicitly.
"""

from __future__ import annotations

import os
from langchain_core.language_models import BaseChatModel

# Default models per provider
_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


def create_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """
    Create and return a chat model for the given provider.

    Args:
        provider: "openai" or "bedrock". Falls back to LLM_PROVIDER env var,
                  then defaults to "openai".
        model:    Model name/ID override. Falls back to provider-specific env var,
                  then a sensible default.

    Returns:
        A LangChain BaseChatModel that supports .bind_tools().
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        return _openai(model)
    elif provider == "bedrock":
        return _bedrock(model)
    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Choose 'openai' or 'bedrock'."
        )


def _openai(model: str | None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", _DEFAULTS["openai"]),
        temperature=0.2,
        api_key=api_key,
    )


def _bedrock(model: str | None) -> BaseChatModel:
    try:
        from langchain_aws import ChatBedrock
    except ImportError:
        raise ImportError(
            "langchain-aws is required for Bedrock support. "
            "Install it with: pip install langchain-aws"
        )

    return ChatBedrock(
        model_id=model or os.getenv("BEDROCK_MODEL", _DEFAULTS["bedrock"]),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"temperature": 0.2},
    )
