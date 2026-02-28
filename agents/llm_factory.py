"""
LLM factory — returns a LangChain-compatible chat model from an LLMConfig.

Supported providers:
  openai   — requires OPENAI_API_KEY env var
  bedrock  — requires AWS credentials and langchain-aws installed
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from config import LLMConfig

_PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


def create_llm(config: LLMConfig) -> BaseChatModel:
    """
    Create and return a LangChain chat model from an LLMConfig.

    Args:
        config: LLMConfig instance with provider, model, and temperature.

    Returns:
        A BaseChatModel that supports .bind_tools().
    """
    provider = config.provider
    model = config.model or _PROVIDER_DEFAULTS.get(provider)
    temperature = config.temperature

    if provider == "openai":
        return _openai(model, temperature)
    elif provider == "bedrock":
        return _bedrock(model, temperature)
    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Choose 'openai' or 'bedrock'."
        )


def _openai(model: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def _bedrock(model: str, temperature: float) -> BaseChatModel:
    try:
        from langchain_aws import ChatBedrock
    except ImportError:
        raise ImportError(
            "langchain-aws is required for Bedrock support. "
            "Install it with: pip install langchain-aws"
        )

    return ChatBedrock(
        model_id=model,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"temperature": temperature},
    )
