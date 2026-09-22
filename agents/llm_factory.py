"""
LLM factory — returns a LangChain-compatible chat model from a tool config.

Accepts any config object that has: provider (str), model (Optional[str]),
temperature (float). Both ChatAgentConfig and ImageAnalyzerConfig qualify.

Supported providers:
  openai     — requires OPENAI_API_KEY env var
  anthropic  — requires ANTHROPIC_API_KEY env var
  google     — requires GOOGLE_API_KEY env var
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

# Default models used when config.model is None
_CHAT_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.0-flash-lite",
}

_VISION_DEFAULTS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-3.6-flash",
}


def create_llm(config) -> BaseChatModel:
    """
    Create a chat LLM from a tool config (ChatAgentConfig or similar).
    Uses config.model if set, otherwise falls back to _CHAT_DEFAULTS.
    """
    provider = config.provider
    model = config.model or _CHAT_DEFAULTS.get(provider)
    temperature = config.temperature
    return _build(provider, model, temperature)


def create_vision_llm(config) -> BaseChatModel:
    """
    Create a vision-capable LLM from a tool config (ImageAnalyzerConfig or similar).
    Uses config.model if set, otherwise falls back to _VISION_DEFAULTS.
    """
    provider = config.provider
    model = config.model or _VISION_DEFAULTS.get(provider)
    temperature = config.temperature
    return _build(provider, model, temperature)


def _build(provider: str, model: str, temperature: float) -> BaseChatModel:
    if provider == "openai":
        return _openai(model, temperature)
    elif provider == "anthropic":
        return _anthropic(model, temperature)
    elif provider == "google":
        return _google(model, temperature)
    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Choose 'openai', 'anthropic', or 'google'."
        )


def _openai(model: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def _anthropic(model: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)


def _google(model: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set.")

    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
