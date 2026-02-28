"""
Application configuration — dataclasses + YAML loader.

Load order:
  1. Defaults baked into the dataclasses
  2. config/defaults.yaml (or any path passed to load_config)
  3. Caller overrides (e.g. CLI flags, API request body)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class LLMConfig:
    provider: str = "openai"                                      # "openai" | "bedrock"
    model: Optional[str] = None                                   # None = use provider default
    temperature: float = 0.2


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AgentConfig:
    max_iterations: int = 10


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def load_config(path: str = "config/defaults.yaml") -> AppConfig:
    """
    Load AppConfig from a YAML file, falling back to dataclass defaults
    for any keys that are missing.

    Args:
        path: Path to the YAML config file. If the file doesn't exist,
              default values are used without error.

    Returns:
        A fully populated AppConfig instance.
    """
    if not os.path.exists(path):
        return AppConfig()

    with open(path) as f:
        data: dict = yaml.safe_load(f) or {}

    llm_data = data.get("llm", {})
    server_data = data.get("server", {})
    agent_data = data.get("agent", {})

    return AppConfig(
        llm=LLMConfig(**llm_data),
        server=ServerConfig(**server_data),
        agent=AgentConfig(**agent_data),
    )
