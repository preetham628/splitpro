"""
Application configuration — dataclasses + YAML loader.

Each tool has its own config class so provider/model/temperature can be
tuned independently (e.g. cheap fast model for chat, best vision model for images).

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


# ---------------------------------------------------------------------------
# Tool-specific configs
# ---------------------------------------------------------------------------

@dataclass
class ChatAgentConfig:
    """Config for the conversational bill-splitting agent."""
    provider: str = "anthropic"
    model: Optional[str] = None    # None → llm_factory picks provider default
    temperature: float = 0.2
    max_iterations: int = 10       # max LLM calls per user turn


@dataclass
class ImageAnalyzerConfig:
    """Config for the image analysis tool (must point to a vision-capable model)."""
    provider: str = "google"
    model: Optional[str] = None    # None → llm_factory picks provider vision default
    temperature: float = 0.1       # lower = more deterministic OCR extraction


# ---------------------------------------------------------------------------
# Infrastructure configs
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Auth config
# ---------------------------------------------------------------------------

@dataclass
class AuthConfig:
    google_client_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_ID", ""))
    google_client_secret: str = field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_SECRET", ""))
    google_redirect_uri: str = field(default_factory=lambda: os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback"))
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", ""))
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080   # 7 days
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "splitpro.db"))


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    chat_agent: ChatAgentConfig = field(default_factory=ChatAgentConfig)
    image_analyzer: ImageAnalyzerConfig = field(default_factory=ImageAnalyzerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/defaults.yaml") -> AppConfig:
    """
    Load AppConfig from a YAML file, falling back to dataclass defaults
    for any keys that are missing or if the file doesn't exist.
    """
    if not os.path.exists(path):
        return AppConfig()

    with open(path) as f:
        data: dict = yaml.safe_load(f) or {}

    return AppConfig(
        chat_agent=ChatAgentConfig(**data.get("chat_agent", {})),
        image_analyzer=ImageAnalyzerConfig(**data.get("image_analyzer", {})),
        server=ServerConfig(**data.get("server", {})),
        auth=AuthConfig(**data.get("auth", {})),
    )
