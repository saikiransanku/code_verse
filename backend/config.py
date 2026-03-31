"""Configuration loader for backend runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(ENV_PATH)


def _resolve_model_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (BACKEND_DIR / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    blur_threshold: float = float(os.getenv("BLUR_THRESHOLD", "80"))
    brightness_threshold: float = float(os.getenv("BRIGHTNESS_THRESHOLD", "40"))
    model_path: Path = _resolve_model_path(os.getenv("MODEL_PATH", "../model/saved_model.pth"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

