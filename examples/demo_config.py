"""OpenRouter-backed HippoRAG config and env helpers for the demo."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def make_demo_config(save_dir: str):
    from hipporag.utils.config_utils import BaseConfig

    return BaseConfig(
        save_dir=save_dir,
        llm_name="openai/gpt-4o-mini",
        llm_base_url=OPENROUTER_BASE,
        embedding_model_name="openai/text-embedding-3-small",
        embedding_base_url=OPENROUTER_BASE,
        retrieval_top_k=20,
        linking_top_k=5,
    )


def load_demo_env() -> None:
    """Load `.env` from repo root and require OPENAI_API_KEY (OpenRouter alias)."""
    load_dotenv(_REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your "
            "OpenRouter key (https://openrouter.ai/keys)."
        )
