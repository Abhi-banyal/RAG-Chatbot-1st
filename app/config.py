from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
LOCAL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INGESTION_DIRS = (DATA_DIR, UPLOADS_DIR)

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

AZURE_ENV_VARS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
)


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def validate_azure_settings() -> dict[str, str]:
    load_environment()
    return {name: get_required_env(name) for name in AZURE_ENV_VARS}


def get_frontend_origins() -> list[str]:
    load_environment()
    raw_value = os.getenv("FRONTEND_ORIGINS", "").strip()
    if not raw_value:
        return list(DEFAULT_FRONTEND_ORIGINS)

    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or list(DEFAULT_FRONTEND_ORIGINS)
