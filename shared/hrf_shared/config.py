"""Central configuration for all HeReFaNMi services (env-prefixed ``HRF_``).

Loaded once at startup via :func:`get_settings`. Because the LLM provider is
just ``base_url``/``model``/``api_key``, switching from the local vLLM Gemma
server to OpenAI or any OpenAI-compatible endpoint is purely an env change.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ChromaMode = Literal["ephemeral", "persistent", "http"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HRF_", env_file=".env", extra="ignore")

    # ---- LLM provider (OpenAI-compatible) ----
    llm_base_url: str = "http://localhost:50033/v1"
    llm_model: str = "google/gemma-4-E4B-it"
    llm_api_key: str = "dummy-key"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: float = 60.0

    # ---- ChromaDB ----
    chroma_mode: ChromaMode = "persistent"
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_path: str = "data/chroma"
    chroma_collection: str = "health_chunks"

    # ---- Embeddings ----
    embed_model: str = "all-MiniLM-L6-v2"

    # ---- Storage ----
    sqlite_path: str = "data/herefanmi.sqlite"

    # ---- Retrieval ----
    default_top_n: int = 5
    rrf_k: int = 60
    dense_candidates: int = 20

    # ---- Scraper scheduling ----
    scraper_autostart: bool = True  # seed default sources on startup
    scheduler_enabled: bool = False  # start the APScheduler tick
    scheduler_tick_seconds: int = 300
    scrape_limit: int | None = None  # max articles per source per run (None = all)

    # ---- Admin ----
    admin_emails: str = ""  # comma-separated allowlist; matched case-insensitively
    scraper_url: str = "http://localhost:8001"

    # ---- Auth (backend) ----
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    admin_uid: str = "admin"
    signup_points: int = 14

    # ---- Inter-service URLs (backend → rag/llm) ----
    rag_url: str = "http://localhost:5000"
    llm_url: str = "http://localhost:5002"

    def admin_email_set(self) -> set[str]:
        """Lowercased set of admin emails from the comma-separated allowlist."""
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (override the cache in tests)."""
    return Settings()
