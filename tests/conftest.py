"""Root pytest fixtures shared across all service test suites."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"

# Use a project-local, writable HuggingFace cache for tests. The machine's
# default HF cache may be a read-only shared path with an inconsistent refs/
# pointer, so we override it here (before sentence-transformers is imported).
_HF_CACHE = REPO_ROOT / ".hf_cache"
os.environ["HF_HOME"] = str(_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(_HF_CACHE / "hub")


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to tests/fixtures (HTML samples, canned LLM outputs)."""
    return FIXTURES


@pytest.fixture
def llm_fixture(fixtures_dir: Path):
    """Return the text content of a file under tests/fixtures/llm/."""

    def _load(name: str) -> str:
        return (fixtures_dir / "llm" / name).read_text(encoding="utf-8")

    return _load


@pytest.fixture
def html_fixture(fixtures_dir: Path):
    """Return the text content of a file under tests/fixtures/html/."""

    def _load(name: str) -> str:
        return (fixtures_dir / "html" / name).read_text(encoding="utf-8")

    return _load
