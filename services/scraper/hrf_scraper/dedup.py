"""Content-based deduplication helpers."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """Stable sha256 hex digest of the trimmed text (used to detect duplicates)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
