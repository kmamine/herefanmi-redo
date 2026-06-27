"""Source-URL validation (recreates the legacy ClearSources, async + mockable).

Each candidate URL is probed; unreachable or error URLs are dropped. The
checker is injectable so tests never make network calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

Checker = Callable[[str], Awaitable[bool]]


async def _default_checker(url: str, timeout: float = 5.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.head(url)
            if resp.status_code >= 400:  # some servers reject HEAD; retry GET
                resp = await client.get(url)
            return resp.status_code < 400
    except httpx.HTTPError:
        return False


async def validate_sources(urls: list[str], *, checker: Checker = _default_checker) -> list[str]:
    """Return the subset of (deduped) URLs that are reachable."""
    unique: list[str] = []
    for url in urls:
        if url and url not in unique:
            unique.append(url)
    if not unique:
        return []
    results = await asyncio.gather(*(checker(u) for u in unique))
    return [u for u, ok in zip(unique, results, strict=True) if ok]
