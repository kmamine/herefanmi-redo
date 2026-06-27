"""Synchronous orchestration: RAG -> LLM -> validate sources -> persist.

This replaces the legacy fire-and-forget callback chain (RAG -> LLM -> Backend
/response polled via a global slot). Here the Backend awaits each step.
"""

from __future__ import annotations

import httpx
from hrf_shared.config import Settings
from hrf_shared.json_utils import ClassificationParseError

NON_MEDICAL_MSG = "Please try to ask something related to the medical field!"
FALLBACK_MSG = (
    "We could not complete your request because the analysis service is "
    "temporarily unavailable. Please try again later."
)


class OutOfPointsError(Exception):
    """Raised when a non-admin user has no remaining query points."""


class Orchestrator:
    def __init__(self, *, rag_client, llm_client, repo, source_validator, settings: Settings):
        self.rag = rag_client
        self.llm = llm_client
        self.repo = repo
        self.source_validator = source_validator
        self.settings = settings

    async def handle(self, *, data: str, opinion: str, uid: str, backend: str) -> dict:
        user = self.repo.get_user_by_uid(uid)
        is_admin = bool(user and user.is_admin)

        if not is_admin and (user is None or user.points <= 0):
            raise OutOfPointsError("No remaining query points")

        chunks = await self.rag.find_similar(data, self.settings.default_top_n)

        try:
            classification = await self.llm.predict([c.text for c in chunks], data)
        except (httpx.HTTPError, ClassificationParseError):
            key = self.repo.create_query(
                uid=uid,
                question=data,
                opinion=opinion,
                medical="True",
                news="False",
                label="Doubtful",
                response=FALLBACK_MSG,
            )
            # Failed attempt: do not charge a point.
            return {
                "data": FALLBACK_MSG,
                "news": "False",
                "label": "Doubtful",
                "source": [],
                "key": key,
            }

        if classification.medical == "False":
            key = self.repo.create_query(
                uid=uid,
                question=data,
                opinion=opinion,
                medical="False",
                news=classification.news,
                label=classification.label,
                response=classification.reasoning,
            )
            if not is_admin:
                self.repo.decrement_points(uid)
            return {"data": NON_MEDICAL_MSG, "key": key}

        # Grounded sources: prefer retrieved chunk URLs, then model-cited URLs.
        candidates: list[str] = []
        for c in chunks:
            if c.url and c.url not in candidates:
                candidates.append(c.url)
        for s in classification.sources:
            if s and s not in candidates:
                candidates.append(s)
        valid_sources = await self.source_validator(candidates)

        key = self.repo.create_query(
            uid=uid,
            question=data,
            opinion=opinion,
            medical="True",
            news=classification.news,
            label=classification.label,
            response=classification.reasoning,
        )
        if not is_admin:
            self.repo.decrement_points(uid)

        return {
            "data": classification.reasoning,
            "news": classification.news,
            "label": classification.label,
            "source": valid_sources,
            "key": key,
        }
