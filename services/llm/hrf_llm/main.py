"""LLM service FastAPI app.

POST /predict  {chunks, question} -> Classification JSON
GET  /health   liveness (does not call the model)
"""

from __future__ import annotations

from functools import lru_cache

import openai
from fastapi import Depends, FastAPI, HTTPException
from hrf_shared.config import Settings, get_settings
from hrf_shared.contracts import Classification, PredictRequest
from hrf_shared.json_utils import ClassificationParseError

from hrf_llm.classifier import classify
from hrf_llm.client import get_llm_client

app = FastAPI(title="HeReFaNMi LLM Service")


@lru_cache
def _cached_client() -> openai.OpenAI:
    return get_llm_client(get_settings())


def get_client() -> openai.OpenAI:
    """Dependency returning the OpenAI-compatible client (overridable in tests)."""
    return _cached_client()


def settings_dep() -> Settings:
    return get_settings()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Classification)
def predict(
    body: PredictRequest,
    client=Depends(get_client),
    settings: Settings = Depends(settings_dep),
) -> Classification:
    try:
        return classify(body.chunks, body.question, client=client, settings=settings)
    except ClassificationParseError as exc:
        raise HTTPException(
            status_code=502, detail=f"LLM produced unparseable output: {exc}"
        ) from exc
    except openai.OpenAIError as exc:  # upstream/provider failure
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc
