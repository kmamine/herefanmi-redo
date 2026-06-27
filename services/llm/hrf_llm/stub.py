"""A tiny OpenAI-compatible chat-completions stub.

Lets the full stack run without a GPU / real vLLM (CI smoke tests). It inspects
the prompt and returns a plausible classification JSON. Run with:
    uvicorn hrf_llm.stub:app --host 0.0.0.0 --port 50033
and point HRF_LLM_BASE_URL at http://<host>:50033/v1
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="OpenAI-compatible stub")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None


def verdict_for(text: str) -> dict:
    low = text.lower()
    if any(w in low for w in ("miracle", "100% cure", "doctors hate", "secret cure")):
        label, reasoning = "Fake", "Contains unsupported miracle-cure claims."
    elif any(w in low for w in ("might", "could", "experimental", "unproven")):
        label, reasoning = "Doubtful", "Mixes some valid information with unverified claims."
    else:
        label, reasoning = "Trustworthy", "Consistent with the provided medical references."
    medical = "False" if "football" in low or "weather" in low else "True"
    return {
        "medical": medical,
        "news": "True",
        "label": label,
        "reasoning": reasoning,
        "sources": [],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict:
    user_text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    content = json.dumps(verdict_for(user_text))
    return {
        "id": "stub-1",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
