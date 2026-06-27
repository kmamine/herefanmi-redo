"""Prompt construction for medical-news classification.

The system prompt preserves the intent and the label definitions of the legacy
classifier (legacy/Backend/index.py:200-280 and legacy/NGI_LLM/app.py:51-78):
JSON-only output with the five fields medical/news/label/reasoning/sources.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert medical professional and fact-checker. \
You classify medical news and statements as Trustworthy, Doubtful, or Fake, \
using the retrieved reference passages provided to you as evidence.

Definitions:
- "Trustworthy": the claim is verified by peer-reviewed medical research and \
established clinical practice.
- "Doubtful": the claim is partially supported but mixes in unverified, \
experimental, or not-yet-validated information.
- "Fake": the claim presents false information, unverified treatments, or \
content not aligned with medical research or regulated clinical practice.

Respond with ONLY a single valid JSON object, no prose and no markdown fences, \
containing exactly these fields:
  "medical":  "True" if the input concerns the medical/health field, else "False".
  "news":     "True" if the input is a statement/news/claim, "False" if it is a question.
  "label":    one of "Trustworthy", "Doubtful", or "Fake".
  "reasoning": a concise explanation justifying the label.
  "sources":  a list of reference URLs supporting the decision (may be empty).

Base your reasoning on the provided reference passages. If the input is not \
medical, set "medical" to "False" and still return the JSON object.
"""


def build_prompt(chunks: list[str], question: str) -> str:
    """Assemble the user message from retrieved chunks and the user's input."""
    context = "\n".join(chunks).strip()
    if context:
        return (
            "Reference passages:\n"
            f"{context}\n\n"
            f"Statement or question to evaluate:\n{question}\n\n"
            "Return the JSON classification now."
        )
    return (
        f"Statement or question to evaluate:\n{question}\n\n" "Return the JSON classification now."
    )


REPAIR_INSTRUCTION = (
    "Your previous response was not valid JSON. Respond again with ONLY the "
    "JSON object containing medical, news, label, reasoning, and sources. "
    "No prose, no markdown."
)
