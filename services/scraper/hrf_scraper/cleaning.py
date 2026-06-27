"""Text normalization for scraped content.

Light mode (default) preserves casing and digits so embeddings retain
information like "COVID-19", dosages, and p-values. Aggressive mode reproduces
the legacy clean.py behavior (lowercase, strip punctuation and digits) and is
opt-in only.
"""

from __future__ import annotations

import re
import unicodedata

_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoji
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators
    "]",
    flags=re.UNICODE,
)

# Contractions expanded in both modes (superset of legacy clean.py).
_CONTRACTIONS = {
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bdoesn't\b": "does not",
    r"\bdon't\b": "do not",
    r"\bdidn't\b": "did not",
    r"\bwon't\b": "will not",
    r"\bcan't\b": "cannot",
}


def _strip_control_chars(text: str) -> str:
    return "".join(
        ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )


def clean_text(text: str, *, aggressive: bool = False) -> str:
    """Normalize text. See module docstring for light vs aggressive modes."""
    if not text:
        return ""

    t = unicodedata.normalize("NFKC", text)
    t = _EMOJI.sub("", t)
    for pattern, repl in _CONTRACTIONS.items():
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    t = _strip_control_chars(t)

    if aggressive:
        t = t.lower()
        t = re.sub(r"[^\w\s]", " ", t)  # strip punctuation
        t = re.sub(r"\d+", " ", t)  # strip digits

    t = re.sub(r"\s+", " ", t).strip()
    return t
