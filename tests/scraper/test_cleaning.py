"""TDD for hrf_scraper.cleaning — text normalization.

Light mode is the default (preserve casing/digits so embeddings keep "COVID-19",
dosages, p-values). Aggressive mode reproduces the legacy clean.py behavior.
"""

from hrf_scraper.cleaning import clean_text


def test_clean_collapses_whitespace():
    assert clean_text("hello   world\n\n\t  again") == "hello world again"


def test_clean_strips_leading_trailing():
    assert clean_text("   padded   ") == "padded"


def test_clean_removes_emojis():
    assert "😀" not in clean_text("great news 😀 today")
    assert clean_text("great news 😀 today") == "great news today"


def test_clean_expands_contractions():
    assert clean_text("it isn't safe") == "it is not safe"
    assert clean_text("studies haven't shown") == "studies have not shown"


def test_clean_light_mode_preserves_casing_and_digits():
    text = "COVID-19 vaccine is 95% effective (p<0.05)"
    out = clean_text(text)
    assert "COVID-19" in out
    assert "95" in out
    assert "0.05" in out


def test_clean_aggressive_mode_lowercases_and_strips_digits():
    out = clean_text("COVID-19 Vaccine 95% Effective", aggressive=True)
    assert out == out.lower()
    assert not any(ch.isdigit() for ch in out)


def test_clean_handles_empty():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
