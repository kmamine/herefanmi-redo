"""TDD for app.orchestrator — synchronous RAG -> LLM -> validate -> persist."""

import pytest
from app.orchestrator import OutOfPointsError
from hrf_shared.contracts import Classification

from tests.backend.conftest import FakeLLM, FakeRAG, chunk, make_orchestrator

TRUSTWORTHY = Classification(
    medical="True",
    news="True",
    label="Trustworthy",
    reasoning="Supported by peer-reviewed evidence.",
    sources=["https://model.example/x"],
)
NON_MEDICAL = Classification(
    medical="False",
    news="False",
    label="Doubtful",
    reasoning="Not a medical topic.",
    sources=[],
)
CHUNKS = [
    chunk("Aspirin reduces heart attack risk.", "cdc", "https://cdc.gov/aspirin"),
    chunk("Daily aspirin guidance.", "nhs", "https://nhs.uk/aspirin"),
]


async def test_medical_response_shape(repo, settings):
    user = repo.create_user("u@e.com", "h", points=14)
    orch = make_orchestrator(repo, FakeRAG(CHUNKS), FakeLLM(TRUSTWORTHY), settings)
    out = await orch.handle(data="is aspirin good?", opinion="3", uid=user.uid, backend="x")
    assert out["label"] == "Trustworthy"
    assert out["news"] == "True"
    assert out["data"] == "Supported by peer-reviewed evidence."
    assert "key" in out
    # sources include grounded chunk URLs
    assert "https://cdc.gov/aspirin" in out["source"]
    # query persisted
    assert repo.get_query(out["key"]).label == "Trustworthy"


async def test_non_medical_response(repo, settings):
    user = repo.create_user("u@e.com", "h", points=14)
    orch = make_orchestrator(repo, FakeRAG(CHUNKS), FakeLLM(NON_MEDICAL), settings)
    out = await orch.handle(data="who won the cup?", opinion="0", uid=user.uid, backend="x")
    assert "medical field" in out["data"]
    assert "label" not in out
    assert "key" in out


async def test_sources_are_validated(repo, settings):
    user = repo.create_user("u@e.com", "h", points=14)

    async def only_cdc(urls):
        return [u for u in urls if "cdc.gov" in u]

    orch = make_orchestrator(
        repo, FakeRAG(CHUNKS), FakeLLM(TRUSTWORTHY), settings, validator=only_cdc
    )
    out = await orch.handle(data="q", opinion="0", uid=user.uid, backend="x")
    assert out["source"] == ["https://cdc.gov/aspirin"]


async def test_points_decremented_on_query(repo, settings):
    user = repo.create_user("u@e.com", "h", points=14)
    orch = make_orchestrator(repo, FakeRAG(CHUNKS), FakeLLM(TRUSTWORTHY), settings)
    await orch.handle(data="q", opinion="0", uid=user.uid, backend="x")
    assert repo.get_user_by_uid(user.uid).points == 13


async def test_out_of_points_blocks(repo, settings):
    user = repo.create_user("u@e.com", "h", points=0)
    orch = make_orchestrator(repo, FakeRAG(CHUNKS), FakeLLM(TRUSTWORTHY), settings)
    with pytest.raises(OutOfPointsError):
        await orch.handle(data="q", opinion="0", uid=user.uid, backend="x")


async def test_admin_unlimited_and_not_decremented(repo, settings):
    admin = repo.create_user("admin@e.com", "h", points=0, is_admin=True)
    orch = make_orchestrator(repo, FakeRAG(CHUNKS), FakeLLM(TRUSTWORTHY), settings)
    out = await orch.handle(data="q", opinion="0", uid=admin.uid, backend="x")
    assert out["label"] == "Trustworthy"
    assert repo.get_user_by_uid(admin.uid).points == 0  # unchanged


async def test_llm_down_returns_graceful_fallback(repo, settings, llm_down_error):
    user = repo.create_user("u@e.com", "h", points=14)
    orch = make_orchestrator(repo, FakeRAG(CHUNKS), FakeLLM(error=llm_down_error), settings)
    out = await orch.handle(data="q", opinion="0", uid=user.uid, backend="x")
    assert out["label"] == "Doubtful"
    assert "key" in out
    # not charged for a failed attempt
    assert repo.get_user_by_uid(user.uid).points == 14
