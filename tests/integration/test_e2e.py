"""In-process end-to-end test of the core pipeline (no Docker, no live vLLM).

Exercises: sample articles -> chunk -> embed -> ChromaDB -> hybrid retrieval ->
orchestrator (RAG + stubbed LLM) -> SQLite persistence. The LLM is stubbed
because a real vLLM server isn't available in CI.
"""

import uuid

import chromadb
import pytest
from app.db import Repository, init_engine
from app.orchestrator import Orchestrator
from hrf_rag.retriever import find_similar_chunks
from hrf_shared.config import Settings
from hrf_shared.contracts import Classification

from ingest.indexing import index_articles
from ingest.sample_articles import sample_articles

pytestmark = pytest.mark.integration


class ChromaRagAdapter:
    """Adapts the in-process RAG retriever to the orchestrator's client API."""

    def __init__(self, collection, settings):
        self.collection = collection
        self.settings = settings

    async def find_similar(self, prompt, top_n=5):
        return find_similar_chunks(
            prompt, top_n, collection=self.collection, settings=self.settings
        )


class StubLLM:
    """Returns a verdict derived from the retrieved evidence (no real model)."""

    def __init__(self, classification):
        self.classification = classification

    async def predict(self, chunks, question):
        return self.classification


async def allow_all(urls):
    return list(urls)


@pytest.fixture(scope="module")
def embed_fn():
    from hrf_shared.chroma_client import get_embedding_function

    return get_embedding_function(Settings())


@pytest.fixture
def collection(embed_fn):
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"e2e_{uuid.uuid4().hex}",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


async def test_full_pipeline_classifies_grounded_claim(collection, tmp_path):
    settings = Settings(default_top_n=3, admin_uid="admin")

    # 1. Ingest bundled sample articles into ChromaDB.
    indexed = index_articles(sample_articles(), collection)
    assert indexed >= 6

    # 2. Sanity-check retrieval surfaces the relevant source.
    chunks = find_similar_chunks(
        "Does vitamin D help bone health?", 3, collection=collection, settings=settings
    )
    assert chunks
    assert any("nhs" in c.url for c in chunks)

    # 3. Wire the orchestrator: real RAG over Chroma + stubbed LLM verdict.
    repo = Repository(init_engine(f"sqlite:///{tmp_path / 'e2e.sqlite'}"))
    user = repo.create_user("e2e@test.com", "h", points=14)
    verdict = Classification(
        medical="True",
        news="False",
        label="Trustworthy",
        reasoning="Vitamin D supports calcium absorption and bone health.",
        sources=[],
    )
    orch = Orchestrator(
        rag_client=ChromaRagAdapter(collection, settings),
        llm_client=StubLLM(verdict),
        repo=repo,
        source_validator=allow_all,
        settings=settings,
    )

    # 4. Run a full /medicalTalk-style request.
    result = await orch.handle(
        data="Does vitamin D help bone health?", opinion="4", uid=user.uid, backend="HeReFaNMi LLM"
    )

    assert result["label"] == "Trustworthy"
    assert result["source"]  # grounded source URLs from retrieved chunks
    assert any("nhs" in s for s in result["source"])
    assert repo.get_query(result["key"]).label == "Trustworthy"
    assert repo.get_user_by_uid(user.uid).points == 13  # charged one point
