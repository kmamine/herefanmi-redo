"""ChromaDB client/collection helpers shared by the ingestion path and RAG.

The same code runs in three modes selected by ``settings.chroma_mode``:
- ``ephemeral``  — in-memory, used by tests
- ``persistent`` — local on-disk store, used for single-process local dev
- ``http``       — connect to a ChromaDB server container (the compose default,
  so the Scraper/ingestion writer and the RAG reader share one store safely)
"""

from __future__ import annotations

from typing import Any

from hrf_shared.config import Settings


def get_embedding_function(settings: Settings) -> Any:
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embed_model)


def get_chroma_client(settings: Settings) -> Any:
    import chromadb

    if settings.chroma_mode == "ephemeral":
        return chromadb.EphemeralClient()
    if settings.chroma_mode == "http":
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return chromadb.PersistentClient(path=settings.chroma_path)


def get_or_create_collection(
    client: Any, settings: Settings, embedding_function: Any | None = None
) -> Any:
    ef = embedding_function or get_embedding_function(settings)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
