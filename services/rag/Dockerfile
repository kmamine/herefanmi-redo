# RAG service + ingestion. Build context must be the repo root.
# This image is "data-capable": it serves RAG retrieval AND can run the
# ingestion/seed scripts (which need the scraper storage + ingest glue).
FROM python:3.11-slim
WORKDIR /app

COPY shared ./shared
COPY services/rag/requirements.txt ./rag-requirements.txt
COPY services/scraper/requirements.txt ./scraper-requirements.txt
RUN pip install --no-cache-dir -r rag-requirements.txt -r scraper-requirements.txt \
    && pip install --no-cache-dir -e ./shared

# Pre-bake the embedding model so the first query isn't a slow download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY services/rag/hrf_rag ./hrf_rag
COPY services/scraper/hrf_scraper ./hrf_scraper
COPY ingest ./ingest

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 5000
CMD ["uvicorn", "hrf_rag.main:app", "--host", "0.0.0.0", "--port", "5000"]
