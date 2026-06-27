# Deployment

The whole stack is dockerized in [`docker-compose.yml`](../docker-compose.yml):
`chromadb`, `scraper`, `rag`, `llm`, `backend`, `frontend`, plus a one-off
`seed` job and an optional `llm-stub` (CI profile).

## Prerequisites

- Docker + Docker Compose.
- For real classification: a vLLM (or other OpenAI-compatible) server reachable
  from the `llm` container. By default the compose file points the LLM service
  at `http://host.docker.internal:50033/v1` and adds a `host-gateway` mapping so
  the container can reach a server on the **host**.

## Bring up the stack

```bash
# Against a host vLLM Gemma on :50033
docker compose up --build

# GPU-free: run the bundled OpenAI-compatible stub instead of vLLM
HRF_LLM_BASE_URL=http://llm-stub:50033/v1 docker compose --profile ci up --build
```

Services and host ports: frontend `3000`, backend `10000`, rag `5000`, llm
`5002`, scraper `8001`, chromadb `8000`. The frontend is served by nginx and
proxies `/api` → `backend:10000`.

### LLM → host vLLM

The `llm` service is configured with:

```yaml
environment:
  - HRF_LLM_BASE_URL=${HRF_LLM_BASE_URL:-http://host.docker.internal:50033/v1}
  - HRF_LLM_MODEL=${HRF_LLM_MODEL:-google/gemma-4-E4B-it}
  - HRF_LLM_API_KEY=${HRF_LLM_API_KEY:-dummy-key}
extra_hosts:
  - "host.docker.internal:host-gateway"
```

On Linux the `host-gateway` mapping lets the container reach a vLLM server on
the host. Override any of those env vars at `docker compose up` time. If your
model is actually `gemma-3n-E4B-it`, set `HRF_LLM_MODEL` accordingly — nothing
else changes.

### ChromaDB as a server

`rag` and `scraper`/`seed` talk to the `chromadb` container over HTTP
(`HRF_CHROMA_MODE=http`, `HRF_CHROMA_HOST=chromadb`). A named volume persists the
store. This keeps a single writer (ingestion) and reader (RAG) safely sharing
one store — never open an embedded persistent Chroma from two processes.

## Seeding

The knowledge base starts empty. Seed it so retrieval and the admin KB stats
have content.

```bash
# Bundled sample articles (CDC/NHS/MedlinePlus/MedPageToday/WebMD/Healthline)
docker compose run --rm seed
```

`seed` runs [`ingest/seed_sample_data.py`](../ingest/seed_sample_data.py) using
the data-capable RAG image. For real content, enable sources in the admin panel
and trigger a scrape (see [admin-and-scheduling.md](admin-and-scheduling.md)),
or run the ingestion CLI:

```bash
# scrape enabled sources then index into Chroma
python ingest/run_ingestion.py --scrape --limit 5
# index already-stored articles only
python ingest/run_ingestion.py
```

## Smoke test

[`scripts/smoke_e2e.sh`](../scripts/smoke_e2e.sh) brings up the `ci` profile
(stub LLM), seeds, signs up, and asserts a classified verdict end-to-end with
no GPU:

```bash
bash scripts/smoke_e2e.sh
```

## First-run checklist

1. `docker compose up --build` (or the `ci` profile).
2. `docker compose run --rm seed` — confirm `GET :5000/stats` shows chunks.
3. Open http://localhost:3000, sign up with an email listed in
   `HRF_ADMIN_EMAILS` to get the admin panel.
4. Submit a claim → expect a colour-coded verdict in a few seconds (real Gemma).

## Production notes

- **Set `HRF_JWT_SECRET`** to a strong secret; never ship the default.
- Put the frontend's nginx (or another reverse proxy) in front so the browser
  uses same-origin `/api` — avoids CORS and keeps the backend off the public
  internet directly.
- Back up the SQLite volume(s) and the Chroma volume.
- Pre-bake the embedding model into the RAG image (the Dockerfile does this) so
  the first query isn't a cold download.
- Scale: this is a single-instance design (SQLite, embedded-or-single Chroma).
  For higher load, move app data to Postgres and Chroma to a dedicated cluster —
  the contracts and the OpenAI-compatible LLM seam don't change.

## Local (no Docker)

See [development.md](development.md#running-services-locally-without-docker) for
running every service under conda with `python -m uvicorn`.
