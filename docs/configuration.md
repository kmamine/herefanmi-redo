# Configuration reference

All configuration is environment-driven, defined once in
[`shared/hrf_shared/config.py`](../shared/hrf_shared/config.py) as a
`pydantic-settings` model. Every variable is prefixed **`HRF_`** and can be set
via the environment, a `.env` file (see [`.env.example`](../.env.example)), or
`docker-compose.yml`. Each service reads only the variables it needs.

`get_settings()` is cached per process; tests construct `Settings(...)` directly
to override.

## LLM provider (OpenAI-compatible)

The provider is fully described by three values, so switching from the local
vLLM Gemma server to OpenAI or any compatible endpoint is purely an env change.

| Variable | Default | Used by | Meaning |
|----------|---------|---------|---------|
| `HRF_LLM_BASE_URL` | `http://localhost:50033/v1` | LLM | OpenAI-compatible base URL. Inside Docker use `http://host.docker.internal:50033/v1`. |
| `HRF_LLM_MODEL` | `google/gemma-4-E4B-it` | LLM | Served model name. If your vLLM serves `gemma-3n-E4B-it`, change only this. |
| `HRF_LLM_API_KEY` | `dummy-key` | LLM | Bearer key sent to the provider. |
| `HRF_LLM_TEMPERATURE` | `0.1` | LLM | Sampling temperature (low → deterministic verdicts). |
| `HRF_LLM_MAX_TOKENS` | `1024` | LLM | Max completion tokens. |
| `HRF_LLM_TIMEOUT` | `60.0` | LLM, Backend | Provider timeout (s); the Backend's LLM client adds headroom on top. |

## ChromaDB

| Variable | Default | Used by | Meaning |
|----------|---------|---------|---------|
| `HRF_CHROMA_MODE` | `persistent` | RAG, ingest | `ephemeral` (in-memory, tests), `persistent` (local file), or `http` (server). |
| `HRF_CHROMA_HOST` | `chromadb` | RAG, ingest | Host when `mode=http` (the compose service name). |
| `HRF_CHROMA_PORT` | `8000` | RAG, ingest | Port when `mode=http`. |
| `HRF_CHROMA_PATH` | `data/chroma` | RAG, ingest | On-disk path when `mode=persistent`. |
| `HRF_CHROMA_COLLECTION` | `health_chunks` | RAG, ingest | Collection name (cosine space). |

> In Docker, run Chroma as a **server** (`mode=http`): only one process may open
> an embedded persistent store safely.

## Embeddings

| Variable | Default | Used by | Meaning |
|----------|---------|---------|---------|
| `HRF_EMBED_MODEL` | `all-MiniLM-L6-v2` | RAG, ingest | sentence-transformers model (384-dim). Changing it changes the embedding dimension — re-index from scratch. |

## Retrieval

| Variable | Default | Used by | Meaning |
|----------|---------|---------|---------|
| `HRF_DEFAULT_TOP_N` | `5` | RAG, Backend | Chunks returned/used as evidence. |
| `HRF_RRF_K` | `60` | RAG | Reciprocal Rank Fusion constant. |
| `HRF_DENSE_CANDIDATES` | `20` | RAG | Dense/BM25 candidate pool size before fusion. |

## Storage

| Variable | Default | Used by | Meaning |
|----------|---------|---------|---------|
| `HRF_SQLITE_PATH` | `data/herefanmi.sqlite` | Backend, Scraper | SQLite file. Backend stores users/queries; Scraper stores articles/sources. Use separate files per service unless you intend them to share. |

## Scraper scheduling

| Variable | Default | Used by | Meaning |
|----------|---------|---------|---------|
| `HRF_SCRAPER_AUTOSTART` | `true` | Scraper | On startup, seed the 8 built-in sources (disabled) if the table is empty. Tests set `false`. |
| `HRF_SCHEDULER_ENABLED` | `false` | Scraper | Start the APScheduler tick. Off by default — admins enable sources explicitly. |
| `HRF_SCHEDULER_TICK_SECONDS` | `300` | Scraper | How often the scheduler checks for due sources. |
| `HRF_SCRAPE_LIMIT` | _(none)_ | Scraper | Max articles per source per run (`None` = no cap). |

## Admin & inter-service URLs

| Variable | Default | Used by | Meaning |
|----------|---------|---------|---------|
| `HRF_ADMIN_EMAILS` | `""` | Backend | Comma-separated allowlist; matching emails get `is_admin` on signup/login (case-insensitive). |
| `HRF_SCRAPER_URL` | `http://localhost:8001` | Backend | Where the admin endpoints proxy source management + scrape triggers. |
| `HRF_RAG_URL` | `http://localhost:5000` | Backend, Scraper | RAG base URL (Backend retrieval; Scraper `POST /index`). |
| `HRF_LLM_URL` | `http://localhost:5002` | Backend | LLM service base URL. |

## Auth (Backend)

| Variable | Default | Meaning |
|----------|---------|---------|
| `HRF_JWT_SECRET` | `change-me-in-production` | HS256 signing secret. **Set a strong value in production.** |
| `HRF_JWT_ALGORITHM` | `HS256` | JWT algorithm. |
| `HRF_JWT_EXPIRE_MINUTES` | `1440` | Token lifetime (minutes). |
| `HRF_SIGNUP_POINTS` | `14` | Starting query points for a new user. |
| `HRF_ADMIN_UID` | `admin` | Legacy fallback admin uid; prefer `HRF_ADMIN_EMAILS`. |

## Test-only environment

Not part of `Settings`, but relevant when running the suite locally — see
[development.md](development.md):

| Variable | Purpose |
|----------|---------|
| `HF_HOME` / `HF_HUB_CACHE` | Point sentence-transformers at a writable cache. `tests/conftest.py` forces a project-local `.hf_cache/`. |
