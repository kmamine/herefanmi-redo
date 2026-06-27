# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**HeReFaNMi** (Health-Related Fake News Mitigation) is an EU NGI Search–funded AI system that classifies medical news as `Trustworthy` / `Doubtful` / `Fake`. This repo (`herefanmi-redo`) has three parts:

- **The rewrite (`shared/`, `services/`, `ingest/`, `frontend/`)** — the *primary active* system. A from-scratch, TDD, dockerized rebuild of the original stack using FastAPI + ChromaDB hybrid search + SQLite + an OpenAI-compatible LLM + React/Vite. **This is where new work goes.** See "Active system: the rewrite" below.
- **`dataset-regenerate/`** — a standalone pipeline that generates a synthetic medical-misinformation dataset (kept as-is; see its own section).
- **`legacy/`** — the *original* production system, kept only as a behavioral reference and slated for deletion. A set of Docker microservices. **It is untracked in git and each subfolder is its own nested git repo** (cloned from the `HeReFanMi` org). Don't `git add legacy/`, don't modify it, and be aware that running git inside a `legacy/*/` folder operates on that nested repo, not this one.

## Active system: the rewrite

The rewrite preserves the legacy *topology and contracts* but fixes its known flaws (Firebase → SQLite, in-process unsloth → OpenAI-compatible API, Postgres+pgvector → ChromaDB hybrid, and the racy fire-and-forget callback chain → synchronous orchestration).

### Services (each a FastAPI app)

```
Frontend(:3000) → Backend(:10000) ──┬─▶ RAG(:5000) ──▶ ChromaDB(:8000)
  React/Vite      orchestrator+JWT   └─▶ LLM(:5002) ──▶ vLLM Gemma (host :50033, OpenAI-compatible)
Scraper(:8001) → SQLite → chunk/embed(MiniLM) → ChromaDB
```

- **`shared/hrf_shared/`** — the single source of truth imported by every service: `contracts.py` (Pydantic models incl. the `Classification` JSON contract `{medical,news,label,reasoning,sources}`), `config.py` (`Settings`, env-prefixed `HRF_`), `json_utils.py` (tolerant model-output parsing), `chroma_client.py` (ephemeral/persistent/http modes).
- **`services/scraper/`** — `BaseScraper` splits `fetch()` (I/O) from `parse()` (pure, per-source CSS selectors in `sources/`); registry of 8 sources; SQLite store with URL+content-hash dedup.
- **`services/rag/`** — `chunking` → `indexer` (Chroma upsert) → hybrid retrieval: dense (`all-MiniLM-L6-v2`) + BM25 fused by Reciprocal Rank Fusion (`hybrid.py`). `/find_similar_chunks` returns chunks **synchronously** (legacy returned bare 200 and fired forward — do not reintroduce that).
- **`services/llm/`** — `classifier.py` calls the OpenAI-compatible provider and parses JSON (one repair retry → 502). `hrf_llm/stub.py` is a GPU-free stand-in. Provider = `HRF_LLM_BASE_URL`/`HRF_LLM_MODEL`/`HRF_LLM_API_KEY` (default the local vLLM Gemma; literal model name lives only in env).
- **`services/backend/`** — `orchestrator.py` runs RAG→LLM→validate-sources→persist in one `await`; JWT auth (bcrypt direct — *not* passlib, which is broken with bcrypt 5.x); SQLite `users`/`queries`; server-side gamification points. `/medicalTalk` is the entry point and preserves the legacy request/response shapes.
- **`ingest/`** — `run_ingestion.py` (scrape→index) and `seed_sample_data.py` (bundled `sample_articles.py` for demos/CI without live scraping).
- **`frontend/`** — Vite+React+TS, two-phase UX, color+icon verdict card (green/amber/red, never color alone). Design follows the `ui-ux-pro-max` skill output (Accessible & Ethical style, cyan+green palette, Figtree/Noto Sans).

### Running it

```bash
# Docker (full stack against host vLLM on :50033)
docker compose up --build
docker compose --profile ci up --build      # GPU-free: uses the OpenAI stub (llm-stub)
docker compose run --rm seed                 # load demo data
bash scripts/smoke_e2e.sh                     # end-to-end smoke test

# Local dev / tests (conda)
conda env create -f environment.yml          # env name: herefanmi
make test                                     # full pytest suite (also: make lint / fmt)
```

### Conventions & gotchas (important for this repo)

- **Run pytest as `conda run -n herefanmi --cwd <repo> python -m pytest`** (the bare `pytest` console script resolves to the base 3.13 interpreter, which lacks the deps and the asyncio plugin). The `Makefile` does this for you.
- Source roots are made importable via a `.pth` file in the env's site-packages **and** the `pythonpath` list in `pyproject.toml` (for pytest). Tests use `--import-mode=importlib` (duplicate test basenames across services would otherwise collide).
- **HuggingFace cache:** the machine's default `HF_HOME` (`/calcul/...`) is read-only with an inconsistent `refs/main`. `tests/conftest.py` forces a project-local `.hf_cache/` so the MiniLM model loads. For running the RAG service outside tests, set `HF_HOME=$PWD/.hf_cache`.
- **ChromaDB `EphemeralClient()` shares one in-memory system per process**, so tests use a unique collection name each (uuid) to stay isolated.
- TDD is the workflow: write/extend tests under `tests/<service>/` first. 141 tests currently pass.

## Active pipeline: dataset-regenerate

[regenrate.py](dataset-regenerate/regenrate.py) drives everything via the `MedicalMisinfoDetectionDatasetGenerator` class. The flow:

1. For each real article, generate false versions at three difficulty levels (`easy`/`medium`/`hard`) — each level has its own prompt template and "marker" phrases (`generate_false_article`).
2. Persist false articles to CSV (`process_csv`), then merge true + false into one shuffled, labelled dataset (`create_balanced_dataset`).

The generator talks to a **local vLLM server** over the OpenAI-compatible API (`openai.OpenAI(base_url=...)`), not the hosted OpenAI API. Model responses are expected to be JSON (the code strips ```` ```json ```` fences before `json.loads`).

### Running it

```bash
# 1. Start the model server (needs 2 GPUs; serves QwQ-32B 4-bit on :8000)
bash dataset-regenerate/launch-vllm-server.sh

# 2. Run the generator
python dataset-regenerate/regenrate.py
```

There is **no test suite, linter, or build step** — this is a single script.

### Gotchas before running

- **Config is hardcoded in `main()`**: `API_KEY`, `BASE_URL` (`http://localhost:8000/v1`), `MODEL_NAME` (`QwQ`), and file paths. The same literal `API-KEY` must match between `launch-vllm-server.sh` (`--api-key`) and `regenrate.py`.
- `main()` reads `medical_blogs.csv` with columns `title`/`content`, but the only data shipped is [data/data.csv](dataset-regenerate/data/data.csv), **which is currently empty**. You must supply the input CSV and align the `TITLE_COLUMN`/`CONTENT_COLUMN` and path constants.
- [requirments.txt](dataset-regenerate/requirments.txt) (note the misspelling) is **empty** — actual deps are `pandas` and `openai`. `vllm` is needed for the server.
- `process_csv` calls `_test_connection()` first and aborts if the server isn't reachable; it sleeps 2s between calls and skips articles under 100 chars.

## Legacy system architecture (reference only)

The original system is 4 cooperating Flask/React services plus a scraper. The request flow for the in-house model path is a **fire-and-forget callback chain**, not a synchronous call graph — understanding this is the key to reading the legacy code:

```
Frontend(:3000) ──POST /medicalTalk──▶ Backend(:10000)
                                          │  if backend == "HeReFaNmi LLM":
                                          ├─ POST /find_similar_chunks ─▶ RAG(:5000)
                                          │                                 │ embeds query (all-MiniLM-L6-v2),
                                          │                                 │ cosine-similarity over Postgres,
                                          │                                 └─ POST /predict ─▶ LLM(:5002)
                                          │                                                      │ NGILlama3 generates
                                          │  ◀── POST /response ────────────────────────────────┘ classification
                                          │  (Backend polls a shared dict until /response fills it, 30s timeout)
                                          │  else: calls OpenAI GPT-4 / GPT-3.5 directly
                                          └─ saves Q/A + label to Firebase Realtime DB
```

- **[legacy/Backend/index.py](legacy/Backend/index.py)** (`:10000`) — orchestrator. `/medicalTalk` is the entry point; it either calls OpenAI directly or kicks off the RAG→LLM chain and waits via `wait_for_response()` polling a `shared_data` dict guarded by `data_lock`. Also handles Firebase persistence, ratings, and signup points. The classification JSON contract (`medical`, `news`, `label`, `reasoning`, `sources`) is defined by the prompt in `Prompt()` and is shared across all services.
- **[legacy/RAG_end_Point/app/app.py](legacy/RAG_end_Point/app/app.py)** (`:5000`) — retrieval. Async SQLAlchemy + asyncpg over Postgres (`chunk_embeddings` table). `/find_similar_chunks` returns `200 OK` immediately and forwards the top-N chunks onward to the LLM rather than returning them to the caller. Has its own [docker-compose.yml](legacy/RAG_end_Point/docker-compose.yml) bundling Postgres on a `shared-network`.
- **[legacy/NGI_LLM/app.py](legacy/NGI_LLM/app.py)** (`:5002`) — generation. Loads `a-hamdi/NGILlama3-merged` (fine-tuned Llama, 4-bit via unsloth `FastLanguageModel`) at startup; `/predict` builds the prompt from chunks + question and POSTs the result back to Backend's `/response`. Requires CUDA.
- **[legacy/Frontend/](legacy/Frontend/)** (`:3000`) — create-react-app SPA, Firebase auth (`firebase.js`), axios to Backend.
- **[legacy/web-scraping-framework/](legacy/web-scraping-framework/)** — Flask + MongoDB (`HeReFaNMiDB`) + BeautifulSoup scraper that harvests ~8 credible health sources (CDC, NHS, MedlinePlus, etc.) to populate the knowledge base.
- **[legacy/ops-testing-repo/](legacy/ops-testing-repo/)** — a trivial hello-world Flask app; Docker/ops scaffolding only.

### Running the legacy stack

```bash
cd legacy/Docker-compose && docker compose up --build
```

Note the top-level [docker-compose.yml](legacy/Docker-compose/docker-compose.yml) expects sibling build contexts named `Frontend/`, `Backend/`, `RAG/`, `LLM/` — the actual folders here are named differently (`RAG_end_Point`, `NGI_LLM`), so the compose file's paths won't resolve as-is. Services are more reliably run individually per each subfolder's own README. Each service has its own `requirements.txt` and `Dockerfile`.

### Known rough edges in legacy code (don't "fix" silently)

- Backend's `RAGrequest` posts to `http://127.0.1:5000/...` (note the malformed `127.0.1`) and the payload key is `prompt`/`top_n` — these inter-service contracts are brittle and version-specific.
- The Backend↔LLM `shared_data` mechanism is a single global slot with no request correlation — concurrent requests would race.
- Service-to-service URLs are hardcoded to `127.0.0.1`, which only works outside Docker; inside compose they'd need service names.

## Secrets

Credentials are referenced via env files that are intentionally absent: legacy Backend needs a `.env` (`API_KEY`, `FIREBASE_KEY`, `DATABASE_URL`) plus a Firebase service-account JSON. The vLLM `--api-key` and the generator's `API_KEY` are placeholder literals (`API-KEY`) in tracked files — replace them locally, don't commit real keys.

## Project background & external resources

HeReFaNMi is one of the funded NGI Search solutions (project ID **NGI-SEARCH:18**), under the EU Next Generation Internet / Horizon Europe programme; the consortium spans France and Italy. The acronym appears as both "Health-Related Fake News **Mitigation**" (repo READMEs) and "**Monitoring**" (official pages) — treat them as the same project.

- Project site: https://sites.google.com/view/herefanmi/ (Home / Mission / Team / Careers / FAQ)
- NGI funded-solution profile: https://ngi.eu/funded_solution/ngi-search18/
- Live demo — the "AI text validator": https://ai-text-validator.vercel.app/ — this is the deployed build of [legacy/Frontend/](legacy/Frontend/) (whose package name is `text-validation`). Use it to see the intended end-user UX of the legacy stack.
- Canonical org/source repos: https://github.com/alessandrobruno10/herefanmi (the `legacy/*/` subfolders were cloned from the related `HeReFanMi` GitHub org).
