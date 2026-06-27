# Development guide

## Prerequisites

- **conda** (the project ships an `environment.yml`)
- A vLLM (or any OpenAI-compatible) server for live LLM calls — or use the
  bundled stub for everything except real classification quality.

## Setup

```bash
conda env create -f environment.yml   # creates env "herefanmi" (Python 3.11, + nodejs)
conda activate herefanmi
```

The env includes the Python deps for all services plus `nodejs` (so `npm` works
for the frontend) and the test toolchain (pytest, respx, ruff, black).

Source roots are importable two ways: the `pyproject.toml` `pythonpath` list
(for pytest) and a `.pth` file the maintainer drops into the env's
site-packages (for scripts/uvicorn). If imports fail outside pytest, add the
roots to `PYTHONPATH`:
`shared:services/scraper:services/rag:services/llm:services/backend:.`

## Running the test suite

```bash
make test            # full suite (≈177 tests: 173 Python + 4 frontend)
make test-rag        # one service: shared | llm | scraper | rag | backend
make lint            # ruff
make fmt             # black + ruff --fix
```

`make` uses **`python -m pytest`** deliberately — the bare `pytest`/`uvicorn`
console scripts can resolve to a different interpreter's site-packages on some
machines. If you invoke pytest directly, use:

```bash
conda run -n herefanmi --cwd "$(pwd)" python -m pytest
```

### Frontend tests

```bash
cd frontend
npm install
npm run lint    # tsc --noEmit
npm run test    # vitest
npm run build   # production bundle
```

## TDD workflow

The whole codebase was built test-first; keep it that way. Tests live under
`tests/<service>/` and mirror the source:

```
tests/shared/      contracts, JSON parsing
tests/llm/         prompt, client, classifier, /predict   (mocked OpenAI client)
tests/scraper/     cleaning, dedup, storage, base+sources (HTML fixtures),
                   sources_store, configurable, scheduler, /ingest+CRUD
tests/rag/         chunking, indexer, bm25, hybrid RRF, retriever, /api  (ephemeral Chroma)
tests/backend/     storage, auth, orchestrator, /medicalTalk, admin       (mocked RAG/LLM)
tests/integration/ in-process end-to-end (seed → retrieve → classify) + stub contract
tests/fixtures/    html/<source>_article.html, llm/*.json
```

Write a failing test first, then implement. Run that service's suite, then the
full suite, before committing.

### Test design seams (reuse these patterns)

- **Scrapers** split `fetch()` (I/O) from `parse()` (pure) — tests call `parse`
  on saved HTML fixtures, never the network.
- **LLM** takes an injectable client; tests pass a fake returning canned text
  (clean / fenced / garbage) — no network, no model.
- **RAG** tests seed an **ephemeral** Chroma collection with known chunks; RRF
  is unit-tested with synthetic rank lists (no embeddings).
- **Backend** overrides FastAPI dependencies (`get_rag_client`,
  `get_llm_client`, `get_admin_gateway`, repos) with fakes/`respx`.

## Gotchas specific to this repo

| Symptom | Cause / fix |
|---------|-------------|
| `ModuleNotFoundError: hrf_*` under `uvicorn`/`pytest` | Bare console script used the wrong interpreter. Use `python -m uvicorn` / `python -m pytest`, or set `PYTHONPATH`. |
| Sentence-transformers can't load / `refs/main` errors | The machine's default `HF_HOME` may be a read-only shared cache. `tests/conftest.py` forces a project-local `.hf_cache/`; for the RAG service set `HF_HOME=$PWD/.hf_cache`. |
| Chroma test state leaks between tests | `chromadb.EphemeralClient()` shares one in-memory system per process — fixtures use a unique collection name (uuid) per test. |
| Scraper tests hit startup side-effects | `tests/scraper/conftest.py` sets `HRF_SCRAPER_AUTOSTART=false` and `HRF_SCHEDULER_ENABLED=false` before the app's lifespan runs. |
| Headless browser tools fail | This kind of box often lacks GUI libs / Chromium; run browser-based steps where a browser is available. |

## Conventions

- Python: ruff + black, line length 100, `--import-mode=importlib`.
- Commits: conventional-ish prefixes (`feat(scope):`, `docs:`, `chore:`), and
  end messages with the `Co-Authored-By: Claude …` trailer.
- Never modify or `git add` the `legacy/` tree (untracked reference, slated for
  deletion). `data/`, `.hf_cache/`, `node_modules/`, and `screenshots/` are
  gitignored.

## Running services locally (without Docker)

Start each in its own shell (after `conda activate herefanmi`). Use
`python -m uvicorn`. Example minimal stack against a host vLLM on `:50033`:

```bash
# LLM → vLLM
HRF_LLM_BASE_URL=http://localhost:50033/v1 HRF_LLM_MODEL=google/gemma-4-E4B-it \
  python -m uvicorn hrf_llm.main:app --port 5002
# RAG (embedded persistent Chroma)
HF_HOME=$PWD/.hf_cache HRF_CHROMA_MODE=persistent HRF_CHROMA_PATH=data/chroma \
  python -m uvicorn hrf_rag.main:app --port 5000
# Scraper
HRF_SQLITE_PATH=data/scraper.sqlite HRF_RAG_URL=http://localhost:5000 \
  python -m uvicorn hrf_scraper.main:app --port 8001
# Backend
HRF_SQLITE_PATH=data/backend.sqlite HRF_RAG_URL=http://localhost:5000 \
  HRF_LLM_URL=http://localhost:5002 HRF_SCRAPER_URL=http://localhost:8001 \
  HRF_ADMIN_EMAILS=you@example.com HRF_JWT_SECRET=dev \
  python -m uvicorn app.main:app --port 10000
# Frontend (proxies /api -> :10000)
cd frontend && npm run dev
```

Then seed the knowledge base (see [deployment.md](deployment.md#seeding)) and
open http://localhost:3000.
