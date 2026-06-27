# HeReFaNMi developer tasks. Run inside the `herefanmi` conda env.
# Use `python -m pytest` (not the bare `pytest` console script, which can resolve
# to a different interpreter's site-packages).
ROOT := $(shell pwd)
CONDA_RUN ?= conda run -n herefanmi --cwd $(ROOT)
PYTEST := $(CONDA_RUN) python -m pytest

.PHONY: help install test test-shared test-llm test-scraper test-rag test-backend lint fmt up down ingest seed

help:
	@echo "install     - create/update the conda env"
	@echo "test        - run the full pytest suite"
	@echo "test-<svc>  - run tests for one service (shared|llm|scraper|rag|backend)"
	@echo "lint        - ruff check"
	@echo "fmt         - black + ruff --fix"
	@echo "up / down   - docker compose up/down"
	@echo "ingest      - scrape sources and index into ChromaDB"
	@echo "seed        - load bundled sample data into SQLite + ChromaDB"

install:
	conda env create -f environment.yml || conda env update -f environment.yml

test:
	$(PYTEST)

test-shared:
	$(PYTEST) tests/shared

test-llm:
	$(PYTEST) tests/llm

test-scraper:
	$(PYTEST) tests/scraper

test-rag:
	$(PYTEST) tests/rag

test-backend:
	$(PYTEST) tests/backend

lint:
	$(CONDA_RUN) ruff check .

fmt:
	$(CONDA_RUN) black .
	$(CONDA_RUN) ruff check --fix .

up:
	docker compose up --build

down:
	docker compose down

ingest:
	$(CONDA_RUN) python ingest/run_ingestion.py

seed:
	$(CONDA_RUN) python ingest/seed_sample_data.py
