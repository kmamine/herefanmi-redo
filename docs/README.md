# HeReFaNMi documentation

Detailed docs for the HeReFaNMi rewrite. Start with the main
[project README](../README.md) for the overview, screenshots, and quick start.

| Doc | What's inside |
|-----|---------------|
| [architecture.md](architecture.md) | Services, the synchronous `/medicalTalk` flow, ChromaDB hybrid search, the classification contract, data stores, and key design decisions (with Mermaid diagrams). |
| [configuration.md](configuration.md) | Every `HRF_*` environment variable — defaults, meaning, and which service uses it. |
| [api-reference.md](api-reference.md) | All endpoints for the Backend, RAG, LLM, and Scraper services, with request/response shapes and `curl` examples. |
| [development.md](development.md) | conda setup, the TDD layout, test seams, running services locally, conventions, and repo-specific gotchas. |
| [deployment.md](deployment.md) | Docker Compose, the `ci` (GPU-free stub) profile, reaching a host vLLM, seeding, the smoke test, and production notes. |
| [admin-and-scheduling.md](admin-and-scheduling.md) | The admin panel, editable sources, the per-source APScheduler, and gamification points. |

`screenshots/` here holds the curated UI images used by the README gallery.
