# API reference

Every Python service is a FastAPI app and serves interactive docs at `/docs`
(Swagger) and `/redoc`. Default ports: Backend `10000`, RAG `5000`, LLM `5002`,
Scraper `8001`. This page lists the contracts and example calls.

Shared models (`hrf_shared/contracts.py`):

- **Classification** — `{medical, news, label, reasoning, sources}` (see
  [architecture](architecture.md#classification-contract)).
- **ScoredChunk** — `{text, score, source, url}`.
- **SourceConfig** — `{name, base_url, listing_url, listing_link_selector,
  title_selector, content_selector, date_selector?, date_attr?, enabled,
  interval_minutes, last_run_at?, last_status?}`.

---

## Backend — `:10000` (the public entry point)

CORS is open; all non-auth endpoints require a `Bearer` JWT. `/admin/*` also
requires `is_admin`.

### `POST /auth/signup` · `POST /auth/login`
Body: `{"email": "...", "password": "..."}` →
`{"access_token","token_type":"bearer","uid","points","is_admin"}`.
Signup fails 400 if the email exists; login fails 401 on bad credentials.
`is_admin` is set when the email is in `HRF_ADMIN_EMAILS`.

```bash
TOKEN=$(curl -s -X POST localhost:10000/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"secret123"}' | jq -r .access_token)
```

### `POST /medicalTalk` (JWT)
Body: `{"data": "<claim>", "opinion": "0", "backend": "HeReFaNMi LLM"}`.

- **Medical** → `{"data": "<reasoning>", "news": "True|False",
  "label": "Trustworthy|Doubtful|Fake", "source": ["<url>"], "key": "<id>"}`
- **Non-medical** → `{"data": "Please try to ask something related to the
  medical field!", "key": "<id>"}`
- **403** if a non-admin user is out of points.

```bash
curl -s -X POST localhost:10000/medicalTalk \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"data":"Does vitamin D support bone health?","opinion":"4","backend":"HeReFaNMi LLM"}'
```

### `POST /save` (JWT)
Body: `{"reference": "<key>", "rating": "5"}` → `{"status":"SUCCESS"}`; 404 if the
reference is unknown. `reference` is the `key` from a prior `/medicalTalk`.

### `POST /pointcheck` (JWT) · `POST /pointsave` (JWT)
`/pointcheck` → `{"points": <int>}` for the current user.
`/pointsave` body `{"points": <int>}` sets them → `{"status":"SUCCESS"}`.

### Admin (`is_admin` JWT)

| Endpoint | Body | Returns |
|----------|------|---------|
| `GET /admin/stats` | — | `{users, queries, chunks, per_source_chunks, sources}` |
| `GET /admin/sources` | — | `{sources: [SourceConfig]}` (proxied to scraper) |
| `POST /admin/sources` | `SourceCreate` | created source (proxied) |
| `PATCH /admin/sources/{name}` | `SourceUpdate` | updated source (proxied) |
| `DELETE /admin/sources/{name}` | — | `{status:"deleted"}` (proxied) |
| `POST /admin/scrape` | `{sources?: [name]}` | scrape report (proxied to scraper `/ingest`) |
| `GET /admin/users` | — | `{users: [{uid,email,points,is_admin}]}` |
| `POST /admin/users/{uid}/points` | `{points}` | `{status, points}`; 404 if unknown |
| `GET /admin/queries?limit=50` | — | `{queries: [{id,uid,question,label,medical,rating}]}` |

---

## RAG — `:5000`

### `POST /find_similar_chunks`
Body: `{"prompt": "<query>", "top_n": 5}` →
`{"chunks": [ScoredChunk], "query": "<prompt>"}`. **400** on empty prompt.
Returns synchronously (it does **not** forward to the LLM — anti-regression
vs. the legacy bare-200 behavior).

### `POST /index`
Body: `{"articles": [{"title","content","url","source","article_id?"}]}` →
`{"indexed": <chunk_count>}`. Chunks + embeds + upserts; idempotent on chunk id.
Used by the scraper after a harvest, and by the ingestion CLIs.

### `GET /stats`
→ `{"chunks": <int>, "sources": {"cdc": 12, ...}}` — total chunks and per-source
breakdown (powers the admin KB cards).

### `GET /health`
→ `{"status":"ok","count": <chunks>}`.

---

## LLM — `:5002`

### `POST /predict`
Body: `{"chunks": ["..."], "question": "..."}` → a **Classification** object.
Builds the system+user prompt, calls the configured provider, parses JSON with
one repair retry. **422** on missing fields; **502** if output stays unparseable
or the provider errors.

### `GET /health`
→ `{"status":"ok"}` — does **not** call the model.

### OpenAI-compatible stub
[`hrf_llm/stub.py`](../services/llm/hrf_llm/stub.py) serves
`POST /v1/chat/completions` (+ `/health`) returning a heuristic verdict, so the
stack runs GPU-free in CI. Point `HRF_LLM_BASE_URL` at it.

---

## Scraper — `:8001`

### Source management
| Endpoint | Body | Returns |
|----------|------|---------|
| `GET /sources` | — | `{sources: [SourceConfig]}` |
| `POST /sources` | `SourceCreate` (`name,base_url,listing_url` required) | `201` SourceConfig; **409** if name exists |
| `PATCH /sources/{name}` | `SourceUpdate` (any subset) | updated SourceConfig; **404** |
| `DELETE /sources/{name}` | — | `{status:"deleted"}`; **404** |

### Harvesting
- `POST /ingest` — body `{"sources?": [name], "limit?": int}`. Scrapes the named
  sources (or all **enabled** if omitted), stores new articles, pushes them to
  RAG `/index`. Returns `{sources: {<name>: {scraped, new}}, scraped, stored, indexed}`.
- `POST /run-now` — scrape all enabled sources now (same report shape).
- `GET /articles?source=&limit=` — read back stored articles.
- `GET /health`.

```bash
# Add a source, enable it, scrape it now
curl -s -X POST localhost:8001/sources -H 'Content-Type: application/json' -d '{
  "name":"mayo","base_url":"https://www.mayoclinic.org",
  "listing_url":"https://www.mayoclinic.org/diseases-conditions",
  "title_selector":"h1","content_selector":"div.content"}'
curl -s -X PATCH localhost:8001/sources/mayo -H 'Content-Type: application/json' -d '{"enabled":true,"interval_minutes":720}'
curl -s -X POST localhost:8001/ingest -H 'Content-Type: application/json' -d '{"sources":["mayo"]}'
```

See [admin-and-scheduling.md](admin-and-scheduling.md) for how these map to the
admin panel and the scheduler.
