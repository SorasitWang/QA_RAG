# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RAG LLM application split into three HTTP services:
- **retrieval** (`retrievalApp.py`, :5001) — owns ChromaDB (vector), SQLite TF-IDF and SQLite graph indexes. Search modes: `vector`, `sparse`, `hybrid` (RRF), `graph`, all followed by MMR. Ingestion runs continuously: a folder watcher re-indexes `Test/RawTextFiles/` on add/edit/delete.
- **web** (`app.py`, :5000) — Flask UI + chat. Keeps per-session history and calls retrieval over HTTP (`Controller/RetrievalClient.py`) before calling Ollama. Does not import chromadb/torch/spaCy.
- **feed-poller** (`feedPollerApp.py`, :5002) — fetches RSS / Reddit / Mastodon / NewsAPI feeds (`feeds.json`) and POSTs new or changed items to retrieval `/ingest`; expires old items via `/ingest/delete`. Keeps only its own state DB, no index, no ML deps.
- **ollama** (:11434) — the LLM, stock image.

Every chunk carries `metadata.sourceType` (`file` | `news` | `blog` | `sns`, defined by `Dto.Source.SourceType`). `RagService.query(sourceTypes=...)` filters on it in all four modes.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything in Docker
docker compose up --build

# Or locally: retrieval service first (indexes + watches Test/RawTextFiles; WATCH_POLLING=1 on WSL /mnt/c)
python retrievalApp.py
# then the web app (http://localhost:5000)
python app.py

# Feed poller (optional; uses feeds.json, falling back to feeds.example.json)
python feedPollerApp.py

# One-shot sync when the retrieval service is not running (or --remote to ask it to rescan)
python -m scripts.updateData

# Run all tests
python -m pytest Test/

# Run a single test file
python -m pytest Test/testRagService.py

# Run a single test case
python -m pytest Test/testRagService.py::TestMmrSelect::test_diversityPrefersDiverseOverRedundant
```

## Architecture

```
app.py            — web service entry point
retrievalApp.py   — retrieval service entry point (createApp() factory + main())
Controller/       — input handling and data loading (SearchData, LLMController, CategoryTagger,
                    RetrievalClient, FolderWatcher, RequestValidation)
DatabaseService/  — business logic layer (RagService: retrieval; IngestionService + IngestionWorker; ChatService)
DatabaseRepository/ — storage adapters (VectorRepository → ChromaDB; TfidfRepository, GraphRepository,
                    IngestManifestRepository → SQLite)
LLMApi/           — thin Ollama SDK wrapper
Dto/              — data transfer objects (Source, VectorDocument, ChatMessage)
View/templates/   — Flask HTML templates
Test/             — unittest tests; RawTextFiles/ holds sample source documents
```

**Key data flow — ingestion (retrieval service):**
`FolderWatcher` (debounced) / startup `reconcile()` / `POST /ingest` → `IngestionWorker` queue (single writer thread) → `IngestionService.ingestFile` → hash vs `IngestManifestRepository` → (changed) `RagService.deleteSource` then chunk → embed → tag → `RagService.addSources` → Vector + TF-IDF + Graph stores.

**Key data flow — query (`app.py` `/api/chat`):**
HTTP POST → `RetrievalClient.query` → retrieval `POST /query` → `RagService.query(mode)` → threshold filter → MMR re-rank → `RetrievalResult.toDict()` (no embeddings) → web → `LLMController.generateResponse` → Ollama (`llama2`) → reply.

**Retrieval pipeline detail (`RagService`):**
- All three modes run Stage 1 (wide candidate pool) → Stage 2 (threshold filter) → MMR re-rank.
- Hybrid mode adds an RRF fusion step between Stage 1 and MMR.
- `Config.py` is the single source of truth for all thresholds, top-K values, model names, and chunk sizes.

**Service settings:** URLs, ports and watcher options come from env vars read in `Config.Services` / `Config.Ingestion` (`RETRIEVAL_SERVICE_URL`, `OLLAMA_HOST`, `WATCH_POLLING`, ...).

**Session management:** `app.py` keeps a `_sessions: dict[str, LLMController]` in memory. `LLMController` stores raw queries (not augmented prompts) in history to keep context compact, truncates old replies beyond the last 2 turns, and compacts the full history via a summary prompt when it exceeds `MaxChatHistoryChars`.

## Code conventions

- camelCase for variables, functions, and file names; PascalCase for global constants and class names; external module names kept as-is.
- Layer architecture: View → Controller → Service → Repository → Dto. Business logic stays in Service; repositories only talk to storage backends.
- Docstrings required on all public functions and classes; inline comments for non-obvious logic only.
- Always annotate non-native types (e.g., `list[Source]`, `dict[str, float]`).

## Things that trip people up

- `embeddings_db/`, `tfidf_db/` and `graph_db/` must hold the same chunks (same chunk IDs) for hybrid/graph mode. Always write through `RagService.addSources` / `RagService.deleteSource`; never touch one store manually.
- Only one process may write the indexes. Inside the retrieval service all writes go through `IngestionWorker`; don't run `scripts/updateData.py` in local mode while the service is up.
- `Config.Model.getEmbeddingFunction()` is lazy on purpose so the web service never imports chromadb/torch. Don't reintroduce import-time model loading in `Config.py`.
- inotify events don't fire for `/mnt/c` under WSL2 or for Windows bind mounts in Docker — use `WATCH_POLLING=1` there.
- Run `retrievalApp.py` with the Flask reloader off (it already does); the reloader would start a second watcher.
- Source type lives in three places: Chroma metadata, the TF-IDF `documents.sourceType` column (+ a `metadata` JSON column), and nowhere in the graph. Graph mode filters after hydrating chunks from Chroma, with a widened candidate pool (`Config.Graph.FilteredPoolMultiplier`). When adding a store, persist `sourceType` there too.
- Feed items are keyed by canonical URL (`Source.title`). They are not in the ingestion manifest, so `reconcile()` never deletes them. Only the poller's retention (`/ingest/delete`) removes them.
- `IngestionService` cleans `file` sources with `cleanText` (drops short lines and lines with URLs) and feed sources with `cleanFeedText`. Don't route feed text through `cleanText`, or SNS posts vanish.
- New connectors go in `Controller/Connectors/` as `fetch(feedConfig, feedState) -> list[FeedItem]`, registered in `Connectors/__init__.py`. They raise on failure; `FeedPollerService` owns retries and backoff.
- `TfidfRepository.search` returns `distances = 1 - normalised_score` (lower = more relevant) to match ChromaDB's L2 convention. The similarity threshold `TfidfSimilarityThreshold = 0.75` therefore keeps chunks with a normalised score ≥ 0.25.
- Sensitive config (API keys, custom model paths) belongs in `.env`, not in `Config.py`.
- Tests in `Test/` require `sys.path.insert` to resolve local imports; keep this pattern when adding new test files.
