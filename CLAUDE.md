# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RAG LLM application with three core functions:
- **Data ingestion** — chunks raw text files, computes sentence embeddings, and stores them in ChromaDB (vector) and SQLite (TF-IDF).
- **Retrieval** — supports three search modes: `vector` (ChromaDB + MMR), `sparse` (TF-IDF + MMR), `hybrid` (RRF fusion of both, then MMR).
- **Chat** — Flask API that maintains per-session conversation history and augments each user turn with retrieved chunks before calling the local Ollama LLM.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Index source documents (run once before starting the app or when adding new files)
python updateData.py

# Run the Flask web app (http://localhost:5000)
python app.py

# Run all tests
python -m pytest Test/

# Run a single test file
python -m pytest Test/testRagService.py

# Run a single test case
python -m pytest Test/testRagService.py::TestMmrSelect::test_diversityPrefersDiverseOverRedundant
```

## Architecture

```
Controller/       — input handling and data loading (SearchData, LLMController, CategoryTagger)
DatabaseService/  — business logic layer (RagService: retrieval pipeline; ChatService: plain chat)
DatabaseRepository/ — storage adapters (VectorRepository → ChromaDB; TfidfRepository → SQLite)
LLMApi/           — thin Ollama SDK wrapper
Dto/              — data transfer objects (Source, VectorDocument, ChatMessage)
View/templates/   — Flask HTML templates
Test/             — unittest tests; RawTextFiles/ holds sample source documents
```

**Key data flow — ingestion (`updateData.py`):**
`RawTextFiles/` → `SearchData.fetchDataFromFolder` → chunk with `RecursiveCharacterTextSplitter` → `SentenceTransformer` embeddings → `RagService.addSources` → `VectorRepository` (ChromaDB `embeddings_db/`) + `TfidfRepository` (SQLite `tfidf_db/`).

**Key data flow — query (`app.py` `/api/chat`):**
HTTP POST → `RagService.query(mode)` → threshold filter → MMR re-rank → top chunks passed to `LLMController.generateResponse` → Ollama (`llama2`) → reply streamed back.

**Retrieval pipeline detail (`RagService`):**
- All three modes run Stage 1 (wide candidate pool) → Stage 2 (threshold filter) → MMR re-rank.
- Hybrid mode adds an RRF fusion step between Stage 1 and MMR.
- `Config.py` is the single source of truth for all thresholds, top-K values, model names, and chunk sizes.

**Session management:** `app.py` keeps a `_sessions: dict[str, LLMController]` in memory. `LLMController` stores raw queries (not augmented prompts) in history to keep context compact, truncates old replies beyond the last 2 turns, and compacts the full history via a summary prompt when it exceeds `MaxChatHistoryChars`.

## Code conventions

- camelCase for variables, functions, and file names; PascalCase for global constants and class names; external module names kept as-is.
- Layer architecture: View → Controller → Service → Repository → Dto. Business logic stays in Service; repositories only talk to storage backends.
- Docstrings required on all public functions and classes; inline comments for non-obvious logic only.
- Always annotate non-native types (e.g., `list[Source]`, `dict[str, float]`).

## Things that trip people up

- `updateData.py` must be run before `app.py` on a fresh setup — the app will crash if the ChromaDB collection `my_docs` does not exist.
- Both `embeddings_db/` and `tfidf_db/` must be populated with the same chunks (same chunk IDs) for hybrid mode to work correctly. `RagService.addSources` handles this atomically; do not insert into one store manually.
- `TfidfRepository.search` returns `distances = 1 - normalised_score` (lower = more relevant) to match ChromaDB's L2 convention. The similarity threshold `TfidfSimilarityThreshold = 0.75` therefore keeps chunks with a normalised score ≥ 0.25.
- Sensitive config (API keys, custom model paths) belongs in `.env`, not in `Config.py`.
- Tests in `Test/` require `sys.path.insert` to resolve local imports; keep this pattern when adding new test files.
