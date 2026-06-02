# atlas-kit-python-rag

> Production-structured Python RAG starter kit — part of [Atlas Systems](https://atlas-systems.uk) P-02 Library.

A foundation for Retrieval-Augmented Generation pipelines. Not a tutorial. Drop it in, wire your documents, swap providers via config.

---

## Architecture
cat > ~/atlas-kit-python-rag/README.md << 'EOF'
# atlas-kit-python-rag

> Production-structured Python RAG starter kit — part of [Atlas Systems](https://atlas-systems.uk) P-02 Library.

A foundation for Retrieval-Augmented Generation pipelines. Not a tutorial. Drop it in, wire your documents, swap providers via config.

---

## Architecture
Documents → Loader → Chunker → Embedder → Vector Store
↓
Query → Embedder → Retriever → Generator → Answer**Provider abstraction:** swap embedding or LLM provider by changing one env var. No code changes.

| Layer | Default | Alternatives |
|---|---|---|
| Embedding | `sentence-transformers` (local) | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB | (extend `BaseStore`) |
| LLM | Ollama (local) | OpenAI GPT-4o-mini |

---

## Quickstart

```bash
git clone https://github.com/AtlasReaper311/atlas-kit-python-rag
cd atlas-kit-python-rag
cp .env.example .env
docker-compose up
```

The API is live at `http://localhost:8080`. ChromaDB at `http://localhost:8000`.

**Ingest a document:**
```bash
curl -X POST http://localhost:8080/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here.", "source_id": "doc-001"}'
```

**Query:**
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

---

## Configuration

All configuration via environment variables. Copy `.env.example` to `.env`.

| Variable | Default | Options |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai` |
| `EMBEDDING_PROVIDER` | `local` | `local`, `openai` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Any model pulled in Ollama |
| `CHUNK_SIZE` | `512` | 64–4096 |
| `RETRIEVAL_TOP_K` | `5` | 1–50 |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest/text` | Ingest raw text |
| `POST` | `/query` | Full RAG query |
| `GET` | `/retrieve?q=...` | Retrieve chunks only |
| `GET` | `/health` | Liveness + store stats |

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
mypy src/
```

---

## Design Notes

**Why abstract base classes?** `BaseEmbedder` and `BaseGenerator` define interfaces independently of implementation. The pipeline composes them — it has no import dependency on OpenAI or Ollama. This is dependency inversion in practice: you can test the pipeline by injecting a mock embedder without touching real APIs.

**Why Pydantic Settings?** All config lives in `config.py` and is validated at startup. A missing required variable fails fast with a clear error, not a silent `None` three layers deep.

**Why multi-stage Docker?** `dev` target mounts source for hot-reload; `prod` strips dev tools and runs as a non-root user. Same Dockerfile, different build target — no duplication.

---

*Part of [atlas-systems.uk](https://atlas-systems.uk) · MIT License*
