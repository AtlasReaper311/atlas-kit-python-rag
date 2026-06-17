<div align="center">
  <img src="https://raw.githubusercontent.com/AtlasReaper311/AtlasReaper311/main/atlas-icon-dark-256.png" width="88" alt="Atlas Systems"/>
</div>

# atlas-kit-python-rag

```
┌─────────────────────────────────────────────┐
│  ATLAS SYSTEMS // atlas-kit-python-rag      │
│  a RAG pipeline as a library:               │
│  swap providers by changing one env var     │
└─────────────────────────────────────────────┘
```

![Python](https://img.shields.io/badge/python-3.12-f5a623?style=flat-square&labelColor=0a0a0f)
[![CI](https://github.com/AtlasReaper311/atlas-kit-python-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/AtlasReaper311/atlas-kit-python-rag/actions)
<!-- coverage-badge-start -->
![Coverage](https://img.shields.io/badge/coverage-unknown-aaa9a0?style=flat-square&labelColor=0a0a0f)
<!-- coverage-badge-end -->
![Vector store](https://img.shields.io/badge/vectors-chromadb-aaa9a0?style=flat-square&labelColor=0a0a0f)
![Cost](https://img.shields.io/badge/cost-%C2%A30-aaa9a0?style=flat-square&labelColor=0a0a0f)

A foundation for retrieval-augmented generation pipelines, structured as a library rather than a script. Drop it in, wire your documents, and swap the embedding or LLM provider by changing one environment variable. This is the P-02 library; [`ollama-rag-kit`](https://github.com/AtlasReaper311/ollama-rag-kit) is the same architecture packaged as a deployable service.

## Architecture

```
Documents ──▶ Loader ──▶ Chunker ──▶ Embedder ──▶ Vector store
                                                       │
                  Query ──▶ Embedder ──▶ Retriever ──▶ Generator ──▶ Answer
```

The pipeline composes its layers behind interfaces; it has no import dependency on any one provider. Change a provider in config and pipeline code stays untouched.

| Layer | Default | Alternatives |
|---|---|---|
| Embedding | `sentence-transformers` (local) | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB | extend `BaseStore` |
| LLM | Ollama (local) | OpenAI `gpt-4o-mini` |

## Quickstart

```bash
git clone https://github.com/AtlasReaper311/atlas-kit-python-rag
cd atlas-kit-python-rag
cp .env.example .env
docker compose up
```

The API is live at `http://localhost:8080`. ChromaDB at `http://localhost:8000`.

Ingest a document:

```bash
curl -X POST http://localhost:8080/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here.", "source_id": "doc-001"}'
```

Query:

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

## Configuration

All configuration is set through environment variables. Copy `.env.example` to `.env`.

| Variable | Default | Options |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai` |
| `EMBEDDING_PROVIDER` | `local` | `local`, `openai` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Any model pulled in Ollama |
| `CHUNK_SIZE` | `512` | 64 to 4096 |
| `RETRIEVAL_TOP_K` | `5` | 1 to 50 |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest/text` | Ingest raw text |
| `POST` | `/query` | Full RAG query |
| `GET` | `/retrieve?q=...` | Retrieve chunks only |
| `GET` | `/health` | Liveness and store stats |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
mypy src/
```

## Design notes

**Why abstract base classes.** `BaseEmbedder` and `BaseGenerator` define interfaces independently of implementation. The pipeline composes them and has no import dependency on OpenAI or Ollama, so a provider swaps without touching pipeline code.

**Why Pydantic settings.** All config lives in `config.py` and is validated at startup. A missing variable fails fast with a clear error rather than surfacing as a silent `None` three layers deep.

**Why multi-stage Docker.** The `dev` target mounts source for hot reload; the `prod` target strips dev tools and runs as a non-root user. One Dockerfile, two build targets, no duplication.

## How it fits into Atlas Systems

This is the reference implementation the rest of the RAG work is built from. [`ollama-rag-kit`](https://github.com/AtlasReaper311/ollama-rag-kit) takes the same pipeline shape and packages it as a multi-container service; the honours project reuses the architecture against UE5 audio documentation.

The transferable pattern is programming to interfaces: when the pipeline depends on a `BaseEmbedder` rather than a named vendor, the vendor becomes a config value instead of a rewrite.

---

Part of [atlas-systems.uk](https://atlas-systems.uk) · MIT License
