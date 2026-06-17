"""
Tests for the FastAPI routes in atlas_rag.api.routes.

The app's lifespan event builds a real RAGPipeline via
RAGPipeline.from_settings(), which needs a real embedding model, vector
store, and LLM client. TestClient only runs lifespan when used as a context
manager (`with TestClient(app) as client:`), so these tests deliberately
skip that form: create_app() builds the app, a fake pipeline is assigned
to app.state.pipeline directly, and TestClient is instantiated plainly.
That's what lets this run with no network, no model download, no live
services, and no API keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from atlas_rag.api.app import create_app


@dataclass
class FakeSearchResult:
    text: str
    score: float = 0.9
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeQueryResult:
    answer: str
    model: str = "fake-model"
    total_tokens: int = 42


class FakeStore:
    def __init__(self, count: int = 7) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class FakePipeline:
    """Duck-types the subset of RAGPipeline the routes actually call.
    fail_ingest / fail_query let individual tests force the 500 path
    without needing a second fake class."""

    def __init__(self) -> None:
        self._store = FakeStore()
        self.ingest_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.retrieve_calls: list[tuple[str, int]] = []
        self.fail_ingest = False
        self.fail_query = False

    def ingest_text(self, text: str, source_id: str, metadata: dict[str, Any] | None = None) -> int:
        if self.fail_ingest:
            raise RuntimeError("simulated ingest failure")
        self.ingest_calls.append((text, source_id, metadata or {}))
        return 3

    def query(self, question: str) -> FakeQueryResult:
        if self.fail_query:
            raise RuntimeError("simulated query failure")
        return FakeQueryResult(answer=f"answer to: {question}")

    def retrieve(self, query: str, top_k: int = 5) -> list[FakeSearchResult]:
        self.retrieve_calls.append((query, top_k))
        return [FakeSearchResult(text=f"result {i}") for i in range(min(top_k, 2))]


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def client(fake_pipeline: FakePipeline) -> TestClient:
    app = create_app()
    app.state.pipeline = fake_pipeline
    # Plain instantiation, deliberately not `with TestClient(app) as c:`,
    # so lifespan never runs and never overwrites the fake with a real
    # RAGPipeline.from_settings() call.
    return TestClient(app)


# ── /ingest/text ─────────────────────────────────────────────────────────

def test_ingest_text_success(client: TestClient, fake_pipeline: FakePipeline) -> None:
    response = client.post(
        "/ingest/text",
        json={"text": "hello world", "source_id": "doc-1", "metadata": {"tag": "test"}},
    )

    assert response.status_code == 200
    assert response.json() == {"source_id": "doc-1", "chunks_stored": 3}
    assert fake_pipeline.ingest_calls == [("hello world", "doc-1", {"tag": "test"})]


def test_ingest_text_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/ingest/text", json={"text": "", "source_id": "doc-1"})

    # min_length=1 on the request schema rejects this before the handler
    # body ever runs; this is FastAPI's own validation, not pipeline logic.
    assert response.status_code == 422


def test_ingest_text_pipeline_error_returns_500(client: TestClient, fake_pipeline: FakePipeline) -> None:
    fake_pipeline.fail_ingest = True

    response = client.post("/ingest/text", json={"text": "hello", "source_id": "doc-1"})

    assert response.status_code == 500
    assert "simulated ingest failure" in response.json()["detail"]


# ── /query ───────────────────────────────────────────────────────────────

def test_query_success(client: TestClient) -> None:
    response = client.post("/query", json={"question": "what is atlas systems?"})

    assert response.status_code == 200
    body = response.json()
    assert body == {"answer": "answer to: what is atlas systems?", "model": "fake-model", "total_tokens": 42}


def test_query_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_pipeline_error_returns_500(client: TestClient, fake_pipeline: FakePipeline) -> None:
    fake_pipeline.fail_query = True

    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 500
    assert "simulated query failure" in response.json()["detail"]


# ── /retrieve ────────────────────────────────────────────────────────────

def test_retrieve_uses_default_top_k(client: TestClient, fake_pipeline: FakePipeline) -> None:
    response = client.get("/retrieve", params={"q": "pets"})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert fake_pipeline.retrieve_calls == [("pets", 5)]  # 5 is the route's own default


def test_retrieve_respects_custom_top_k(client: TestClient, fake_pipeline: FakePipeline) -> None:
    response = client.get("/retrieve", params={"q": "pets", "top_k": 1})

    assert response.status_code == 200
    assert fake_pipeline.retrieve_calls == [("pets", 1)]


def test_retrieve_missing_query_param_is_rejected(client: TestClient) -> None:
    response = client.get("/retrieve")

    # q has no default in the route signature, so FastAPI treats it as
    # required and rejects the request before the handler ever runs.
    assert response.status_code == 422


# ── /health ──────────────────────────────────────────────────────────────

def test_health_reports_ok_and_store_count(client: TestClient, fake_pipeline: FakePipeline) -> None:
    fake_pipeline._store = FakeStore(count=11)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "store_count": 11}
