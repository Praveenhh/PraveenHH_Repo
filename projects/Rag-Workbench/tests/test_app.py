from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app import RAGEngine, create_app


def fake_embed(texts: list[str]) -> np.ndarray:
    # Deterministic two-dimensional vectors make retrieval tests model-free.
    return np.array([[len(text), text.lower().count("return") + 1] for text in texts], dtype=np.float32)


def test_chunks_documents_and_retrieves_sources() -> None:
    engine = RAGEngine(fake_embed, lambda prompt: "Returns are accepted within 30 days.", chunk_size=40, chunk_overlap=5)
    count = engine.add_document("Policy", "Customers may return unused products within 30 days.")

    result = engine.ask("What is the return period?", top_k=2)

    assert count >= 1
    assert result["answer"] == "Returns are accepted within 30 days."
    assert result["sources"][0]["title"] == "Policy"
    assert result["retrieval_count"] == len(result["sources"])


def test_api_health_and_empty_index_error() -> None:
    engine = RAGEngine(fake_embed, lambda prompt: "A response.")
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/chat", json={"question": "Hello"})

    assert response.status_code == 422
    assert "Add at least one document" in response.json()["detail"]


def test_api_ingests_and_answers() -> None:
    engine = RAGEngine(fake_embed, lambda prompt: "The policy allows returns.")
    with TestClient(create_app(engine)) as client:
        added = client.post("/documents", json={"title": "Policy", "text": "Returns are allowed within 30 days."})
        answer = client.post("/chat", json={"question": "What is the return policy?", "top_k": 1})

    assert added.status_code == 200
    assert added.json()["chunks_added"] >= 1
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["title"] == "Policy"


def test_invalid_chunk_configuration_is_rejected() -> None:
    try:
        RAGEngine(fake_embed, lambda prompt: "answer", chunk_size=10, chunk_overlap=10)
    except ValueError as error:
        assert "chunk_size" in str(error)
    else:
        raise AssertionError("Invalid chunk configuration should fail")
