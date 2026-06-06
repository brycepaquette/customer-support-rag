from __future__ import annotations

from typing import Any

from customer_support_rag.config import REFUSAL_TEXT
from customer_support_rag.metrics import (
    mean_confidence,
    recall_at_k,
    refusal_accuracy,
    summarize,
)


def _factual(
    qid: str, expected: str, retrieved: list[str], conf: float = 0.7
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question_type": "factual",
        "expected_source": expected,
        "actual_answer": "some answer",
        "actual_sources": retrieved[:1],
        "confidence": conf,
        "retrieved_sources": retrieved,
    }


def _edge(qid: str, refused: bool, conf: float = 0.0) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question_type": "edge",
        "expected_source": "N/A",
        "actual_answer": REFUSAL_TEXT if refused else "I made something up",
        "actual_sources": [] if refused else ["https://example.com"],
        "confidence": conf,
        "retrieved_sources": ["https://other.com"],
    }


def test_recall_at_k_all_hit() -> None:
    results = [
        _factual("q001", "https://a", ["https://a", "https://b"]),
        _factual("q002", "https://b", ["https://x", "https://b"]),
    ]
    assert recall_at_k(results, 5) == 1.0


def test_recall_at_k_respects_k_slice() -> None:
    results = [
        _factual("q001", "https://target", ["https://noise", "https://target"]),
    ]
    assert recall_at_k(results, 1) == 0.0
    assert recall_at_k(results, 2) == 1.0


def test_recall_at_k_ignores_edge_cases() -> None:
    results = [
        _factual("q001", "https://a", ["https://a"]),
        _edge("q002", refused=True),
    ]
    assert recall_at_k(results, 5) == 1.0


def test_recall_at_k_no_non_edge() -> None:
    assert recall_at_k([_edge("q001", refused=True)], 5) == 0.0


def test_recall_at_k_skips_errored_cases() -> None:
    results = [
        {"question_id": "q001", "question_type": "factual", "error": "boom"},
        _factual("q002", "https://a", ["https://a"]),
    ]
    assert recall_at_k(results, 5) == 1.0


def test_refusal_accuracy_strict_on_answer_and_sources() -> None:
    results = [
        _edge("q001", refused=True),
        _edge("q002", refused=False),
        _edge("q003", refused=True),
    ]
    assert refusal_accuracy(results) == 2 / 3


def test_refusal_accuracy_rejects_refusal_text_with_citation() -> None:
    bad = _edge("q001", refused=True)
    bad["actual_sources"] = ["https://hallucinated"]
    assert refusal_accuracy([bad]) == 0.0


def test_refusal_accuracy_no_edges() -> None:
    assert refusal_accuracy([_factual("q001", "https://a", ["https://a"])]) == 0.0


def test_mean_confidence_ignores_missing() -> None:
    results = [
        {"question_id": "q001", "question_type": "factual", "error": "boom"},
        _factual("q002", "https://a", ["https://a"], conf=0.6),
        _factual("q003", "https://b", ["https://b"], conf=1.0),
    ]
    assert mean_confidence(results) == 0.8


def test_summarize_counts_and_metrics() -> None:
    results = [
        _factual("q001", "https://a", ["https://a"], conf=0.9),
        _factual("q002", "https://b", ["https://noise"], conf=0.4),
        _edge("q003", refused=True),
        _edge("q004", refused=False),
        {"question_id": "q005", "question_type": "factual", "error": "boom"},
    ]
    s = summarize(results)
    assert s.n_cases == 5
    assert s.n_non_edge == 3
    assert s.n_edge == 2
    assert s.n_errors == 1
    assert s.recall_at_5 == 0.5
    assert s.refusal_accuracy == 0.5
    assert s.mean_confidence == (0.9 + 0.4 + 0.0 + 0.0) / 4
