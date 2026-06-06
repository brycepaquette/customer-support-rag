"""Pure scoring functions over eval-run results (evals/results.json shape)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import REFUSAL_TEXT


class RunSummary(BaseModel):
    n_cases: int = Field(ge=0)
    n_non_edge: int = Field(ge=0)
    n_edge: int = Field(ge=0)
    n_errors: int = Field(ge=0)
    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    refusal_accuracy: float = Field(ge=0.0, le=1.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    n_judged: int = Field(default=0, ge=0)
    mean_judge_score: float | None = Field(default=None, ge=1.0, le=5.0)


def _scoreable_non_edge(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("question_type") != "edge" and "error" not in r]


def _scoreable_edge(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("question_type") == "edge" and "error" not in r]


def recall_at_k(results: list[dict[str, Any]], k: int) -> float:
    non_edge = _scoreable_non_edge(results)
    if not non_edge:
        return 0.0
    hits = sum(
        1
        for r in non_edge
        if r["expected_source"] in set((r.get("retrieved_sources") or [])[:k])
    )
    return hits / len(non_edge)


def refusal_accuracy(results: list[dict[str, Any]]) -> float:
    edges = _scoreable_edge(results)
    if not edges:
        return 0.0
    correct = sum(
        1
        for r in edges
        if r.get("actual_answer") == REFUSAL_TEXT and r.get("actual_sources") == []
    )
    return correct / len(edges)


def mean_confidence(results: list[dict[str, Any]]) -> float:
    confidences = [float(r["confidence"]) for r in results if "confidence" in r]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def mean_judge_score(judgements: list[dict[str, Any]]) -> float:
    scores = [int(j["score"]) for j in judgements if "score" in j]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def summarize(results: list[dict[str, Any]]) -> RunSummary:
    return RunSummary(
        n_cases=len(results),
        n_non_edge=sum(1 for r in results if r.get("question_type") != "edge"),
        n_edge=sum(1 for r in results if r.get("question_type") == "edge"),
        n_errors=sum(1 for r in results if "error" in r),
        recall_at_1=recall_at_k(results, 1),
        recall_at_5=recall_at_k(results, 5),
        recall_at_10=recall_at_k(results, 10),
        refusal_accuracy=refusal_accuracy(results),
        mean_confidence=mean_confidence(results),
    )


def summarize_run(results_path: Path) -> RunSummary:
    return summarize(json.loads(results_path.read_text()))
