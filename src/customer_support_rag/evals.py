"""Eval set models and I/O for the RAG regression suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["factual", "conceptual", "multi_hop", "edge"]
Difficulty = Literal["easy", "medium", "hard"]


class EvalCase(BaseModel):
    question_id: str = Field(pattern=r"^q\d{3}$")
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    source_doc: str = Field(min_length=1)
    difficulty: Difficulty
    question_type: QuestionType


def load_eval_set(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text())
    return [EvalCase.model_validate(item) for item in data]


def save_eval_set(eval_cases: list[EvalCase], path: Path) -> None:
    path.write_text(json.dumps([c.model_dump() for c in eval_cases], indent=2) + "\n")
