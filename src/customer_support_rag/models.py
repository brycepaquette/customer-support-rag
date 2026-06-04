from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    source_url: str
    word_count: int


class TicketClassification(BaseModel):
    issue_type: Literal["INCIDENT", "QUESTION", "SERVICE_REQUEST"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10)

    @field_validator("issue_type", mode="before")
    @classmethod
    def normalize_issue_type(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "_")


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    token_count: int = Field(ge=1)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text using the ~0.75 words per token heuristic."""
        return max(1, int(len(text.split()) / 0.75))


class RetrievedChunk(Chunk):
    similarity_score: float = Field(ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk]
    confidence: float = Field(ge=0.0, le=1.0)
