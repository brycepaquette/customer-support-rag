from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TicketClassification(BaseModel):
    issue_type: Literal["INCIDENT", "QUESTION", "SERVICE_REQUEST"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10)

    @field_validator("issue_type", mode="before")
    @classmethod
    def normalize_issue_type(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "_")
