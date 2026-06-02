import pytest
from pydantic import ValidationError

from customer_support_rag.models import TicketClassification


def test_normalization_works() -> None:
    tc = TicketClassification(
        issue_type="incident",
        confidence=0.95,
        reasoning="System is completely down, affecting all users",
    )
    assert tc.issue_type == "INCIDENT"
    assert tc.confidence == pytest.approx(0.95)


def test_validation_catches_bad_confidence() -> None:
    with pytest.raises(ValidationError):
        TicketClassification(
            issue_type="INCIDENT",
            confidence=1.5,
            reasoning="test",
        )
