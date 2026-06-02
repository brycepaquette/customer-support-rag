import json
from unittest.mock import MagicMock

import pytest

from customer_support_rag.classifier import classify_ticket


def test_classify_incident(mocker):
    mock_response = json.dumps(
        {
            "issue_type": "INCIDENT",
            "confidence": 0.95,
            "reasoning": "System is completely down affecting all users",
        }
    )

    mocker.patch(
        "customer_support_rag.classifier.prompt_tester", return_value=mock_response
    )
    result = classify_ticket(
        mocker.MagicMock(), "Our system is down and we can't access anything!"
    )
    assert result.issue_type == "INCIDENT"
    assert result.confidence == pytest.approx(0.95)


def test_classify_normalizes_lowercase(mocker):
    mock_response = json.dumps(
        {
            "issue_type": "question",
            "confidence": 0.9,
            "reasoning": "Customer is asking for documentation help",
        }
    )
    mocker.patch(
        "customer_support_rag.classifier.prompt_tester", return_value=mock_response
    )
    result = classify_ticket(mocker.MagicMock(), "How do I reset my password?")
    assert result.issue_type == "QUESTION"
    assert result.confidence == pytest.approx(0.9)


def test_classify_invalid_json(mocker):
    mocker.patch(
        "customer_support_rag.classifier.prompt_tester",
        return_value="Not a JSON response",
    )
    with pytest.raises(ValueError, match="Failed to parse Claude response as JSON"):
        classify_ticket(mocker.MagicMock(), "This is a test ticket")


def test_classify_bad_confidence_raises(mocker):
    mock_response = json.dumps(
        {"issue_type": "INCIDENT", "confidence": 1.5, "reasoning": "System is down"}
    )
    mocker.patch(
        "customer_support_rag.classifier.prompt_tester", return_value=mock_response
    )
    with pytest.raises(ValueError, match="failed validation"):
        classify_ticket(mocker.MagicMock(), "Some ticket")


def test_classify_empty_ticket_text():
    with pytest.raises(ValueError, match="Ticket text cannot be empty or whitespace"):
        classify_ticket(MagicMock(), ticket_text="")


def test_classify_valid_confidence(mocker):
    mock_response = json.dumps(
        {"issue_type": "QUESTION", "confidence": 0.0, "reasoning": "Asking for help"}
    )
    mocker.patch(
        "customer_support_rag.classifier.prompt_tester", return_value=mock_response
    )
    result = classify_ticket(
        mocker.MagicMock(), ticket_text="How do I reset my password?"
    )
    assert result.confidence == pytest.approx(0.0)


def test_classify_bad_issue_type(mocker):
    mock_response = json.dumps(
        {"issue_type": "UNKNOWN", "confidence": 0.5, "reasoning": "Unknown issue type"}
    )
    mocker.patch(
        "customer_support_rag.classifier.prompt_tester", return_value=mock_response
    )
    with pytest.raises(ValueError, match="failed validation"):
        classify_ticket(mocker.MagicMock(), ticket_text="Some ticket")
