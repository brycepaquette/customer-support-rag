import json

from anthropic import Anthropic
from pydantic import ValidationError

from .models import TicketClassification
from .prompt_tester import prompt_tester

CLASSIFICATION_SYSTEM_PROMPT = """You are a Celonis support engineer classifying 
tickets.

<classification_rules>
- INCIDENT: An active, unexpected system failure or degradation affecting the user.
- QUESTION: A request for information, guidance, or how-to help — no system failure.
- SERVICE_REQUEST: A deliberate ask for an action such as provisioning or access.
</classification_rules>

Respond only with valid JSON matching this exact schema:
{"issue_type": "INCIDENT|QUESTION|SERVICE_REQUEST", "confidence": 0.0-1.0,
 "reasoning": "..."}"""


def classify_ticket(client: Anthropic, ticket_text: str) -> TicketClassification:

    if not ticket_text.strip():
        raise ValueError("Ticket text cannot be empty or whitespace")

    response_text = prompt_tester(
        client=client,
        system_message=CLASSIFICATION_SYSTEM_PROMPT,
        message=ticket_text,
    )
    clean = response_text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Claude response as JSON: {e}\nResponse: {clean}"
        ) from e
    try:
        return TicketClassification.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Claude response failed validation: {e}\nData: {data}") from e
