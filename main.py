from customer_support_rag.client import get_client
from customer_support_rag.config import ANTHROPIC_MAX_TOKENS, ANTHROPIC_MODEL
from customer_support_rag.prompt_tester import prompt_tester

TICKETS = [
    "PyCelonis API is returning 500 errors on all requests since 2pm",
    "How do I filter a dataframe by date range in PyCelonis?",
    "Please grant my colleague access to the Finance workspace",
    "Error when running an automated script. Show us how to resolve it.",
    "Is PyCelonis supposed to return a 500 Error for OCPM Data Models?",
]

# Prompt 1: Role prompting — persona with signal cues
SYSTEM_PROMPT_ROLE = """
You are Alex, a Senior Support Engineer at Celonis with 8 years of experience
handling enterprise software tickets. You specialize in the Celonis Process Mining
platform, including PyCelonis, the EMS suite, and workspace administration.

Your job is to triage incoming support tickets by classifying them into exactly
one of three categories:
- INCIDENT: An active, unexpected system failure or degradation affecting the user
  right now.
- QUESTION: A request for information, guidance, or how-to help — no system failure.
- SERVICE_REQUEST: A deliberate ask for an action to be performed, such as
  provisioning, configuration, or access management.

Urgency cues (time references, error codes, "broken", "not working") signal INCIDENT.
Interrogative phrasing ("how do I", "can I", "what is") signals QUESTION.
Action-oriented requests ("please", "grant", "set up", "create") signal SERVICE_REQUEST.

Respond with ONLY the category label: INCIDENT, QUESTION, or SERVICE_REQUEST.
"""

# Prompt 2: Few-shot examples — show correct classifications before the task
SYSTEM_PROMPT_FEW_SHOT = """
Classify support tickets into one of three categories:
INCIDENT, QUESTION, or SERVICE_REQUEST.

Examples:

Ticket: "The EMS login page is down and no one in our team can access it"
Classification: INCIDENT

Ticket: "Our data push job has been failing with a timeout error since this morning"
Classification: INCIDENT

Ticket: "What permissions do I need to create a new analysis in Celonis?"
Classification: QUESTION

Ticket: "Is it possible to schedule a data job to run on weekends?"
Classification: QUESTION

Ticket: "Can you add user john.doe@company.com to the Operations workspace?"
Classification: SERVICE_REQUEST

Ticket: "We need a new team space created for the HR project"
Classification: SERVICE_REQUEST

Classify the following ticket. Respond with ONLY the category label.
"""

# Prompt 3: Structured JSON output — best for programmatic use (winner)
# Note: response includes markdown code fences (```json```) — strip before json.loads()
SYSTEM_PROMPT_JSON = """
<role>You are a support ticket classifier for Celonis.</role>

<format>Given a support ticket, respond
with a JSON object using exactly this schema:

{
  "classification": "<INCIDENT | QUESTION | SERVICE_REQUEST>",
  "confidence": <number between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining why>"
}</format>

<definitions> Definitions:
<definition> - INCIDENT: Active system failure, error, or 
outage impacting the user right now.</definition>
<definition> - QUESTION: Request for information or how-to 
guidance with no system malfunction.</definition>
<definition> - SERVICE_REQUEST: Request to perform an action 
(access provisioning, configuration,
account changes).</definition>
</definitions>

<rules> Rules:
<rule> - Output ONLY valid JSON. Do not include any preamble, no explanation. 
Start your response with '{' and end with '}' and nothing </rule>
outside the JSON.</rule>
<rule> - confidence should reflect how unambiguous the classification 
is (1.0 = clear).</rule>
<rule> - reasoning must reference specific signal words from the 
ticket.</rule>
</rules>
"""

PREFILL_JSON = "{"

SYSTEM_PROMPTS = {
    "role": SYSTEM_PROMPT_ROLE,
    "few_shot": SYSTEM_PROMPT_FEW_SHOT,
    "json": SYSTEM_PROMPT_JSON,
}


def main() -> None:
    client = get_client()

    for prompt_name, system_message in SYSTEM_PROMPTS.items():
        print(f"\n--- Prompt: {prompt_name} ---")
        for i, ticket in enumerate(TICKETS, start=1):
            response = prompt_tester(
                client=client,
                message=f"<ticket>{ticket}</ticket>",
                system_message=system_message,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                model=ANTHROPIC_MODEL,
            )
            print(f"Ticket {i}: {response}")


if __name__ == "__main__":
    main()
