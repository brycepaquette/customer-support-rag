# customer-support-rag

A RAG pipeline over Celonis public docs with Anthropic SDK, Pydantic structured outputs, Chroma vector store, and Langfuse observability.

## Prompt Engineering Experiments

### Day 2 — Technique comparison

Compared three prompting techniques for support ticket classification (INCIDENT / QUESTION / SERVICE_REQUEST):

| Technique | Output | Result |
|-----------|--------|--------|
| Role prompting | Label only | Correct |
| Few-shot examples | Label only | Correct |
| Structured JSON output | JSON with classification, confidence, reasoning | Correct |

All three techniques produced correct classifications. **Structured JSON output was selected as the production approach** — it returns a parseable object with a confidence score and reasoning, not just a bare label.

> **Note:** Claude may wrap JSON responses in markdown code fences (` ```json ``` `). To prevent this, add an explicit rule to the system prompt: `"Always start your response with '{' and end with '}'. Output nothing after the closing brace."` This is more reliable than stripping fences after the fact and works regardless of gateway support for response prefilling.

### Day 3 — XML-structured prompt

Rewrote the structured JSON prompt using XML tags (`<role>`, `<format>`, `<definitions>`, `<rules>`) per Anthropic's prompt engineering guidelines. Each ticket is passed as `<ticket>{ticket}</ticket>` in the user message.

**Delta vs Day 2:** Classifications and confidence scores were identical across the original 3 tickets. The key improvement was output reliability — clean JSON without markdown fences, achieved via an explicit rule rather than post-processing.

**5-ticket test results:**

| Ticket | Classification | Confidence | Notes |
|--------|---------------|------------|-------|
| PyCelonis API returning 500 errors since 2pm | INCIDENT | 0.97 | Clear — time reference + error code |
| How do I filter a dataframe by date range? | QUESTION | 0.97 | Clear — interrogative phrasing |
| Please grant my colleague access to Finance workspace | SERVICE_REQUEST | 0.99 | Clear — action request |
| Error when running automated script, show us how to resolve it | INCIDENT | 0.60 | Ambiguous — "error" signals INCIDENT, "show us how to resolve" pulls toward QUESTION |
| Is PyCelonis supposed to return a 500 Error for OCPM Data Models? | QUESTION | 0.72 | Ambiguous — "supposed to" signals QUESTION but underlying 500 error may indicate an INCIDENT |

Lower confidence on tickets 4 and 5 correctly reflects genuine ambiguity — both could be classified differently depending on context not present in the ticket text. These are good candidates for the ground-truth eval set in Week 3.

## Setup

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
uv sync --dev
uv run python main.py
```
