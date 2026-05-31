# customer-support-rag

A RAG pipeline over Celonis public docs with Anthropic SDK, Pydantic structured outputs, Chroma vector store, and Langfuse observability.

## Prompt Engineering Experiments

Compared three prompting techniques for support ticket classification (INCIDENT / QUESTION / SERVICE_REQUEST):

| Technique | Output | Result |
|-----------|--------|--------|
| Role prompting | Label only | Correct |
| Few-shot examples | Label only | Correct |
| Structured JSON output | JSON with classification, confidence, reasoning | Correct |

All three techniques produced correct classifications. **Structured JSON output was selected as the production approach** — it is the most useful for application development as it returns a parseable object including a confidence score and reasoning, not just a bare label.

> **Note:** Claude wraps JSON responses in markdown code fences (` ```json ``` `) even when instructed not to. Strip these before passing to `json.loads()`:
> ```python
> text.strip().removeprefix("```json").removesuffix("```").strip()
> ```

## Setup

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
uv sync --dev
uv run python main.py
```
