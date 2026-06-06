import json

from langfuse import observe

from .client import get_client
from .config import REFUSAL_TEXT
from .models import RAGResponse
from .prompt_tester import prompt_tester, strip_code_fences
from .retrieval import retrieve
from .vector_store import HybridStore

SYSTEM_PROMPT = (
    f"""You are a Celonis support assistant. Answer questions using ONLY the information in the <context> tags provided in the user message.

Rules:
1. You may only use information explicitly stated in the context. Do not use prior knowledge.
2. If the answer is not directly supported by the context, you MUST refuse. Set answer to "{REFUSAL_TEXT}", confidence to 0.0, and sources to [].
3. When you do answer, cite only the source URLs whose content you actually used. Do not cite sources you did not draw from.
4. Estimate your confidence (0.0-1.0) based on how directly the context answers the question.

Respond with a single JSON object matching this schema (no markdown, no prose outside the JSON):
"""
    + """{
  "answer": "<your answer string>",
  "sources": ["<url1>", "<url2>"],
  "confidence": <float between 0.0 and 1.0>
}"""
)


@observe(name="rag_query")
def rag_query(
    query: str,
    store: HybridStore,
    top_k: int = 5,
) -> RAGResponse:
    retrieved = retrieve(query, store, top_k=top_k)

    context = "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in retrieved)
    user_message = f"<context>\n{context}\n</context>\n\n<question>{query}</question>"

    raw = prompt_tester(
        client=get_client(),
        system_message=SYSTEM_PROMPT,
        message=user_message,
    )

    parsed = json.loads(strip_code_fences(raw))

    valid_sources = {c.source for c in retrieved}
    cited = [s for s in parsed["sources"] if s in valid_sources]

    # Apply hybrid confidence: min(claude, top_similarity)
    claude_conf = float(parsed["confidence"])
    top_similarity = retrieved[0].similarity_score if retrieved else 0.0
    confidence = min(claude_conf, top_similarity)

    return RAGResponse(
        answer=parsed["answer"],
        sources=cited,
        retrieved_chunks=retrieved,
        confidence=confidence,
    )


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if Claude wrapped the JSON."""
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence (with or without language)
        text = text.split("\n", 1)[1] if "\n" in text else text
        # drop closing fence
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()
