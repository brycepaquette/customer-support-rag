from functools import lru_cache

from langfuse import observe
from sentence_transformers import CrossEncoder

from .config import RERANK_MODEL
from .models import RetrievedChunk


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder. Cached so we pay the load cost once."""
    return CrossEncoder(RERANK_MODEL)


@observe(name="rerank")
def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Reorder candidates by cross-encoder relevance; return the top_k.

    Preserves each chunk's existing `similarity_score` (dense cosine). Only the
    ordering changes — confidence/score semantics elsewhere stay intact.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return [c for c, _ in ranked[:top_k]]
