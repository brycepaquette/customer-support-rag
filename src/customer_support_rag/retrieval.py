from collections import defaultdict

import bm25s
from langfuse import get_client, observe

from .embedder import embed_texts
from .models import Chunk, RetrievedChunk
from .vector_store import HybridStore

RRF_K = 60
FETCH_PER_RANKER = 20  # over-fetch then fuse down to top_k


def _dense_search(store: HybridStore, query: str, n: int) -> list[tuple[str, float]]:
    """Return [(chunk_id, similarity_score), ...] from Chroma."""
    query_embedding = embed_texts([query])[0]
    results = store.chroma.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["distances"],
    )
    ids = results["ids"][0]
    distances = results["distances"]
    if distances is None:
        return []
    return [
        (cid, max(0.0, min(1.0, 1.0 - d)))
        for cid, d in zip(ids, distances[0], strict=True)
    ]


def _sparse_search(store: HybridStore, query: str, n: int) -> list[tuple[str, float]]:
    """Return [(chunk_id, bm25_score), ...] from BM25."""
    tokenized = bm25s.tokenize([query], stopwords="en")
    # bm25s returns (positions, scores) of shape (1, n)
    positions, scores = store.bm25.retrieve(tokenized, k=min(n, len(store.chunks)))
    return [
        (store.chunks[int(pos)].chunk_id, float(score))
        for pos, score in zip(positions[0], scores[0], strict=True)
    ]


def _rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[str]:
    """Reciprocal Rank Fusion. Returns chunk_ids sorted by fused score desc."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


@observe(name="retrieve")
def retrieve(
    query: str,
    store: HybridStore,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    dense_hits = _dense_search(store, query, FETCH_PER_RANKER)
    sparse_hits = _sparse_search(store, query, FETCH_PER_RANKER)

    fused_ids = _rrf_fuse([dense_hits, sparse_hits])[:top_k]

    # Build lookup tables for assembling the final RetrievedChunk objects.
    dense_scores = dict(dense_hits)
    chunk_by_id = {c.chunk_id: c for c in store.chunks}

    retrieved: list[RetrievedChunk] = []
    for chunk_id in fused_ids:
        c = chunk_by_id.get(chunk_id)
        if c is None:
            continue  # safety: id appeared in Chroma but not in chunk list
        # Use the dense similarity if available, otherwise 0.0. This keeps the
        # downstream `min(claude_conf, top_similarity)` heuristic well-defined.
        sim = dense_scores.get(chunk_id, 0.0)
        retrieved.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                chunk_index=c.chunk_index,
                token_count=Chunk.estimate_tokens(c.text),
                similarity_score=sim,
            )
        )

    get_client().update_current_span(
        metadata={
            "top_k": top_k,
            "fetch_per_ranker": FETCH_PER_RANKER,
            "num_results": len(retrieved),
            "top_score": retrieved[0].similarity_score if retrieved else None,
            "sources": [r.source for r in retrieved],
            "dense_top_sources": [
                chunk_by_id[cid].source
                for cid, _ in dense_hits[:5]
                if cid in chunk_by_id
            ],
            "sparse_top_sources": [
                chunk_by_id[cid].source
                for cid, _ in sparse_hits[:5]
                if cid in chunk_by_id
            ],
        }
    )
    return retrieved


if __name__ == "__main__":
    from pathlib import Path

    from .ingestion import chunk_corpus, load_corpus
    from .vector_store import build_vector_store

    chunks = chunk_corpus(load_corpus(Path("./corpus")))
    store = build_vector_store(chunks)

    query = "How do I authenticate with the Celonis API?"
    results = retrieve(query, store, top_k=3)

    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [score={r.similarity_score:.3f}] {r.source}")
        print(f"   {r.text[:150]}...\n")
