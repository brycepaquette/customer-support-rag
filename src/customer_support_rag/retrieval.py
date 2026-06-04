import chromadb
from langfuse import get_client, observe

from .embedder import embed_texts
from .models import Chunk, RetrievedChunk


@observe(name="retrieve")
def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    retrieved_chunks: list[RetrievedChunk] = []
    ids = results["ids"][0]
    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]
    if documents is None or metadatas is None or distances is None:
        raise RuntimeError("Chroma query returned None for an included field")

    docs_row = documents[0]
    metas_row = metadatas[0]
    dists_row = distances[0]

    for chunk_id, doc, meta, dist in zip(
        ids, docs_row, metas_row, dists_row, strict=True
    ):
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=doc,
                source=str(meta["source"]),
                chunk_index=int(str(meta["chunk_index"])),
                token_count=Chunk.estimate_tokens(doc),
                similarity_score=max(0.0, min(1.0, 1.0 - dist)),
            )
        )
    get_client().update_current_span(
        metadata={
            "top_k": top_k,
            "num_results": len(retrieved_chunks),
            "top_score": retrieved_chunks[0].similarity_score
            if retrieved_chunks
            else None,
            "sources": [r.source for r in retrieved_chunks],
        }
    )
    return retrieved_chunks


if __name__ == "__main__":
    from pathlib import Path

    from .ingestion import chunk_corpus, load_corpus
    from .vector_store import build_vector_store

    chunks = chunk_corpus(load_corpus(Path("./corpus")))
    collection = build_vector_store(chunks)

    query = "How do I authenticate with the Celonis API?"
    results = retrieve(query, collection, top_k=3)

    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [score={r.similarity_score:.3f}] {r.source}")
        print(f"   {r.text[:150]}...\n")
