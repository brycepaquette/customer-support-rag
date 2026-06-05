"""End-to-end RAG smoke test — runs 5 known questions + 1 out-of-domain probe."""

from pathlib import Path

import chromadb
from langfuse import observe, propagate_attributes

from customer_support_rag.generator import rag_query
from customer_support_rag.ingestion import chunk_corpus, load_corpus
from customer_support_rag.vector_store import build_vector_store

QUESTIONS = [
    "How do I authenticate with the Celonis API?",
    "How can I install a REST Client?",
    "What are the rate limits for the MCP API?",
    "How do I export data from a data pool?",
    "How to retrieve a record by its ID?",
    "What is the capital of France?",  # out-of-domain probe — must refuse
]


@observe(name="eval_rag_question")
def run_question(question: str, collection: chromadb.Collection) -> None:
    with propagate_attributes(
        tags=["test-rag-eval"],
        metadata={"question": question},
    ):
        print(f"\n{'=' * 80}")
        print(f"Q: {question}")
        print("=" * 80)

        response = rag_query(question, collection, top_k=3)

        print(f"\nAnswer (confidence={response.confidence:.3f}):")
        print(f"  {response.answer}")
        print(f"\nSources cited ({len(response.sources)}):")
        for s in response.sources:
            print(f"  - {s}")
        print(f"\nRetrieved chunks ({len(response.retrieved_chunks)}):")
        for c in response.retrieved_chunks:
            print(f"  [score={c.similarity_score:.3f}] {c.source}")


def main() -> None:
    chunks = chunk_corpus(load_corpus(Path("./corpus")))
    collection = build_vector_store(chunks)

    for question in QUESTIONS:
        run_question(question, collection)


if __name__ == "__main__":
    main()
