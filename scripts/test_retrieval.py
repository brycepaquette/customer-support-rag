"""Manual retrieval smoke test - runs 5 known queries and prints the top 3 chunks.

Day 11 retrieval observations:
1. Authentication queries: excellent (scores 0.7+, all correct sources)
2. "Install REST client": correct top-1 but low score (0.389). Boilerplate
   nav/footer text in chunks is creating noise — many pages match on the
   "Copy for LLM / Open in ChatGPT" header repeated everywhere.
3. Rare acronyms (MCP): corpus has 139 mentions, but embedding model
   (all-MiniLM-L6-v2, ~2020 training data) has no strong representation
   for the acronym. It embeds "MCP" as a low-information token and falls
   back to matching on "API" / "rate limits". Textbook case for BM25 hybrid.
4. Directional verbs (export vs ingest): embeddings conflate opposites.
   This is the strongest argument for hybrid (BM25 + embeddings) in Week 3.
5. Phrasing mismatch ("retrieve by ID" vs "retrieving record data"):
   correct answer ranks #2 instead of #1. Re-ranker would help.

Week 3 priority order:
  a) Strip boilerplate from chunks before embedding
  b) Hybrid retrieval (BM25 + dense)
  c) Cross-encoder re-ranker on top-10
"""

from pathlib import Path

import chromadb
from langfuse import observe, propagate_attributes

from customer_support_rag.ingestion import chunk_corpus, load_corpus
from customer_support_rag.retrieval import retrieve
from customer_support_rag.vector_store import build_vector_store

QUESTIONS = [
    "How do I authenticate with the Celonis API?",
    "How can I install a REST Client?",
    "What are the rate limits for the MCP API?",
    "How do I export data from a data pool?",
    "How to retrieve a record by its ID?",
]


@observe(name="eval_question")
def run_question(question: str, collection: chromadb.Collection) -> None:
    with propagate_attributes(
        tags=["test-retrieval-eval"],
        metadata={"question": question},
    ):
        print(f"\n{'=' * 80}")
        print(f"Q: {question}")
        print("=" * 80)

        results = retrieve(question, collection, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  {i}. [score={r.similarity_score:.3f}] {r.source}")
            print(f"     {r.text[:180].strip()}...")


def main() -> None:
    chunks = chunk_corpus(load_corpus(Path("./corpus")))
    collection = build_vector_store(chunks)

    for question in QUESTIONS:
        run_question(question, collection)


if __name__ == "__main__":
    main()
