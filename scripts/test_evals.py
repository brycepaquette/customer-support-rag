"""Run the full eval set through the RAG pipeline and write results to JSON."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from langfuse import observe, propagate_attributes

from customer_support_rag.evals import EvalCase, load_eval_set
from customer_support_rag.generator import rag_query
from customer_support_rag.ingestion import chunk_corpus, load_corpus
from customer_support_rag.models import RAGResponse
from customer_support_rag.vector_store import build_vector_store

EVAL_SET_PATH = Path("evals/eval_set.json")
RESULTS_PATH = Path("evals/results.json")
CORPUS_PATH = Path("./corpus")
TOP_K = 10


@observe(name="eval_case")
def run_case(case: EvalCase, collection: chromadb.Collection) -> RAGResponse:
    with propagate_attributes(
        tags=["eval-run"],
        metadata={
            "question_id": case.question_id,
            "difficulty": case.difficulty,
            "question_type": case.question_type,
        },
    ):
        return rag_query(case.question, collection, top_k=TOP_K)


def main() -> None:
    cases = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(cases)} eval cases")

    print("Building vector store from corpus...")
    collection = build_vector_store(chunk_corpus(load_corpus(CORPUS_PATH)))

    results: list[dict] = []
    for i, case in enumerate(cases, 1):
        label = f"{case.question_id} ({case.difficulty}/{case.question_type})"
        print(f"\n[{i}/{len(cases)}] {label}")
        print(f"  Q: {case.question}")
        try:
            response = run_case(case, collection)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "question_id": case.question_id,
                    "question": case.question,
                    "difficulty": case.difficulty,
                    "question_type": case.question_type,
                    "expected_answer": case.expected_answer,
                    "expected_source": case.source_doc,
                    "error": str(e),
                }
            )
            continue

        print(f"  A: ({response.confidence:.2f}): {response.answer[:140]}")
        print(f"  cited: {response.sources}")
        results.append(
            {
                "question_id": case.question_id,
                "question": case.question,
                "difficulty": case.difficulty,
                "question_type": case.question_type,
                "expected_answer": case.expected_answer,
                "expected_source": case.source_doc,
                "actual_answer": response.answer,
                "actual_sources": response.sources,
                "confidence": response.confidence,
                "retrieved_sources": [c.source for c in response.retrieved_chunks],
            }
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {len(results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
