"""Score a saved eval run (evals/results.json) and write evals/metrics.json."""

from __future__ import annotations

from pathlib import Path

from customer_support_rag.metrics import summarize_run

RESULTS_PATH = Path("evals/results.json")
METRICS_PATH = Path("evals/metrics.json")


def main() -> None:
    summary = summarize_run(RESULTS_PATH)
    counts = f"{summary.n_non_edge} non-edge, {summary.n_edge} edge"
    print(f"Eval run summary ({RESULTS_PATH}):")
    print(f"  cases:           {summary.n_cases} ({counts})")
    print(f"  errors:          {summary.n_errors}")
    print(f"  recall@1:        {summary.recall_at_1:.3f}")
    print(f"  recall@5:        {summary.recall_at_5:.3f}")
    print(f"  recall@10:       {summary.recall_at_10:.3f}")
    print(f"  refusal_acc:     {summary.refusal_accuracy:.3f}")
    print(f"  mean_confidence: {summary.mean_confidence:.3f}")

    METRICS_PATH.write_text(summary.model_dump_json(indent=2) + "\n")
    print(f"\nWrote summary to {METRICS_PATH}")


if __name__ == "__main__":
    main()
