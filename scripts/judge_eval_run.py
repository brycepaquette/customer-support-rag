"""Run the LLM-as-judge over a saved eval run; write evals/judgements.json.

Judges only non-edge, non-errored cases. Edge-case correctness is already
covered by refusal_accuracy in metrics.py — judging refusals would
double-count.
"""

from __future__ import annotations

import json
from pathlib import Path

from customer_support_rag.judge import judge_case

RESULTS_PATH = Path("evals/results.json")
JUDGEMENTS_PATH = Path("evals/judgements.json")


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    to_judge = [
        r for r in results if r.get("question_type") != "edge" and "error" not in r
    ]
    print(f"Judging {len(to_judge)} non-edge cases from {RESULTS_PATH}")

    judgements: list[dict[str, object]] = []
    for i, r in enumerate(to_judge, 1):
        qid = r["question_id"]
        print(f"[{i}/{len(to_judge)}] {qid}")
        try:
            j = judge_case(
                question_id=qid,
                question=r["question"],
                expected_answer=r["expected_answer"],
                actual_answer=r["actual_answer"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            judgements.append({"question_id": qid, "error": str(e)})
            continue
        print(f"  score={j.score}  {j.reasoning[:100]}")
        judgements.append(j.model_dump())

    JUDGEMENTS_PATH.write_text(json.dumps(judgements, indent=2) + "\n")
    print(f"\nWrote {len(judgements)} judgements to {JUDGEMENTS_PATH}")


if __name__ == "__main__":
    main()
