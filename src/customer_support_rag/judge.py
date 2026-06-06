import json

from langfuse import observe
from pydantic import BaseModel, Field

from .client import get_client
from .prompt_tester import prompt_tester, strip_code_fences

JUDGE_SYSTEM_PROMPT = """You are an impartial grader scoring a customer-support assistant's answer against a reference answer.

You will be given:
- <question>: the user's question
- <expected>: the reference answer (ground truth)
- <actual>: the assistant's answer

Score the assistant's answer 1–5 on factual correctness and completeness relative to the expected answer:
  5 = fully correct and complete; conveys the same facts as the expected answer
  4 = mostly correct; minor omission or imprecision, no factual error
  3 = partially correct; meaningful gap or one notable inaccuracy
  2 = largely wrong; some thread of truth but mostly inaccurate or misleading
  1 = wrong or fabricated

Ignore stylistic differences, ordering, and phrasing. Score only on factual alignment.

Respond with a single JSON object (no markdown, no prose outside the JSON):
{
  "score": <integer 1-5>,
  "reasoning": "<one or two sentences explaining the score>"
}"""


class EvalJudgement(BaseModel):
    question_id: str
    score: int = Field(ge=1, le=5)
    reasoning: str = Field(min_length=1)


@observe(name="judge_case")
def judge_case(
    question_id: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> EvalJudgement:
    user_message = (
        f"<question>{question}</question>\n"
        f"<expected>{expected_answer}</expected>\n"
        f"<actual>{actual_answer}</actual>"
    )
    raw = prompt_tester(
        client=get_client(),
        system_message=JUDGE_SYSTEM_PROMPT,
        message=user_message,
    )
    parsed = json.loads(strip_code_fences(raw))
    return EvalJudgement(
        question_id=question_id,
        score=int(parsed["score"]),
        reasoning=str(parsed["reasoning"]),
    )
