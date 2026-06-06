from anthropic import Anthropic, types
from langfuse import get_client, observe

from .config import ANTHROPIC_MAX_TOKENS, ANTHROPIC_MODEL, ANTHROPIC_TEMPERATURE


@observe(as_type="generation", name="anthropic_messages")
def prompt_tester(
    client: Anthropic,
    system_message: str,
    message: str,
    temperature: float = ANTHROPIC_TEMPERATURE,
    max_tokens: int = ANTHROPIC_MAX_TOKENS,
    model: str = ANTHROPIC_MODEL,
) -> str:
    langfuse = get_client()
    langfuse.update_current_generation(
        model=model,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ],
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
    )

    response = client.messages.create(
        max_tokens=max_tokens,
        model=model,
        temperature=temperature,
        system=system_message,
        messages=[{"role": "user", "content": message}],
    )

    block = response.content[0]
    if not isinstance(block, types.TextBlock):
        raise ValueError(f"Unexpected block type: {type(block)}")

    langfuse.update_current_generation(
        output=block.text,
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    )
    return block.text


def strip_code_fences(s: str) -> str:
    """Strip ```...``` fences (with or without a language tag) from a model response."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    return s.strip()
