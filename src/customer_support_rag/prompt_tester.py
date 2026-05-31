from anthropic import Anthropic, types

from .config import ANTHROPIC_MAX_TOKENS, ANTHROPIC_MODEL, ANTHROPIC_TEMPERATURE


def prompt_tester(
    client: Anthropic,
    message: str,
    system_message: str,
    temperature: float = ANTHROPIC_TEMPERATURE,
    max_tokens: int = ANTHROPIC_MAX_TOKENS,
    model: str = ANTHROPIC_MODEL,
) -> str:
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
    return block.text
