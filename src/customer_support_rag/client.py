import anthropic

from .config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL


def get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY must be set in the environment variables.")
    return anthropic.Anthropic(base_url=ANTHROPIC_BASE_URL, api_key=ANTHROPIC_API_KEY)
