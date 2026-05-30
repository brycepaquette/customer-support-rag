import anthropic

from .config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL


def get_client() -> anthropic.Anthropic:
    client = anthropic.Anthropic(base_url=ANTHROPIC_BASE_URL, api_key=ANTHROPIC_API_KEY)
    return client
