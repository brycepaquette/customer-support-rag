from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return cast(SentenceTransformer, SentenceTransformer(EMBEDDING_MODEL))


def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return cast(NDArray[np.float32], embeddings)
