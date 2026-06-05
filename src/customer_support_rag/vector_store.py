from dataclasses import dataclass

import bm25s
import chromadb
from chromadb import Collection

from .config import COLLECTION_NAME
from .embedder import embed_texts
from .models import Chunk


@dataclass
class HybridStore:
    """Dense (Chroma) + sparse (BM25) indices over the same chunk set.

    `chunks` is parallel to the BM25 index: position i in BM25 corresponds
    to chunks[i]. Chroma is keyed by chunk_id.
    """

    chroma: Collection
    bm25: bm25s.BM25
    chunks: list[Chunk]


def build_vector_store(
    chunks: list[Chunk],
    persist_dir: str = ".chroma",
) -> HybridStore:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == len(chunks):
        print(f"Collection already has {len(chunks)} chunks. Skipping upsert.")
    else:
        if collection.count() > 0:
            print(
                f"Chunk count changed ({collection.count()} -> {len(chunks)}); "
                "recreating collection to clear stale ids."
            )
            client.delete_collection(COLLECTION_NAME)
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "chunk_index": c.chunk_index} for c in chunks
            ],
        )
        print(f"Upserted {len(chunks)} chunks to Chroma")

    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize([c.text for c in chunks], stopwords="en"))
    print(f"Indexed {len(chunks)} chunks in BM25")

    return HybridStore(chroma=collection, bm25=bm25, chunks=chunks)


if __name__ == "__main__":
    from pathlib import Path

    from .ingestion import chunk_corpus, load_corpus

    chunks = chunk_corpus(load_corpus(Path("./corpus")))
    print(f"Loaded {len(chunks)} chunks")
    store = build_vector_store(chunks)
    print(f"Collection count: {store.chroma.count()}")
