import chromadb
from chromadb import Collection

from .config import COLLECTION_NAME
from .embedder import embed_texts
from .models import Chunk


def build_vector_store(
    chunks: list[Chunk],
    persist_dir: str = ".chroma",
) -> Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == len(chunks):
        print(f"Collection already has {len(chunks)} chunks. Skipping upsert.")
        return collection
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    print(f"Upserted {len(chunks)} chunks to Chroma")
    return collection


if __name__ == "__main__":
    from pathlib import Path

    from .ingestion import chunk_corpus, load_corpus

    chunks = chunk_corpus(load_corpus(Path("./corpus")))
    print(f"Loaded {len(chunks)} chunks")
    collection = build_vector_store(chunks)
    print(f"Collection count: {collection.count()}")
