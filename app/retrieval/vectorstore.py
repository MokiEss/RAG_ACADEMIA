from functools import lru_cache

import chromadb

from app.config import get_settings


@lru_cache
def get_collection():
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )