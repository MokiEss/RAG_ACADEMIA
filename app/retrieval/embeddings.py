from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

# BGE models work best when queries (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # unit length, so cosine similarity works cleanly
        show_progress_bar=len(texts) > 64,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([QUERY_PREFIX + query])[0]