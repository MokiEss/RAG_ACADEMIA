from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import get_settings


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(get_settings().reranker_model, max_length=512)


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    model = get_reranker()
    pairs = [(query, p) for p in passages]
    scores = model.predict(pairs, batch_size=16)
    return [float(s) for s in scores]