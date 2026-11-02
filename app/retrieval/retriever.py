from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.retrieval.embeddings import embed_query
from app.retrieval.reranker import rerank_scores
from app.retrieval.vectorstore import get_collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: int
    vector_score: float                    # cosine similarity from vector search
    vector_rank: int                       # position before reranking (1 = best)
    rerank_score: Optional[float] = None   # cross-encoder score (None if not reranked)

    @property
    def citation(self) -> str:
        return f"{self.source}, p.{self.page}"


def vector_search(query: str, k: int) -> list[RetrievedChunk]:
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                source=meta["source"],
                page=meta["page"],
                vector_score=1 - dist,
                vector_rank=rank,
            )
        )
    return chunks


def retrieve(
    query: str,
    k: Optional[int] = None,
    top_n: Optional[int] = None,
    rerank: bool = True,
) -> list[RetrievedChunk]:
    settings = get_settings()
    k = k or settings.retrieve_k
    top_n = top_n or settings.rerank_top_n

    candidates = vector_search(query, k)

    if not rerank:
        return candidates[:top_n]

    scores = rerank_scores(query, [c.text for c in candidates])
    for chunk, score in zip(candidates, scores):
        chunk.rerank_score = score

    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:top_n]