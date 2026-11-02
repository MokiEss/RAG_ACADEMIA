import sys

from app.retrieval.embeddings import embed_query
from app.retrieval.vectorstore import get_collection


def search(query: str, k: int = 5):
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = 1 - dist  # cosine distance -> similarity
        print(f"\n[{similarity:.3f}] {meta['source']} (page {meta['page']})")
        print(doc[:300].replace("\n", " ") + "...")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    search(question)