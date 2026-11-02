import sys

from app.retrieval.retriever import retrieve


def show(title: str, chunks):
    print(f"\n=== {title} ===")
    for i, c in enumerate(chunks, start=1):
        rr = f"rerank={c.rerank_score:.3f} " if c.rerank_score is not None else ""
        print(
            f"{i}. {rr}vec={c.vector_score:.3f} (was #{c.vector_rank}) "
            f"{c.citation}"
        )
        print("   " + c.text[:200].replace("\n", " ") + "...")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What retriever does the RAG paper use?"
    print(f"Question: {question}")
    show("VECTOR SEARCH ONLY", retrieve(question, rerank=False))
    show("VECTOR SEARCH + RERANKER", retrieve(question, rerank=True))