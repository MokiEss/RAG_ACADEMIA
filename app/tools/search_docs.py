from langchain_core.tools import tool

from app.retrieval.retriever import RetrievedChunk, retrieve


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Turn retrieved chunks into text the LLM can read and cite."""
    if not chunks:
        return "No relevant passages found in the document collection."

    blocks = []
    for c in chunks:
        text = " ".join(c.text.split())  # collapse newlines and extra spaces
        blocks.append(f"[{c.citation}]\n{text}")
    return "\n\n".join(blocks)


@tool
def search_docs(query: str) -> str:
    """Search the local research paper collection (LLMs, RAG, agents, fine-tuning).

    Use this FIRST for any question about the content of the papers. Pass a
    specific, keyword-rich query rather than a vague question, for example
    'RAG retriever DPR bi-encoder BERT' instead of 'what retriever is used'.
    Returns passages, each preceded by a citation tag like [rag_original.pdf, p.3].
    """
    return format_chunks(retrieve(query))