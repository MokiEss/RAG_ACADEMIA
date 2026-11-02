import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


def chunk_documents(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    counters: dict[tuple, int] = {}
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata["page"])
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        chunk.metadata["chunk_index"] = idx

        raw = f"{key[0]}|{key[1]}|{idx}|{chunk.page_content}"
        chunk.metadata["chunk_id"] = hashlib.sha1(raw.encode()).hexdigest()[:16]

    return chunks