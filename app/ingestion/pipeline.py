from app.config import get_settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_directory
from app.retrieval.embeddings import embed_texts
from app.retrieval.vectorstore import get_collection

BATCH_SIZE = 128


def run_ingestion() -> int:
    settings = get_settings()

    docs = load_directory(settings.docs_dir)
    if not docs:
        print(f"No documents found in {settings.docs_dir}")
        return 0

    chunks = chunk_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    collection = get_collection()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c.page_content for c in batch]
        collection.upsert(
            ids=[c.metadata["chunk_id"] for c in batch],
            embeddings=embed_texts(texts),
            documents=texts,
            metadatas=[
                {
                    "source": c.metadata["source"],
                    "page": c.metadata["page"],
                    "chunk_index": c.metadata["chunk_index"],
                }
                for c in batch
            ],
        )
        print(f"Stored {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection now holds {collection.count()} chunks.")
    return len(chunks)


if __name__ == "__main__":
    run_ingestion()