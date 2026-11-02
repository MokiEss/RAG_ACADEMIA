from app.retrieval.retriever import RetrievedChunk
from app.tools.search_docs import format_chunks
from app.tools.web_search import format_web_results


def test_format_chunks_includes_citation():
    chunk = RetrievedChunk(
        chunk_id="abc",
        text="The retriever is\nDPR.",
        source="rag_original.pdf",
        page=3,
        vector_score=0.8,
        vector_rank=1,
    )
    out = format_chunks([chunk])
    assert "[rag_original.pdf, p.3]" in out
    assert "The retriever is DPR." in out  # newline collapsed


def test_format_chunks_empty():
    assert "No relevant passages" in format_chunks([])


def test_format_web_results():
    out = format_web_results(
        [{"href": "https://example.com", "title": "Example", "body": "Hello  world"}]
    )
    assert "[web: https://example.com]" in out
    assert "Hello world" in out


def test_format_web_results_empty():
    assert "No web results" in format_web_results([])