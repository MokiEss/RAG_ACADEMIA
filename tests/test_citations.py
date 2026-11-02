from app.agents.citations import extract_citations, invalid_citations


def test_extract_document_and_web_tags():
    text = "RAG uses DPR [rag_original.pdf, p.3]. See also [web: https://example.com]."
    assert extract_citations(text) == ["rag_original.pdf, p.3", "web: https://example.com"]


def test_extract_ignores_non_citations_and_duplicates():
    text = "Array [1] and list [a, b] then [bert.pdf, p.2] again [bert.pdf, p.2]."
    assert extract_citations(text) == ["bert.pdf, p.2"]


def test_invalid_citations_detects_hallucinated_tag():
    evidence = "[rag_original.pdf, p.3]\nThe retriever is DPR."
    answer = "It is DPR [rag_original.pdf, p.3] and fast [lora.pdf, p.9]."
    assert invalid_citations(answer, evidence) == ["lora.pdf, p.9"]