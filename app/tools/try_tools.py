from app.tools import ALL_TOOLS, search_docs, web_search

if __name__ == "__main__":
    print("=== Tool descriptions the LLM will see ===")
    for t in ALL_TOOLS:
        print(f"\n{t.name}: {t.args}")

    print("\n=== search_docs ===")
    print(search_docs.invoke({"query": "RAG retriever DPR bi-encoder BERT"}))

    print("\n=== web_search ===")
    print(web_search.invoke({"query": "LangGraph multi-agent tutorial"}))