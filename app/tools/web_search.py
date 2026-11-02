from langchain_core.tools import tool

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS


def format_web_results(results: list[dict]) -> str:
    if not results:
        return "No web results found."

    blocks = []
    for r in results:
        url = r.get("href", "")
        title = r.get("title", "")
        body = " ".join((r.get("body") or "").split())
        blocks.append(f"[web: {url}]\n{title}\n{body}")
    return "\n\n".join(blocks)


@tool
def web_search(query: str) -> str:
    """Search the public web with DuckDuckGo.

    Use this ONLY when search_docs does not contain the answer, for example for
    recent events, papers that are not in the collection, or general facts.
    Returns results, each preceded by a citation tag like [web: https://...].
    """
    try:
        results = list(DDGS().text(query, max_results=5))
    except Exception as exc:  # network errors, rate limits
        return f"Web search failed: {exc}"
    return format_web_results(results)