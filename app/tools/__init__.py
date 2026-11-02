from app.tools.search_docs import search_docs
from app.tools.web_search import web_search

ALL_TOOLS = [search_docs, web_search]

__all__ = ["search_docs", "web_search", "ALL_TOOLS"]