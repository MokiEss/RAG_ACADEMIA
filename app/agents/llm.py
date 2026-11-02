from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


@lru_cache
def get_llm():
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def message_text(message) -> str:
    """Return plain text from a model message (content can be a string or a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
    )