from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    google_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"

    # Retrieval models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Storage
    chroma_dir: str = "./data/chroma"
    docs_dir: str = "./data/docs"
    collection_name: str = "research_docs"

    # Retrieval params
    retrieve_k: int = 20      # candidates from vector search
    rerank_top_n: int = 5     # kept after reranking

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()