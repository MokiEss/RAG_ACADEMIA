from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED = {".pdf", ".md", ".txt"}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        docs = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_num},
                    )
                )
        return docs

    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        return [Document(page_content=text, metadata={"source": path.name, "page": 1})]

    return []


def load_directory(directory: str) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            loaded = load_file(path)
            print(f"Loaded {path.name}: {len(loaded)} page(s)")
            docs.extend(loaded)
    return docs