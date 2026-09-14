from __future__ import annotations

import re
from pathlib import Path
from typing import List

from src.models import Document


def normalize_text(text: str) -> str:
    """Normalize newlines and trailing whitespace without flattening paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def load_txt_documents(documents_dir: Path) -> List[Document]:
    """Load every non-empty .txt file in deterministic filename order."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

    documents: List[Document] = []
    for path in sorted(documents_dir.glob("*.txt")):
        text = normalize_text(path.read_text(encoding="utf-8"))
        if text:
            documents.append(Document(path=path, filename=path.name, text=text))

    if not documents:
        raise ValueError(f"No non-empty .txt documents found in {documents_dir}")
    return documents

