from pathlib import Path

import pytest

from src.ingestion.loaders import load_txt_documents, normalize_text


def test_normalize_text_preserves_paragraphs():
    assert normalize_text("First  \r\n\r\n\r\nSecond\n") == "First\n\nSecond"


def test_loads_only_non_empty_txt_files(tmp_path: Path):
    (tmp_path / "b.txt").write_text("Second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("First", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("Ignored", encoding="utf-8")

    documents = load_txt_documents(tmp_path)

    assert [document.filename for document in documents] == ["a.txt", "b.txt"]


def test_missing_documents_directory_is_clear(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_txt_documents(tmp_path / "missing")

