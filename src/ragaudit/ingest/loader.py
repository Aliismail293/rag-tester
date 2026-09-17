"""Walks a directory and loads .txt/.md files into Document objects."""

import hashlib
import logging
from pathlib import Path

from ragaudit.models.document import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents(root: str | Path) -> list[Document]:
    """Walk `root` and load every supported file into a Document.

    Only .txt and .md files are read. Files that can't be decoded as UTF-8
    text, or can't be read at all, are skipped with a logged warning rather
    than raising.
    """
    root_path = Path(root)
    documents: list[Document] = []

    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable file %s: %s", file_path, exc)
            continue

        content_hash = _sha256(content)
        doc_id = _sha256(f"{file_path}:{content_hash}")
        documents.append(
            Document(
                doc_id=doc_id,
                source_path=str(file_path),
                content=content,
                content_hash=content_hash,
            )
        )

    return documents
