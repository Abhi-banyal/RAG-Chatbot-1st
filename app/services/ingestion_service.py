from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.chunker import chunk_documents
from app.config import DATA_DIR, UPLOADS_DIR
from app.document_builder import build_documents
from app.file_reader import FILE_TYPE_LABELS
from app.ingest import ensure_vectorstore_dir, get_embeddings


SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt", ".png", ".jpg", ".jpeg"}


@dataclass(slots=True)
class IngestionResult:
    documents_indexed: int
    chunks_indexed: int
    vectorstore_dir: str


def _unique_destination(base_dir: Path, filename: str) -> Path:
    target = base_dir / Path(filename).name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = base_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_upload_bytes(filename: str, content: bytes, destination_dir: Path | None = None) -> Path:
    base_dir = destination_dir or UPLOADS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(base_dir, filename)
    destination.write_bytes(content)
    return destination


def list_available_documents() -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for location, base_dir in (("data", DATA_DIR), ("uploads", UPLOADS_DIR)):
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in FILE_TYPE_LABELS:
                continue
            documents.append(
                {
                    "name": path.name,
                    "location": location,
                    "file_type": FILE_TYPE_LABELS[suffix].lower(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return documents


def rebuild_vectorstore() -> IngestionResult:
    documents = build_documents()
    chunks = chunk_documents(documents)

    if not documents:
        raise ValueError("No supported documents were found to index.")
    if not chunks:
        raise ValueError("Document extraction succeeded, but no chunks were created.")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    base_dir = ensure_vectorstore_dir()
    vectorstore.save_local(str(base_dir))

    return IngestionResult(
        documents_indexed=len(documents),
        chunks_indexed=len(chunks),
        vectorstore_dir=str(base_dir),
    )
