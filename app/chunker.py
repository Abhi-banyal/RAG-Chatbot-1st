from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.document_builder import build_documents


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _build_chunk_id(metadata: dict[str, object], chunk_index: int) -> str:
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    if page is None:
        page_label = "page-na"
    else:
        page_label = f"page-{page}"

    return f"{source}:{page_label}:chunk-{chunk_index}"


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    chunk_counters: dict[tuple[str, object], int] = defaultdict(int)
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if "page" not in metadata:
            metadata["page"] = metadata.get("page_number")

        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        key = (source, page)
        chunk_counters[key] += 1
        metadata["chunk_id"] = _build_chunk_id(metadata, chunk_counters[key])
        metadata["chunk_index"] = chunk_counters[key]

    print(f"Total original Documents: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        sample = chunks[0]
        print("Sample chunk text:")
        print(sample.page_content[:1000])
        print("Sample chunk metadata:")
        print(sample.metadata)

    return chunks


def build_chunks() -> list[Document]:
    documents = build_documents()
    return chunk_documents(documents)


def main() -> None:
    build_chunks()


if __name__ == "__main__":
    main()
