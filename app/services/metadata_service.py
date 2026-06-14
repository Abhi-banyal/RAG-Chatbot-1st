from __future__ import annotations

from app.retriever import RetrievedChunk


def chunk_to_source_item(chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "source": chunk.source,
        "page": chunk.page,
        "figure_number": chunk.figure_number,
        "content_type": chunk.content_type,
        "image_id": chunk.image_id,
        "chunk_id": chunk.chunk_id,
        "score": chunk.score,
    }


def chunks_to_source_items(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [chunk_to_source_item(chunk) for chunk in chunks]


def format_source_label(chunk: RetrievedChunk) -> str:
    page_label = f"page {chunk.page}" if chunk.page is not None else "page not available"
    figure_label = f", {chunk.figure_number}" if chunk.figure_number else ""
    return f"{chunk.source}, {page_label}{figure_label}"
