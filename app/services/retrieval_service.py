from __future__ import annotations

from app.chatbot import filter_relevant_chunks
from app.retriever import RetrievedChunk, extract_figure_number, find_figure_chunks, retrieve_chunks


def get_relevant_chunks(question: str, limit: int = 4) -> list[RetrievedChunk]:
    figure_number = extract_figure_number(question)
    if figure_number:
        return find_figure_chunks(question, debug=False)

    chunks = retrieve_chunks(question, k=max(limit, 4))
    return filter_relevant_chunks(question, chunks, limit=limit)
