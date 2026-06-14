from __future__ import annotations

from app.chatbot import FALLBACK_ANSWER, generate_answer, get_chat_model
from app.retriever import RetrievedChunk


def answer_with_context(question: str, chunks: list[RetrievedChunk]) -> str:
    return generate_answer(question, chunks)


def build_chat_model():
    return get_chat_model()


def fallback_answer() -> str:
    return FALLBACK_ANSWER
