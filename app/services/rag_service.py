from __future__ import annotations

from dataclasses import dataclass

from app.chatbot import FALLBACK_ANSWER, resolve_question
from app.services.llm_service import answer_with_context, fallback_answer
from app.services.metadata_service import chunks_to_source_items
from app.services.retrieval_service import get_relevant_chunks


DEFAULT_SESSION_ID = "default"
SESSION_PENDING_QUESTIONS: dict[str, str | None] = {}


@dataclass(slots=True)
class ChatResult:
    answer: str
    sources: list[dict[str, object]]
    needs_clarification: bool
    suggested_question: str | None
    session_id: str


def _get_session_id(session_id: str | None) -> str:
    return session_id.strip() if session_id and session_id.strip() else DEFAULT_SESSION_ID


def process_question(question: str, session_id: str | None = None) -> ChatResult:
    active_session_id = _get_session_id(session_id)
    pending_question = SESSION_PENDING_QUESTIONS.get(active_session_id)

    try:
        resolved_question, clarification, updated_pending = resolve_question(question, pending_question)
    except FileNotFoundError:
        return ChatResult(
            answer=fallback_answer(),
            sources=[],
            needs_clarification=False,
            suggested_question=None,
            session_id=active_session_id,
        )

    SESSION_PENDING_QUESTIONS[active_session_id] = updated_pending

    if clarification:
        return ChatResult(
            answer=clarification,
            sources=[],
            needs_clarification=True,
            suggested_question=updated_pending,
            session_id=active_session_id,
        )

    if resolved_question is None:
        return ChatResult(
            answer=fallback_answer(),
            sources=[],
            needs_clarification=False,
            suggested_question=None,
            session_id=active_session_id,
        )

    try:
        chunks = get_relevant_chunks(resolved_question)
    except FileNotFoundError:
        return ChatResult(
            answer=fallback_answer(),
            sources=[],
            needs_clarification=False,
            suggested_question=None,
            session_id=active_session_id,
        )

    answer = answer_with_context(resolved_question, chunks)
    sources = chunks_to_source_items(chunks) if answer != FALLBACK_ANSWER else []

    return ChatResult(
        answer=answer,
        sources=sources,
        needs_clarification=False,
        suggested_question=None,
        session_id=active_session_id,
    )
