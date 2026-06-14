from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.config import validate_azure_settings
from app.retriever import (
    RetrievedChunk,
    extract_figure_number,
    find_figure_chunks,
    load_vectorstore,
    retrieve_chunks,
)


FALLBACK_ANSWER = "I could not find this information in the uploaded documents."
AFFIRMATIVE_RESPONSES = {
    "yes",
    "yeah",
    "yep",
    "correct",
    "right",
    "okay",
    "ok",
    "sure",
    "y",
}
NEGATIVE_RESPONSES = {
    "no",
    "nope",
    "nah",
    "not really",
}
VAGUE_PATTERNS = (
    "tell me about it",
    "explain this",
    "what about policy",
    "what about it",
    "what is machine",
    "what is it",
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "about",
    "be",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "the",
    "this",
    "that",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
VISUAL_QUERY_WORDS = {"figure", "fig", "diagram", "chart", "image", "graph", "plot", "illustration"}
TOPIC_KEYWORDS = {
    "taxonomy",
    "hierarchy",
    "relationship",
    "relationships",
    "category",
    "categories",
    "discipline",
    "disciplines",
    "nested",
    "circles",
    "ai",
    "artificial",
    "intelligence",
    "machine",
    "learning",
    "deep",
    "generative",
    "genai",
}
VISUAL_CONTENT_HINTS = {
    "figure",
    "diagram",
    "chart",
    "taxonomy",
    "hierarchy",
    "nested circles",
    "relationship",
    "category",
    "discipline",
}


def get_chat_model() -> AzureChatOpenAI:
    azure_settings = validate_azure_settings()
    return AzureChatOpenAI(
        azure_endpoint=azure_settings["AZURE_OPENAI_ENDPOINT"],
        api_key=azure_settings["AZURE_OPENAI_API_KEY"],
        api_version=azure_settings["AZURE_OPENAI_API_VERSION"],
        azure_deployment=azure_settings["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _figure_sort_value(figure_number: str | None) -> int | None:
    if not figure_number:
        return None
    match = re.search(r"(\d+)", figure_number)
    if not match:
        return None
    return int(match.group(1))


def _source_hints(question: str) -> set[str]:
    stems: set[str] = set()
    for token in _tokenize(question):
        if len(token) >= 3 and token not in STOP_WORDS:
            stems.add(token)
    return stems


def _looks_visual(question: str) -> bool:
    tokens = set(_tokenize(question))
    return bool(tokens & VISUAL_QUERY_WORDS)


def _looks_taxonomy_question(question: str) -> bool:
    tokens = set(_tokenize(question))
    return bool(tokens & {"taxonomy", "hierarchy", "relationship", "relationships", "categories", "category", "discipline", "disciplines"})


def _chunk_text_tokens(chunk: RetrievedChunk) -> set[str]:
    text = f"{chunk.text} {chunk.source} {chunk.figure_number or ''}"
    return {token for token in _tokenize(text) if token not in STOP_WORDS}


def _chunk_mentions_visual_theme(chunk: RetrievedChunk) -> bool:
    text = chunk.text.lower()
    return any(hint in text for hint in VISUAL_CONTENT_HINTS)


def _score_chunk_relevance(question: str, chunk: RetrievedChunk) -> float:
    question_tokens = {token for token in _tokenize(question) if token not in STOP_WORDS}
    chunk_tokens = _chunk_text_tokens(chunk)
    overlap = len(question_tokens & chunk_tokens)
    score = float(overlap)

    if _looks_visual(question):
        if chunk.content_type == "image_description":
            score += 2.5
        if chunk.figure_number:
            score += 1.2
        if extract_figure_number(question) and chunk.figure_number and chunk.figure_number.lower() == extract_figure_number(question).lower():
            score += 3.0

    if _looks_taxonomy_question(question):
        if _chunk_mentions_visual_theme(chunk):
            score += 2.0
        if chunk.content_type == "image_description":
            score += 0.8
        if any(token in chunk_tokens for token in {"taxonomy", "hierarchy", "relationship", "categories", "discipline", "ai", "machine", "learning", "deep", "generative"}):
            score += 1.5
        if "genai" in chunk_tokens and len(question_tokens & {"taxonomy", "hierarchy", "relationship", "category", "categories", "discipline", "disciplines"}) > 0:
            score += 0.25
        if "midjourney" in chunk_tokens or "codex" in chunk_tokens:
            score -= 0.5

    if any(token in question_tokens for token in {"midjourney", "codex"}) and any(token in chunk_tokens for token in {"midjourney", "codex"}):
        score += 2.0

    if chunk.content_type == "image_description" and "genai" in chunk_tokens and not _looks_visual(question) and not _looks_taxonomy_question(question):
        score -= 0.4

    return score


def filter_relevant_chunks(question: str, chunks: list[RetrievedChunk], limit: int = 4) -> list[RetrievedChunk]:
    if not chunks:
        return []

    ranked = sorted(chunks, key=lambda chunk: (_score_chunk_relevance(question, chunk), -(chunk.score or 0.0)), reverse=True)

    filtered: list[RetrievedChunk] = []
    seen_sources: set[tuple[str, int | None, str | None]] = set()
    best_score = _score_chunk_relevance(question, ranked[0])
    min_score = 1.0 if _looks_visual(question) or _looks_taxonomy_question(question) else 0.5

    for chunk in ranked:
        score = _score_chunk_relevance(question, chunk)
        key = (chunk.source, chunk.page, chunk.figure_number)
        if key in seen_sources:
            continue
        if score < min_score and filtered:
            continue
        if score < 0 and filtered:
            continue
        if _looks_taxonomy_question(question) and chunk.content_type == "image_description" and not _chunk_mentions_visual_theme(chunk):
            continue
        if _looks_taxonomy_question(question) and "genai" in _chunk_text_tokens(chunk) and not _chunk_mentions_visual_theme(chunk):
            continue

        seen_sources.add(key)
        filtered.append(chunk)
        if len(filtered) >= limit:
            break

    if not filtered and ranked:
        top_chunk = ranked[0]
        if _score_chunk_relevance(question, top_chunk) >= best_score:
            filtered.append(top_chunk)

    return filtered


def is_affirmative_response(text: str) -> bool:
    normalized = _normalize_text(text)
    tokens = _tokenize(normalized)
    if not tokens:
        return False
    if normalized in AFFIRMATIVE_RESPONSES:
        return True
    return tokens[0] in AFFIRMATIVE_RESPONSES and len(tokens) <= 3


def is_negative_response(text: str) -> bool:
    normalized = _normalize_text(text)
    tokens = _tokenize(normalized)
    if not tokens:
        return False
    if normalized in NEGATIVE_RESPONSES:
        return True
    return tokens[0] in NEGATIVE_RESPONSES and len(tokens) <= 3


def is_incomplete_question(question: str) -> bool:
    normalized = _normalize_text(question)
    if not normalized:
        return True

    if normalized in VAGUE_PATTERNS:
        return True

    if any(normalized.startswith(prefix) for prefix in ("what is ", "tell me about ", "explain ", "what about ")):
        remainder = normalized.split(" ", 2)[-1]
        remainder_tokens = [token for token in _tokenize(remainder) if token not in STOP_WORDS]
        if len(remainder_tokens) <= 1:
            return True

    content_tokens = [token for token in _tokenize(normalized) if token not in STOP_WORDS]
    return len(content_tokens) <= 1


def build_follow_up_question(question: str, chunks: list[RetrievedChunk]) -> str | None:
    if not chunks:
        return None

    top_chunk = chunks[0]
    question_tokens = {token for token in _tokenize(question) if token not in STOP_WORDS}
    source_tokens = {token for token in _tokenize(Path(top_chunk.source).stem) if len(token) >= 3}
    if not question_tokens or not source_tokens or not (question_tokens & source_tokens):
        return None

    topic = Path(top_chunk.source).stem.replace("_", " ").replace("-", " ").strip()
    if not topic:
        return None

    return f"What is {topic}?"


def build_visual_follow_up(question: str, chunks: list[RetrievedChunk]) -> tuple[str | None, str | None]:
    target_figure = extract_figure_number(question)
    if not target_figure or not chunks:
        return None, None

    source_hints = _source_hints(question)
    visual_chunks = [chunk for chunk in chunks if chunk.content_type == "image_description" or chunk.figure_number]
    if not visual_chunks:
        return None, None

    exact_matches = [chunk for chunk in visual_chunks if chunk.figure_number and chunk.figure_number.lower() == target_figure.lower()]
    if exact_matches:
        return None, None

    matching_source_chunks = [
        chunk
        for chunk in visual_chunks
        if source_hints and any(hint in Path(chunk.source).stem.lower() for hint in source_hints)
    ]
    candidate_pool = matching_source_chunks or visual_chunks

    def sort_key(chunk: RetrievedChunk) -> tuple[int, int, str]:
        figure_value = _figure_sort_value(chunk.figure_number)
        target_value = _figure_sort_value(target_figure)
        distance = abs((figure_value or target_value or 0) - (target_value or 0))
        return (0 if chunk.content_type == "image_description" else 1, distance, chunk.source)

    best_chunk = sorted(candidate_pool, key=sort_key)[0]
    if best_chunk.figure_number:
        clarification = (
            f"I found {best_chunk.figure_number} in {best_chunk.source}, "
            f"but not {target_figure}. Did you mean {best_chunk.figure_number}?"
        )
        suggested_question = f"What does {best_chunk.figure_number} in {best_chunk.source} show?"
        return clarification, suggested_question

    clarification = f"I found a visual in {best_chunk.source}, but not {target_figure}. Did you mean that one?"
    suggested_question = f"What does the visual in {best_chunk.source} show?"
    return clarification, suggested_question


def format_context(chunks: list[RetrievedChunk]) -> str:
    context_parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        context_parts.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"Source: {chunk.source}",
                    f"Page: {chunk.page if chunk.page is not None else 'not available'}",
                    f"Content type: {chunk.content_type}",
                    f"Figure number: {chunk.figure_number or 'not available'}",
                    f"Image ID: {chunk.image_id or 'not available'}",
                    f"Chunk ID: {chunk.chunk_id or 'not available'}",
                    f"Score: {chunk.score if chunk.score is not None else 'not available'}",
                    f"Metadata: {metadata}",
                    f"Text: {chunk.text}",
                ]
            )
        )
    return "\n\n---\n\n".join(context_parts)


def get_source_lines(chunks: list[RetrievedChunk]) -> list[str]:
    sources = OrderedDict()
    for chunk in chunks:
        page_label = f"page {chunk.page}" if chunk.page is not None else "page not available"
        figure_label = f", {chunk.figure_number}" if chunk.figure_number else ""
        key = (chunk.source, chunk.page, chunk.figure_number)
        if key not in sources:
            sources[key] = f"{chunk.source}, {page_label}{figure_label}"
    return list(sources.values())


def has_reliable_context(question: str, chunks: list[RetrievedChunk]) -> bool:
    question_tokens = {token for token in _tokenize(question) if token not in STOP_WORDS}
    if not question_tokens:
        return False

    if _looks_visual(question):
        for chunk in chunks:
            if chunk.content_type == "image_description":
                return True
            if chunk.figure_number and extract_figure_number(question) and chunk.figure_number.lower() == extract_figure_number(question).lower():
                return True

    context_tokens: set[str] = set()
    for chunk in chunks:
        context_tokens.update(token for token in _tokenize(chunk.text) if token not in STOP_WORDS)
        context_tokens.update(token for token in _tokenize(chunk.source) if token not in STOP_WORDS)

    return bool(question_tokens & context_tokens)


def resolve_question(question: str, pending_question: str | None) -> tuple[str | None, str | None, str | None]:
    stripped = question.strip()
    if not stripped:
        return None, None, pending_question

    figure_number = extract_figure_number(stripped)
    if figure_number:
        figure_chunks = find_figure_chunks(stripped, debug=False)
        if figure_chunks:
            return stripped, None, None
        return (
            None,
            f"I could not find {figure_number} in the uploaded documents. Please mention the document name or try another figure number.",
            None,
        )

    if pending_question:
        if is_affirmative_response(stripped):
            return pending_question, None, None
        if is_negative_response(stripped):
            return None, "Please ask a new question.", None

    if is_incomplete_question(stripped):
        chunks = retrieve_chunks(stripped, k=3)
        visual_follow_up, suggested_question = build_visual_follow_up(stripped, chunks)
        if visual_follow_up and suggested_question:
            return None, visual_follow_up, suggested_question
        suggestion = build_follow_up_question(stripped, chunks)
        if suggestion:
            return None, f'Did you mean: "{suggestion}"?', suggestion
        return None, "Could you clarify your question?", None

    if _looks_visual(stripped):
        chunks = retrieve_chunks(stripped, k=5)
        visual_follow_up, suggested_question = build_visual_follow_up(stripped, chunks)
        if visual_follow_up and suggested_question:
            return None, visual_follow_up, suggested_question

    return stripped, None, None


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return FALLBACK_ANSWER

    if not has_reliable_context(question, chunks):
        return FALLBACK_ANSWER

    context = format_context(chunks)
    system_prompt = (
        "You answer questions using only the provided context from uploaded documents. "
        "Answer only from the retrieved context. "
        f"If the answer is not explicitly supported by the context, reply with exactly: {FALLBACK_ANSWER}. "
        "Do not use outside knowledge. Do not hallucinate. "
        "Do not create citations, source names, or page numbers in the answer body. "
        "If the context describes a figure, chart, image, or diagram but is not enough to answer fully, say that the figure was found but the description is not enough to answer fully."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer using only the context above."
    )

    model = get_chat_model()
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    answer = str(response.content).strip()

    if not answer or FALLBACK_ANSWER.lower() in answer.lower():
        answer = FALLBACK_ANSWER

    return answer


def answer_question(question: str) -> tuple[str, list[str]]:
    figure_number = extract_figure_number(question)
    if figure_number:
        relevant_chunks = find_figure_chunks(question, debug=True)
        if not relevant_chunks:
            return FALLBACK_ANSWER, []
        answer = generate_answer(question, relevant_chunks)
        sources = [] if answer == FALLBACK_ANSWER else get_source_lines(relevant_chunks)
        return answer, sources

    chunks = retrieve_chunks(question, k=4)
    relevant_chunks = filter_relevant_chunks(question, chunks, limit=4)
    answer = generate_answer(question, relevant_chunks)
    sources = [] if answer == FALLBACK_ANSWER else get_source_lines(relevant_chunks)
    return answer, sources


def print_answer(question: str) -> None:
    answer, sources = answer_question(question)
    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    if sources:
        for index, source in enumerate(sources, start=1):
            print(f"{index}. {source}")
    else:
        print("No relevant source found")


def main() -> None:
    try:
        load_vectorstore()
    except FileNotFoundError as exc:
        print(str(exc))
        return

    print("Text RAG chatbot is ready with Azure OpenAI. Type a question or press Enter to quit.")

    pending_question: str | None = None
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            print("Exiting chatbot.")
            break

        lowered = question.lower()
        if lowered in {"exit", "quit"}:
            print("Exiting chatbot.")
            break

        resolved_question, clarification, updated_pending = resolve_question(question, pending_question)
        pending_question = updated_pending

        if clarification:
            print(f"\n{clarification}")
            continue

        if resolved_question is None:
            continue

        print_answer(resolved_question)


if __name__ == "__main__":
    main()
