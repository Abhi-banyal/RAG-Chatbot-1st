from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import LOCAL_EMBEDDING_MODEL_NAME, VECTORSTORE_DIR
from app.image_processor import IMAGE_DESCRIPTION_CONTENT_TYPE


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL_NAME)


FIGURE_QUERY_PATTERN = re.compile(
    r"\b(?:figure|fig\.?)\s*(?:no\.?|number)?\s*(\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
FIGURE_TEXT_PATTERN_TEMPLATE = r"\b(?:figure|fig\.?)\s*(?:no\.?|number)?\s*{figure_number}\b"


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    content_type: str
    figure_number: str | None
    image_id: str | None
    chunk_id: str | None
    score: float | None
    metadata: dict[str, object]


def vectorstore_exists(vectorstore_dir: Path | None = None) -> bool:
    base_dir = vectorstore_dir or VECTORSTORE_DIR
    index_file = base_dir / "index.faiss"
    store_file = base_dir / "index.pkl"
    return index_file.exists() and store_file.exists()


def load_vectorstore(vectorstore_dir: Path | None = None) -> FAISS:
    base_dir = vectorstore_dir or VECTORSTORE_DIR
    if not vectorstore_exists(base_dir):
        raise FileNotFoundError(
            f"FAISS vectorstore not found in {base_dir}. Run `python -m app.ingest` first."
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(base_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _document_to_retrieved_chunk(document: Document, score: float | None = None) -> RetrievedChunk:
    metadata = document.metadata or {}
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    if page is None:
        page = metadata.get("page_number")

    if isinstance(page, str) and page.isdigit():
        page = int(page)

    if not isinstance(page, int):
        page = None

    content_type = str(metadata.get("content_type") or metadata.get("file_type") or "text")
    figure_number = metadata.get("figure_number")
    if figure_number is not None:
        figure_number = str(figure_number)

    image_id = metadata.get("image_id")
    if image_id is not None:
        image_id = str(image_id)

    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        chunk_id = str(chunk_id)

    return RetrievedChunk(
        text=document.page_content,
        source=source,
        page=page,
        content_type=content_type,
        figure_number=figure_number,
        image_id=image_id,
        chunk_id=chunk_id,
        score=score,
        metadata=dict(metadata),
    )


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().lower()


def extract_figure_number(question: str) -> str | None:
    match = FIGURE_QUERY_PATTERN.search(question)
    if not match:
        return None
    return f"Figure {match.group(1).upper()}"


def _normalize_figure_label(label: str | None) -> str | None:
    if label is None:
        return None
    cleaned = _normalize_question(str(label))
    match = re.search(r"\bfigure\s*(\d+[a-z]?)\b", cleaned)
    if match:
        return f"Figure {match.group(1).upper()}"
    return cleaned


def _source_hints(question: str) -> set[str]:
    normalized = _normalize_question(question)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    hints: set[str] = set()
    for token in tokens:
        if len(token) >= 3:
            hints.add(token)
    return hints


def _rank_chunk(question: str, chunk: RetrievedChunk, score: float | None) -> float:
    ranked_score = score if score is not None else float("inf")
    normalized = _normalize_question(question)
    hints = _source_hints(question)
    figure_number = extract_figure_number(question)
    source_stem = Path(chunk.source).stem.lower()
    source_tokens = set(re.findall(r"[a-z0-9]+", source_stem))

    if chunk.content_type == IMAGE_DESCRIPTION_CONTENT_TYPE:
        ranked_score -= 0.3

    if any(token in normalized for token in source_tokens if len(token) >= 3):
        ranked_score -= 0.25
    elif any(token in source_stem for token in hints):
        ranked_score -= 0.2

    if figure_number and chunk.figure_number:
        if chunk.figure_number.lower() == figure_number.lower():
            ranked_score -= 0.6
        else:
            ranked_score -= 0.15

    if figure_number and chunk.content_type == IMAGE_DESCRIPTION_CONTENT_TYPE:
        ranked_score -= 0.1

    if any(keyword in normalized for keyword in ("figure", "diagram", "chart", "image", "graph", "plot")) and chunk.content_type != IMAGE_DESCRIPTION_CONTENT_TYPE:
        ranked_score += 0.1

    return ranked_score


def _iter_vectorstore_documents(vectorstore: FAISS) -> list[Document]:
    docstore = getattr(vectorstore, "docstore", None)
    if not docstore or not hasattr(docstore, "_dict"):
        return []
    return list(docstore._dict.values())


def _chunk_matches_figure_text(chunk: Document, figure_number: str) -> bool:
    pattern = re.compile(
        FIGURE_TEXT_PATTERN_TEMPLATE.format(figure_number=re.escape(figure_number.split()[-1])),
        flags=re.IGNORECASE,
    )
    text = chunk.page_content or ""
    return bool(pattern.search(text))


def _debug_chunk_summary(label: str, chunks: list[RetrievedChunk]) -> None:
    print(f"{label}: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        print(
            {
                "index": index,
                "source": chunk.source,
                "page": chunk.page,
                "figure_number": chunk.figure_number,
                "content_type": chunk.content_type,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id,
            }
        )


def find_figure_chunks(
    question: str,
    vectorstore: FAISS | None = None,
    debug: bool = False,
) -> list[RetrievedChunk]:
    figure_number = extract_figure_number(question)
    if debug:
        print(f"Detected figure number: {figure_number or 'None'}")

    if not figure_number:
        if debug:
            print("Exact metadata matches: 0")
            print("Exact text/caption matches: 0")
        return []

    store = vectorstore or load_vectorstore()
    documents = _iter_vectorstore_documents(store)

    exact_metadata_docs = [
        doc
        for doc in documents
        if _normalize_figure_label(doc.metadata.get("figure_number")) == figure_number
    ]

    exact_metadata_chunks = [_document_to_retrieved_chunk(doc) for doc in exact_metadata_docs]
    if debug:
        _debug_chunk_summary("Exact metadata matches", exact_metadata_chunks)

    if exact_metadata_chunks:
        if debug:
            _debug_chunk_summary("Returned figure chunks", exact_metadata_chunks)
        return exact_metadata_chunks

    caption_docs = [doc for doc in documents if _chunk_matches_figure_text(doc, figure_number)]
    caption_chunks = [_document_to_retrieved_chunk(doc) for doc in caption_docs]
    for chunk in caption_chunks:
        if not chunk.figure_number:
            chunk.figure_number = figure_number
            if chunk.metadata is not None:
                chunk.metadata["figure_number"] = figure_number

    if debug:
        _debug_chunk_summary("Exact text/caption matches", caption_chunks)
        _debug_chunk_summary("Returned figure chunks", caption_chunks)

    return caption_chunks


def retrieve_chunks(question: str, k: int = 4) -> list[RetrievedChunk]:
    vectorstore = load_vectorstore()

    if hasattr(vectorstore, "similarity_search_with_score"):
        results = vectorstore.similarity_search_with_score(question, k=max(k * 6, 24))
        retrieved = [_document_to_retrieved_chunk(document, float(score)) for document, score in results]
        ranked = sorted(retrieved, key=lambda chunk: _rank_chunk(question, chunk, chunk.score))
        unique: list[RetrievedChunk] = []
        seen: set[tuple[str, int | None, str | None, str | None]] = set()
        for chunk in ranked:
            key = (chunk.source, chunk.page, chunk.figure_number, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
            if len(unique) >= k:
                break
        return unique

    documents = vectorstore.similarity_search(question, k=k)
    retrieved = [_document_to_retrieved_chunk(document) for document in documents]
    return retrieved[:k]


def print_retrieved_chunks(question: str) -> None:
    print(f"Question: {question}")
    chunks = retrieve_chunks(question, k=4)

    if not chunks:
        print("No relevant chunks found.")
        return

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}:")
        print(chunk.text)
        print("Metadata:")
        print(
            {
                "source": chunk.source,
                "page": chunk.page,
                "content_type": chunk.content_type,
                "figure_number": chunk.figure_number,
                "image_id": chunk.image_id,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
        )


def main() -> None:
    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        return

    print_retrieved_chunks(question)


if __name__ == "__main__":
    main()
