from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOADS_DIR
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    UploadResponse,
    UploadedFileItem,
)
from app.services.ingestion_service import (
    SUPPORTED_UPLOAD_SUFFIXES,
    ingest_single_file,
    list_available_documents,
    rebuild_vectorstore,
    save_upload_bytes,
)
from app.services.rag_service import process_question


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", message="RAG chatbot backend is running")


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    result = process_question(payload.question, payload.session_id)
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        needs_clarification=result.needs_clarification,
        suggested_question=result.suggested_question,
        session_id=result.session_id,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    uploaded_files: list[UploadedFileItem] = []
    failed_files: list[str] = []
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            failed_files.append(file.filename or "unknown")
            continue

        content = await file.read()
        if not content:
            failed_files.append(file.filename or "unknown")
            continue

        destination = save_upload_bytes(file.filename or "upload", content)
        print(f"Uploaded file name: {file.filename or 'unknown'}")
        print(f"Saved path: {destination}")

        try:
            ingest_single_file(destination)
        except Exception as exc:
            failed_files.append(file.filename or "unknown")
            print(f"Failed to index {destination.name}: {exc}")
            continue

        uploaded_files.append(
            UploadedFileItem(
                name=destination.name,
                location="uploads",
                file_type=suffix.lstrip("."),
                size_bytes=destination.stat().st_size,
            )
        )

    if uploaded_files and failed_files:
        message = "Some files were uploaded and indexed, but others failed."
    elif uploaded_files:
        message = "Files uploaded and indexed successfully."
    else:
        message = "No files were uploaded."
    return UploadResponse(message=message, uploaded_files=uploaded_files, failed_files=failed_files)


@router.get("/documents", response_model=DocumentListResponse)
def documents() -> DocumentListResponse:
    return DocumentListResponse(documents=list_available_documents())


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents() -> IngestResponse:
    try:
        result = rebuild_vectorstore()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        status="ok",
        documents_indexed=result.documents_indexed,
        chunks_indexed=result.chunks_indexed,
        vectorstore_dir=result.vectorstore_dir,
    )
