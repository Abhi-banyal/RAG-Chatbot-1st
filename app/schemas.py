from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for the RAG chatbot")
    session_id: str | None = Field(
        default=None,
        description="Optional browser/session identifier used to preserve follow-up clarification state",
    )


class SourceItem(BaseModel):
    source: str
    page: int | None = None
    figure_number: str | None = None
    content_type: str
    image_id: str | None = None
    chunk_id: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    needs_clarification: bool = False
    suggested_question: str | None = None
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    message: str


class UploadedFileItem(BaseModel):
    name: str
    location: Literal["data", "uploads"]
    file_type: str
    size_bytes: int


class UploadResponse(BaseModel):
    message: str
    uploaded_files: list[UploadedFileItem] = Field(default_factory=list)
    failed_files: list[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: list[UploadedFileItem] = Field(default_factory=list)


class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_indexed: int
    vectorstore_dir: str
