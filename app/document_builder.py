from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.extractor import ExtractedContent, extract_all_content, extract_file
from app.config import INGESTION_DIRS
from app.image_processor import build_image_documents, build_image_documents_for_file


def extracted_content_to_document(item: ExtractedContent) -> Document:
    metadata: dict[str, object] = {
        "source": item.source,
        "file_type": item.file_type,
        "page": item.page_number,
        "page_number": item.page_number,
        "content_type": "text",
    }

    if item.page_number is not None:
        metadata["page_number"] = item.page_number

    if item.row_number is not None:
        metadata["row_number"] = item.row_number

    return Document(page_content=item.text, metadata=metadata)


def build_documents() -> list[Document]:
    extracted_items = extract_all_content(INGESTION_DIRS)
    documents = [extracted_content_to_document(item) for item in extracted_items if item.text]
    image_documents = build_image_documents(INGESTION_DIRS)
    documents.extend(image_documents)

    print(f"Total Document objects created: {len(documents)}")

    if documents:
        sample = documents[0]
        print("Sample document text:")
        print(sample.page_content[:1000])
        print("Sample document metadata:")
        print(sample.metadata)

    if image_documents:
        print(f"Image Document objects created: {len(image_documents)}")

    return documents


def build_documents_for_file(file_path: Path) -> list[Document]:
    extracted_items = extract_file(file_path)
    documents = [extracted_content_to_document(item) for item in extracted_items if item.text]
    image_documents = build_image_documents_for_file(file_path)
    documents.extend(image_documents)

    print(f"Total Document objects created for {file_path.name}: {len(documents)}")

    return documents


def main() -> None:
    build_documents()


if __name__ == "__main__":
    main()
