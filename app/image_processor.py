from __future__ import annotations

import base64
import io
import mimetypes
import re
from collections.abc import Iterable
from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.config import DATA_DIR, INGESTION_DIRS, validate_azure_settings


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_PDF_EXTENSION = ".pdf"
IMAGE_DESCRIPTION_CONTENT_TYPE = "image_description"
TEXT_CONTENT_TYPE = "text"
VISUAL_KEYWORDS = ("figure", "fig.", "diagram", "chart", "graph", "plot", "image", "illustration")
FIGURE_PATTERN = re.compile(r"\b(?:Figure|Fig\.?)\s*(\d+[A-Za-z]?)\b", re.IGNORECASE)
FIGURE_SECTION_PATTERN = re.compile(r"(?:^|\n)(?:###\s*)?(Figure\s*\d+[A-Za-z]?\s*:)", re.IGNORECASE)


def get_vision_chat_model() -> AzureChatOpenAI:
    azure_settings = validate_azure_settings()

    return AzureChatOpenAI(
        azure_endpoint=azure_settings["AZURE_OPENAI_ENDPOINT"],
        api_key=azure_settings["AZURE_OPENAI_API_KEY"],
        api_version=azure_settings["AZURE_OPENAI_API_VERSION"],
        azure_deployment=azure_settings["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
    )


def _iter_supported_media_files(
    data_dirs: Iterable[Path] | None,
    suffixes: set[str],
) -> list[Path]:
    base_dirs = list(data_dirs) if data_dirs is not None else [DATA_DIR]
    results: list[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        results.extend(
            path
            for path in base_dir.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        )
    return sorted(results)


def iter_image_files(data_dirs: Iterable[Path] | None = None) -> Iterable[Path]:
    return _iter_supported_media_files(data_dirs, SUPPORTED_IMAGE_EXTENSIONS)


def iter_pdf_files(data_dirs: Iterable[Path] | None = None) -> Iterable[Path]:
    return _iter_supported_media_files(data_dirs, {SUPPORTED_PDF_EXTENSION})


def image_path_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/png"

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def render_pdf_page_to_data_url(pdf_path: Path, page_number: int, zoom: float = 2.0) -> str:
    return f"data:image/png;base64,{base64.b64encode(render_pdf_page_to_bytes(pdf_path, page_number, zoom)).decode('utf-8')}"


def render_pdf_page_to_bytes(pdf_path: Path, page_number: int, zoom: float = 2.0) -> bytes:
    with fitz.open(pdf_path) as pdf:
        page = pdf.load_page(page_number)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")

    return image_bytes


def extract_figure_number(text: str) -> str | None:
    match = FIGURE_PATTERN.search(text)
    if not match:
        return None
    return f"Figure {match.group(1)}"


def extract_figure_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(FIGURE_SECTION_PATTERN.finditer(text))
    if not matches:
        return [(None, text.strip())]

    sections: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        figure_number = extract_figure_number(match.group(1))
        sections.append((figure_number, section_text))

    return sections


def page_contains_visual_content(page: fitz.Page, page_text: str) -> bool:
    if page.get_images(full=True) or page.get_drawings():
        return True
    lowered = page_text.lower()
    return any(keyword in lowered for keyword in VISUAL_KEYWORDS)


def extract_caption_text(page_text: str) -> str | None:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    caption_lines = [line for line in lines if any(keyword in line.lower() for keyword in VISUAL_KEYWORDS)]
    if caption_lines:
        return caption_lines[0]
    return None


def optional_ocr_image_bytes(image_bytes: bytes) -> str | None:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            text = pytesseract.image_to_string(image).strip()
    except Exception:
        return None

    return text or None


def optional_ocr_image_path(image_path: Path) -> str | None:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image).strip()
    except Exception:
        return None

    return text or None


def _shorten_text(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _build_visual_description(
    *,
    source: str,
    page_number: int | None,
    figure_number: str | None,
    body: str,
    fallback_note: str | None = None,
) -> str:
    prefix_parts = ["Figure/diagram description:"]
    if figure_number:
        prefix_parts.append(f"{figure_number}.")
    if page_number is not None:
        prefix_parts.append(f"Page {page_number}.")
    prefix = " ".join(prefix_parts)

    details = [prefix, body.strip()]
    if fallback_note:
        details.append(fallback_note.strip())
    return " ".join(part for part in details if part)


def _build_pdf_image_documents(
    *,
    pdf_path: Path,
    page_number: int,
    description: str,
    page_text: str,
) -> list[Document]:
    sections = extract_figure_sections(description)
    documents: list[Document] = []

    for index, (section_figure_number, section_text) in enumerate(sections, start=1):
        figure_number = section_figure_number or extract_figure_number(page_text)
        image_id = f"{pdf_path.stem}_page_{page_number}_image_{index}"
        document = Document(
            page_content=section_text,
            metadata={
                **_figure_metadata(
                    source=pdf_path.name,
                    page_number=page_number,
                    index=index,
                    figure_number=figure_number,
                    content_type=IMAGE_DESCRIPTION_CONTENT_TYPE,
                ),
                "file_type": "pdf_image",
                "render_type": "page_screenshot",
            },
        )
        documents.append(document)

    return documents


def _figure_metadata(
    *,
    source: str,
    page_number: int | None,
    index: int,
    figure_number: str | None,
    content_type: str,
) -> dict[str, object]:
    image_id = f"{Path(source).stem}_page_{page_number}_image_{index}" if page_number is not None else f"{Path(source).stem}_image_{index}"
    return {
        "source": source,
        "page": page_number,
        "page_number": page_number,
        "content_type": content_type,
        "figure_number": figure_number,
        "image_id": image_id,
        "chunk_id": image_id,
    }


def describe_image(image_path: Path, model: AzureChatOpenAI) -> str:
    data_url = image_path_to_data_url(image_path)

    messages = [
        SystemMessage(
            content=(
                "You are a vision assistant for multimodal RAG. Produce a detailed factual description of the image. "
                "Include visible objects, layout, labels, text, relationships, and any diagram structure. "
                "Do not invent details that are not visible."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe this image in detail for retrieval in a RAG system.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    },
                },
            ]
        ),
    ]

    try:
        response = model.invoke(messages)
    except Exception as exc:  # pragma: no cover - depends on Azure deployment capabilities
        raise RuntimeError(
            "This Azure chat deployment does not appear to support vision/image input. "
            "Please use a vision-capable Azure OpenAI deployment before enabling image RAG."
        ) from exc

    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def describe_pdf_page(
    pdf_path: Path,
    page_number: int,
    model: AzureChatOpenAI | None,
    *,
    page_text: str = "",
    figure_number: str | None = None,
) -> str:
    data_url = render_pdf_page_to_data_url(pdf_path, page_number)

    if model is not None:
        messages = [
            SystemMessage(
                content=(
                    "You are a vision assistant for multimodal RAG. Produce a detailed factual description of this PDF page image. "
                    "Include visible text, charts, diagrams, labels, layout, and relationships between elements. "
                    "Do not invent details that are not visible."
                )
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe this PDF page image in detail for retrieval in a RAG system. "
                            "Pay special attention to charts, diagrams, and any embedded figures."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ]
            ),
        ]

        try:
            response = model.invoke(messages)
        except Exception as exc:  # pragma: no cover - depends on Azure deployment capabilities
            print(
                "Vision description failed for a PDF page. Falling back to page text and caption extraction. "
                "For best diagram understanding, use a vision-capable Azure OpenAI deployment."
            )
            print(str(exc))
        else:
            content = response.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()

    caption_text = extract_caption_text(page_text)
    ocr_text = optional_ocr_image_bytes(render_pdf_page_to_bytes(pdf_path, page_number))
    figure_hint = figure_number or extract_figure_number(page_text) or extract_figure_number(caption_text or "")
    body_parts = []
    if caption_text:
        body_parts.append(f"Nearby caption/text: {_shorten_text(caption_text)}")
    if ocr_text:
        body_parts.append(f"OCR text: {_shorten_text(ocr_text)}")
    if page_text:
        body_parts.append(f"Page text: {_shorten_text(page_text)}")
    if not body_parts:
        body_parts.append("Visual description unavailable from local text extraction.")

    fallback_note = (
        "A vision-capable model is required for richer diagram understanding."
        if not model
        else "The image description was generated from local extraction fallback."
    )
    return _build_visual_description(
        source=pdf_path.name,
        page_number=page_number + 1,
        figure_number=figure_hint,
        body=" ".join(body_parts),
        fallback_note=fallback_note,
    )


def describe_standalone_image(
    image_path: Path,
    model: AzureChatOpenAI | None,
) -> str:
    if model is not None:
        try:
            return describe_image(image_path, model)
        except Exception as exc:  # pragma: no cover - depends on Azure deployment capabilities
            print(
                f"Vision description failed for image {image_path.name}. Falling back to OCR or filename-based text."
            )
            print(str(exc))

    ocr_text = optional_ocr_image_path(image_path)
    body_parts = []
    if ocr_text:
        body_parts.append(f"OCR text: {_shorten_text(ocr_text)}")
    body_parts.append(f"Standalone image file: {image_path.name}.")
    body_parts.append("A vision-capable model is required for richer image understanding.")
    return _build_visual_description(
        source=image_path.name,
        page_number=None,
        figure_number=extract_figure_number(ocr_text or ""),
        body=" ".join(body_parts),
        fallback_note=None,
    )


def build_image_documents(data_dirs: Iterable[Path] | None = None) -> list[Document]:
    data_dirs = INGESTION_DIRS if data_dirs is None else data_dirs
    image_paths = list(iter_image_files(data_dirs))
    pdf_paths = list(iter_pdf_files(data_dirs))
    if not image_paths and not pdf_paths:
        print("No supported visual files found.")
        return []

    vision_model = _get_optional_vision_model()
    image_docs: list[Document] = []

    for image_path in image_paths:
        image_docs.extend(build_image_documents_for_file(image_path, vision_model=vision_model))

    for pdf_path in pdf_paths:
        image_docs.extend(build_image_documents_for_file(pdf_path, vision_model=vision_model))

    print(f"Total image documents created: {len(image_docs)}")
    return image_docs


def _get_optional_vision_model() -> AzureChatOpenAI | None:
    try:
        return get_vision_chat_model()
    except ValueError as exc:
        print(
            "Azure chat settings are incomplete or a vision deployment is unavailable. "
            "Using fallback image descriptions from OCR and nearby PDF text. "
            "Add a vision-capable deployment to enable richer diagram understanding."
        )
        print(str(exc))
        return None


def build_image_documents_for_file(
    file_path: Path,
    vision_model: AzureChatOpenAI | None = None,
) -> list[Document]:
    suffix = file_path.suffix.lower()
    model = vision_model

    if model is None and suffix in SUPPORTED_IMAGE_EXTENSIONS.union({SUPPORTED_PDF_EXTENSION}):
        model = _get_optional_vision_model()

    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        description = describe_standalone_image(file_path, model)
        document = Document(
            page_content=description,
            metadata={
                **_figure_metadata(
                    source=file_path.name,
                    page_number=None,
                    index=1,
                    figure_number=extract_figure_number(description),
                    content_type=IMAGE_DESCRIPTION_CONTENT_TYPE,
                ),
                "file_type": "image",
                "render_type": "standalone_image",
            },
        )
        print(f"Processed image: {file_path.name}")
        print(f"Description length: {len(description)} characters")
        return [document]

    if suffix == SUPPORTED_PDF_EXTENSION:
        image_docs: list[Document] = []
        with fitz.open(file_path) as pdf:
            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                page_text = page.get_text("text").strip()
                if not page_contains_visual_content(page, page_text):
                    continue

                figure_number = extract_figure_number(page_text)
                description = describe_pdf_page(
                    file_path,
                    page_index,
                    model,
                    page_text=page_text,
                    figure_number=figure_number,
                )
                page_documents = _build_pdf_image_documents(
                    pdf_path=file_path,
                    page_number=page_index + 1,
                    description=description,
                    page_text=page_text,
                )
                image_docs.extend(page_documents)
                print(f"Processed PDF page image: {file_path.name} page {page_index + 1}")
                print(f"Description length: {len(description)} characters")

        return image_docs

    return []


def main() -> None:
    data_dirs = INGESTION_DIRS
    images = list(iter_image_files(data_dirs))
    pdfs = list(iter_pdf_files(data_dirs))

    if not images and not pdfs:
        print(f"No supported visual files found in {data_dir}")
        return

    print(f"Found {len(images)} standalone image file(s) and {len(pdfs)} PDF file(s) in {data_dir}")
    for image_path in images:
        print(f"Found IMAGE: {image_path.name}")
    for pdf_path in pdfs:
        print(f"Found PDF for visual extraction: {pdf_path.name}")

    documents = build_image_documents(data_dirs)
    if documents:
        sample = documents[0]
        print("Sample image document text:")
        print(sample.page_content[:1000])
        print("Sample image document metadata:")
        print(sample.metadata)


if __name__ == "__main__":
    main()
