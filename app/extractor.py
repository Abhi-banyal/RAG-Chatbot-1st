from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path

import fitz
import pandas as pd

from app.config import DATA_DIR, INGESTION_DIRS
from app.file_reader import FILE_TYPE_LABELS, iter_supported_files


@dataclass(slots=True)
class ExtractedContent:
    text: str
    source: str
    file_type: str
    page_number: int | None = None
    row_number: int | None = None


def extract_pdf(file_path: Path) -> list[ExtractedContent]:
    extracted_items: list[ExtractedContent] = []

    with fitz.open(file_path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            text = page.get_text("text").strip()
            extracted_items.append(
                ExtractedContent(
                    text=text,
                    source=file_path.name,
                    file_type="pdf",
                    page_number=page_index + 1,
                )
            )

    print(f"Extracted PDF: {file_path.name} | pages: {len(extracted_items)}")
    return extracted_items


def extract_txt(file_path: Path) -> list[ExtractedContent]:
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    extracted_items = [
        ExtractedContent(
            text=text,
            source=file_path.name,
            file_type="txt",
        )
    ]
    print(f"Extracted TXT: {file_path.name} | documents: 1")
    return extracted_items


def row_to_text(row: pd.Series) -> str:
    parts = []
    for column, value in row.items():
        cleaned_value = "" if pd.isna(value) else str(value).strip()
        parts.append(f"{column}: {cleaned_value}")
    return "; ".join(parts)


def extract_csv(file_path: Path) -> list[ExtractedContent]:
    dataframe = pd.read_csv(file_path, dtype=str).fillna("")
    extracted_items: list[ExtractedContent] = []

    for index, row in dataframe.iterrows():
        extracted_items.append(
            ExtractedContent(
                text=row_to_text(row),
                source=file_path.name,
                file_type="csv",
                row_number=index + 1,
            )
        )

    print(f"Extracted CSV: {file_path.name} | rows: {len(extracted_items)}")
    return extracted_items


def extract_file(file_path: Path) -> list[ExtractedContent]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(file_path)
    if suffix == ".txt":
        return extract_txt(file_path)
    if suffix == ".csv":
        return extract_csv(file_path)

    print(f"Skipped image for now: {file_path.name}")
    return []


def extract_all_content(data_dirs: Iterable[Path] | None = None) -> list[ExtractedContent]:
    base_dirs = list(data_dirs) if data_dirs is not None else [DATA_DIR]
    extracted_content: list[ExtractedContent] = []

    for base_dir in base_dirs:
        for file_path in iter_supported_files(base_dir):
            if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                print(f"Skipped image for now: {file_path.name}")
                continue

            extracted_content.extend(extract_file(file_path))

    return extracted_content


def print_extraction_summary(items: list[ExtractedContent]) -> None:
    if not items:
        print("No text content extracted.")
        return

    summary: dict[str, int] = {}
    for item in items:
        summary[item.file_type] = summary.get(item.file_type, 0) + 1

    print("Extraction summary:")
    for file_type, count in summary.items():
        print(f"- {file_type.upper()}: {count}")


def main() -> None:
    data_dir = DATA_DIR
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    items = extract_all_content(INGESTION_DIRS)
    print_extraction_summary(items)


if __name__ == "__main__":
    main()
