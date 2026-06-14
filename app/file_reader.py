from __future__ import annotations

from pathlib import Path

from app.config import DATA_DIR


FILE_TYPE_LABELS = {
    ".pdf": "PDF",
    ".txt": "TXT",
    ".csv": "CSV",
    ".png": "IMAGE",
    ".jpg": "IMAGE",
    ".jpeg": "IMAGE",
}


def iter_supported_files(data_dir: Path | None = None) -> list[Path]:
    base_dir = data_dir or DATA_DIR
    if not base_dir.exists():
        return []

    return sorted(
        path
        for path in base_dir.iterdir()
        if path.is_file() and path.suffix.lower() in FILE_TYPE_LABELS
    )


def print_file_types(data_dir: Path | None = None) -> int:
    files = iter_supported_files(data_dir)

    if not files:
        print(f"No supported files found in {DATA_DIR}")
        return 0

    for path in files:
        label = FILE_TYPE_LABELS[path.suffix.lower()]
        print(f"Found {label}: {path.name}")

    return len(files)


def main() -> None:
    total = print_file_types()
    print(f"Total supported files found: {total}")


if __name__ == "__main__":
    main()
