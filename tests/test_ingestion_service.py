from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.services.ingestion_service import ingest_single_file


class IngestionServiceTests(unittest.TestCase):
    @patch("app.services.ingestion_service.append_documents_to_vectorstore")
    @patch("app.services.ingestion_service.chunk_documents")
    @patch("app.services.ingestion_service.build_documents_for_file")
    def test_ingest_single_file_processes_only_the_uploaded_file(
        self,
        mock_build_documents_for_file,
        mock_chunk_documents,
        mock_append_documents_to_vectorstore,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            uploaded_file = Path(tmp_dir) / "sample.pdf"
            uploaded_file.write_bytes(b"%PDF-1.4 test file")

            source_document = Document(
                page_content="hello world",
                metadata={"source": "sample.pdf", "content_type": "text"},
            )
            chunk_document = Document(
                page_content="hello world",
                metadata={"source": "sample.pdf", "content_type": "text", "chunk_id": "sample.pdf:page-na:chunk-1"},
            )

            mock_build_documents_for_file.return_value = [source_document]
            mock_chunk_documents.return_value = [chunk_document]
            mock_append_documents_to_vectorstore.return_value = Path(tmp_dir) / "vectorstore"

            result = ingest_single_file(uploaded_file)

            mock_build_documents_for_file.assert_called_once_with(uploaded_file)
            mock_chunk_documents.assert_called_once_with([source_document])
            mock_append_documents_to_vectorstore.assert_called_once_with([chunk_document])
            self.assertEqual(result.documents_indexed, 1)
            self.assertEqual(result.chunks_indexed, 1)


if __name__ == "__main__":
    unittest.main()
