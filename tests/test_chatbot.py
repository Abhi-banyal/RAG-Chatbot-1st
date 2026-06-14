from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.chatbot import answer_question, filter_relevant_chunks, resolve_question
from app.retriever import RetrievedChunk


class FakeModel:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages):  # noqa: D401 - test helper signature mirrors LangChain
        return SimpleNamespace(content=self._content)


def make_chunk(
    *,
    text: str,
    source: str,
    page: int | None,
    content_type: str,
    figure_number: str | None,
    image_id: str | None,
    chunk_id: str,
    score: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source=source,
        page=page,
        content_type=content_type,
        figure_number=figure_number,
        image_id=image_id,
        chunk_id=chunk_id,
        score=score,
        metadata={
            "source": source,
            "page": page,
            "content_type": content_type,
            "figure_number": figure_number,
            "image_id": image_id,
            "chunk_id": chunk_id,
        },
    )


class ChatbotBehaviorTests(unittest.TestCase):
    @patch("app.chatbot.find_figure_chunks")
    @patch("app.chatbot.get_chat_model")
    def test_example_1_explain_figure_8_in_genai_uses_visual_context(
        self,
        mock_get_chat_model,
        mock_find_figure_chunks,
    ):
        mock_find_figure_chunks.return_value = [
            make_chunk(
                text="Figure 8 shows a pipeline diagram linking inputs, model training, and deployment.",
                source="genai.pdf",
                page=8,
                content_type="image_description",
                figure_number="Figure 8",
                image_id="genai_page_8_image_1",
                chunk_id="genai.pdf:page-8:chunk-1",
            )
        ]
        mock_get_chat_model.return_value = FakeModel("Figure 8 shows a GenAI pipeline with inputs, training, and deployment.")

        answer, sources = answer_question("explain figure 8 in genai")

        self.assertEqual(answer, "Figure 8 shows a GenAI pipeline with inputs, training, and deployment.")
        self.assertEqual(sources, ["genai.pdf, page 8, Figure 8"])
        mock_find_figure_chunks.assert_called_once_with("explain figure 8 in genai", debug=True)

    @patch("app.chatbot.get_chat_model")
    @patch("app.chatbot.retrieve_chunks")
    def test_example_2_diagram_question_uses_image_description_chunks(
        self,
        mock_retrieve_chunks,
        mock_get_chat_model,
    ):
        mock_retrieve_chunks.return_value = [
            make_chunk(
                text="Figure/diagram description: This chart shows the model lifecycle and deployment stages.",
                source="genai.pdf",
                page=6,
                content_type="image_description",
                figure_number=None,
                image_id="genai_page_6_image_1",
                chunk_id="genai.pdf:page-6:chunk-1",
            )
        ]
        mock_get_chat_model.return_value = FakeModel("The diagram shows the model lifecycle and deployment stages.")

        answer, sources = answer_question("what does the diagram in genai.pdf show?")

        self.assertEqual(answer, "The diagram shows the model lifecycle and deployment stages.")
        self.assertEqual(sources, ["genai.pdf, page 6"])

    @patch("app.chatbot.get_chat_model")
    @patch("app.chatbot.retrieve_chunks")
    def test_example_3_company_chart_answer_includes_pdf_source_and_page(
        self,
        mock_retrieve_chunks,
        mock_get_chat_model,
    ):
        mock_retrieve_chunks.return_value = [
            make_chunk(
                text="Figure 3 shows quarterly revenue growth for the company chart.",
                source="company_chart.pdf",
                page=1,
                content_type="image_description",
                figure_number="Figure 3",
                image_id="company_chart_page_1_image_1",
                chunk_id="company_chart.pdf:page-1:chunk-1",
            )
        ]
        mock_get_chat_model.return_value = FakeModel("The chart shows quarterly revenue growth.")

        answer, sources = answer_question("explain company chart")

        self.assertEqual(answer, "The chart shows quarterly revenue growth.")
        self.assertEqual(sources, ["company_chart.pdf, page 1, Figure 3"])

    @patch("app.chatbot.find_figure_chunks")
    def test_example_4_exact_figure_7_uses_metadata_without_nearby_suggestions(self, mock_find_figure_chunks):
        mock_find_figure_chunks.return_value = [
            make_chunk(
                text="Figure 7 shows word embeddings and semantic relationships in vector space.",
                source="genai.pdf",
                page=8,
                content_type="image_description",
                figure_number="Figure 7",
                image_id="genai_page_8_image_1",
                chunk_id="genai.pdf:page-8:chunk-3",
            )
        ]
        with patch("app.chatbot.get_chat_model", return_value=FakeModel("Figure 7 shows word embeddings and semantic relationships.")):
            answer, sources = answer_question("what figure 7 is showing in genai")

        self.assertEqual(answer, "Figure 7 shows word embeddings and semantic relationships.")
        self.assertEqual(sources, ["genai.pdf, page 8, Figure 7"])
        mock_find_figure_chunks.assert_called_once_with("what figure 7 is showing in genai", debug=True)

    @patch("app.chatbot.find_figure_chunks")
    def test_example_5_missing_figure_returns_generic_clarification(
        self,
        mock_find_figure_chunks,
    ):
        mock_find_figure_chunks.return_value = []

        resolved_question, clarification, pending_question = resolve_question("figure 7", None)

        self.assertIsNone(resolved_question)
        self.assertEqual(
            clarification,
            "I could not find Figure 7 in the uploaded documents. Please mention the document name or try another figure number.",
        )
        self.assertIsNone(pending_question)

    @patch("app.chatbot.get_chat_model")
    @patch("app.chatbot.retrieve_chunks")
    def test_example_6_taxonomy_question_filters_weak_genai_distractors(
        self,
        mock_retrieve_chunks,
        mock_get_chat_model,
    ):
        mock_retrieve_chunks.return_value = [
            make_chunk(
                text="A taxonomy of GenAI-related disciplines includes Artificial Intelligence, Machine Learning, Deep Learning, and Generative AI in nested circles.",
                source="genai.pdf",
                page=1,
                content_type="image_description",
                figure_number="Figure 1",
                image_id="genai_page_1_image_1",
                chunk_id="genai.pdf:page-1:chunk-1",
            ),
            make_chunk(
                text="Midjourney generates visual content from prompts.",
                source="genai.pdf",
                page=2,
                content_type="image_description",
                figure_number="Figure 2",
                image_id="genai_page_2_image_1",
                chunk_id="genai.pdf:page-2:chunk-1",
            ),
            make_chunk(
                text="OpenAI Codex helps with code generation examples.",
                source="genai.pdf",
                page=2,
                content_type="image_description",
                figure_number="Figure 3",
                image_id="genai_page_2_image_2",
                chunk_id="genai.pdf:page-2:chunk-2",
            ),
        ]
        mock_get_chat_model.return_value = FakeModel(
            "The taxonomy groups GenAI into AI, Machine Learning, Deep Learning, and Generative AI."
        )

        answer, sources = answer_question("what is the taxonomy and hierarchy of genai disciplines?")

        self.assertEqual(
            answer,
            "The taxonomy groups GenAI into AI, Machine Learning, Deep Learning, and Generative AI.",
        )
        self.assertEqual(sources, ["genai.pdf, page 1, Figure 1"])
        filtered = filter_relevant_chunks("what is the taxonomy and hierarchy of genai disciplines?", mock_retrieve_chunks.return_value)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].figure_number, "Figure 1")
        self.assertNotIn("Midjourney", filtered[0].text)
        self.assertNotIn("Codex", filtered[0].text)


if __name__ == "__main__":
    unittest.main()
