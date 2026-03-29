# pytest tests/test_pipeline.py -v
# Ollama and FAISS are mocked — no live services needed.

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from langchain.schema import Document


class TestPdfLoader:

    def test_load_pdfs_returns_list_of_documents(self, tmp_path, monkeypatch):
        fake_doc = Document(page_content="Capital One annual report", metadata={"page": 0})

        with patch("ingestion.pdf_loader.PyPDFLoader") as mock_loader_cls, \
             patch("ingestion.pdf_loader.DATA_DIR", tmp_path):

            for name in ["capital_one_10k.pdf", "discover_10k.pdf", "synchrony_10k.pdf"]:
                (tmp_path / name).touch()

            mock_loader_cls.return_value.load.return_value = [fake_doc]

            from ingestion.pdf_loader import load_pdfs
            chunks = load_pdfs()

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_load_pdfs_adds_source_metadata(self, tmp_path):
        fake_doc = Document(page_content="Some text", metadata={"page": 1})

        with patch("ingestion.pdf_loader.PyPDFLoader") as mock_loader_cls, \
             patch("ingestion.pdf_loader.DATA_DIR", tmp_path):

            (tmp_path / "capital_one_10k.pdf").touch()
            (tmp_path / "discover_10k.pdf").touch()
            (tmp_path / "synchrony_10k.pdf").touch()

            mock_loader_cls.return_value.load.return_value = [fake_doc]

            from ingestion.pdf_loader import load_pdfs
            chunks = load_pdfs()

        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_load_pdfs_skips_missing_files(self, tmp_path, capsys):
        with patch("ingestion.pdf_loader.DATA_DIR", tmp_path):
            from ingestion.pdf_loader import load_pdfs
            chunks = load_pdfs()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert chunks == []


class TestCsvLoader:

    def _make_csv(self, tmp_path):
        content = (
            "company,year,net_income_millions,total_revenue_millions,"
            "provision_for_credit_losses_millions,net_charge_off_rate_pct\n"
            "Capital One,2024,5000,35000,3000,2.5\n"
            "Discover Financial,2024,4000,16000,2500,3.1\n"
        )
        csv_file = tmp_path / "financial_summary.csv"
        csv_file.write_text(content)
        return csv_file

    def test_load_csv_returns_documents(self, tmp_path):
        self._make_csv(tmp_path)

        with patch("ingestion.csv_loader.CSV_FILE", tmp_path / "financial_summary.csv"):
            from ingestion.csv_loader import load_csv
            chunks = load_csv()

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_load_csv_metadata_fields(self, tmp_path):
        self._make_csv(tmp_path)

        with patch("ingestion.csv_loader.CSV_FILE", tmp_path / "financial_summary.csv"):
            from ingestion.csv_loader import load_csv
            chunks = load_csv()

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "company" in chunk.metadata
            assert "year" in chunk.metadata

    def test_load_csv_content_contains_company(self, tmp_path):
        self._make_csv(tmp_path)

        with patch("ingestion.csv_loader.CSV_FILE", tmp_path / "financial_summary.csv"):
            from ingestion.csv_loader import load_csv
            chunks = load_csv()

        companies = {c.metadata["company"] for c in chunks}
        assert "Capital One" in companies
        assert "Discover Financial" in companies


class TestVectorStore:

    def test_load_index_raises_when_missing(self, tmp_path):
        with patch("embeddings.vector_store.INDEX_DIR", tmp_path / "nonexistent"):
            from embeddings.vector_store import load_index
            with pytest.raises(FileNotFoundError):
                load_index()

    def test_query_index_calls_similarity_search(self):
        mock_index = MagicMock()
        mock_index.similarity_search.return_value = [
            Document(page_content="result chunk", metadata={"source": "capital_one_10k.pdf"})
        ]

        from embeddings.vector_store import query_index
        results = query_index(mock_index, "net income", k=3)

        mock_index.similarity_search.assert_called_once_with("net income", k=3)
        assert len(results) == 1
        assert results[0].page_content == "result chunk"

    def test_build_index_saves_to_disk(self, tmp_path):
        fake_docs = [Document(page_content="text", metadata={})]

        with patch("embeddings.vector_store.load_pdfs", return_value=fake_docs), \
             patch("embeddings.vector_store.load_csv", return_value=[]), \
             patch("embeddings.vector_store.get_embeddings", return_value=MagicMock()), \
             patch("embeddings.vector_store.FAISS") as mock_faiss_cls, \
             patch("embeddings.vector_store.INDEX_DIR", tmp_path / "faiss_index"):

            mock_index = MagicMock()
            mock_faiss_cls.from_documents.return_value = mock_index

            from embeddings.vector_store import build_index
            build_index()

        mock_faiss_cls.from_documents.assert_called_once()
        mock_index.save_local.assert_called_once()


class TestQaChain:

    def test_ask_returns_answer_and_sources(self):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "Capital One net income was $5 billion.",
            "source_documents": [
                Document(
                    page_content="Net income was $5 billion in 2024.",
                    metadata={"source": "capital_one_10k.pdf", "page": 42},
                )
            ],
        }

        from chains.qa_chain import ask
        response = ask(mock_chain, "What was Capital One's net income?")

        assert "answer" in response
        assert "sources" in response
        assert response["answer"] == "Capital One net income was $5 billion."
        assert len(response["sources"]) == 1
        assert response["sources"][0]["source"] == "capital_one_10k.pdf"
        assert response["sources"][0]["page"] == 42

    def test_ask_source_snippet_truncated(self):
        long_text = "x" * 500
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "Answer.",
            "source_documents": [
                Document(page_content=long_text, metadata={"source": "discover_10k.pdf", "page": 1})
            ],
        }

        from chains.qa_chain import ask
        response = ask(mock_chain, "Some question?")

        assert len(response["sources"][0]["snippet"]) <= 200

    def test_ask_handles_no_source_documents(self):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "I don't have enough information.",
            "source_documents": [],
        }

        from chains.qa_chain import ask
        response = ask(mock_chain, "Unanswerable question?")

        assert response["sources"] == []


class TestApi:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "Net income was $5B.",
            "source_documents": [
                Document(
                    page_content="Net income text here.",
                    metadata={"source": "capital_one_10k.pdf", "page": 10},
                )
            ],
        }

        with patch("api.main._chain", mock_chain):
            from api.main import app
            yield TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ask_endpoint_returns_answer(self, client):
        response = client.post("/ask", json={"question": "What is Capital One's net income?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_seconds" in data

    def test_ask_endpoint_rejects_empty_question(self, client):
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400

    def test_ask_endpoint_question_echoed_in_response(self, client):
        q = "What is Discover's provision for credit losses?"
        response = client.post("/ask", json={"question": q})
        assert response.json()["question"] == q
