# RAG Knowledge Assistant

Answers natural language questions over three financial 10-K filings:
- Capital One (`capital_one_10k.pdf`)
- Discover Financial (`discover_10k.pdf`)
- Synchrony Financial (`synchrony_10k.pdf`)

Retrieval is done with FAISS. Generation uses LLaMA 3 via Ollama. The API is served with FastAPI.

---

## Project Structure

```
.
├── api/
│   └── main.py                 # FastAPI app (POST /ask, GET /health)
├── chains/
│   └── qa_chain.py             # RetrievalQA chain (LLaMA 3 + FAISS retriever)
├── data/
│   ├── capital_one_10k.pdf
│   ├── discover_10k.pdf
│   ├── synchrony_10k.pdf
│   └── financial_summary.csv
├── embeddings/
│   └── vector_store.py         # FAISS index build/load/query
├── evaluation/
│   └── retrieval_eval.py       # Hit rate, source accuracy, MRR
├── exploration/
│   └── explore_data.py         # Data inspection script
├── faiss_index/                # Pre-built index (do not delete)
├── ingestion/
│   ├── pdf_loader.py           # PDF → chunks
│   └── csv_loader.py           # CSV → Documents
├── monitoring/
│   └── mlflow_logger.py        # MLflow query logging
├── tests/
│   └── test_pipeline.py        # Unit tests (14 tests, no Ollama required)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) running locally with two models pulled:
  ```
  ollama pull llama3
  ollama pull nomic-embed-text
  ```

---

## Setup

```bash
git clone <repo-url>
cd "RAG Knowledge Assistant"

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The FAISS index is pre-built in `faiss_index/`. If you need to rebuild it (e.g. after changing the data files):

```bash
python -m embeddings.vector_store
```

---

## Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

### Endpoints

#### `GET /health`
```json
{ "status": "ok", "chain_loaded": true }
```

#### `POST /ask`

Request:
```json
{ "question": "What was Capital One's net income in 2024?" }
```

Response:
```json
{
  "question": "What was Capital One's net income in 2024?",
  "answer": "...",
  "sources": [
    {
      "source": "capital_one_10k.pdf",
      "page": <page_number>,
      "snippet": "..."
    }
  ],
  "latency_seconds": <measured>
}
```

---

## Running with Docker

```bash
docker build -t rag-assistant .

# macOS / Windows (Docker Desktop)
docker run -p 8000:8000 rag-assistant

# Linux (Ollama on host)
docker run -p 8000:8000 \
  -e OLLAMA_HOST=http://172.17.0.1:11434 \
  rag-assistant
```

---

## Tests

```bash
pytest tests/test_pipeline.py -v
```

14 unit tests across all pipeline components. No live Ollama instance required — external calls are mocked.

---

## Retrieval Evaluation

```bash
python -m evaluation.retrieval_eval
```

Evaluates 8 questions (covering all 3 companies) against the live FAISS index.

| Metric | Description |
|--------|-------------|
| Hit Rate | Fraction of queries where a retrieved chunk contains an expected keyword |
| Source Accuracy | Fraction where the correct company's PDF appears in top-k results |
| MRR | Mean Reciprocal Rank of the first matching source |

Expected output format:
```
num_questions: 8
k: 4
hit_rate: <measured>
source_accuracy: <measured>
mrr: <measured>
avg_chunks_returned: 4.0
```

---

## MLflow Tracking

```bash
python -m monitoring.mlflow_logger   # runs one example query and logs it

mlflow ui                             # open http://localhost:5000
```

Each query is logged as a run under the `rag-knowledge-assistant` experiment with:
- **Params**: question, num_sources, source_files
- **Metrics**: latency_seconds, answer_length_chars, sources_returned
- **Artifact**: full answer text file

---

## Data Exploration

```bash
python -m exploration.explore_data
```

Prints CSV contents, per-PDF page/chunk counts, FAISS index size, and chunk character distribution.

---

## Models

| Role | Model | Provider |
|------|-------|----------|
| Embeddings | `nomic-embed-text` | Ollama (local) |
| Generation | `llama3` | Ollama (local) |

No external API calls are made at inference time.
