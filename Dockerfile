# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# faiss-cpu needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion/   ingestion/
COPY embeddings/  embeddings/
COPY chains/      chains/
COPY api/         api/
COPY monitoring/  monitoring/
COPY evaluation/  evaluation/
COPY exploration/ exploration/

COPY faiss_index/ faiss_index/
COPY data/        data/

# Ollama runs on the host — override at runtime for Linux:
#   docker run -e OLLAMA_HOST=http://172.17.0.1:11434 ...
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
