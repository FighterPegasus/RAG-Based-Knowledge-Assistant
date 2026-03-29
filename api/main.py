import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chains.qa_chain import build_chain, ask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# loaded once at startup, reused across requests
_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain
    logger.info("Loading RAG chain...")
    _chain = build_chain()
    logger.info("RAG chain ready.")
    yield
    _chain = None


app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Q&A over Capital One, Discover, and Synchrony 10-K filings.",
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    source: str
    page: str | int
    snippet: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok", "chain_loaded": _chain is not None}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if _chain is None:
        raise HTTPException(status_code=503, detail="Chain not loaded yet.")

    start = time.time()
    try:
        result = ask(_chain, request.question)
    except Exception as e:
        logger.error(f"Chain error: {e}")
        raise HTTPException(status_code=500, detail=f"Chain error: {str(e)}")
    latency = round(time.time() - start, 3)

    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        latency_seconds=latency,
    )
