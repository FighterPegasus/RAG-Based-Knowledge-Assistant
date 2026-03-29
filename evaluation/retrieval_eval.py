# Offline retrieval eval — no LLM judge needed.
# Metrics: hit rate, source accuracy, MRR.
# Run: python -m evaluation.retrieval_eval

from embeddings.vector_store import load_index, query_index

# each entry: question, keywords (any match = hit), expected source file
EVAL_QUESTIONS = [
    {
        "question": "What was Capital One's net income?",
        "expected_keywords": ["net income", "capital one"],
        "expected_source": "capital_one_10k.pdf",
    },
    {
        "question": "What is Discover Financial's provision for credit losses?",
        "expected_keywords": ["provision", "credit losses", "discover"],
        "expected_source": "discover_10k.pdf",
    },
    {
        "question": "What was Synchrony Financial's net charge-off rate?",
        "expected_keywords": ["charge-off", "synchrony"],
        "expected_source": "synchrony_10k.pdf",
    },
    {
        "question": "What is Capital One's total revenue?",
        "expected_keywords": ["total revenue", "net revenue", "capital one"],
        "expected_source": "capital_one_10k.pdf",
    },
    {
        "question": "What are the risk factors for Discover Financial?",
        "expected_keywords": ["risk", "discover"],
        "expected_source": "discover_10k.pdf",
    },
    {
        "question": "How does Synchrony Financial describe its business segments?",
        "expected_keywords": ["segment", "synchrony", "platform"],
        "expected_source": "synchrony_10k.pdf",
    },
    {
        "question": "What is Capital One's allowance for credit losses?",
        "expected_keywords": ["allowance", "credit losses", "capital one"],
        "expected_source": "capital_one_10k.pdf",
    },
    {
        "question": "What dividends did Discover Financial pay?",
        "expected_keywords": ["dividend", "discover"],
        "expected_source": "discover_10k.pdf",
    },
]


def _chunk_hits_keywords(chunk_text: str, keywords: list[str]) -> bool:
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _source_rank(docs, expected_source: str) -> int | None:
    """Return 1-based rank of first chunk from expected_source, or None."""
    for rank, doc in enumerate(docs, start=1):
        if doc.metadata.get("source", "") == expected_source:
            return rank
    return None


def evaluate(k: int = 4, verbose: bool = True) -> dict:
    index = load_index()

    hits = 0
    source_hits = 0
    reciprocal_ranks = []
    total_chunks = 0

    for i, item in enumerate(EVAL_QUESTIONS):
        q = item["question"]
        keywords = item["expected_keywords"]
        expected_src = item["expected_source"]

        docs = query_index(index, q, k=k)
        total_chunks += len(docs)

        # hit rate: any retrieved chunk contains a keyword?
        hit = any(_chunk_hits_keywords(d.page_content, keywords) for d in docs)
        if hit:
            hits += 1

        # source accuracy + MRR
        rank = _source_rank(docs, expected_src)
        if rank is not None:
            source_hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        if verbose:
            status = "HIT" if hit else "MISS"
            src_status = f"rank {rank}" if rank else "not found"
            print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {status} | source: {src_status} | {q[:60]}")

    n = len(EVAL_QUESTIONS)
    results = {
        "num_questions": n,
        "k": k,
        "hit_rate": round(hits / n, 3),
        "source_accuracy": round(source_hits / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "avg_chunks_returned": round(total_chunks / n, 1),
    }

    if verbose:
        print("\n--- Retrieval Evaluation Results ---")
        for key, val in results.items():
            print(f"  {key}: {val}")

    return results


if __name__ == "__main__":
    print("Running retrieval evaluation (k=4)...\n")
    results = evaluate(k=4)
