# Quick data inspection — run with: python -m exploration.explore_data

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_FILES = [
    "capital_one_10k.pdf",
    "discover_10k.pdf",
    "synchrony_10k.pdf",
]


def explore_csv():
    csv_path = DATA_DIR / "financial_summary.csv"
    print("=" * 60)
    print("CSV: financial_summary.csv")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}\n")
    print(df.to_string(index=False))

    print(f"\n  Null values per column:")
    for col, n in df.isnull().sum().items():
        print(f"    {col}: {n} nulls")


def explore_pdfs():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    print("\n" + "=" * 60)
    print("PDF Documents")
    print("=" * 60)

    total_pages = 0
    total_chunks = 0

    for filename in PDF_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  WARNING: {filename} not found, skipping.")
            continue

        loader = PyPDFLoader(str(path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        size_mb = path.stat().st_size / (1024 * 1024)

        print(f"\n  {filename}")
        print(f"    File size : {size_mb:.1f} MB")
        print(f"    Pages     : {len(pages)}")
        print(f"    Chunks    : {len(chunks)}")
        print(f"    Avg chunk : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

        total_pages += len(pages)
        total_chunks += len(chunks)

    print(f"\n  TOTAL — pages: {total_pages}  |  chunks: {total_chunks}")


def explore_index():
    from embeddings.vector_store import load_index

    print("\n" + "=" * 60)
    print("FAISS Index")
    print("=" * 60)

    index = load_index()
    total_vectors = index.index.ntotal
    print(f"\n  Total vectors in index: {total_vectors}")

    # TODO: expose k as a param if we want to dig deeper
    sample_queries = {
        "Capital One":         "Capital One net income revenue",
        "Discover Financial":  "Discover Financial credit losses provision",
        "Synchrony Financial": "Synchrony Financial charge-off rate",
    }

    for company, query in sample_queries.items():
        docs = index.similarity_search(query, k=2)
        print(f"\n  Sample chunks for '{company}':")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:180].replace("\n", " ")
            print(f"    [{i}] {src} (p.{page}): {snippet}...")


def explore_chunk_distribution():
    from ingestion.pdf_loader import load_pdfs
    from ingestion.csv_loader import load_csv

    print("\n" + "=" * 60)
    print("Chunk Character Distribution")
    print("=" * 60)

    all_chunks = load_pdfs() + load_csv()
    lengths = [len(c.page_content) for c in all_chunks]

    print(f"\n  Total chunks : {len(lengths)}")
    print(f"  Min chars    : {min(lengths)}")
    print(f"  Max chars    : {max(lengths)}")
    print(f"  Mean chars   : {sum(lengths) // len(lengths)}")

    buckets = {"<200": 0, "200-500": 0, "500-800": 0, "800-1000": 0, ">1000": 0}
    for l in lengths:
        if l < 200:
            buckets["<200"] += 1
        elif l < 500:
            buckets["200-500"] += 1
        elif l < 800:
            buckets["500-800"] += 1
        elif l <= 1000:
            buckets["800-1000"] += 1
        else:
            buckets[">1000"] += 1

    print("\n  Distribution:")
    for bucket, count in buckets.items():
        bar = "#" * (count // 10)
        print(f"    {bucket:>10}  {count:>5}  {bar}")


if __name__ == "__main__":
    explore_csv()
    explore_pdfs()
    explore_index()
    explore_chunk_distribution()
