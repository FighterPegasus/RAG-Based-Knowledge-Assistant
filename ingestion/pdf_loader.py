from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent.parent / "data"

PDF_FILES = [
    "capital_one_10k.pdf",
    "discover_10k.pdf",
    "synchrony_10k.pdf",
]

# NOTE: chunk size may need tuning for larger docs
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdfs():
    """Load and chunk all PDFs from the data directory."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for filename in PDF_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"WARNING: {filename} not found at {path}, skipping")
            continue

        loader = PyPDFLoader(str(path))
        pages = loader.load()

        # tag each chunk with source and page
        for doc in pages:
            doc.metadata["source"] = filename

        file_chunks = splitter.split_documents(pages)
        print(f"  {filename}: {len(pages)} pages → {len(file_chunks)} chunks")
        chunks.extend(file_chunks)

    return chunks


if __name__ == "__main__":
    print("Loading PDFs...")
    docs = load_pdfs()
    print(f"\nTotal chunks: {len(docs)}")
    print(f"\nSample chunk:\n{docs[0].page_content[:300]}")
    print(f"Metadata: {docs[0].metadata}")
