from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from ingestion.pdf_loader import load_pdfs
from ingestion.csv_loader import load_csv

INDEX_DIR = Path(__file__).parent.parent / "faiss_index"

# NOTE: nomic-embed-text is the recommended embedding model for Ollama
EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_index():
    """Load all docs, embed them, and save the FAISS index to disk."""
    print("Loading documents...")
    docs = load_pdfs() + load_csv()
    print(f"Total documents to embed: {len(docs)}")

    print("Building FAISS index (this may take a few minutes)...")
    embeddings = get_embeddings()
    index = FAISS.from_documents(docs, embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    index.save_local(str(INDEX_DIR))
    print(f"Index saved to {INDEX_DIR}")
    return index


def load_index():
    """Load existing FAISS index from disk."""
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"No index found at {INDEX_DIR}. Run build_index() first.")
    embeddings = get_embeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def query_index(index, question: str, k: int = 4):
    return index.similarity_search(question, k=k)


if __name__ == "__main__":
    # TODO: add retry logic for Ollama connection errors
    index = build_index()

    print("\nTest query: 'What is Capital One net income?'")
    results = query_index(index, "What is Capital One net income?")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] source={doc.metadata.get('source')} page={doc.metadata.get('page')}")
        print(doc.page_content[:200])
