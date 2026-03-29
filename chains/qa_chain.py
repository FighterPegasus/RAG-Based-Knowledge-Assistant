from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from embeddings.vector_store import load_index

LLM_MODEL = "llama3"

PROMPT_TEMPLATE = """You are a financial analyst assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite the source document and page number when referencing specific data.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_chain(k: int = 4):
    """Build RetrievalQA chain using local FAISS index and Ollama LLaMA 3."""
    index = load_index()
    retriever = index.as_retriever(search_kwargs={"k": k})
    llm = OllamaLLM(model=LLM_MODEL)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def ask(chain, question: str) -> dict:
    result = chain.invoke({"query": question})
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "snippet": doc.page_content[:200],
        }
        for doc in result.get("source_documents", [])
    ]
    return {"answer": result["result"], "sources": sources}


if __name__ == "__main__":
    print("Loading chain...")
    chain = build_chain()

    questions = [
        "What was Capital One's net income in 2024?",
        "What is Discover Financial's provision for credit losses?",
        "What was Synchrony Financial's net charge-off rate in 2024?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        response = ask(chain, q)
        print(f"A: {response['answer']}")
        print("Sources:")
        for s in response["sources"]:
            print(f"  - {s['source']} (page {s['page']})")
