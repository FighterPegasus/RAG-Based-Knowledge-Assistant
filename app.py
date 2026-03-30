import time
import streamlit as st
from chains.qa_chain import build_chain, ask

st.set_page_config(page_title="RAG Knowledge Assistant", layout="centered")
st.title("RAG Knowledge Assistant")
st.caption("Ask natural language questions across your documents")

# load chain once per session
if "chain" not in st.session_state:
    with st.spinner("Loading index and chain..."):
        st.session_state.chain = build_chain()

if "history" not in st.session_state:
    st.session_state.history = []

# render chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"Sources — {msg['latency']}s"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** — page {s['page']}")

# input
if question := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            result = ask(st.session_state.chain, question)
            latency = round(time.time() - start, 2)

        st.markdown(result["answer"])
        with st.expander(f"Sources — {latency}s"):
            for s in result["sources"]:
                st.markdown(f"- **{s['source']}** — page {s['page']}")

    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "latency": latency,
    })
