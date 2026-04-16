import os
import pickle
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Paths
# -----------------------------
FAISS_INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_store.pkl"
TFIDF_PATH = "tfidf.pkl"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -----------------------------
# Load FAISS + metadata + TF-IDF
# -----------------------------
def load_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return index, meta, vectorizer


def embed_query(query, vectorizer):
    return vectorizer.transform([query]).toarray().astype("float32")


# -----------------------------
# Query Rewriting (Groq LLaMA)
# -----------------------------
def rewrite_query(query):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,
    )

    prompt = f"""
Rewrite the following query to be more specific and suitable for SAP enterprise documentation search.

Original Query: {query}

Rewritten Query:
"""
    rewritten = llm.invoke(prompt).content.strip()
    return rewritten


# -----------------------------
# Retrieval (FAISS)
# -----------------------------
def retrieve(query, index, meta, vectorizer, top_k=50):
    q_vec = embed_query(query, vectorizer)
    distances, indices = index.search(q_vec, top_k)
    return [meta[i] for i in indices[0]]


# -----------------------------
# Offline Reranking (TF-IDF Cosine Similarity)
# -----------------------------
def rerank(query, retrieved_chunks, vectorizer, top_k=5):
    q_vec = vectorizer.transform([query]).toarray()
    chunk_texts = [c["text"] for c in retrieved_chunks]
    chunk_vecs = vectorizer.transform(chunk_texts).toarray()

    scores = cosine_similarity(q_vec, chunk_vecs)[0]
    ranked = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)

    return [c for _, c in ranked[:top_k]]


# -----------------------------
# Build Context
# -----------------------------
def build_context(chunks):
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)


# -----------------------------
# LLM Answering (Groq + LLaMA)
# -----------------------------
def answer_question(query, index, meta, vectorizer):
    rewritten = rewrite_query(query)
    raw_chunks = retrieve(rewritten, index, meta, vectorizer, top_k=50)
    chunks = rerank(rewritten, raw_chunks, vectorizer, top_k=5)
    context = build_context(chunks)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
    )

    prompt = f"""
You are an SAP expert. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}
"""

    resp = llm.invoke(prompt)
    return resp.content


# -----------------------------
# Premium UI CSS
# -----------------------------
PREMIUM_CSS = """
<style>

body {
    background-color: #f3f4f6;
}

/* Main container */
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding-bottom: 120px;
}

/* Chat box */
.chat-box {
    padding: 20px;
    background-color: #ffffff;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* User message */
.user-msg {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    padding: 14px 18px;
    border-radius: 18px;
    margin: 10px 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    animation: fadeIn 0.4s ease-in-out;
}

/* Assistant message */
.assistant-msg {
    background: #f1f5f9;
    padding: 14px 18px;
    border-radius: 18px;
    margin: 10px 0;
    max-width: 75%;
    margin-right: auto;
    font-size: 15px;
    line-height: 1.5;
    border: 1px solid #e2e8f0;
    animation: fadeIn 0.4s ease-in-out;
}

/* Fade-in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Floating input bar */
.input-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 18px 0;
    background: rgba(243,244,246,0.85);
    backdrop-filter: blur(8px);
    border-top: 1px solid #e5e7eb;
}

.input-inner {
    max-width: 900px;
    margin: 0 auto;
    background: white;
    border-radius: 999px;
    padding: 10px 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

</style>
"""


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="SAP RAG Assistant", layout="wide")

    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # Load resources
    if "resources" not in st.session_state:
        st.session_state.resources = load_resources()
    index, meta, vectorizer = st.session_state.resources

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar
    st.sidebar.title("📚 Chat History")

    if st.sidebar.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.sidebar.markdown(f"🧑 **You:** {msg['content'][:40]}...")
        else:
            st.sidebar.markdown(f"🤖 **Bot:** {msg['content'][:40]}...")

    # Main Chat UI
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Floating Input Bar
    st.markdown("<div class='input-bar'><div class='input-inner'>", unsafe_allow_html=True)
    user_query = st.chat_input("Ask anything about your SAP PDFs...")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Handle Query
    if user_query and user_query.strip():
        st.session_state.history.append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):
            answer = answer_question(user_query, index, meta, vectorizer)

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.rerun()


if __name__ == "__main__":
    main()
