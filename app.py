# -*- coding: utf-8 -*-

from pathlib import Path
import os
from collections import OrderedDict

import streamlit as st
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# =========================================================
# Base configuration
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "data" / "index"
ACTIVE_CHROMA_PATH_FILE = INDEX_DIR / "active_chroma_path.txt"

COLLECTION_NAME = "cgspace_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "models/gemini-flash-lite-latest"
TOP_K_DEFAULT = 8


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🌿",
    layout="wide"
)


# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #F7FAF8 !important;
        color: #17231D !important;
    }

    [data-testid="stHeader"] {
        background: #F7FAF8 !important;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .hero {
        background: linear-gradient(135deg, #003D2B 0%, #00543C 55%, #63C08A 100%);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 30px rgba(0, 61, 43, 0.13);
    }

    .hero-label {
        font-size: 0.78rem;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        opacity: 0.9;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 800;
        line-height: 1.15;
    }

    .hero h2 {
        font-size: 1.02rem;
        margin-top: 1.2rem;
        line-height: 1.5;
        font-weight: 550;
        max-width: 850px;
        opacity: 0.96;
    }

    .hero-credit {
        font-size: 0.84rem;
        margin-top: 1.4rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.24);
        opacity: 0.9;
    }

    .note-box {
        background: #FFFFFF;
        border: 1px solid #DCEBE4;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #31483B;
        font-size: 0.93rem;
        margin-bottom: 1.1rem;
    }

    [data-testid="stSidebar"] {
        background: #EFF8F2 !important;
        color: #17231D !important;
    }

    .sidebar-title {
        color: #003D2B;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .sidebar-text {
        color: #30483B;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .fake-logo {
        display: flex;
        width: 100%;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 1.1rem;
        border: 1px solid #DCEBE4;
    }

    .fake-logo-left {
        background:#003D2B;
        color:white;
        font-weight:800;
        font-size:1.45rem;
        padding:1.05rem 0.8rem;
        width:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        letter-spacing:0.02rem;
    }

    .fake-logo-right {
        background:#63C08A;
        color:#003D2B;
        font-weight:800;
        font-size:0.82rem;
        line-height:1.05;
        padding:1rem 0.75rem;
        width:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        text-align:left;
    }

    .source-card {
        background-color: #FFFFFF;
        border: 1px solid #DCEBE4;
        border-left: 5px solid #63C08A;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.75rem;
        color: #17231D;
        box-shadow: 0 3px 10px rgba(0,0,0,0.035);
    }

    .source-title {
        font-weight: 750;
        color: #003D2B;
        margin-bottom: 0.25rem;
    }

    .source-meta {
        font-size: 0.88rem;
        color: #5B6B63;
        margin-bottom: 0.35rem;
    }

    .source-card a {
        color: #00543C !important;
        font-weight: 700;
        text-decoration: none;
    }

    .source-card a:hover {
        text-decoration: underline;
    }

    div[data-testid="stChatMessageContent"] {
        color: #17231D !important;
    }

    .stMarkdown {
        color: #17231D !important;
    }

    textarea, input {
        background-color: #FFFFFF !important;
        color: #17231D !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Environment and model loading
# =========================================================
load_dotenv(BASE_DIR / ".env")


def get_google_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            key = None
    return key


def load_chroma_path():
    if ACTIVE_CHROMA_PATH_FILE.exists():
        path = ACTIVE_CHROMA_PATH_FILE.read_text(encoding="utf-8").strip()
        if path:
            return path
    return str(INDEX_DIR / "chroma")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_chroma_collection():
    chroma_path = load_chroma_path()
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION_NAME)
    return collection


@st.cache_resource
def load_gemini_model():
    api_key = get_google_api_key()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


# =========================================================
# RAG functions
# =========================================================
def retrieve_context(question: str, top_k: int):
    embedding_model = load_embedding_model()
    collection = load_chroma_collection()

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        chunks.append({
            "raw_id": i,
            "text": doc,
            "title": meta.get("title", "") or meta.get("document", ""),
            "document": meta.get("document", ""),
            "page": meta.get("page", ""),
            "item_url": meta.get("item_url", ""),
            "pdf_url": meta.get("pdf_url", ""),
            "pdf_name": meta.get("pdf_name", ""),
            "similarity": round(1 - dist, 4)
        })

    return chunks


def group_sources(chunks: list[dict]):
    grouped = OrderedDict()

    for chunk in chunks:
        key = chunk.get("item_url") or chunk.get("title") or chunk.get("document")

        if key not in grouped:
            grouped[key] = {
                "id": len(grouped) + 1,
                "title": chunk.get("title", "") or "Untitled document",
                "document": chunk.get("document", ""),
                "item_url": chunk.get("item_url", ""),
                "pdf_url": chunk.get("pdf_url", ""),
                "pages": set(),
                "texts": []
            }

        page = chunk.get("page", "")
        if page != "":
            grouped[key]["pages"].add(str(page))

        grouped[key]["texts"].append(chunk.get("text", ""))

    sources = list(grouped.values())

    for source in sources:
        source["pages"] = sorted(
            source["pages"],
            key=lambda x: int(x) if str(x).isdigit() else 999999
        )

    return sources


def pages_to_text(pages):
    if not pages:
        return "n.d."
    return ", ".join(pages)


def build_prompt(question: str, grouped_sources: list[dict]) -> str:
    context_blocks = []

    for source in grouped_sources:
        excerpts = "\n\n".join(
            [f"Excerpt {i+1}: {txt}" for i, txt in enumerate(source["texts"][:4])]
        )

        context_blocks.append(
            f"[{source['id']}]\n"
            f"Title: {source['title']}\n"
            f"Pages retrieved: {pages_to_text(source['pages'])}\n"
            f"CGSpace URL: {source['item_url']}\n"
            f"PDF URL: {source['pdf_url']}\n"
            f"Text excerpts:\n{excerpts}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a research assistant agent for the CGIAR Science Program on Multifunctional Landscapes.

Your job is to answer the user's question using ONLY the retrieved CGSpace document excerpts below.
Do not use web sources or outside knowledge. Use only the retrieved context.

Core rules:
1. Do not invent facts, outcomes, countries, indicators, targets, dates, partners, or conclusions.
2. If the retrieved context is insufficient, say so clearly.
3. Distinguish between stated outcomes, intended outcomes, achieved results, objectives, assumptions, recommendations, and supporting evidence.
4. If documents describe objectives or theory of change, do not present them as achieved results.
5. Use simple numeric citations in the text, like [1], [2], [3].
6. Do not include long document titles or URLs inside the main answer.
7. Every substantive claim must be traceable to one of the retrieved sources.
8. Do not cite sources that are not included in the retrieved context.
9. In the "Sources used" section, group repeated evidence from the same document into one source entry.
10. Use this source format:
   [1] Title of the document. Pages: 5, 6. CGSpace.
11. The word "CGSpace" should refer to the source URL provided in the retrieved context.
12. Keep the answer concise, academic, and useful for research or program review.

Write naturally in an academic and analytical style.
Use structured paragraphs when useful, but do not force bullet points or rigid sections.
Only include an evidence limitation if truly necessary.
Do not generate a "Sources used" section in the answer.
Source cards will be displayed separately in the interface.


User question:
{question}

Retrieved context:
{context}

Answer:
"""


def answer_question(question: str, top_k: int):
    llm = load_gemini_model()

    if llm is None:
        raise ValueError("GOOGLE_API_KEY is not configured in .env or Streamlit secrets.")

    chunks = retrieve_context(question, top_k=top_k)
    grouped_sources = group_sources(chunks)
    prompt = build_prompt(question, grouped_sources)

    response = llm.generate_content(prompt)

    return response.text, grouped_sources


def render_source_card(source):
    title = source["title"] or source["document"] or "Untitled source"
    item_url = source.get("item_url", "")
    pdf_url = source.get("pdf_url", "")
    pages = pages_to_text(source.get("pages", []))

    cgspace_link = f'<a href="{item_url}" target="_blank">CGSpace</a>' if item_url else "CGSpace"
    pdf_link = f' &nbsp;|&nbsp; <a href="{pdf_url}" target="_blank">PDF</a>' if pdf_url else ""

    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-title">[{source['id']}] {title}</div>
            <div class="source-meta">
                Pages: {pages}. {cgspace_link}.{pdf_link}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="fake-logo">
            <div class="fake-logo-left">CGIAR</div>
            <div class="fake-logo-right">MULTIFUNCTIONAL<br>LANDSCAPES</div>
        </div>

        <div class="sidebar-title">Research Assistant</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-text">
        Select the number of document segments used to generate the answer.
        </div>
        """,
        unsafe_allow_html=True
    )

    top_k_user = st.slider(
        "Document segments",
        min_value=3,
        max_value=12,
        value=TOP_K_DEFAULT
    )


# =========================================================
# Main UI
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">CGIAR Science Program on Multifunctional Landscapes</div>
        <h1>Research Assistant Agent</h1>
        <h2>
        Explore program documents, reports, briefs, and evidence through questions grounded
        in sources retrieved from the official program repository in CGSpace.
        </h2>
        <div class="hero-credit">
        Source corpus: official CGSpace collection for the CGIAR Science Program on Multifunctional Landscapes.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# =========================================================
# Chat state
# =========================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("CGSpace document sources"):
                for source in msg["sources"]:
                    render_source_card(source)


user_question = st.chat_input("Ask a question about the MFL document collection...")


if user_question:
    st.session_state["messages"].append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence from CGSpace documents and generating answer..."):
            try:
                answer, grouped_sources = answer_question(user_question, top_k=top_k_user)

                st.markdown(answer)

                with st.expander("CGSpace document sources"):
                    for source in grouped_sources:
                        render_source_card(source)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": grouped_sources
                })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg
                })