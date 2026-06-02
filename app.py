# -*- coding: utf-8 -*-

from pathlib import Path
import os
from collections import OrderedDict

import pandas as pd
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "data" / "index"
CHUNKS_CSV = INDEX_DIR / "chunks.csv"

COLLECTION_NAME = "cgspace_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_CHROMA = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "models/gemini-flash-lite-latest"
TOP_K_DEFAULT = 8


st.set_page_config(
    page_title="MFL Research Assistant",
    page_icon="🌿",
    layout="wide"
)


st.markdown(
    """
    <style>
        :root {
            --radius: 8px;
            --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.8rem;
            padding-bottom: 6rem;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .hero {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: var(--radius);
            padding: 1.6rem 1.8rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 8px rgba(0,0,0,0.08);
        }

        .hero-top {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.8rem;
        }

        .hero-badge {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.6;
        }

        .hero-status {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
            color: #1F5E3B;
            background-color: #E6F4EA;
        }

        .hero-title {
            font-family: var(--font);
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 760;
            letter-spacing: -0.035em;
            margin: 0 0 0.85rem 0;
            line-height: 1.12;
        }

        .hero-desc {
            font-family: var(--font);
            font-size: 1.02rem;
            line-height: 1.55;
            opacity: 0.78;
            margin: 0;
            max-width: 970px;
        }

        .hero-meta {
            margin-top: 1.1rem;
            padding-top: 0.85rem;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            font-family: var(--font);
            font-size: 0.78rem;
            opacity: 0.58;
        }

        .source-card {
            display: flex;
            gap: 0.65rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-left: 3px solid #63C08A;
            border-radius: 0 var(--radius) var(--radius) 0;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 1px 5px rgba(0,0,0,0.05);
        }

        .source-card-icon {
            width: 16px;
            height: 16px;
            flex-shrink: 0;
            margin-top: 0.15rem;
            opacity: 0.35;
        }

        .source-card-body {
            flex: 1;
            min-width: 0;
        }

        .source-title {
            font-family: var(--font);
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
            line-height: 1.4;
        }

        .source-meta {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            opacity: 0.65;
            line-height: 1.5;
        }

        .source-meta a {
            color: #007A53 !important;
            font-weight: 650;
            text-decoration: none;
        }

        .source-meta a:hover {
            text-decoration: underline;
        }

        div[data-testid="stChatMessageContent"] {
            font-family: var(--font);
            font-size: 0.94rem;
            line-height: 1.65;
        }

        div[data-testid="stChatMessageContent"] p {
            margin: 0.35rem 0;
        }

        .stMarkdown {
            font-family: var(--font);
        }

        textarea, input {
            font-family: var(--font) !important;
        }

        .stSlider label {
            font-weight: 500;
        }

        div[data-testid="stChatInput"] {
            max-width: 1180px;
            margin: 0 auto;
        }

        div[data-testid="stExpander"] {
            border-radius: var(--radius);
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }

        @media (prefers-color-scheme: dark) {
            .hero {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.12);
                box-shadow: 0 1px 10px rgba(0,0,0,0.35);
            }

            .source-card {
                background: rgba(255,255,255,0.03);
                border-color: rgba(255,255,255,0.12);
                border-left-color: #63C08A;
            }

            .source-meta a {
                color: #7BE0A3 !important;
            }

            .hero-status {
                color: #9BE7B5;
                background-color: rgba(99, 192, 138, 0.18);
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


load_dotenv(BASE_DIR / ".env")


def get_google_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            key = None
    return key


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_gemini_model():
    api_key = get_google_api_key()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def clean_metadata_value(value):
    if pd.isna(value):
        return ""
    return str(value)


@st.cache_resource
def load_chroma_collection():
    chroma_path = str(INDEX_DIR / "chroma")

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_CHROMA
    )

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False)
    )

    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

    if not CHUNKS_CSV.exists():
        raise FileNotFoundError("data/index/chunks.csv was not found.")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    df = pd.read_csv(CHUNKS_CSV)

    required_cols = [
        "chunk_id", "chunk_text", "title", "document",
        "page", "item_url", "pdf_url", "pdf_name"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in chunks.csv: {missing}")

    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)
    df = df[df["chunk_text"].str.strip() != ""].copy()

    ids, documents, metadatas = [], [], []

    for _, row in df.iterrows():
        ids.append(str(row["chunk_id"]))
        documents.append(str(row["chunk_text"]))
        metadatas.append({
            "title": clean_metadata_value(row.get("title", "")),
            "document": clean_metadata_value(row.get("document", "")),
            "page": int(row.get("page", 0)),
            "item_url": clean_metadata_value(row.get("item_url", "")),
            "pdf_url": clean_metadata_value(row.get("pdf_url", "")),
            "pdf_name": clean_metadata_value(row.get("pdf_name", "")),
        })

    batch_size = 100

    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


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
9. Do not generate a "Sources used", "Fuentes utilizadas", bibliography, or references section in the answer.
10. Source cards will be displayed separately in the interface.
11. Keep the answer natural, concise, academic, and useful for research or program review.
12. Do not force bullet points. Use paragraphs or bullets only when they help answer the question.

User question:
{question}

Retrieved context:
{context}

Answer:
"""


def answer_question(question: str, top_k: int):
    llm = load_gemini_model()

    if llm is None:
        raise ValueError("GOOGLE_API_KEY is not configured in Streamlit secrets.")

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
    pdf_link = f' <span class="meta-sep">|</span> <a href="{pdf_url}" target="_blank">PDF</a>' if pdf_url else ""

    st.markdown(
        f"""
        <div class="source-card">
            <svg class="source-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                 <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                 <polyline points="14 2 14 8 20 8"/>
                 <line x1="16" y1="13" x2="8" y2="13"/>
                 <line x1="16" y1="17" x2="8" y2="17"/>
            </svg>
            <div class="source-card-body">
                <div class="source-title">[{source['id']}] {title}</div>
                <div class="source-meta">
                    Pages: {pages}. {cgspace_link}.{pdf_link}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-top">
            <span class="hero-badge">CGIAR · MFL</span>
            <span class="hero-status">Active Agent</span>
        </div>
        <h1 class="hero-title">MFL Research Assistant</h1>
        <p class="hero-desc">
        Explore the 328 PDF files in the official CGSpace collection of the CGIAR Scientific Program on
        Multifunctional Landscapes. Ask a question or enter keywords; the assistant will search the
        documents and generate a response.
        </p>
        <div class="hero-meta">
            Source corpus: Official CGSpace collection for the CGIAR Science Program on Multifunctional Landscapes.
            &nbsp;·&nbsp; Gemini Flash Lite
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("CGSpace document sources"):
                for source in msg["sources"]:
                    render_source_card(source)


top_k_user = TOP_K_DEFAULT

with st.expander("Settings", expanded=False):
    top_k_user = st.slider(
        "Number of document segments retrieved per query",
        min_value=3,
        max_value=12,
        value=TOP_K_DEFAULT
    )


user_question = st.chat_input("Ask a question about the MFL document collection...")


if user_question:
    st.session_state["messages"].append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the CGSpace document collection and generating an answer..."):
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
