# -*- coding: utf-8 -*-

from pathlib import Path
import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "data" / "index" / "chroma"

COLLECTION_NAME = "cgspace_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 8
MODEL_NAME = "models/gemini-flash-lite-latest"


load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("No se encontró GOOGLE_API_KEY en el archivo .env")

genai.configure(api_key=GOOGLE_API_KEY)

llm = genai.GenerativeModel(MODEL_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client_chroma = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)

collection = client_chroma.get_collection(COLLECTION_NAME)


def retrieve_context(question: str, top_k: int = TOP_K):
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
            "id": i,
            "text": doc,
            "title": meta.get("title", ""),
            "document": meta.get("document", ""),
            "page": meta.get("page", ""),
            "item_url": meta.get("item_url", ""),
            "pdf_url": meta.get("pdf_url", ""),
            "pdf_name": meta.get("pdf_name", ""),
            "similarity": round(1 - dist, 4)
        })

    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []

    for c in chunks:
        context_blocks.append(
            f"[Source {c['id']}]\n"
            f"Title: {c['title']}\n"
            f"Document: {c['document']}\n"
            f"Page: {c['page']}\n"
            f"CGSpace URL: {c['item_url']}\n"
            f"PDF URL: {c['pdf_url']}\n"
            f"Similarity: {c['similarity']}\n"
            f"Text:\n{c['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a research assistant for the CGIAR Multifunctional Landscapes Science Program.

Your job is to answer the user's question using ONLY the retrieved context below.
You help users understand what the Science Program has worked on, based on the retrieved documents from the program corpus.

Use ONLY the retrieved context. Do not use outside knowledge.

Core rules:
1. Do not invent facts, outcomes, countries, indicators, targets, dates, partners, or conclusions.
2. If the retrieved context is insufficient, say so clearly.
3. Distinguish carefully between:
   - stated program outcomes,
   - intended outcomes,
   - achieved results,
   - objectives,
   - risks,
   - assumptions,
   - recommendations,
   - supporting evidence,
   - examples from specific documents.
4. If the documents describe objectives or theory of change, do not present them as achieved results.
5. If the user asks a broad or ambiguous question, answer based on the dominant patterns in the retrieved documents and clearly say that this is not an exhaustive classification of the full corpus.
6. Prefer precise, evidence-based language over generic summaries.
7. Use a concise, leadership-oriented tone.
8. Every substantive claim must be traceable to at least one retrieved source.
9. Cite sources using this format: [Source X: Document title, page Y].
10. Do not cite sources that are not included in the retrieved context.
11. If sources disagree or cover different geographies/cases, mention that the answer reflects multiple document types or case examples.
12. Do not overstate certainty.
13. At the end, include a "Sources used" section with:
    - source number
    - document title
    - page
    - CGSpace URL
    - PDF URL when available

Response structure:
1. Direct answer
2. Key points
3. Caveat / evidence limitation
4. Sources used

User question:
{question}

Retrieved context:
{context}

Answer:
"""


def answer_question(question: str):
    chunks = retrieve_context(question, top_k=TOP_K)
    prompt = build_prompt(question, chunks)

    response = llm.generate_content(prompt)

    return response.text, chunks


def main():
    print("\nRAG chat con Gemini listo.")
    print(f"Modelo: {MODEL_NAME}")
    print("Escribe 'exit' para salir.")

    while True:
        question = input("\nPregunta: ").strip()

        if question.lower() in ["exit", "quit", "salir"]:
            break

        if not question:
            continue

        answer, chunks = answer_question(question)

        print("\n" + "=" * 90)
        print("ANSWER")
        print("=" * 90)
        print(answer)

        print("\n" + "=" * 90)
        print("RETRIEVED SOURCES")
        print("=" * 90)

        for c in chunks:
            print(f"[{c['id']}] {c['title']}")
            print(f"    Document: {c['document']}")
            print(f"    Page: {c['page']}")
            print(f"    Similarity: {c['similarity']}")
            print(f"    CGSpace: {c['item_url']}")
            print(f"    PDF: {c['pdf_url']}")
            print("-" * 90)


if __name__ == "__main__":
    main()