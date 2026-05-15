# -*- coding: utf-8 -*-

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[1]

INDEX_DIR = BASE_DIR / "data" / "index"
ACTIVE_CHROMA_PATH_FILE = INDEX_DIR / "active_chroma_path.txt"

COLLECTION_NAME = "cgspace_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chroma_path():
    if not ACTIVE_CHROMA_PATH_FILE.exists():
        raise FileNotFoundError(
            f"No existe {ACTIVE_CHROMA_PATH_FILE}. "
            "Primero corre scripts/03_build_vectorstoreV2.py"
        )

    chroma_path = ACTIVE_CHROMA_PATH_FILE.read_text(encoding="utf-8").strip()

    if not chroma_path:
        raise ValueError("active_chroma_path.txt está vacío.")

    return chroma_path


def load_collection():
    chroma_path = load_chroma_path()

    print("Usando ChromaDB:", chroma_path)
    print("Cargando modelo de embeddings:", EMBEDDING_MODEL)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=chroma_path)

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    print("Colección:", COLLECTION_NAME)
    print("Chunks disponibles:", collection.count())

    return collection


def search(collection, query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    rows = []

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        rows.append({
            "rank": i,
            "distance": dist,
            "title": meta.get("title", ""),
            "document": meta.get("document", ""),
            "page": meta.get("page", ""),
            "item_url": meta.get("item_url", ""),
            "pdf_url": meta.get("pdf_url", ""),
            "text": doc
        })

    return rows


def print_results(query, rows):
    print("\n" + "=" * 90)
    print("QUERY:")
    print(query)
    print("=" * 90)

    if not rows:
        print("No se encontraron resultados.")
        return

    for row in rows:
        print(f"\n[{row['rank']}] distance={row['distance']:.4f}")
        print(f"Title: {row['title']}")
        print(f"Document: {row['document']}")
        print(f"Page: {row['page']}")

        if row["item_url"]:
            print(f"CGSpace: {row['item_url']}")

        if row["pdf_url"]:
            print(f"PDF: {row['pdf_url']}")

        print("\nText preview:")
        print(row["text"][:900])
        print("-" * 90)


def main():
    collection = load_collection()

    print("\nEscribe una pregunta para probar el retrieval.")
    print("Ejemplo: What are the main outcomes of the program?")
    print("Escribe 'exit' para salir.\n")

    while True:
        query = input("Pregunta: ").strip()

        if query.lower() in ["exit", "quit", "salir"]:
            break

        if not query:
            continue

        n_raw = input("Número de resultados [default=5]: ").strip()
        n_results = int(n_raw) if n_raw else 5

        rows = search(collection, query, n_results=n_results)
        print_results(query, rows)


if __name__ == "__main__":
    main()