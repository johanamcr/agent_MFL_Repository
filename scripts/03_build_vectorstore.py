# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[1]

INDEX_DIR = BASE_DIR / "data" / "index"
CHUNKS_CSV = INDEX_DIR / "chunks.csv"

COLLECTION_NAME = "cgspace_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean_metadata_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def main():
    if not CHUNKS_CSV.exists():
        print("No existe chunks.csv:", CHUNKS_CSV)
        return

    df = pd.read_csv(CHUNKS_CSV)

    required_cols = [
        "chunk_id",
        "document",
        "page",
        "chunk_order",
        "word_count",
        "chunk_text",
        "title",
        "item_url",
        "pdf_url",
        "pdf_name",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("Faltan columnas en chunks.csv:", missing)
        return

    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)
    df = df[df["chunk_text"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"Chunks cargados: {len(df)}")

    # Crear una carpeta nueva para evitar bloqueos de OneDrive/Windows
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    CHROMA_DIR = INDEX_DIR / f"chroma_{timestamp}"
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("ChromaDB path:", CHROMA_DIR)
    print("Cargando modelo de embeddings:", EMBEDDING_MODEL)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "description": "CGSpace document chunks with source metadata"
        }
    )

    ids = []
    documents = []
    metadatas = []

    for _, row in df.iterrows():
        ids.append(str(row["chunk_id"]))
        documents.append(str(row["chunk_text"]))

        metadatas.append({
            "document": clean_metadata_value(row["document"]),
            "page": int(row["page"]),
            "chunk_order": int(row["chunk_order"]),
            "word_count": int(row["word_count"]),
            "title": clean_metadata_value(row["title"]),
            "item_url": clean_metadata_value(row["item_url"]),
            "pdf_url": clean_metadata_value(row["pdf_url"]),
            "pdf_name": clean_metadata_value(row["pdf_name"]),
        })

    batch_size = 100

    print("Agregando chunks a ChromaDB...")

    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"  Agregados {end} / {len(ids)}")

    # Guardar ruta activa para que app.py sepa cuál Chroma usar
    active_path_file = INDEX_DIR / "active_chroma_path.txt"
    active_path_file.write_text(str(CHROMA_DIR), encoding="utf-8")

    print("\n==============================")
    print("Vectorstore construido correctamente")
    print("ChromaDB:", CHROMA_DIR)
    print("Colección:", COLLECTION_NAME)
    print("Chunks indexados:", collection.count())
    print("Ruta activa guardada en:", active_path_file)
    print("==============================")


if __name__ == "__main__":
    main()