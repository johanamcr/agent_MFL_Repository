# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"
META_DIR = BASE_DIR / "data" / "metadata"
LOGS_DIR = BASE_DIR / "data" / "logs"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_CSV = INDEX_DIR / "chunks.csv"
ERROR_LOG = LOGS_DIR / "chunk_errors.log"

METADATA_CSV = META_DIR / "collection_pdfs_metadata.csv"

PAGE_PATTERN = re.compile(r"--- PAGE (\d+) ---")

CHUNK_SIZE_WORDS = 300
OVERLAP_WORDS = 50


# =========================================================
# Utilidades
# =========================================================
def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(name):
    return Path(name).stem.lower().strip()


# =========================================================
# Lectura de páginas
# =========================================================
def split_pages(txt_path):
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    parts = PAGE_PATTERN.split(text)

    pages = []

    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = clean_text(parts[i + 1])

        if page_text:
            pages.append((page_num, page_text))

    return pages


# =========================================================
# Chunking
# =========================================================
def chunk_words(words, chunk_size=300, overlap=50):
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = words[start:end]

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


def build_chunks_for_page(
    document_name,
    page_num,
    page_text,
    metadata_row
):
    words = page_text.split()

    word_chunks = chunk_words(
        words,
        CHUNK_SIZE_WORDS,
        OVERLAP_WORDS
    )

    rows = []

    for idx, word_list in enumerate(word_chunks, start=1):

        chunk_text = " ".join(word_list).strip()

        if len(word_list) < 40:
            continue

        row = {
            "chunk_id": f"{Path(document_name).stem}_p{page_num}_c{idx}",
            "document": document_name,
            "page": page_num,
            "chunk_order": idx,
            "word_count": len(word_list),
            "chunk_text": chunk_text,

            # ==========================
            # Metadata de fuente
            # ==========================
            "title": metadata_row.get("title", ""),
            "item_url": metadata_row.get("item_url", ""),
            "pdf_url": metadata_row.get("pdf_url", ""),
            "pdf_name": metadata_row.get("pdf_name", "")
        }

        rows.append(row)

    return rows


# =========================================================
# Main
# =========================================================
def main():

    txt_files = sorted(PROCESSED_DIR.glob("*.txt"))

    if not txt_files:
        print("No se encontraron TXT en:", PROCESSED_DIR)
        return

    # =====================================================
    # Metadata documentos
    # =====================================================
    if not METADATA_CSV.exists():
        print("No existe metadata CSV:", METADATA_CSV)
        return

    metadata_df = pd.read_csv(METADATA_CSV)

    metadata_lookup = {}

    for _, row in metadata_df.iterrows():

        pdf_name = row.get("pdf_name", "")

        key = normalize_name(pdf_name)

        metadata_lookup[key] = row.to_dict()

    print(f"Metadata cargada: {len(metadata_lookup)} documentos")

    # =====================================================
    # Construcción chunks
    # =====================================================
    all_rows = []

    with open(ERROR_LOG, "w", encoding="utf-8") as elog:

        for txt_file in txt_files:

            print(f"\nProcesando: {txt_file.name}")

            try:

                key = normalize_name(txt_file.name)

                metadata_row = metadata_lookup.get(key)

                if metadata_row is None:
                    print("WARNING: no metadata encontrada para:", txt_file.name)

                    metadata_row = {
                        "title": "",
                        "item_url": "",
                        "pdf_url": "",
                        "pdf_name": ""
                    }

                pages = split_pages(txt_file)

                for page_num, page_text in pages:

                    rows = build_chunks_for_page(
                        document_name=txt_file.name,
                        page_num=page_num,
                        page_text=page_text,
                        metadata_row=metadata_row
                    )

                    all_rows.extend(rows)

            except Exception as e:

                elog.write(
                    f"{txt_file.name}\t{type(e).__name__}\t{str(e)}\n"
                )

                print(f"ERROR en {txt_file.name}: {e}")

    # =====================================================
    # Guardar CSV
    # =====================================================
    with open(CHUNKS_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id",
                "document",
                "page",
                "chunk_order",
                "word_count",
                "chunk_text",

                # metadata
                "title",
                "item_url",
                "pdf_url",
                "pdf_name"
            ]
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\n==============================")
    print(f"Chunks generados: {len(all_rows)}")
    print(f"Archivo: {CHUNKS_CSV}")
    print(f"Errores: {ERROR_LOG}")
    print("==============================")


if __name__ == "__main__":
    main()