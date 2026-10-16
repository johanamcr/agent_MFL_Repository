# -*- coding: utf-8 -*-
"""MFL: descarga VERIFICADA del snapshot privado de Hugging Face.

No publica datos ni cambia GitHub o la aplicacion existente.

Prueba local, aislada de los indices en uso:
    python mfl_cloud_data.py --test-download

Uso posterior en el entrypoint Streamlit (cuando se prepare el despliegue):
    from mfl_cloud_data import ensure_mfl_data
    ensure_mfl_data()  # usa st.secrets['HF_TOKEN'] o variable de entorno HF_TOKEN

No almacena tokens en archivos del proyecto. Mantener huggingface_hub>=1.3,<2.
"""
from __future__ import annotations

import argparse
from collections import Counter
from getpass import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Optional

REPO_ID = "jcastillor/mfl-agent-data"
REPO_TYPE = "dataset"
MANIFEST_NAME = "mfl_data_manifest.json"
SNAPSHOT_ID = "mfl-2026-10-16-453"
CATALOG_PATH = "data_rebuild/catalog/cgspace_catalog_enriched.jsonl"
FRAMEWORK_RECORDS_PATH = "data_rebuild/private_results_framework/results_framework_records.jsonl"
EXPECTED_FILE_COUNT = 20
INDEXES = {
    "data_rebuild/vectorstore_public_clean/": ("mfl_cgspace_public_clean_v2", 15952),
    "data_rebuild/vectorstore_framework_private/": ("mfl_results_framework_v1", 1174),
    "data_rebuild/vectorstore_tracker/": ("mfl_tracker_deliverables_v1", 1848),
}
ALLOWED_SINGLE_FILES = {CATALOG_PATH, FRAMEWORK_RECORDS_PATH}
APP_ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_from_environment() -> Optional[str]:
    """Read secret safely; never log or print token value."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        import streamlit as st
        token = str(st.secrets.get("HF_TOKEN", "") or "").strip()
        if token:
            return token
    except Exception:
        pass
    return None


def validate_manifest(manifest: dict) -> list[dict]:
    if manifest.get("bundle_id") != SNAPSHOT_ID or manifest.get("catalog_count") != 453:
        raise ValueError("El manifiesto remoto no corresponde al snapshot de 453 outputs.")
    items = manifest.get("files", [])
    if not isinstance(items, list) or len(items) != EXPECTED_FILE_COUNT:
        raise ValueError(f"Se esperaban {EXPECTED_FILE_COUNT} archivos en el manifiesto.")
    seen = set()
    by_prefix = Counter()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Formato de registro de archivo no válido.")
        raw = item.get("path", "")
        if not isinstance(raw, str):
            raise ValueError("Ruta de archivo no válida.")
        rel = PurePosixPath(raw)
        if raw.startswith("/") or "\\" in raw or ".." in rel.parts or "." in rel.parts:
            raise ValueError("Ruta no permitida en manifiesto.")
        if len(rel.parts) < 3 or rel.as_posix() != raw or raw in seen:
            raise ValueError("Ruta malformada o repetida en manifiesto.")
        allowed = raw in ALLOWED_SINGLE_FILES
        for prefix in INDEXES:
            if raw.startswith(prefix):
                allowed = True
                by_prefix[prefix] += 1
        if not allowed:
            raise ValueError(f"Ruta no autorizada: {raw}")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            raise ValueError(f"Tamaño inválido: {raw}")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(item.get("sha256", ""))):
            raise ValueError(f"SHA256 inválido: {raw}")
        seen.add(raw)
    if not ALLOWED_SINGLE_FILES.issubset(seen):
        raise ValueError("Falta archivo bibliográfico o de registros de Excel.")
    for prefix, (_, _) in INDEXES.items():
        if by_prefix[prefix] != 6 or f"{prefix}chroma.sqlite3" not in seen:
            raise ValueError(f"El índice no tiene los seis archivos previstos: {prefix}")
    return items


def verify_local_files(root: Path, items: list[dict]) -> bool:
    """Comparar tamaño y SHA256; nunca aceptar un índice parcial."""
    for item in items:
        path = root.joinpath(*PurePosixPath(item["path"]).parts)
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            return False
        if sha256_file(path).lower() != item["sha256"].lower():
            return False
    return True


def inspect_chroma_and_catalog(root: Path) -> None:
    """Comprobar apertura de SQLite/Chroma y cardinalidades esperadas."""
    import chromadb
    from chromadb.config import Settings

    for prefix, (collection, count) in INDEXES.items():
        path = root.joinpath(*PurePosixPath(prefix.rstrip("/")).parts)
        client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        got = client.get_collection(name=collection).count()
        if got != count:
            raise RuntimeError(f"{prefix}: {got} vectores; esperados {count}.")
        print(f"  {path.name}: {got} vectores — OK")

    catalog = root.joinpath(*PurePosixPath(CATALOG_PATH).parts)
    with catalog.open(encoding="utf-8-sig") as f:
        records = [json.loads(line) for line in f if line.strip()]
    ids = [r.get("item_uuid") for r in records]
    if len(ids) != 453 or len(set(ids)) != 453:
        raise RuntimeError("Catálogo inválido: no contiene 453 UUID distintos.")
    print("  CGSpace: 453 outputs — OK")
    framework = root.joinpath(*PurePosixPath(FRAMEWORK_RECORDS_PATH).parts)
    with framework.open(encoding="utf-8-sig") as f:
        framework_count = sum(bool(line.strip()) for line in f)
    if framework_count != 796:
        raise RuntimeError(f"Results Framework: {framework_count} registros; esperados 796.")
    print("  Results Framework: 796 registros — OK")


def fetch_manifest(token: str, root: Path) -> tuple[dict, list[dict], str]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    if info.private is not True:
        raise RuntimeError("ALTO: el dataset no está privado.")
    revision = info.sha  # fija todos los archivos a un mismo commit remoto
    path = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, revision=revision,
        filename=MANIFEST_NAME, token=token, local_dir=str(root)
    ))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = validate_manifest(manifest)
    return manifest, entries, revision


def download_and_verify(root: Path, token: str) -> None:
    """Descargar únicamente los 20 archivos aprobados y comprobarlos.

    Apto para uso con st.cache_resource en Streamlit, antes de abrir Chroma.
    """
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    _, entries, revision = fetch_manifest(token, root)
    print(f"Repositorio privado confirmado: {REPO_ID}")
    print(f"Snapshot fijo: {revision[:12]}")
    if verify_local_files(root, entries):
        print("Todos los archivos ya están íntegros; no se necesitan descargas.")
    else:
        print(f"Descargando {len(entries)} archivos verificados a {root} ...")
        snapshot_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE, revision=revision,
            token=token, local_dir=str(root),
            allow_patterns=[entry["path"] for entry in entries],
            max_workers=4
        )
        if not verify_local_files(root, entries):
            raise RuntimeError(
                "Integridad SHA256 fallida o archivos incompletos. "
                "No iniciar la aplicación con estos índices."
            )
    print("Integridad SHA256 de los 20 archivos: OK")
    inspect_chroma_and_catalog(root)


def ensure_mfl_data() -> None:
    """Entrada futura para Streamlit. Exige HF_TOKEN en secretos o ambiente.

    Invocar una sola vez al iniciar la app, idealmente via @st.cache_resource,
    ANTES de la primera consulta al catálogo o a Chroma.
    """
    token = token_from_environment()
    if not token:
        raise RuntimeError(
            "Falta HF_TOKEN. Configura el token de solo lectura en los secretos "
            "de Streamlit Cloud; no lo incluyas en app.py."
        )
    download_and_verify(APP_ROOT, token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-download", action="store_true",
        help="Prueba completa de descarga en carpeta aislada, sin alterar índices locales"
    )
    args = parser.parse_args()
    if not args.test_download:
        print("Uso: python mfl_cloud_data.py --test-download")
        print("No se descargó ningún archivo.")
        return
    print("\nPASO 90 — PRUEBA DE DESCARGA AUTENTICADA (SIN CAMBIOS EN APP)")
    print("=" * 71)
    root = APP_ROOT / "deploy_artifacts_private" / "hf_cloud_download_test"
    print("Destino de prueba:", root)
    print("La prueba descargará aproximadamente 239 MiB; no tocará los índices en uso.")
    token = getpass("Token mfl-streamlit-read (no visible): ").strip()
    if not token:
        raise RuntimeError("Token de lectura vacío.")
    download_and_verify(root, token)
    print("\nRESULTADO FINAL")
    print("20 archivos autenticados y SHA256 verificados")
    print("CGSpace: 15952 vectores y 453 outputs")
    print("Results Framework: 1174 vectores y 796 registros")
    print("MFL Tracker: 1848 vectores")
    print("Descarga de prueba aislada — app.py y GitHub SIN CAMBIOS")
    print("DESCARGA PRIVADA LISTA PARA STREAMLIT CLOUD")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
