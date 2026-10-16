# -*- coding: utf-8 -*-
"""Entrada de prueba para desplegar MFL sin cambiar el diseno del agente.

Uso local:
    python -m streamlit run app_cloud_mfl_preview.py

Uso futuro en Streamlit Cloud (tras validar y revisar requirements.txt):
    - El contenido de este archivo puede convertirse en app.py.
    - app_mfl_integrado_v6.py y mfl_cloud_data.py deben estar en el mismo directorio.
    - HF_TOKEN, token de LECTURA, va en Streamlit Cloud > App settings > Secrets.

El script no descarga datos si ya existen localmente los archivos minimos.
Cuando faltan, mfl_cloud_data.ensure_mfl_data() autentica la descarga y valida
SHA256, cantidad de vectores y registros contra el manifiesto privado.
"""
from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st

HERE = Path(__file__).resolve().parent
INTEGRATED_APP = HERE / "app_mfl_integrado_v6.py"
DATA_REQUIRED = (
    HERE / "data_rebuild" / "catalog" / "cgspace_catalog_enriched.jsonl",
    HERE / "data_rebuild" / "private_results_framework" / "results_framework_records.jsonl",
    HERE / "data_rebuild" / "vectorstore_public_clean" / "chroma.sqlite3",
    HERE / "data_rebuild" / "vectorstore_framework_private" / "chroma.sqlite3",
    HERE / "data_rebuild" / "vectorstore_tracker" / "chroma.sqlite3",
)


@st.cache_resource(show_spinner=False)
def prepare_mfl_data() -> bool:
    """Preparar datos una vez por proceso, antes de cargar el agente.

    En el PC local, los datos completos ya existen y no necesita HF_TOKEN.
    En una instalacion nueva, el modulo descarga y verifica los 20 archivos.
    """
    if all(path.is_file() for path in DATA_REQUIRED):
        return True

    from mfl_cloud_data import ensure_mfl_data

    ensure_mfl_data()
    if not all(path.is_file() for path in DATA_REQUIRED):
        raise RuntimeError("Faltan archivos obligatorios tras la descarga verificada.")
    return True


if not INTEGRATED_APP.is_file():
    st.error("Missing application file: app_mfl_integrado_v6.py")
    st.stop()

try:
    prepare_mfl_data()
except Exception as exc:
    # Nunca mostramos tokens en pantalla ni imprimimos valores de st.secrets.
    st.error("Could not initialize the MFL data. Check Streamlit secrets and logs.")
    st.caption(f"Initialization error: {type(exc).__name__}: {exc}")
    st.stop()

# Ejecutar el agente tal como esta. Su propio st.set_page_config, CSS,
# busquedas, estadisticas y los modos /framework y /tracker se conservan.
runpy.run_path(str(INTEGRATED_APP), run_name="__main__")
