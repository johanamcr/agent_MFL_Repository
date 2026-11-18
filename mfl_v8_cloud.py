
"""Start the validated MFL v8 app with private cloud data."""

from pathlib import Path
import runpy

from mfl_cloud_data import (
    ensure_mfl_data,
    inspect_chroma_and_catalog
)

ROOT = Path(__file__).resolve().parent

REQUIRED = [
    ROOT / "data_rebuild/catalog/cgspace_catalog_enriched.jsonl",
    ROOT / "data_rebuild/compare/all_aow_commitments.jsonl",
    ROOT / "data_rebuild/compare/all_aow_crosswalk_reviewed.jsonl",
    ROOT / "data_rebuild/compare/all_aow_tracker_inventory.jsonl",
    ROOT / "data_rebuild/compare/all_aow_activity_status.jsonl",
    ROOT / "data_rebuild/compare/aow4_commitments.jsonl",
    ROOT / "data_rebuild/compare/aow4_crosswalk_reviewed.jsonl",
    ROOT / "data_rebuild/compare/aow4_tracker_inventory.jsonl",
    ROOT / "data_rebuild/compare/mfl_country_filter.py",
    ROOT / "data_rebuild/compare/mfl_country_sources.py",
    ROOT / "data_rebuild/compare/mfl_country_prompt.py",
    ROOT / "data_rebuild/compare/mfl_center_filter.py",
    ROOT / "data_rebuild/compare/mfl_center_sources.py",
    ROOT / "data_rebuild/compare/mfl_center_prompt.py"
]

# MFL_BOOTSTRAP_VERIFIED_FIRST_V1
# Verify/download the pinned snapshot BEFORE opening ChromaDB.
# Cache initialization across Streamlit reruns for this snapshot.
import streamlit as st
from mfl_cloud_data import SNAPSHOT_ID

@st.cache_resource(show_spinner=False)
def _initialize_mfl_snapshot(snapshot_id):
    ensure_mfl_data()
    return snapshot_id

_initialize_mfl_snapshot(SNAPSHOT_ID)

runpy.run_path(
    str(ROOT / "app_mfl_integrado_v8_compare_estructurado.py"),
    run_name="__main__"
)
