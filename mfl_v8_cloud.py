
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
    ROOT / "data_rebuild/compare/mfl_country_filter.py",
    ROOT / "data_rebuild/compare/mfl_country_sources.py",
    ROOT / "data_rebuild/compare/mfl_country_prompt.py",
    ROOT / "data_rebuild/compare/mfl_center_filter.py",
    ROOT / "data_rebuild/compare/mfl_center_sources.py",
    ROOT / "data_rebuild/compare/mfl_center_prompt.py"
]

if all(path.is_file() for path in REQUIRED):
    try:
        inspect_chroma_and_catalog(ROOT)
    except (OSError, RuntimeError, ValueError):
        ensure_mfl_data()
else:
    ensure_mfl_data()

runpy.run_path(
    str(ROOT / "app_mfl_integrado_v8_compare_estructurado.py"),
    run_name="__main__"
)
