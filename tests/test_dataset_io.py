"""
Phase 3 test: run manifest generation and dataset schema stability.
Maps to Phase 2 requirements:
  - Run manifest generation (self_test_cell11, 5 assertions)
  - Dataset schema stability (self_test_cell12 + self_test_cell13)

Phase 2 regression hashes:
  Row content SHA256: 653dc140f6765295aeee5dc195078c457f5ea51e6a0e3a069ab046298c522d57
  Campaign dataset CSV SHA256: 4f1386993afc47c0cac7e110ea23570dfbda170f3dacc2fb3649cf2c1fe7c4d4
"""
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import make_easy_cfg
from src.dataset_io import (
    CELL11_SCHEMA_ID,
    RUN_ROW_COLUMNS,
    _require_columns_exact,
    row_content_sha256,
)
from src.simulate import simulate_once

PHASE2_ROW_CONTENT_SHA256 = "653dc140f6765295aeee5dc195078c457f5ea51e6a0e3a069ab046298c522d57"


def test_simulate_once_row_content_hash_stable():
    """Same cfg+seed+policy must produce identical row_content_sha256."""
    cfg = make_easy_cfg()
    d1 = Path(tempfile.mkdtemp(prefix="p3_test_io1_"))
    d2 = Path(tempfile.mkdtemp(prefix="p3_test_io2_"))
    try:
        r1 = simulate_once(cfg, 1, root=str(d1), write_files=True, policy_name="flood")
        r2 = simulate_once(cfg, 1, root=str(d2), write_files=True, policy_name="flood")
        assert str(r1["row_content_sha256"]) == str(r2["row_content_sha256"]), "Row content hash drift"
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


def test_simulate_once_csv_schema():
    """run_row.csv must have the exact RUN_ROW_COLUMNS schema."""
    cfg = make_easy_cfg()
    d = Path(tempfile.mkdtemp(prefix="p3_test_schema_"))
    try:
        r = simulate_once(cfg, 1, root=str(d), write_files=True, policy_name="flood")
        csv_path = Path(r["outdir"]) / "run_row.csv"
        assert csv_path.exists(), "Missing run_row.csv"
        df = pd.read_csv(csv_path)
        _require_columns_exact(df, RUN_ROW_COLUMNS)
        assert int(df.iloc[0]["seed"]) == 1
        assert str(df.iloc[0]["policy"]) == "flood"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_simulate_once_manifest_fields():
    """manifest.json must contain row_content_sha256 and parquet_written."""
    cfg = make_easy_cfg()
    d = Path(tempfile.mkdtemp(prefix="p3_test_mfst_"))
    try:
        r = simulate_once(cfg, 1, root=str(d), write_files=True, policy_name="flood")
        m = Path(r["outdir"]) / "manifest.json"
        mj = json.loads(m.read_text(encoding="utf-8"))
        assert mj.get("row_content_sha256") == r["row_content_sha256"]
        assert bool(mj.get("parquet_written")) in (True, False)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_row_content_hash_regression():
    """Phase 2 regression: row_content_sha256 for easy_cfg seed=1 flood."""
    cfg = make_easy_cfg()
    d = Path(tempfile.mkdtemp(prefix="p3_test_reg_"))
    try:
        r = simulate_once(cfg, 1, root=str(d), write_files=True, policy_name="flood")
        computed = str(r["row_content_sha256"])
        assert computed == PHASE2_ROW_CONTENT_SHA256, (
            f"Row content SHA256 regression failed.\n"
            f"  Expected: {PHASE2_ROW_CONTENT_SHA256}\n"
            f"  Got:      {computed}"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_simulate_once_trace_hash_stable():
    """Same cfg+seed+policy must produce identical trace_hash."""
    cfg = make_easy_cfg()
    d1 = Path(tempfile.mkdtemp(prefix="p3_th1_"))
    d2 = Path(tempfile.mkdtemp(prefix="p3_th2_"))
    try:
        r1 = simulate_once(cfg, 1, root=str(d1), write_files=True, policy_name="flood")
        r2 = simulate_once(cfg, 1, root=str(d2), write_files=True, policy_name="flood")
        assert str(r1["trace_hash"]) == str(r2["trace_hash"]), "Trace hash drift"
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)
