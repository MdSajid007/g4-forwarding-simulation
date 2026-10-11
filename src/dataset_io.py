"""
Dataset schema + writers + per-run manifest (Phase E).

Extracted from notebook Cell 11, Phase 3.

Contract:
- Writes outputs to: runs/<cfg_hash>/<seed>/
- Exports a stable per-run table (one row per run) + manifest.json
- Determinism: row-content hash stable for same (cfg, seed, policy) in same codebase.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import SimCfg
from .hashing import canonical_json, MOBILITY_DIGEST_NDIGITS


# ----------------------------
# Schema (stable)
# ----------------------------

CELL11_SCHEMA_ID = "run_row_v1"

RUN_ROW_COLUMNS: Tuple[str, ...] = (
    # identity
    "schema",
    "cfg_hash",
    "seed",
    "policy",
    # scenario/context
    "sim_duration_s",
    "n_nodes",
    # determinism inputs/derived
    "inj_hash",
    "mobility_digest_hash",
    "trace_hash",
    # core outcomes
    "app_injected",
    "delivered",
    "dropped",
    "pdr",
    # radio/mac counters
    "tx_attempts",
    "tx_success",
    "airtime_s",
    "airtime_data_s",
    "airtime_ctrl_s",
    "energy_j",
    "budget_deferrals",
    "cca_busy_events",
    # scanning/overlap (legacy + v4)
    "not_scanning_events",
    "not_scanning_dst_events",
    "not_scanning_rx_opportunities",
    "not_scanning_waveforms",
    "overlap_events",
    "overlap_waveforms",
    "overlap_rx_evals",
    "overlap_interferer_pairs",
    # rx accounting
    "rx_opportunities",
    "rx_scans",
    "rx_total",
    "rx_ok",
    "rx_fail",
)


def _require_columns_exact(df: pd.DataFrame, columns: Sequence[str]) -> None:
    got = list(df.columns)
    exp = list(columns)
    if got != exp:
        raise AssertionError(f"Schema drift: expected columns={exp} but got={got}")


# ----------------------------
# Hash helpers (Gate R)
# ----------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_any(obj: Any) -> str:
    try:
        return str(canonical_json(obj))
    except Exception:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def row_content_sha256(row: Dict[str, Any]) -> str:
    s = canonical_json_any(row)
    return sha256_bytes(s.encode("utf-8"))


# ----------------------------
# Deterministic derived-hash helpers (self-contained)
# ----------------------------

def _trace_hash_cell11(trace: Any) -> str:
    rows: List[Tuple[float, int, str, int]] = []
    for (t, prio, etype, src) in list(trace):
        rows.append((round(float(t), 12), int(prio), str(etype), int(src)))
    return sha256_bytes(repr(rows).encode("utf-8"))


def _inj_hash_cell11(injs: Any) -> str:
    rows: List[Dict[str, Any]] = []
    for inj in list(injs):
        rows.append(
            {
                "t": round(float(getattr(inj, "t")), 12),
                "flow_id": int(getattr(inj, "flow_id")),
                "src": int(getattr(inj, "src")),
                "dst": int(getattr(inj, "dst")),
                "payload_bytes": int(getattr(inj, "payload_bytes")),
                "ttl_hops": int(getattr(inj, "ttl_hops")),
            }
        )
    s = canonical_json_any(rows)
    return sha256_bytes(s.encode("utf-8"))


def _mobility_digest_hash_cell11(plan: Any, *, n_nodes: int, times: Sequence[float]) -> str:
    nd = MOBILITY_DIGEST_NDIGITS
    dig: Dict[int, List[Tuple[float, float, float]]] = {}
    for i in range(int(n_nodes)):
        pts: List[Tuple[float, float, float]] = []
        for t in times:
            x, y = plan.pos_at(int(i), float(t))
            pts.append((round(float(t), 12), round(float(x), nd), round(float(y), nd)))
        dig[int(i)] = pts

    s = canonical_json_any(dig)
    return sha256_bytes(s.encode("utf-8"))


# ----------------------------
# Paths
# ----------------------------

def run_dir(root: Path, cfg_hash_str: str, seed: int) -> Path:
    return Path(root) / "runs" / str(cfg_hash_str) / str(int(seed))


def ensure_run_dir(root: Path, cfg_hash_str: str, seed: int) -> Path:
    d = run_dir(root, cfg_hash_str, seed)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------
# Row builder
# ----------------------------

def make_run_row(
    *,
    cfg: SimCfg,
    cfg_hash_str: str,
    seed: int,
    policy: str,
    inj_hash: str,
    mobility_digest_hash: str,
    trace_hash: str,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    def gi(k: str, default: int = 0) -> int:
        v = summary.get(k, default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def gf(k: str, default: float = 0.0) -> float:
        v = summary.get(k, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    row: Dict[str, Any] = {c: None for c in RUN_ROW_COLUMNS}
    row["schema"] = CELL11_SCHEMA_ID
    row["cfg_hash"] = str(cfg_hash_str)
    row["seed"] = int(seed)
    row["policy"] = str(policy)

    row["sim_duration_s"] = float(cfg.sim_duration_s)
    row["n_nodes"] = int(cfg.n_nodes)

    row["inj_hash"] = str(inj_hash)
    row["mobility_digest_hash"] = str(mobility_digest_hash)
    row["trace_hash"] = str(trace_hash)

    row["app_injected"] = gi("app_injected", 0)
    row["delivered"] = gi("delivered", 0)
    row["dropped"] = gi("dropped", 0)
    row["pdr"] = gf("pdr", 0.0)

    row["tx_attempts"] = gi("tx_attempts", 0)
    row["tx_success"] = gi("tx_success", 0)
    row["airtime_s"] = gf("airtime_s", 0.0)
    row["airtime_data_s"] = gf("airtime_data_s", 0.0)
    row["airtime_ctrl_s"] = gf("airtime_ctrl_s", 0.0)
    row["energy_j"] = gf("energy_j", 0.0)
    row["budget_deferrals"] = gi("budget_deferrals", 0)
    row["cca_busy_events"] = gi("cca_busy_events", 0)

    row["not_scanning_events"] = gi("not_scanning_events", 0)
    row["not_scanning_dst_events"] = gi("not_scanning_dst_events", 0)
    row["not_scanning_rx_opportunities"] = gi("not_scanning_rx_opportunities", 0)
    row["not_scanning_waveforms"] = gi("not_scanning_waveforms", 0)

    row["overlap_events"] = gi("overlap_events", 0)
    row["overlap_waveforms"] = gi("overlap_waveforms", 0)
    row["overlap_rx_evals"] = gi("overlap_rx_evals", 0)
    row["overlap_interferer_pairs"] = gi("overlap_interferer_pairs", 0)

    row["rx_opportunities"] = gi("rx_opportunities", 0)
    row["rx_scans"] = gi("rx_scans", 0)
    row["rx_total"] = gi("rx_total", 0)
    row["rx_ok"] = gi("rx_ok", 0)
    row["rx_fail"] = gi("rx_fail", 0)

    for k in RUN_ROW_COLUMNS:
        if k not in row:
            raise AssertionError(f"Missing column in row: {k}")
    return row


# ----------------------------
# Parquet optional writer
# ----------------------------

def _detect_parquet_engine() -> Optional[str]:
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        pass
    return None


def write_run_artifacts(
    *,
    root: Path,
    row: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg_hash_str = str(row["cfg_hash"])
    seed = int(row["seed"])
    outdir = ensure_run_dir(root, cfg_hash_str, seed)

    # --- CSV (canonical) ---
    csv_path = outdir / "run_row.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(RUN_ROW_COLUMNS))
        w.writeheader()
        w.writerow({k: row.get(k, None) for k in RUN_ROW_COLUMNS})

    # --- Parquet (optional) ---
    parquet_engine = _detect_parquet_engine()
    parquet_path = outdir / "run_row.parquet"
    parquet_written = False
    if parquet_engine is not None:
        df = pd.DataFrame([{k: row.get(k, None) for k in RUN_ROW_COLUMNS}], columns=list(RUN_ROW_COLUMNS))
        _require_columns_exact(df, RUN_ROW_COLUMNS)
        if parquet_engine == "pyarrow":
            df.to_parquet(parquet_path, engine="pyarrow", index=False, compression=None)
        else:
            df.to_parquet(parquet_path, engine="fastparquet", index=False, compression=None)
        parquet_written = True

    # --- Manifest ---
    row_sha = row_content_sha256(row)
    files: Dict[str, str] = {"run_row_csv": "run_row.csv", "manifest": "manifest.json"}
    file_sha256_map: Dict[str, str] = {"run_row_csv": sha256_file(csv_path)}
    if parquet_written:
        files["run_row_parquet"] = "run_row.parquet"
        file_sha256_map["run_row_parquet"] = sha256_file(parquet_path)

    manifest = {
        "schema": "run_manifest_v1",
        "run_row_schema": CELL11_SCHEMA_ID,
        "cfg_hash": cfg_hash_str,
        "seed": int(seed),
        "policy": str(row["policy"]),
        "row_content_sha256": row_sha,
        "parquet_written": bool(parquet_written),
        "parquet_engine": parquet_engine,
        "files": files,
        "file_sha256": file_sha256_map,
        "extra": extra or {},
    }
    manifest_json = canonical_json_any(manifest)
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(manifest_json, encoding="utf-8")
    file_sha256_map["manifest"] = sha256_file(manifest_path)

    return {
        "ok": True,
        "outdir": str(outdir),
        "row_content_sha256": row_sha,
        "parquet_written": bool(parquet_written),
        "parquet_engine": parquet_engine,
        "file_sha256": file_sha256_map,
        "files": files,
    }
