"""
simulate_once: canonical run entrypoint.

Extracted from notebook Cell 12, Phase 3.

Contract:
- Only entrypoint used by campaigns.
- Deterministic for same (cfg, seed, policy) given same codebase.
- Writes outputs to runs/<cfg_hash>/<seed>/ via dataset_io writers when write_files=True.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .airtime import build_airtime_buckets
from .config import SimCfg, validate_cfg
from .dataset_io import (
    CELL11_SCHEMA_ID,
    RUN_ROW_COLUMNS,
    _inj_hash_cell11,
    _mobility_digest_hash_cell11,
    _require_columns_exact,
    _trace_hash_cell11,
    make_run_row,
    row_content_sha256,
    write_run_artifacts,
)
from .hashing import cfg_hash
from .mac_engine import SimEngine
from .mobility import build_plan
from .policies import policy_factory
from .seeds import SeedManager
from .traffic import build_injections


def simulate_once(
    cfg: SimCfg,
    seed: int,
    *,
    root: str = ".",
    policy_name: Optional[str] = None,
    write_files: bool = True,
) -> Dict[str, Any]:
    """
    Canonical run entrypoint.
    """
    validate_cfg(cfg)
    sd = int(seed)
    pol_name = str(policy_name if policy_name is not None else getattr(cfg, "policy", "flood")).strip().lower()

    h = str(cfg_hash(cfg))
    sm = SeedManager(cfg_hash=h, seed=sd)

    plan = build_plan(cfg, sm)  # Phase 12B-0: dispatches rwp->build_rwp_plan, static->build_static_plan
    injs = build_injections(cfg, sm)

    inj_h = str(_inj_hash_cell11(injs))
    mob_h = str(
        _mobility_digest_hash_cell11(
            plan,
            n_nodes=int(cfg.n_nodes),
            times=(0.0, float(cfg.sim_duration_s) / 2.0, float(cfg.sim_duration_s)),
        )
    )

    pol = policy_factory(pol_name)
    buckets = build_airtime_buckets(cfg)

    eng = SimEngine(cfg, seed=sd, plan=plan, policy=pol, cfg_hash_for_rng=h, airtime_buckets=buckets)
    out = eng.run(injs)

    tr_h = str(_trace_hash_cell11(out["trace_head"]))

    row = make_run_row(
        cfg=cfg,
        cfg_hash_str=h,
        seed=sd,
        policy=pol_name,
        inj_hash=inj_h,
        mobility_digest_hash=mob_h,
        trace_hash=tr_h,
        summary=out["summary"],
    )

    rep: Dict[str, Any] = {}
    if write_files:
        rep = write_run_artifacts(
            root=Path(root),
            row=row,
            extra={"note": "simulate_once"},
        )
    else:
        rep = {
            "ok": True,
            "outdir": str(Path(root) / "runs" / h / str(sd)),
            "row_content_sha256": row_content_sha256(row),
            "parquet_written": False,
            "parquet_engine": None,
            "file_sha256": {},
            "files": {},
        }

    return {
        "cfg_hash": h,
        "seed": sd,
        "policy": pol_name,
        "outdir": rep.get("outdir"),
        "row_content_sha256": rep.get("row_content_sha256"),
        "trace_hash": tr_h,
        "summary": out["summary"],
        "trace_head": out["trace_head"],
        "parquet_written": bool(rep.get("parquet_written", False)),
        "parquet_engine": rep.get("parquet_engine"),
        "file_sha256": rep.get("file_sha256", {}),
    }
