"""
SimConfig dataclasses, validation, scenario builders, and project-level logging/env setup.

Extracted from notebook Cell 1 (logging/env) and Cell 2 (config), Phase 3.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .hashing import canonical_json, cfg_hash  # noqa: F401  (re-exported for callers)

# ----------------------------
# Project paths (resolved at import time from cwd)
# ----------------------------

PROJECT_ROOT: Path = Path.cwd().resolve()
RUNS_DIR: Path = PROJECT_ROOT / "runs"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
DATA_DIR: Path = PROJECT_ROOT / "data"
FIGS_DIR: Path = PROJECT_ROOT / "figures"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

_LOGGERS: Dict[str, logging.Logger] = {}


def ensure_dirs() -> None:
    """Create required project directories (idempotent)."""
    for p in (RUNS_DIR, ARTIFACTS_DIR, DATA_DIR, FIGS_DIR, LOGS_DIR):
        p.mkdir(parents=True, exist_ok=True)


class UTCFormatter(logging.Formatter):
    """Logging formatter that renders timestamps in UTC with stable ISO-like format."""

    def formatTime(self, record, datefmt=None):  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def get_logger(name: str = "sim", *, file_mode: str = "w") -> logging.Logger:
    """
    Return a configured logger.
    - No duplicate handlers on re-run
    - Console + file handlers
    - UTC timestamps
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    if file_mode not in ("w", "a"):
        raise ValueError("file_mode must be 'w' or 'a'")

    ensure_dirs()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = UTCFormatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh_path = LOGS_DIR / f"{name}.log"
    fh = logging.FileHandler(fh_path, mode=file_mode, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def env_summary() -> Dict[str, Any]:
    """Return a concise environment summary dict (serializable)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(PROJECT_ROOT),
        "paths": {
            "runs": str(RUNS_DIR),
            "artifacts": str(ARTIFACTS_DIR),
            "data": str(DATA_DIR),
            "figures": str(FIGS_DIR),
            "logs": str(LOGS_DIR),
        },
    }


def write_json(path: Path, obj: Any) -> None:
    """Write JSON with stable formatting (sorted keys, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)


# ----------------------------
# Config schema
# ----------------------------

@dataclass(frozen=True)
class MobilityCfg:
    model: str  # "rwp"
    area_w: float
    area_h: float
    v_min: float
    v_max: float
    pause_s: float


@dataclass(frozen=True)
class StaticMobilityCfg(MobilityCfg):
    """
    Phase 12B-0 (A-025): additive subclass for deterministic static-placement
    diagnostic topologies. Used ONLY by explicit diagnostic scenarios.

    CLAIM BOUNDARY: Static placement is a controlled diagnostic topology only.
    It is NOT realistic disaster/emergency mobility, NOT Bluetooth Mesh compliance,
    NOT IEEE compliance, NOT a real deployment model, NOT true articulation-point /
    cut-vertex detection, NOT CDS, NOT MPR, and NOT a connectivity guarantee. Node
    positions are researcher-defined test fixtures placed by the experimenter, not
    discovered by any algorithm. See docs/assumption_ledger.md A-025 and
    docs/phase12b0_static_placement_plan.md.

    GOLDEN-HASH SAFETY: This is an additive *subclass*. Existing RWP scenarios
    construct plain MobilityCfg, whose asdict() is unchanged (only the 6 base
    fields), so all existing cfg_hash() / golden hashes are preserved. The extra
    fields below exist only on StaticMobilityCfg instances.

    Construct with model="static". Nodes do not move (require v_min == v_max == 0.0).
    node_positions must contain exactly n_nodes (x, y) pairs within the area bounds.
    """
    # New fields are defaulted so they legally follow the required base fields.
    node_positions: Tuple[Tuple[float, float], ...] = ()
    placement_tag: str = ""  # human-readable topology label, e.g. "barbell_2c_3i"


@dataclass(frozen=True)
class TrafficFlowCfg:
    src: int
    dst: int
    model: str  # "cbr" or "poisson"
    rate_pps: float
    start_s: float
    stop_s: float
    payload_bytes: int


@dataclass(frozen=True)
class TrafficCfg:
    flows: Tuple[TrafficFlowCfg, ...]


@dataclass(frozen=True)
class PhyCfg:
    freq_mhz: float
    tx_power_dbm: float
    noise_floor_dbm: float
    pathloss_exp: float
    ref_dist_m: float
    ref_pl_db: float
    sinr0_db: float
    per_k: float

    fading_model: str = "none"
    rician_k_db: float = 6.0
    shadow_sigma_db: float = 0.0
    small_scale_fading: bool = False


@dataclass(frozen=True)
class MacCfg:
    scan_interval_s: float
    scan_window_s: float
    adv_interval_s: float
    adv_channels: Tuple[int, int, int]
    max_retries: int
    cca_threshold_dbm: float
    backoff_min_s: float
    backoff_max_s: float

    t_ifs_s: float = 0.000150
    drift_ppm: float = 20.0

    airtime_budget_enabled: bool = False
    airtime_budget_capacity_s: float = 0.0
    airtime_budget_refill_sps: float = 0.0
    airtime_budget_init_s: float = 0.0


@dataclass(frozen=True)
class SimCfg:
    sim_duration_s: float
    n_nodes: int
    seed_base: int
    mobility: MobilityCfg
    traffic: TrafficCfg
    phy: PhyCfg
    mac: MacCfg
    policy: str
    run_tag: str = "dev"
    notes: str = ""


# ----------------------------
# Validation
# ----------------------------

class CfgError(ValueError):
    pass


def validate_cfg(cfg: SimCfg) -> None:
    errs: List[str] = []

    if not (cfg.sim_duration_s > 0):
        errs.append("sim_duration_s must be > 0")
    if not (cfg.n_nodes >= 2):
        errs.append("n_nodes must be >= 2")
    if not (cfg.seed_base >= 0):
        errs.append("seed_base must be >= 0")

    m = cfg.mobility
    # Phase 12B-0 (A-025): "static" added as an additive diagnostic model.
    # RWP validation behavior is unchanged: an "rwp" config produces exactly the
    # same (empty) error contribution from this check as before.
    if m.model not in ("rwp", "static"):
        errs.append("mobility.model must be 'rwp' or 'static'")
    for field_name in ("area_w", "area_h"):
        v = getattr(m, field_name)
        if not (v > 0):
            errs.append(f"mobility.{field_name} must be > 0")
    if not (0 <= m.v_min <= m.v_max):
        errs.append("mobility.v_min must satisfy 0 <= v_min <= v_max")
    if not (m.pause_s >= 0):
        errs.append("mobility.pause_s must be >= 0")

    # Phase 12B-0 (A-025): static-placement diagnostic validation (additive).
    # Only reached for model == "static"; RWP configs never enter this branch.
    if m.model == "static":
        positions = getattr(m, "node_positions", None)
        if not positions:
            errs.append(
                "mobility.model 'static' requires non-empty node_positions "
                "(use StaticMobilityCfg)"
            )
        else:
            if len(positions) != int(cfg.n_nodes):
                errs.append(
                    f"static node_positions length {len(positions)} != n_nodes {cfg.n_nodes}"
                )
            for i, p in enumerate(positions):
                try:
                    px, py = float(p[0]), float(p[1])
                except Exception:
                    errs.append(f"static node_positions[{i}] must be an (x, y) pair")
                    continue
                if not (0.0 <= px <= m.area_w):
                    errs.append(f"static node_positions[{i}].x={px} out of [0, area_w={m.area_w}]")
                if not (0.0 <= py <= m.area_h):
                    errs.append(f"static node_positions[{i}].y={py} out of [0, area_h={m.area_h}]")
        if not (m.v_min == 0.0 and m.v_max == 0.0):
            errs.append("static placement requires v_min == v_max == 0.0 (nodes do not move)")

    if len(cfg.traffic.flows) == 0:
        errs.append("traffic.flows must be non-empty (use at least 1 flow)")
    for i, f in enumerate(cfg.traffic.flows):
        if not (0 <= f.src < cfg.n_nodes):
            errs.append(f"traffic.flows[{i}].src out of range")
        if not (0 <= f.dst < cfg.n_nodes):
            errs.append(f"traffic.flows[{i}].dst out of range")
        if f.src == f.dst:
            errs.append(f"traffic.flows[{i}] src==dst not allowed")
        if f.model not in ("cbr", "poisson"):
            errs.append(f"traffic.flows[{i}].model must be 'cbr' or 'poisson'")
        if not (f.rate_pps > 0):
            errs.append(f"traffic.flows[{i}].rate_pps must be > 0")
        if not (0 <= f.start_s < f.stop_s <= cfg.sim_duration_s):
            errs.append(f"traffic.flows[{i}] must satisfy 0<=start<stop<=sim_duration_s")
        if not (1 <= f.payload_bytes <= 2048):
            errs.append(f"traffic.flows[{i}].payload_bytes must be 1..2048")

    p = cfg.phy
    if not (p.freq_mhz > 0):
        errs.append("phy.freq_mhz must be > 0")
    if not (math.isfinite(p.tx_power_dbm) and math.isfinite(p.noise_floor_dbm)):
        errs.append("phy tx_power_dbm/noise_floor_dbm must be finite")
    if not (p.pathloss_exp > 0):
        errs.append("phy.pathloss_exp must be > 0")
    if not (p.ref_dist_m > 0):
        errs.append("phy.ref_dist_m must be > 0")
    if not (p.ref_pl_db >= 0):
        errs.append("phy.ref_pl_db must be >= 0")
    if not (math.isfinite(p.sinr0_db) and math.isfinite(p.per_k) and p.per_k > 0):
        errs.append("phy.sinr0_db must be finite and per_k must be > 0")

    fm = str(getattr(p, "fading_model", "none"))
    if not isinstance(fm, str) or not fm:
        errs.append("phy.fading_model must be non-empty string")
    if not math.isfinite(float(getattr(p, "rician_k_db", 6.0))):
        errs.append("phy.rician_k_db must be finite")
    ss = float(getattr(p, "shadow_sigma_db", 0.0))
    if not (math.isfinite(ss) and ss >= 0.0):
        errs.append("phy.shadow_sigma_db must be finite and >= 0")
    _ = bool(getattr(p, "small_scale_fading", False))

    mac = cfg.mac
    if not (mac.scan_interval_s > 0):
        errs.append("mac.scan_interval_s must be > 0")
    if not (0 < mac.scan_window_s <= mac.scan_interval_s):
        errs.append("mac.scan_window_s must be in (0, scan_interval_s]")
    if not (mac.adv_interval_s > 0):
        errs.append("mac.adv_interval_s must be > 0")
    if tuple(mac.adv_channels) != (37, 38, 39):
        errs.append("mac.adv_channels must be (37,38,39) for BLE ADV (Phase C)")
    if not (mac.max_retries >= 0):
        errs.append("mac.max_retries must be >= 0")
    if not (mac.backoff_min_s >= 0 and mac.backoff_max_s >= mac.backoff_min_s):
        errs.append("mac.backoff_min_s/max_s must satisfy 0<=min<=max")

    if not (float(mac.t_ifs_s) >= 0.0):
        errs.append("mac.t_ifs_s must be >= 0")
    if not (math.isfinite(float(mac.drift_ppm)) and float(mac.drift_ppm) >= 0.0):
        errs.append("mac.drift_ppm must be finite and >= 0")

    if bool(mac.airtime_budget_enabled):
        cap = float(mac.airtime_budget_capacity_s)
        r = float(mac.airtime_budget_refill_sps)
        init = float(mac.airtime_budget_init_s)
        if not (math.isfinite(cap) and cap > 0.0):
            errs.append("mac.airtime_budget_capacity_s must be finite and > 0 when enabled")
        if not (math.isfinite(r) and r > 0.0):
            errs.append("mac.airtime_budget_refill_sps must be finite and > 0 when enabled")
        if not (math.isfinite(init) and init >= 0.0):
            errs.append("mac.airtime_budget_init_s must be finite and >= 0 when enabled")
        if math.isfinite(cap) and math.isfinite(init) and init > cap + 1e-15:
            errs.append("mac.airtime_budget_init_s must be <= mac.airtime_budget_capacity_s when enabled")
    else:
        for fn in ("airtime_budget_capacity_s", "airtime_budget_refill_sps", "airtime_budget_init_s"):
            v = float(getattr(mac, fn))
            if not math.isfinite(v):
                errs.append(f"mac.{fn} must be finite")

    if not isinstance(cfg.policy, str) or not cfg.policy:
        errs.append("policy must be non-empty string")

    if errs:
        raise CfgError("Invalid SimCfg:\n- " + "\n- ".join(errs))


# ----------------------------
# Minimal scenario builders
# ----------------------------

def make_easy_cfg() -> SimCfg:
    return SimCfg(
        sim_duration_s=30.0,
        n_nodes=10,
        seed_base=0,
        mobility=MobilityCfg(
            model="rwp",
            area_w=6.0,
            area_h=6.0,
            v_min=0.0,
            v_max=0.2,
            pause_s=2.0,
        ),
        traffic=TrafficCfg(
            flows=(
                TrafficFlowCfg(
                    src=0,
                    dst=1,
                    model="cbr",
                    rate_pps=2.0,
                    start_s=1.0,
                    stop_s=25.0,
                    payload_bytes=50,
                ),
            )
        ),
        phy=PhyCfg(
            freq_mhz=2402.0,
            tx_power_dbm=0.0,
            noise_floor_dbm=-95.0,
            pathloss_exp=2.0,
            ref_dist_m=1.0,
            ref_pl_db=35.0,
            sinr0_db=6.0,
            per_k=1.0,
        ),
        mac=MacCfg(
            scan_interval_s=0.1,
            scan_window_s=0.08,
            adv_interval_s=0.2,
            adv_channels=(37, 38, 39),
            max_retries=1,
            cca_threshold_dbm=-75.0,
            backoff_min_s=0.001,
            backoff_max_s=0.01,
            t_ifs_s=0.000150,
            drift_ppm=20.0,
        ),
        policy="flood",
        run_tag="easy",
    )


def make_moderate_cfg() -> SimCfg:
    return SimCfg(
        sim_duration_s=60.0,
        n_nodes=40,
        seed_base=0,
        mobility=MobilityCfg(model="rwp", area_w=80.0, area_h=80.0, v_min=0.2, v_max=1.5, pause_s=0.5),
        traffic=TrafficCfg(
            flows=(
                TrafficFlowCfg(src=0, dst=10, model="poisson", rate_pps=3.0, start_s=2.0, stop_s=55.0, payload_bytes=80),
                TrafficFlowCfg(src=5, dst=20, model="cbr", rate_pps=2.0, start_s=5.0, stop_s=55.0, payload_bytes=80),
            )
        ),
        phy=PhyCfg(
            freq_mhz=2402.0,
            tx_power_dbm=0.0,
            noise_floor_dbm=-95.0,
            pathloss_exp=2.2,
            ref_dist_m=1.0,
            ref_pl_db=40.0,
            sinr0_db=6.0,
            per_k=1.2,
        ),
        mac=MacCfg(
            scan_interval_s=0.2,
            scan_window_s=0.08,
            adv_interval_s=0.25,
            adv_channels=(37, 38, 39),
            max_retries=2,
            cca_threshold_dbm=-78.0,
            backoff_min_s=0.001,
            backoff_max_s=0.02,
            t_ifs_s=0.000150,
            drift_ppm=20.0,
        ),
        policy="batman_lite",
        run_tag="moderate",
    )


def make_stress_cfg() -> SimCfg:
    return SimCfg(
        sim_duration_s=60.0,
        n_nodes=70,
        seed_base=0,
        mobility=MobilityCfg(model="rwp", area_w=120.0, area_h=120.0, v_min=0.5, v_max=2.5, pause_s=0.2),
        traffic=TrafficCfg(
            flows=(
                TrafficFlowCfg(src=0, dst=30, model="poisson", rate_pps=6.0, start_s=2.0, stop_s=55.0, payload_bytes=120),
                TrafficFlowCfg(src=10, dst=50, model="poisson", rate_pps=6.0, start_s=2.0, stop_s=55.0, payload_bytes=120),
                TrafficFlowCfg(src=20, dst=60, model="cbr", rate_pps=5.0, start_s=5.0, stop_s=55.0, payload_bytes=120),
            )
        ),
        phy=PhyCfg(
            freq_mhz=2402.0,
            tx_power_dbm=0.0,
            noise_floor_dbm=-95.0,
            pathloss_exp=2.4,
            ref_dist_m=1.0,
            ref_pl_db=40.0,
            sinr0_db=6.0,
            per_k=1.5,
        ),
        mac=MacCfg(
            scan_interval_s=0.5,
            scan_window_s=0.05,
            adv_interval_s=0.2,
            adv_channels=(37, 38, 39),
            max_retries=3,
            cca_threshold_dbm=-80.0,
            backoff_min_s=0.001,
            backoff_max_s=0.05,
            t_ifs_s=0.000150,
            drift_ppm=20.0,
        ),
        policy="batman_lite",
        run_tag="stress",
    )
