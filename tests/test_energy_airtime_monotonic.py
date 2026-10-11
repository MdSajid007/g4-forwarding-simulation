"""
Phase 4 trend test: airtime vs estimated energy (monotonic).

Validation matrix row: Energy model
Sources: HW-NRF52840
Assumption: A-005

Energy model: energy_j = sum(airtime_per_tx * tx_current_mA * tx_voltage)
The simulator computes: energy_j += dt * dbm_to_mw(tx_power_dbm) per TX event
(treating TX power in mW as a proxy for energy rate, consistent with the
airtime-proportional model from A-005).

Expected trends:
1. airtime_s increases monotonically with offered load (more TX events).
2. energy_j increases monotonically with airtime_s across the load sweep.
3. The ratio energy_j / airtime_s is consistent (within +/-15% relative spread)
   confirming the linear energy model is preserved.

Claim level: hardware-parameterized (energy model uses HW-NRF52840, A-005).
Limitation: simplified energy model; sleep/idle sub-states not modeled;
nRF52840-specific current values (A-005 outstanding verification).
"""
import dataclasses

from src.airtime import build_airtime_buckets
from src.config import make_easy_cfg, TrafficCfg, TrafficFlowCfg
from src.hashing import cfg_hash
from src.mac_engine import SimEngine
from src.mobility import build_rwp_plan
from src.policies import policy_factory
from src.seeds import SeedManager
from src.traffic import build_injections

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOAD_SWEEP_PPS = [0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = list(range(1, 6))
N_NODES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(rate_pps: float):
    base = make_easy_cfg()
    new_flow = dataclasses.replace(
        base.traffic.flows[0],
        src=0,
        dst=min(1, N_NODES - 1),
        rate_pps=rate_pps,
    )
    return dataclasses.replace(
        base,
        n_nodes=N_NODES,
        traffic=TrafficCfg(flows=(new_flow,)),
    )


def _run(cfg, seed):
    h = cfg_hash(cfg)
    sm = SeedManager(cfg_hash=h, seed=seed)
    plan = build_rwp_plan(cfg, sm)
    injs = build_injections(cfg, sm)
    pol = policy_factory("flood")
    buckets = build_airtime_buckets(cfg)
    eng = SimEngine(cfg, seed=seed, plan=plan, policy=pol, cfg_hash_for_rng=h, airtime_buckets=buckets)
    return eng.run(injs)


def _mostly_monotonic_increasing(values, min_fraction=0.75):
    pairs = len(values) - 1
    if pairs < 1:
        return True, 1.0
    good = sum(1 for i in range(pairs) if values[i + 1] >= values[i])
    frac = good / pairs
    return frac >= min_fraction, frac


def _avg(cfg, key) -> float:
    vals = [float(_run(cfg, s)["summary"].get(key, 0)) for s in SEEDS]
    return sum(vals) / len(vals)


def _sweep_avgs(metric_key: str):
    return [_avg(_make_cfg(r), metric_key) for r in LOAD_SWEEP_PPS]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_airtime_increases_with_load():
    """
    Sweep offered load 0.5 -> 8.0 pps; airtime_s must increase mostly
    monotonically (>= 75% of consecutive pairs).
    Higher load -> more TX events -> more total airtime.
    """
    avgs = _sweep_avgs("airtime_s")

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "Airtime trend FAIL: airtime_s not mostly increasing with load "
        f"(fraction={frac:.2f}).\n"
        f"  Rates (pps): {LOAD_SWEEP_PPS}\n"
        f"  Avg airtime_s: {[round(v, 6) for v in avgs]}"
    )


def test_energy_increases_with_load():
    """
    Sweep offered load; energy_j must increase mostly monotonically.
    Energy is proportional to airtime via the TX power model.
    """
    avgs = _sweep_avgs("energy_j")

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "Energy trend FAIL: energy_j not mostly increasing with load "
        f"(fraction={frac:.2f}).\n"
        f"  Rates (pps): {LOAD_SWEEP_PPS}\n"
        f"  Avg energy_j: {[round(v, 9) for v in avgs]}"
    )


def test_energy_proportional_to_airtime():
    """
    Verify the linear energy model: energy_j / airtime_s ratio must be
    consistent across all load sweep points (within +/-15% relative to mean).
    This confirms energy_j = airtime_s * constant, as per the A-005 model.
    """
    ratios = []
    for rate in LOAD_SWEEP_PPS:
        cfg = _make_cfg(rate)
        at_vals = [float(_run(cfg, s)["summary"].get("airtime_s", 0.0)) for s in SEEDS]
        en_vals = [float(_run(cfg, s)["summary"].get("energy_j", 0.0)) for s in SEEDS]
        avg_at = sum(at_vals) / len(at_vals)
        avg_en = sum(en_vals) / len(en_vals)
        if avg_at > 1e-12:
            ratios.append(avg_en / avg_at)

    assert len(ratios) >= 3, "Not enough valid ratio points to check linearity"

    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio > 0.0, "Mean energy/airtime ratio must be positive"

    max_dev = max(abs(r - mean_ratio) / mean_ratio for r in ratios)
    assert max_dev <= 0.15, (
        f"Energy/airtime ratio inconsistency: max relative deviation={max_dev:.3f} "
        f"exceeds 15% tolerance.\n"
        f"  Ratios: {[round(r, 6) for r in ratios]}\n"
        f"  Mean:   {mean_ratio:.6f}"
    )


def test_energy_extreme_comparison():
    """
    Sanity check: rate=8.0 must produce more energy and airtime than rate=0.5,
    summed over all seeds.
    """
    cfg_lo = _make_cfg(0.5)
    cfg_hi = _make_cfg(8.0)

    at_lo = sum(float(_run(cfg_lo, s)["summary"].get("airtime_s", 0.0)) for s in SEEDS)
    at_hi = sum(float(_run(cfg_hi, s)["summary"].get("airtime_s", 0.0)) for s in SEEDS)
    en_lo = sum(float(_run(cfg_lo, s)["summary"].get("energy_j", 0.0)) for s in SEEDS)
    en_hi = sum(float(_run(cfg_hi, s)["summary"].get("energy_j", 0.0)) for s in SEEDS)

    assert at_hi > at_lo, (
        f"Airtime extreme FAIL: rate=8.0 sum={at_hi:.6f} must be > rate=0.5 sum={at_lo:.6f}"
    )
    assert en_hi > en_lo, (
        f"Energy extreme FAIL: rate=8.0 sum={en_hi:.9f} must be > rate=0.5 sum={en_lo:.9f}"
    )
