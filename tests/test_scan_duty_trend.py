"""
Phase 4 trend test: scan duty vs NOT_SCANNING drops.

Validation matrix row: Scan duty / NOT_SCANNING drops
Sources: BT-CORE-54, BT-MESH-PROFILE-11, GOMEZ-2012, BLE-MESH-PERF-001
Assumption: A-002

Expected trend: lower scan duty -> higher not_scanning_rx_opportunities.
As duty increases (more scanning), not_scanning_rx_opportunities must decrease
monotonically or mostly monotonically (>= 75% of consecutive pairs).

Claim level: standards-aligned / literature-consistent controlled trend validation.
Limitation: simplified scan window model; passive scanning only; single-source scenario.
"""
import dataclasses

from src.airtime import build_airtime_buckets
from src.config import make_easy_cfg
from src.hashing import cfg_hash
from src.mac_engine import SimEngine
from src.mobility import build_rwp_plan
from src.policies import policy_factory
from src.seeds import SeedManager
from src.traffic import build_injections

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 0.21s avoids alignment with CBR at 0.5s (multiples of 0.5 mod 0.21 ≠ 0).
# With dt_cbr=0.5s and interval=0.21s: phases cycle through ~{0.16, 0.0, 0.08, ...}
# covering a range broad enough to distinguish low-duty from high-duty scenarios.
SCAN_INTERVAL_S = 0.21
# Sweep duties ascending; metric should decrease as duty increases.
SCAN_DUTIES = [0.10, 0.20, 0.40, 0.60, 0.90]
SEEDS = list(range(1, 6))   # 5 seeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cfg, seed):
    h = cfg_hash(cfg)
    sm = SeedManager(cfg_hash=h, seed=seed)
    plan = build_rwp_plan(cfg, sm)
    injs = build_injections(cfg, sm)
    pol = policy_factory("no_fwd")
    buckets = build_airtime_buckets(cfg)
    eng = SimEngine(cfg, seed=seed, plan=plan, policy=pol, cfg_hash_for_rng=h, airtime_buckets=buckets)
    return eng.run(injs)


def _make_cfg(duty: float):
    base = make_easy_cfg()
    return dataclasses.replace(
        base,
        mac=dataclasses.replace(
            base.mac,
            scan_interval_s=SCAN_INTERVAL_S,
            scan_window_s=round(duty * SCAN_INTERVAL_S, 9),
        ),
    )


def _mostly_monotonic_decreasing(values, min_fraction=0.75):
    """Return (pass_bool, fraction) for a non-increasing sequence."""
    pairs = len(values) - 1
    if pairs < 1:
        return True, 1.0
    good = sum(1 for i in range(pairs) if values[i + 1] <= values[i])
    frac = good / pairs
    return frac >= min_fraction, frac


def _avg_metric(duty: float, key: str) -> float:
    cfg = _make_cfg(duty)
    vals = [float(int(_run(cfg, s)["summary"].get(key, 0))) for s in SEEDS]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scan_duty_monotonic_not_scanning():
    """
    Sweep scan duty 0.10 -> 0.90; not_scanning_rx_opportunities must decrease
    monotonically or mostly monotonically (>= 75% of consecutive pairs).
    """
    avgs = [_avg_metric(d, "not_scanning_rx_opportunities") for d in SCAN_DUTIES]

    ok, frac = _mostly_monotonic_decreasing(avgs)
    assert ok, (
        "Scan duty trend FAIL: not_scanning_rx_opportunities not mostly decreasing "
        f"as duty increases (fraction={frac:.2f}).\n"
        f"  Duties:      {SCAN_DUTIES}\n"
        f"  Avg metric:  {[round(v, 1) for v in avgs]}"
    )


def test_scan_duty_extreme_comparison():
    """
    Sanity check: duty=0.10 must give strictly more not_scanning_rx_opportunities
    than duty=0.90, summed over all seeds.
    """
    cfg_lo = _make_cfg(0.10)
    cfg_hi = _make_cfg(0.90)

    ns_lo = sum(int(_run(cfg_lo, s)["summary"].get("not_scanning_rx_opportunities", 0))
                for s in SEEDS)
    ns_hi = sum(int(_run(cfg_hi, s)["summary"].get("not_scanning_rx_opportunities", 0))
                for s in SEEDS)

    assert ns_lo > ns_hi, (
        f"Extreme comparison FAIL: duty=0.10 not_scanning={ns_lo} "
        f"must be > duty=0.90 not_scanning={ns_hi}"
    )


def test_scan_duty_not_scanning_dst_events():
    """
    Secondary metric: not_scanning_dst_events must also increase when duty is lower.
    Lower duty -> more often the destination is not scanning when a waveform arrives.
    """
    avg_lo = _avg_metric(0.10, "not_scanning_dst_events")
    avg_hi = _avg_metric(0.90, "not_scanning_dst_events")
    assert avg_lo >= avg_hi, (
        f"not_scanning_dst_events: duty=0.10 avg={avg_lo:.1f} "
        f"should be >= duty=0.90 avg={avg_hi:.1f}"
    )
