"""
Phase 4 trend test: offered load vs collision / channel-busy pressure.

Validation matrix row: Offered load / collision pressure
Sources: BLE-MESH-PERF-001, GOMEZ-2012, BLE-BCAST-STORM-NI-1999
Assumption: A-010

Metric: cca_busy_events — counts how often a node finds the channel occupied
when attempting a transmission. With multiple simultaneous CBR sources, higher
load produces more concurrent TX events, raising CCA-busy collisions.

Scenario: 8 concurrent CBR sources with independent destination nodes; no_fwd
policy (no relay complication). Concurrent source transmissions interact at
the channel level, producing CCA-busy events that scale with injection rate.

Expected trend: higher offered load -> higher cca_busy_events (mostly monotonic).

Claim level: literature-consistent.
Limitation: CBR traffic model; 8-source concurrent scenario; CCA model is
simplified (single-threshold energy detection per A-010).
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

# 8 concurrent sources all inject at the same CBR times.
# Their TX attempts are staggered by individual adv_delay (0-10ms), creating
# channel contention that is captured by cca_busy_events.
N_SOURCES = 8
LOAD_SWEEP_PPS = [0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = list(range(1, 6))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(rate_pps: float):
    base = make_easy_cfg()
    # Each source has its own destination to avoid policy/relay interaction.
    flows = tuple(
        TrafficFlowCfg(
            src=i,
            dst=i + N_SOURCES,
            model="cbr",
            rate_pps=rate_pps,
            start_s=1.0,
            stop_s=25.0,
            payload_bytes=50,
        )
        for i in range(N_SOURCES)
    )
    return dataclasses.replace(
        base,
        n_nodes=N_SOURCES * 2,   # sources (0..7) + destinations (8..15)
        traffic=TrafficCfg(flows=flows),
    )


def _run(cfg, seed):
    h = cfg_hash(cfg)
    sm = SeedManager(cfg_hash=h, seed=seed)
    plan = build_rwp_plan(cfg, sm)
    injs = build_injections(cfg, sm)
    pol = policy_factory("no_fwd")
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


def _avg_cca(rate_pps: float) -> float:
    cfg = _make_cfg(rate_pps)
    vals = [float(int(_run(cfg, s)["summary"].get("cca_busy_events", 0))) for s in SEEDS]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_cca_busy_trend():
    """
    Sweep offered load 0.5 -> 8.0 pps with 8 concurrent CBR sources.
    cca_busy_events must increase mostly monotonically (>= 75% of pairs).
    Higher rate -> more concurrent TX events -> more channel-busy collisions.
    """
    avgs = [_avg_cca(r) for r in LOAD_SWEEP_PPS]

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "Load/collision trend FAIL: cca_busy_events not mostly increasing "
        f"with offered load (fraction={frac:.2f}).\n"
        f"  Rates (pps): {LOAD_SWEEP_PPS}\n"
        f"  Avg cca_busy:{[round(v, 1) for v in avgs]}"
    )


def test_load_extreme_comparison():
    """
    Sanity check: rate=8.0 pps must produce more cca_busy_events
    than rate=0.5 pps, summed over all seeds.
    """
    cfg_lo = _make_cfg(0.5)
    cfg_hi = _make_cfg(8.0)

    cca_lo = sum(int(_run(cfg_lo, s)["summary"].get("cca_busy_events", 0)) for s in SEEDS)
    cca_hi = sum(int(_run(cfg_hi, s)["summary"].get("cca_busy_events", 0)) for s in SEEDS)

    assert cca_hi > cca_lo, (
        f"Extreme comparison FAIL: rate=8.0 cca_busy={cca_hi} "
        f"must be > rate=0.5 cca_busy={cca_lo}"
    )


def test_load_tx_attempts_trend():
    """
    Secondary metric: total tx_attempts must also increase with offered load.
    Higher injection rate -> more packets queued -> more TX attempts.
    """
    avgs = []
    for rate in LOAD_SWEEP_PPS:
        cfg = _make_cfg(rate)
        vals = [int(_run(cfg, s)["summary"].get("tx_attempts", 0)) for s in SEEDS]
        avgs.append(sum(vals) / len(vals))

    ok, frac = _mostly_monotonic_increasing(avgs)
    assert ok, (
        "tx_attempts not mostly increasing with load "
        f"(fraction={frac:.2f}): {[round(v, 1) for v in avgs]}"
    )
