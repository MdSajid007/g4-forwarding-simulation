"""
Phase 4 trend test: flood density vs duplicate transmissions and airtime.

Validation matrix row: Flood density / duplicate transmissions
Sources: BLE-BCAST-STORM-NI-1999, BLE-MESH-PERF-001
Assumption: none (experimental design choice -- TTL as flood-depth control)

TTL controls flood depth in the flood policy:
- ttl_hops=1  -> src transmits, only 1 relay hop allowed -> low tx_attempts
- ttl_hops=10 -> maximal relay cascade -> many tx_attempts
Each relay node that receives a packet and has ttl > 0 will forward once.
More relay hops -> more concurrent transmissions -> higher tx_attempts and airtime.

Scenario:
- 20x20m area, 20 nodes, flood policy.
- Dense network so relaying creates many concurrent forwarding events.
- Fixed traffic rate; sweep ttl_hops.

Acceptance criterion: tx_attempts and airtime_s both increase mostly monotonically
with ttl_hops (>= 75% of consecutive pairs non-decreasing).

Claim level: literature-consistent controlled trend validation.
Limitation: TTL as flood-depth proxy; BLE-BCAST-STORM-NI-1999 is not BLE-specific;
real broadcast-storm dynamics differ from this simplified model.
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

TTL_SWEEP = [1, 2, 3, 5, 8, 10]
SEEDS = list(range(1, 6))
N_NODES = 20


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _make_flood_density_cfg():
    """
    Multi-hop scenario: 100x100m area with limited PHY range (~15-20m).
    pathloss_exp=3.0, ref_pl_db=40, per_k=1.5 -> effective ~50% PER radius ~40m.
    src=0, dst=19 are statistically distant; direct delivery fails for most seeds.

    This is essential: with direct delivery suppressed by _delivered_keys, relay
    transmissions only proceed when dst has NOT yet been reached. In a large area
    with limited range, higher TTL enables deeper relay cascades before delivery
    occurs, causing tx_attempts and airtime to scale monotonically with TTL.

    pause_s=100 keeps nodes near their initial positions for the 30s simulation.
    """
    base = make_easy_cfg()
    flow = TrafficFlowCfg(
        src=0,
        dst=N_NODES - 1,
        model="cbr",
        rate_pps=2.0,
        start_s=1.0,
        stop_s=25.0,
        payload_bytes=50,
    )
    return dataclasses.replace(
        base,
        n_nodes=N_NODES,
        mobility=dataclasses.replace(
            base.mobility,
            area_w=100.0,
            area_h=100.0,
            v_min=0.01,
            v_max=0.1,
            pause_s=100.0,
        ),
        traffic=TrafficCfg(flows=(flow,)),
        phy=dataclasses.replace(
            base.phy,
            pathloss_exp=3.0,
            ref_pl_db=40.0,
            noise_floor_dbm=-95.0,
            sinr0_db=6.0,
            per_k=1.5,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cfg, seed, ttl_hops):
    h = cfg_hash(cfg)
    sm = SeedManager(cfg_hash=h, seed=seed)
    plan = build_rwp_plan(cfg, sm)
    injs = build_injections(cfg, sm, ttl_hops=ttl_hops)
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


def _avg_metric(cfg, ttl_hops, key) -> float:
    vals = [float(_run(cfg, s, ttl_hops)["summary"].get(key, 0)) for s in SEEDS]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flood_density_tx_attempts_trend():
    """
    Sweep ttl_hops 1 -> 10 with flood policy.
    tx_attempts must increase mostly monotonically (>= 75% of consecutive pairs).
    More relay hops -> more forwarding events -> more TX attempts.
    """
    cfg = _make_flood_density_cfg()
    avgs = [_avg_metric(cfg, ttl, "tx_attempts") for ttl in TTL_SWEEP]

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "Flood density trend FAIL (tx_attempts): not mostly increasing with TTL "
        f"(fraction={frac:.2f}).\n"
        f"  TTL sweep:  {TTL_SWEEP}\n"
        f"  Avg tx_attempts: {[round(v, 1) for v in avgs]}"
    )


def test_flood_density_airtime_trend():
    """
    Sweep ttl_hops 1 -> 10; airtime_s must increase mostly monotonically.
    More relay transmissions -> more total airtime consumed.
    """
    cfg = _make_flood_density_cfg()
    avgs = [_avg_metric(cfg, ttl, "airtime_s") for ttl in TTL_SWEEP]

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "Flood density trend FAIL (airtime_s): not mostly increasing with TTL "
        f"(fraction={frac:.2f}).\n"
        f"  TTL sweep:  {TTL_SWEEP}\n"
        f"  Avg airtime_s: {[round(v, 6) for v in avgs]}"
    )


def test_flood_density_extreme_comparison():
    """
    Sanity check: ttl_hops=10 must produce more tx_attempts and more airtime_s
    than ttl_hops=1, summed over all seeds.
    """
    cfg = _make_flood_density_cfg()

    tx_lo = sum(int(_run(cfg, s, 1)["summary"].get("tx_attempts", 0)) for s in SEEDS)
    tx_hi = sum(int(_run(cfg, s, 10)["summary"].get("tx_attempts", 0)) for s in SEEDS)

    at_lo = sum(float(_run(cfg, s, 1)["summary"].get("airtime_s", 0.0)) for s in SEEDS)
    at_hi = sum(float(_run(cfg, s, 10)["summary"].get("airtime_s", 0.0)) for s in SEEDS)

    assert tx_hi > tx_lo, (
        f"Extreme comparison FAIL (tx_attempts): ttl=10 sum={tx_hi} "
        f"must be > ttl=1 sum={tx_lo}"
    )
    assert at_hi > at_lo, (
        f"Extreme comparison FAIL (airtime_s): ttl=10 sum={at_hi:.6f} "
        f"must be > ttl=1 sum={at_lo:.6f}"
    )
