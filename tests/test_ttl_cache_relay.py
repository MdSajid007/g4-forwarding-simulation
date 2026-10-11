"""
Phase 4 trend test: TTL vs multi-hop delivery reach.

Validation matrix row: Relay / TTL
Sources: BT-MESH-PROFILE-11 (S3.4.5, S3.4.6), BT-MESH-MODEL-11
Assumption: A-003

Expected trend: lower TTL -> reduced multi-hop reach -> lower PDR.
ttl_hops=N allows up to N relay hops beyond the source (N+1 total transmissions).

Scenario design:
- Large area (80x80m), limited PHY range (~40-50m effective radius).
- With n_nodes=20 and effective range < diagonal (~113m), multi-hop is required
  for src=0 to dst=19 across random topologies.
- Flood policy; seeds 1-5 average over random topologies.
- Sweep ttl_hops: 1, 2, 3, 5, 8.

Acceptance criterion: avg_pdr[ttl=8] > avg_pdr[ttl=1];
at least 3/4 consecutive TTL-ascending pairs show PDR non-decreasing.

Claim level: standards-aligned (TTL decrement/cache rule from BT-MESH-PROFILE-11).
Limitation: simplified relay logic; no segmentation/IV-index; RWP mobility
introduces topology randomness (A-003, A-009).
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

TTL_SWEEP = [1, 2, 3, 5, 8]
SEEDS = list(range(1, 6))
N_NODES = 20


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _make_multihop_cfg():
    """
    Multi-hop scenario:
    - 80x80m area; 20 nodes; slow mobility (mostly static topology).
    - PHY: pathloss_exp=3.0, ref_pl_db=40, tx_power=0, noise_floor=-95.
      Effective ~50% PER radius ~ 40m; paths >56m (diagonal of 40m grid cell)
      require relaying.
    - src=0, dst=19 (statistically distant across seeds).
    - pause_s=100 keeps nodes near their initial positions for the 30s sim.
    """
    base = make_easy_cfg()
    flow = TrafficFlowCfg(
        src=0,
        dst=N_NODES - 1,
        model="cbr",
        rate_pps=1.0,
        start_s=1.0,
        stop_s=25.0,
        payload_bytes=50,
    )
    return dataclasses.replace(
        base,
        n_nodes=N_NODES,
        mobility=dataclasses.replace(
            base.mobility,
            area_w=80.0,
            area_h=80.0,
            v_min=0.01,
            v_max=0.1,
            pause_s=100.0,   # stay near initial position for 30s sim
        ),
        traffic=TrafficCfg(flows=(flow,)),
        phy=dataclasses.replace(
            base.phy,
            pathloss_exp=3.0,
            ref_pl_db=40.0,
            sinr0_db=6.0,
            per_k=1.5,
            noise_floor_dbm=-95.0,
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


def _pdr(summary: dict) -> float:
    inj = int(summary.get("app_injected", 0))
    dlv = int(summary.get("delivered", 0))
    return float(dlv) / float(max(1, inj))


def _mostly_monotonic_increasing(values, min_fraction=0.75):
    pairs = len(values) - 1
    if pairs < 1:
        return True, 1.0
    good = sum(1 for i in range(pairs) if values[i + 1] >= values[i])
    frac = good / pairs
    return frac >= min_fraction, frac


def _avg_pdr(cfg, ttl_hops) -> float:
    vals = [_pdr(_run(cfg, s, ttl_hops)["summary"]) for s in SEEDS]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ttl_pdr_trend():
    """
    Sweep ttl_hops 1 -> 8; avg PDR over seeds must increase mostly monotonically
    (>= 75% of consecutive pairs non-decreasing).
    """
    cfg = _make_multihop_cfg()
    avgs = [_avg_pdr(cfg, ttl) for ttl in TTL_SWEEP]

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "TTL trend FAIL: PDR not mostly increasing with TTL "
        f"(fraction={frac:.2f}).\n"
        f"  TTL sweep:  {TTL_SWEEP}\n"
        f"  Avg PDR:    {[round(v, 4) for v in avgs]}"
    )


def test_ttl_extreme_comparison():
    """
    Sanity check: ttl_hops=8 must produce higher PDR than ttl_hops=1.
    With a large area and limited PHY range, TTL=1 restricts to 2-hop delivery
    while TTL=8 enables up to 9-hop relay chains.
    """
    cfg = _make_multihop_cfg()
    pdr_lo = _avg_pdr(cfg, 1)
    pdr_hi = _avg_pdr(cfg, 8)
    assert pdr_hi > pdr_lo, (
        f"Extreme TTL comparison FAIL: ttl=8 avg_pdr={pdr_hi:.4f} "
        f"must be > ttl=1 avg_pdr={pdr_lo:.4f}"
    )


def test_ttl_cache_duplicate_suppression():
    """
    Verify that the message cache (duplicate suppression) operates correctly.
    With ttl_hops=8, delivered must be <= app_injected (no double-counting).
    """
    cfg = _make_multihop_cfg()
    for s in SEEDS:
        out = _run(cfg, s, ttl_hops=8)
        inj = int(out["summary"].get("app_injected", 0))
        dlv = int(out["summary"].get("delivered", 0))
        assert dlv <= inj, (
            f"Cache invariant FAIL at seed={s}: delivered={dlv} > app_injected={inj}"
        )
