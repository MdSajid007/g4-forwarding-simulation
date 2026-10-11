"""
Phase 4 trend test: interference (SINR degradation) vs PHY fail rate.

Validation matrix row: Interference
Sources: BLE-MESH-PERF-001 (primary), COEX-IEEE-802-15-2 (conceptual framing)
Assumption: A-011

Interference is modeled as controlled elevation of noise_floor_dbm (additive noise
contribution to SINR, per A-011). Higher noise floor -> lower SINR -> higher PHY
fail rate.

Claim level: literature-consistent controlled trend validation + controlled stress
assumption (A-011: additive noise abstraction).
Limitation: not a full RF coexistence simulator; additive noise floor is a simplified
abstraction; real 802.11/BLE coexistence dynamics not modeled.
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

# PHY config: pathloss_exp=3.0 and reduced area give SINR values in the
# logistic curve's sensitive region (5-15 dB) for the noise floor sweep.
# At 10m with n=3.0, ref_pl_db=40: RSSI ~ -70 dBm.
# With sinr0_db=6, per_k=1.5 the transition from low to high PER occurs
# approximately between noise_floor=-80 and noise_floor=-65 dBm.
NOISE_FLOOR_SWEEP_DBM = [-115.0, -100.0, -85.0, -78.0, -72.0]
SEEDS = list(range(1, 6))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(noise_floor_dbm: float):
    base = make_easy_cfg()
    return dataclasses.replace(
        base,
        n_nodes=15,
        mobility=dataclasses.replace(
            base.mobility,
            area_w=30.0,
            area_h=30.0,
        ),
        phy=dataclasses.replace(
            base.phy,
            pathloss_exp=3.0,
            ref_pl_db=40.0,
            per_k=1.5,
            noise_floor_dbm=noise_floor_dbm,
        ),
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


def _phy_fail_rate(summary: dict) -> float:
    rx_ok = int(summary.get("rx_ok", 0))
    rx_fail = int(summary.get("rx_fail", 0))
    total = rx_ok + rx_fail
    return float(rx_fail) / float(max(1, total))


def _mostly_monotonic_increasing(values, min_fraction=0.75):
    pairs = len(values) - 1
    if pairs < 1:
        return True, 1.0
    good = sum(1 for i in range(pairs) if values[i + 1] >= values[i])
    frac = good / pairs
    return frac >= min_fraction, frac


def _avg_fail_rate(noise_floor_dbm: float) -> float:
    cfg = _make_cfg(noise_floor_dbm)
    vals = [_phy_fail_rate(_run(cfg, s)["summary"]) for s in SEEDS]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_interference_phy_fail_rate_trend():
    """
    Sweep noise_floor_dbm -115 -> -72 dBm (increasing interference/noise).
    PHY fail rate (rx_fail / rx_total) must increase mostly monotonically
    (>= 75% of consecutive pairs).
    """
    avgs = [_avg_fail_rate(nf) for nf in NOISE_FLOOR_SWEEP_DBM]

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "Interference trend FAIL: PHY fail rate not mostly increasing "
        f"with noise floor (fraction={frac:.2f}).\n"
        f"  Noise floors (dBm): {NOISE_FLOOR_SWEEP_DBM}\n"
        f"  Avg fail rate:      {[round(v, 4) for v in avgs]}"
    )


def test_interference_extreme_comparison():
    """
    Sanity check: highest noise floor (-72 dBm) must produce strictly higher
    PHY fail rate than lowest noise floor (-115 dBm), summed over all seeds.
    """
    cfg_clean = _make_cfg(-115.0)
    cfg_noisy = _make_cfg(-72.0)

    fr_clean = sum(_phy_fail_rate(_run(cfg_clean, s)["summary"]) for s in SEEDS)
    fr_noisy = sum(_phy_fail_rate(_run(cfg_noisy, s)["summary"]) for s in SEEDS)

    assert fr_noisy > fr_clean, (
        f"Extreme comparison FAIL: noise=-72dBm fail_rate_sum={fr_noisy:.4f} "
        f"must be > noise=-115dBm fail_rate_sum={fr_clean:.4f}"
    )


def test_interference_rx_fail_count_trend():
    """
    Secondary metric: absolute rx_fail count must also increase with noise.
    This confirms the signal is large enough to observe with these seeds.
    """
    avgs = []
    for nf in NOISE_FLOOR_SWEEP_DBM:
        cfg = _make_cfg(nf)
        vals = [int(_run(cfg, s)["summary"].get("rx_fail", 0)) for s in SEEDS]
        avgs.append(sum(vals) / len(vals))

    ok, frac = _mostly_monotonic_increasing(avgs, min_fraction=0.75)
    assert ok, (
        "rx_fail count not mostly increasing with noise floor "
        f"(fraction={frac:.2f}): {[round(v, 1) for v in avgs]}"
    )
