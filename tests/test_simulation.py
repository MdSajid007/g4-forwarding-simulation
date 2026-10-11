"""
Phase 3 test: minimal simulation repeatability.
Maps to Phase 2 requirement: Minimal simulation repeatability (self_test_cell8, 6 assertions).

Phase 2 regression hashes:
  Gate A summary hash: c9077676352c5eda984ba9e3dd8a666d7d3738b01eacf9e437ab36b3be085070
  Minimal run trace hash: 784e5436222fcec34bb1a9ab22435a68411fcaec935774a3040d8451d78aabf1
"""
import hashlib
import pytest

from src.airtime import build_airtime_buckets
from src.config import make_easy_cfg
from src.hashing import cfg_hash
from src.mac_engine import SimEngine, is_scanning
from src.mobility import build_rwp_plan
from src.policies import policy_factory
from src.seeds import SeedManager
from src.traffic import AppInjection, build_injections


PHASE2_TRACE_HASH = "784e5436222fcec34bb1a9ab22435a68411fcaec935774a3040d8451d78aabf1"


def _run_engine(cfg, seed, injs, policy_name):
    h = cfg_hash(cfg)
    plan = build_rwp_plan(cfg, SeedManager(cfg_hash=h, seed=seed))
    pol = policy_factory(policy_name)
    buckets = build_airtime_buckets(cfg)
    eng = SimEngine(cfg, seed=seed, plan=plan, policy=pol, cfg_hash_for_rng=h, airtime_buckets=buckets)
    return eng.run(injs)


def _trace_hash(trace):
    rows = [(round(float(t), 12), int(p), str(e), int(src)) for (t, p, e, src) in trace]
    b = repr(rows).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _pick_summary(s):
    keys = ["app_injected", "delivered", "dropped", "pdr", "tx_attempts",
            "tx_success", "airtime_s", "not_scanning_events", "cca_busy_events",
            "overlap_events", "rx_total", "rx_ok", "rx_fail", "budget_deferrals"]
    return {k: s.get(k, None) for k in keys}


def test_minimal_run_repeatability():
    """Two identical runs must produce identical summaries and trace hashes."""
    cfg = make_easy_cfg()
    h = cfg_hash(cfg)

    injs1 = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))
    injs2 = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))

    r1 = _run_engine(cfg, 1, injs1, "flood")
    r2 = _run_engine(cfg, 1, injs2, "flood")

    assert _pick_summary(r1["summary"]) == _pick_summary(r2["summary"]), "Cell8 minimal summary drift"
    assert _trace_hash(r1["trace_head"]) == _trace_hash(r2["trace_head"]), "Cell8 trace drift"


def test_trace_hash_regression():
    """Phase 2 regression: trace hash for easy_cfg seed=1 flood must match locked value."""
    cfg = make_easy_cfg()
    h = cfg_hash(cfg)
    injs = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))
    r = _run_engine(cfg, 1, injs, "flood")
    computed = _trace_hash(r["trace_head"])
    assert computed == PHASE2_TRACE_HASH, (
        f"Trace hash regression failed.\n"
        f"  Expected: {PHASE2_TRACE_HASH}\n"
        f"  Got:      {computed}"
    )



def test_scan_duty_affects_not_scanning():
    """Lower scan duty must increase not_scanning_rx_opportunities."""
    import dataclasses
    cfg = make_easy_cfg()
    cfg_hi = dataclasses.replace(
        cfg,
        mac=dataclasses.replace(cfg.mac, scan_interval_s=2.0, scan_window_s=0.8),
    )
    cfg_lo = dataclasses.replace(
        cfg,
        mac=dataclasses.replace(cfg.mac, scan_interval_s=2.0, scan_window_s=0.2),
    )

    injs_scan = []
    fid = 0
    for t0 in [0.0, 0.5, 1.0, 1.5]:
        for src in range(1, min(6, int(cfg.n_nodes))):
            injs_scan.append(
                AppInjection(t=float(t0), flow_id=int(fid), src=int(src), dst=int(0), payload_bytes=80, ttl_hops=4)
            )
            fid += 1

    out_hi = _run_engine(cfg_hi, 1, list(injs_scan), "no_fwd")
    out_lo = _run_engine(cfg_lo, 1, list(injs_scan), "no_fwd")

    ns_hi = int(out_hi["summary"].get("not_scanning_rx_opportunities", 0))
    ns_lo = int(out_lo["summary"].get("not_scanning_rx_opportunities", 0))
    assert ns_lo > ns_hi, (
        f"Lower scan duty must increase not_scanning_rx_opportunities (hi={ns_hi}, lo={ns_lo})"
    )


def test_worse_sinr_increases_rx_fail():
    """Worse SINR (lower power, higher noise) must increase rx_fail."""
    import dataclasses
    cfg_good = make_easy_cfg()
    cfg_bad = dataclasses.replace(
        cfg_good,
        phy=dataclasses.replace(
            cfg_good.phy,
            tx_power_dbm=float(cfg_good.phy.tx_power_dbm) - 30.0,
            noise_floor_dbm=float(cfg_good.phy.noise_floor_dbm) + 20.0,
        ),
    )
    h0 = cfg_hash(cfg_good)
    injs0 = build_injections(cfg_good, SeedManager(cfg_hash=h0, seed=1))
    out_good = _run_engine(cfg_good, 1, injs0, "no_fwd")
    out_bad = _run_engine(cfg_bad, 1, injs0, "no_fwd")
    fg = int(out_good["summary"].get("rx_fail", 0))
    fb = int(out_bad["summary"].get("rx_fail", 0))
    assert fb > fg, "Gate C: worse SINR must increase rx_fail"
