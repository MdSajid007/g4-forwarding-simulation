"""
Phase 3 test: mobility determinism.
Maps to Phase 2 requirement: Mobility determinism (self_test_cell5, 5 assertions).
"""
import pytest

from src.config import make_easy_cfg
from src.hashing import cfg_hash
from src.mobility import build_rwp_plan
from src.seeds import SeedManager


def test_mobility_reproducible_same_seed():
    cfg = make_easy_cfg()
    sm1 = SeedManager(cfg_hash=cfg_hash(cfg), seed=1)
    sm2 = SeedManager(cfg_hash=cfg_hash(cfg), seed=1)

    p1 = build_rwp_plan(cfg, sm1)
    p2 = build_rwp_plan(cfg, sm2)

    d1 = p1.digest_first_segments(k=3)
    d2 = p2.digest_first_segments(k=3)
    assert d1 == d2, "Mobility plan not reproducible for same seed"


def test_mobility_differs_different_seeds():
    cfg = make_easy_cfg()
    sm1 = SeedManager(cfg_hash=cfg_hash(cfg), seed=1)
    sm3 = SeedManager(cfg_hash=cfg_hash(cfg), seed=2)

    p1 = build_rwp_plan(cfg, sm1)
    p3 = build_rwp_plan(cfg, sm3)

    d1 = p1.digest_first_segments(k=3)
    d3 = p3.digest_first_segments(k=3)
    assert d1 != d3, "Mobility plan unexpectedly identical across different seeds"


def test_mobility_positions_in_bounds():
    cfg = make_easy_cfg()
    sm = SeedManager(cfg_hash=cfg_hash(cfg), seed=1)
    p1 = build_rwp_plan(cfg, sm)

    for nid in range(cfg.n_nodes):
        x, y = p1.pos_at(nid, t=cfg.sim_duration_s * 0.5)
        assert 0.0 <= x <= cfg.mobility.area_w + 1e-6
        assert 0.0 <= y <= cfg.mobility.area_h + 1e-6


def test_mobility_segments_time_monotone():
    cfg = make_easy_cfg()
    sm = SeedManager(cfg_hash=cfg_hash(cfg), seed=1)
    p1 = build_rwp_plan(cfg, sm)

    for nid, segs in p1.segments_by_node.items():
        for i in range(1, len(segs)):
            assert segs[i].t0 >= segs[i - 1].t0, f"Non-monotonic t0 for node {nid}"
            assert segs[i].t1 >= segs[i].t0, f"Invalid segment duration for node {nid}"


def test_mobility_build_once_guard():
    cfg = make_easy_cfg()
    smg = SeedManager(cfg_hash=cfg_hash(cfg), seed=1)
    _ = build_rwp_plan(cfg, smg)
    with pytest.raises(RuntimeError):
        _ = build_rwp_plan(cfg, smg)
