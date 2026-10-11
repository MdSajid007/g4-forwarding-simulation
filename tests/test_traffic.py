"""
Phase 3 test: traffic injection determinism.
Maps to Phase 2 requirement: Traffic injection determinism (self_test_cell6, 6 assertions).

Phase 2 regression hash:
  Injection hash: e393e8a7c7b5a26b2f4a514948fe9f4b4944ba62a63df3064b9b2e9613fac28b
"""
import pytest

from src.config import make_easy_cfg, make_moderate_cfg
from src.hashing import cfg_hash
from src.scheduler import EventScheduler
from src.seeds import SeedManager
from src.traffic import build_injections, injections_hash, schedule_injections

PHASE2_INJECTION_HASH = "e393e8a7c7b5a26b2f4a514948fe9f4b4944ba62a63df3064b9b2e9613fac28b"


def test_cbr_injections_deterministic():
    cfg = make_easy_cfg()
    h = cfg_hash(cfg)
    inj1 = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))
    inj2 = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))
    assert [(i.t, i.flow_id, i.src, i.dst) for i in inj1] == [(i.t, i.flow_id, i.src, i.dst) for i in inj2], \
        "Injections not deterministic for same seed"
    assert injections_hash(inj1) == injections_hash(inj2), "injections_hash not deterministic for same seed"


def test_cbr_injections_seed_independent():
    cfg = make_easy_cfg()
    h = cfg_hash(cfg)
    inj1 = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))
    inj3 = build_injections(cfg, SeedManager(cfg_hash=h, seed=2))
    assert [(i.t, i.flow_id) for i in inj1] == [(i.t, i.flow_id) for i in inj3], \
        "CBR should be seed-independent but differed"
    assert injections_hash(inj1) == injections_hash(inj3), "CBR injections_hash should be seed-independent but differed"


def test_poisson_injections_seed_dependent():
    cfgm = make_moderate_cfg()
    hm = cfg_hash(cfgm)
    p1 = build_injections(cfgm, SeedManager(cfg_hash=hm, seed=1))
    p2 = build_injections(cfgm, SeedManager(cfg_hash=hm, seed=2))
    assert [(i.t, i.flow_id) for i in p1[:10]] != [(i.t, i.flow_id) for i in p2[:10]], \
        "Poisson injections unexpectedly identical across seeds"
    assert injections_hash(p1) != injections_hash(p2), "Poisson injections_hash unexpectedly identical across seeds"


def test_schedule_injections_count():
    cfg = make_easy_cfg()
    h = cfg_hash(cfg)
    inj1 = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))
    sch = EventScheduler()
    schedule_injections(sch, inj1)
    assert len(sch) == len(inj1), "Scheduler injection count mismatch"
    types = sch.drain_types()
    assert all(t == "APP_INJECT" for t in types), "Non APP_INJECT in scheduled traffic"


def test_build_injections_build_once_guard():
    cfg = make_easy_cfg()
    h = cfg_hash(cfg)
    sm = SeedManager(cfg_hash=h, seed=1)
    _ = build_injections(cfg, sm)
    with pytest.raises(RuntimeError):
        _ = build_injections(cfg, sm)


def test_injection_hash_regression():
    """Phase 2 regression: inj_hash for moderate_cfg seed=1 must match locked value.
    Note: Phase 2 self_test_cell6 minimal_run used make_moderate_cfg().
    """
    from src.config import make_moderate_cfg
    cfg = make_moderate_cfg()
    h = cfg_hash(cfg)
    inj = build_injections(cfg, SeedManager(cfg_hash=h, seed=1))
    computed = injections_hash(inj)
    assert computed == PHASE2_INJECTION_HASH, (
        f"Injection hash regression failed.\n"
        f"  Expected: {PHASE2_INJECTION_HASH}\n"
        f"  Got:      {computed}"
    )
