"""
Phase 3 test: seed manager determinism.
Maps to Phase 2 requirement: Seed manager determinism (self_test_cell2, 9 assertions).
"""
import numpy as np
import pytest

from src.seeds import SeedManager, STD_STREAMS


def test_rng_deterministic_across_instances():
    sm1 = SeedManager(cfg_hash="abc", seed=1)
    sm2 = SeedManager(cfg_hash="abc", seed=1)
    r1 = sm1.rng("mobility").random(5)
    r2 = sm2.rng("mobility").random(5)
    assert np.allclose(r1, r2), "rng() not deterministic across instances"


def test_different_streams_differ():
    sm1 = SeedManager(cfg_hash="abc", seed=1)
    a = sm1.rng("mobility").random(3)
    b = sm1.rng("traffic").random(3)
    assert not np.allclose(a, b), "Different streams unexpectedly identical"


def test_different_seeds_differ():
    sm1 = SeedManager(cfg_hash="abc", seed=1)
    sm3 = SeedManager(cfg_hash="abc", seed=2)
    a = sm1.rng("mobility").random(3)
    c = sm3.rng("mobility").random(3)
    assert not np.allclose(a, c), "Different run seeds unexpectedly identical"


def test_spawn_provides_all_std_streams():
    sm1 = SeedManager(cfg_hash="abc", seed=1)
    d = sm1.spawn(STD_STREAMS)
    assert set(d.keys()) == set(STD_STREAMS), "spawn missing streams"


def test_seed_u64_in_range():
    sm1 = SeedManager(cfg_hash="abc", seed=1)
    u = sm1.seed_u64("core")
    assert 0 <= u < 2**64, "seed_u64 out of range"


def test_rng_cached_persistence():
    smc = SeedManager(cfg_hash="abc", seed=1)
    g1 = smc.rng_cached("mac")
    x1 = g1.integers(0, 10, size=5).tolist()
    g2 = smc.rng_cached("mac")
    x2 = g2.integers(0, 10, size=5).tolist()
    assert g1 is g2, "rng_cached should return same Generator instance per stream"


def test_rng_cached_vs_fresh():
    smc = SeedManager(cfg_hash="abc", seed=1)
    g1 = smc.rng_cached("mac")
    x1 = g1.integers(0, 10, size=5).tolist()
    x2 = g1.integers(0, 10, size=5).tolist()

    fresh = SeedManager(cfg_hash="abc", seed=1).rng("mac")
    f1 = fresh.integers(0, 10, size=5).tolist()
    f2 = fresh.integers(0, 10, size=5).tolist()
    assert x1 == f1, "First cached draws should match fresh stream origin"
    assert x2 == f2, "Second cached draws should match fresh stream continued sequence"


def test_guard_build_once_raises_on_second_call():
    smg = SeedManager(cfg_hash="abc", seed=1)
    smg.guard_build_once("mobility", "builderA")
    with pytest.raises(RuntimeError):
        smg.guard_build_once("mobility", "builderA")
