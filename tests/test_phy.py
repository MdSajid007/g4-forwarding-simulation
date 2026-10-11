"""
Phase 3 test: PHY/PER monotonicity.
Maps to Phase 2 requirement: PHY/PER monotonicity (self_test_cell7, 5 assertions).
"""
import numpy as np
import pytest

from src.config import make_easy_cfg
from src.phy import PhyModel


def test_per_monotone_decreasing_with_sinr():
    cfg = make_easy_cfg()
    phy = PhyModel(cfg)
    sins = [-20.0, -10.0, 0.0, 6.0, 12.0, 20.0]
    pers = [phy.per_from_sinr(s) for s in sins]
    assert all(pers[i] >= pers[i + 1] for i in range(len(pers) - 1)), "PER not monotone decreasing with SINR"


def test_rssi_decreases_with_distance():
    cfg = make_easy_cfg()
    phy = PhyModel(cfg)
    txp = (1.0, 1.0)
    rx_near = (1.5, 1.0)
    rx_far = (50.0, 50.0)
    r_near = phy.rssi_dbm(cfg.phy.tx_power_dbm, txp, rx_near, rng=None)
    r_far = phy.rssi_dbm(cfg.phy.tx_power_dbm, txp, rx_far, rng=None)
    assert r_near > r_far, "RSSI not decreasing with distance"


def test_sinr_decreases_with_interference():
    cfg = make_easy_cfg()
    phy = PhyModel(cfg)
    sig = -60.0
    s0 = phy.sinr_db(sig, [], cfg.phy.noise_floor_dbm)
    s1 = phy.sinr_db(sig, [-70.0], cfg.phy.noise_floor_dbm)
    assert s0 > s1, "SINR did not decrease with added interference"


def test_fading_deterministic_when_enabled():
    cfg = make_easy_cfg()
    phy = PhyModel(cfg)
    txp = (1.0, 1.0)
    rx_near = (1.5, 1.0)

    enabled = (
        str(getattr(cfg.phy, "fading_model", "none")).strip().lower() not in ("none", "", "off", "disabled")
        or bool(getattr(cfg.phy, "small_scale_fading", False))
        or float(getattr(cfg.phy, "shadow_sigma_db", 0.0)) > 0.0
    )
    if enabled:
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        a1 = phy.rssi_dbm(cfg.phy.tx_power_dbm, txp, rx_near, rng=rng1)
        a2 = phy.rssi_dbm(cfg.phy.tx_power_dbm, txp, rx_near, rng=rng2)
        assert abs(a1 - a2) < 1e-12, "Fading/shadowing not reproducible for same RNG seed/order"
    else:
        pytest.skip("Fading not enabled in easy_cfg (expected — pathloss-only mode)")
