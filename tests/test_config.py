"""
Phase 3 test: config hash stability.
Maps to Phase 2 requirement: Config hash stability (self_test_cell1, 27 assertions).
"""
import dataclasses
import pytest

from src.config import (
    CfgError,
    SimCfg,
    make_easy_cfg,
    make_moderate_cfg,
    make_stress_cfg,
    validate_cfg,
)
from src.hashing import (
    CFG_FLOAT_NDIGITS,
    INJECT_T_NDIGITS,
    MOBILITY_DIGEST_NDIGITS,
    canonical_json,
    cfg_hash,
)


def test_cfg_hash_stable_identical_configs():
    h1 = cfg_hash(make_easy_cfg())
    h2 = cfg_hash(make_easy_cfg())
    assert h1 == h2, "cfg_hash not stable for identical cfg"


def test_cfg_hash_changes_on_field_change():
    cfg1 = make_easy_cfg()
    h1 = cfg_hash(cfg1)
    cfg2 = dataclasses.replace(cfg1, sim_duration_s=cfg1.sim_duration_s + 1.0)
    h3 = cfg_hash(cfg2)
    assert h3 != h1, "cfg_hash should change when sim_duration_s changes"


def test_canonical_json_order_invariant():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert canonical_json(d1) == canonical_json(d2), "canonical_json not order-invariant"


def test_validate_cfg_rejects_n_nodes_1():
    cfg1 = make_easy_cfg()
    bad = dataclasses.replace(cfg1, n_nodes=1)
    with pytest.raises(CfgError):
        validate_cfg(bad)


def test_all_scenario_builders_validate():
    for builder in (make_easy_cfg, make_moderate_cfg, make_stress_cfg):
        validate_cfg(builder())


def test_constants_sanity():
    assert CFG_FLOAT_NDIGITS == 12
    assert MOBILITY_DIGEST_NDIGITS == 6
    assert INJECT_T_NDIGITS == 12


def test_mac_fields_present():
    cfg1 = make_easy_cfg()
    assert abs(float(cfg1.mac.t_ifs_s) - 0.000150) < 1e-15
    assert float(cfg1.mac.drift_ppm) >= 0.0


def test_phy_fading_fields_present():
    cfg1 = make_easy_cfg()
    assert str(cfg1.phy.fading_model) == "none"
    assert abs(float(cfg1.phy.rician_k_db) - 6.0) < 1e-12
    assert abs(float(cfg1.phy.shadow_sigma_db) - 0.0) < 1e-12
    assert bool(cfg1.phy.small_scale_fading) is False


def test_phy_fading_changes_cfg_hash():
    cfg1 = make_easy_cfg()
    h1 = cfg_hash(cfg1)
    cfg_phy2 = dataclasses.replace(cfg1, phy=dataclasses.replace(cfg1.phy, shadow_sigma_db=1.0))
    validate_cfg(cfg_phy2)
    assert cfg_hash(cfg_phy2) != h1, "cfg_hash should change when phy fading/shadowing fields change"


def test_airtime_budget_fields_present():
    cfg1 = make_easy_cfg()
    assert bool(cfg1.mac.airtime_budget_enabled) is False
    assert abs(float(cfg1.mac.airtime_budget_capacity_s) - 0.0) < 1e-15
    assert abs(float(cfg1.mac.airtime_budget_refill_sps) - 0.0) < 1e-15
    assert abs(float(cfg1.mac.airtime_budget_init_s) - 0.0) < 1e-15


def test_airtime_budget_changes_cfg_hash():
    cfg1 = make_easy_cfg()
    h1 = cfg_hash(cfg1)
    cfg_b = dataclasses.replace(
        cfg1,
        mac=dataclasses.replace(
            cfg1.mac,
            airtime_budget_enabled=True,
            airtime_budget_capacity_s=0.02,
            airtime_budget_refill_sps=0.001,
            airtime_budget_init_s=0.02,
        ),
    )
    validate_cfg(cfg_b)
    assert cfg_hash(cfg_b) != h1, "cfg_hash should change when mac airtime budget fields change"


def test_canonical_json_unsupported_type_raises():
    class _X:
        pass
    with pytest.raises(TypeError):
        canonical_json({"x": _X()})


# Phase 2 regression: Gate A summary hash requires stable cfg_hash for make_easy_cfg
GATE_A_EXPECTED_CFG_HASH = None  # not pinned — just verify it's a 64-char hex string

def test_cfg_hash_format():
    h = cfg_hash(make_easy_cfg())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
