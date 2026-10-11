"""G4c tests for mode-level paired comparison artifacts."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from scenario_configs.phase5_scenarios import get_all_scenarios

from src.config import SimCfg, make_easy_cfg
from src.counterfactual.mode_comparison import (
    ARM_A_POLICY_NAME,
    ARM_B_POLICY_NAME,
    ModeComparisonResult,
    compare_flood_vs_controlled_modes,
)


G4C_SEED = 1

EXPECTED_SCENARIOS: Tuple[str, ...] = (
    "easy",
    "p5_baseline_normal",
    "p5_congestion_burst",
    "p5_low_power_survival",
    "p5_interference_zone",
    "p5_partition_reconnect",
    "p5_node_failure",
    "p5_group_mobility",
    "p5_priority_emergency",
)

REQUIRED_RESULT_FIELDS = {
    "scenario_name",
    "seed",
    "arm_a_policy",
    "arm_b_policy",
    "arm_a_injected_count",
    "arm_b_injected_count",
    "arm_a_delivered_count",
    "arm_b_delivered_count",
    "arm_a_pdr",
    "arm_b_pdr",
    "delta_pdr",
    "arm_a_tx_attempts",
    "arm_b_tx_attempts",
    "delta_tx_attempts",
    "arm_a_airtime_proxy_s",
    "arm_b_airtime_proxy_s",
    "delta_airtime_proxy_s",
    "arm_a_mean_delay_s",
    "arm_b_mean_delay_s",
    "delta_delay_s",
    "arm_a_decision_log_hash",
    "arm_b_decision_log_hash",
    "arm_a_summary_hash",
    "arm_b_summary_hash",
    "mobility_digest_match_status",
    "traffic_digest_match_status",
    "mac_drift_init_match_status",
    "notes_limitations",
}


def _scenario_cases() -> List[Tuple[str, SimCfg]]:
    cases: List[Tuple[str, SimCfg]] = [("easy", make_easy_cfg())]
    cases.extend(get_all_scenarios().items())
    return cases


SCENARIO_CASES = _scenario_cases()
SCENARIO_BY_NAME: Dict[str, SimCfg] = dict(SCENARIO_CASES)
SCENARIO_NAMES = tuple(name for name, _ in SCENARIO_CASES)

assert SCENARIO_NAMES == EXPECTED_SCENARIOS


@lru_cache(maxsize=None)
def _comparison_for(scenario_name: str) -> ModeComparisonResult:
    return compare_flood_vs_controlled_modes(
        SCENARIO_BY_NAME[scenario_name],
        G4C_SEED,
        scenario_name=scenario_name,
    )


def _all_keys(value: Any) -> List[str]:
    if isinstance(value, dict):
        keys: List[str] = []
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(item))
        return keys
    if isinstance(value, (list, tuple)):
        keys = []
        for item in value:
            keys.extend(_all_keys(item))
        return keys
    return []


def test_g4c_mode_comparison_result_schema() -> None:
    result = _comparison_for("easy")
    payload = result.to_dict()

    assert isinstance(result, ModeComparisonResult)
    assert REQUIRED_RESULT_FIELDS.issubset(payload.keys())
    assert result.scenario_name == "easy"
    assert result.seed == G4C_SEED
    assert result.notes_limitations


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g4c_runs_flood_and_controlled_arms(scenario_name: str) -> None:
    result = _comparison_for(scenario_name)

    assert result.arm_a_policy == ARM_A_POLICY_NAME
    assert result.arm_b_policy == ARM_B_POLICY_NAME
    assert result.arm_a_injected_count > 0
    assert result.arm_b_injected_count > 0
    assert result.arm_a.policy_name == ARM_A_POLICY_NAME
    assert result.arm_b.policy_name == ARM_B_POLICY_NAME


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g4c_same_seed_and_scenario_pairing(scenario_name: str) -> None:
    result = _comparison_for(scenario_name)

    assert result.scenario_name == scenario_name
    assert result.seed == G4C_SEED
    assert result.arm_a.seed == G4C_SEED
    assert result.arm_b.seed == G4C_SEED
    assert result.arm_a.cfg_hash == result.arm_b.cfg_hash
    assert result.arm_a_injected_count == result.arm_b_injected_count


def test_g4c_records_delta_pdr_and_delta_tx_attempts() -> None:
    result = _comparison_for("p5_congestion_burst")

    assert result.delta_pdr == pytest.approx(result.arm_b_pdr - result.arm_a_pdr)
    assert result.delta_tx_attempts == result.arm_b_tx_attempts - result.arm_a_tx_attempts
    assert result.delta_tx_attempts != 0
    if result.delta_airtime_proxy_s is not None:
        assert result.delta_airtime_proxy_s == pytest.approx(
            result.arm_b_airtime_proxy_s - result.arm_a_airtime_proxy_s  # type: ignore[operator]
        )


def test_g4c_records_hashes() -> None:
    result = _comparison_for("easy")

    assert len(result.arm_a_decision_log_hash) == 64
    assert len(result.arm_b_decision_log_hash) == 64
    assert len(result.arm_a_summary_hash) == 64
    assert len(result.arm_b_summary_hash) == 64
    assert result.arm_a_decision_log_hash == result.arm_a.decision_log_hash
    assert result.arm_b_summary_hash == result.arm_b.summary_hash


def test_g4c_records_pairing_digests_when_accessible() -> None:
    result = _comparison_for("easy")

    assert result.mobility_digest_match_status == "match"
    assert result.traffic_digest_match_status == "match"
    assert result.mac_drift_init_match_status == "match"
    assert result.arm_a_mobility_digest == result.arm_b_mobility_digest
    assert result.arm_a_traffic_digest == result.arm_b_traffic_digest
    assert result.arm_a_mac_drift_hash == result.arm_b_mac_drift_hash


def test_g4c_does_not_require_byte_exact_mac_phy_after_policy_divergence() -> None:
    result = _comparison_for("p5_congestion_burst")
    payload = result.to_dict()

    assert result.arm_a_decision_log_hash != result.arm_b_decision_log_hash
    assert result.arm_a_summary_hash != result.arm_b_summary_hash
    assert result.pairing_checks == {
        "mobility_digest_match_status": "match",
        "traffic_digest_match_status": "match",
        "mac_drift_init_match_status": "match",
    }
    assert "runtime_mac_phy_match_status" not in payload
    assert "trace_identity_required" not in payload


def test_g4c_no_selector_or_ml_artifacts() -> None:
    result = _comparison_for("easy")
    forbidden_names = {
        "selector",
        "selector_eligibility",
        "ml_model",
        "rl_policy",
        "drl_policy",
        "gnn_model",
        "training_table",
    }

    for key in _all_keys(result.to_dict()):
        key_l = key.lower()
        assert key_l not in forbidden_names
        assert not key_l.startswith(("selector_", "ml_", "rl_", "drl_", "gnn_"))


def test_g4c_no_oracle_gap_verdict() -> None:
    result = _comparison_for("easy")
    keys = {key.lower() for key in _all_keys(result.to_dict())}

    assert "oracle_gap_verdict" not in keys
    assert "oracle_gap_success" not in keys
    assert "oracle_gap_failure" not in keys
    assert "selector_eligibility" not in keys
