"""G4d tests for CI-based mode-level oracle-gap gating."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from src.counterfactual.g4d_oracle_gap import (
    COMPARISON_LEVEL,
    CONTROLLED_ORACLE_EXPERT,
    DEFAULT_SEEDS,
    EPSILON_GAP,
    EPSILON_PDR,
    EPSILON_TX_REDUCTION,
    G4dScenarioSummary,
    evaluate_g4d_verdict,
    run_g4d_probe,
    summarize_by_scenario,
    write_g4d_outputs,
)
from src.counterfactual.mode_comparison import (
    G4C_LIMITATIONS,
    ModeArmRun,
    ModeComparisonResult,
)
from src.config import make_easy_cfg


def _hash(char: str) -> str:
    return char * 64


def _arm(
    *,
    seed: int,
    policy_name: str,
    pdr: float,
    tx_attempts: int,
    delivered_count: int,
) -> ModeArmRun:
    return ModeArmRun(
        cfg_hash="cfg-hash",
        seed=seed,
        policy_name=policy_name,
        summary={
            "app_injected": 10,
            "delivered": delivered_count,
            "pdr": pdr,
            "tx_attempts": tx_attempts,
            "airtime_s": float(tx_attempts) / 1000.0,
        },
        summary_hash=_hash("a" if policy_name == "flood" else "b"),
        decision_log_hash=_hash("c" if policy_name == "flood" else "d"),
        decision_rows_hash=_hash("e" if policy_name == "flood" else "f"),
        injected_count=10,
        delivered_count=delivered_count,
        pdr=pdr,
        tx_attempts=tx_attempts,
        airtime_proxy_s=float(tx_attempts) / 1000.0,
        mean_delay_s=0.1,
        mobility_digest="mobility",
        traffic_digest="traffic",
        mac_drift_factors=((0, 1.0),),
        mac_drift_hash="mac",
    )


def _mode_result(
    *,
    scenario_name: str,
    seed: int,
    pdr_flood: float,
    pdr_controlled: float,
    tx_flood: int,
    tx_controlled: int,
) -> ModeComparisonResult:
    arm_a = _arm(
        seed=seed,
        policy_name="flood",
        pdr=pdr_flood,
        tx_attempts=tx_flood,
        delivered_count=round(pdr_flood * 10),
    )
    arm_b = _arm(
        seed=seed,
        policy_name="controlled_retransmission_expert",
        pdr=pdr_controlled,
        tx_attempts=tx_controlled,
        delivered_count=round(pdr_controlled * 10),
    )
    return ModeComparisonResult(
        scenario_name=scenario_name,
        seed=seed,
        arm_a_policy=arm_a.policy_name,
        arm_b_policy=arm_b.policy_name,
        arm_a_injected_count=arm_a.injected_count,
        arm_b_injected_count=arm_b.injected_count,
        arm_a_delivered_count=arm_a.delivered_count,
        arm_b_delivered_count=arm_b.delivered_count,
        arm_a_pdr=arm_a.pdr,
        arm_b_pdr=arm_b.pdr,
        delta_pdr=arm_b.pdr - arm_a.pdr,
        arm_a_tx_attempts=arm_a.tx_attempts,
        arm_b_tx_attempts=arm_b.tx_attempts,
        delta_tx_attempts=arm_b.tx_attempts - arm_a.tx_attempts,
        arm_a_airtime_proxy_s=arm_a.airtime_proxy_s,
        arm_b_airtime_proxy_s=arm_b.airtime_proxy_s,
        delta_airtime_proxy_s=arm_b.airtime_proxy_s - arm_a.airtime_proxy_s,
        arm_a_mean_delay_s=arm_a.mean_delay_s,
        arm_b_mean_delay_s=arm_b.mean_delay_s,
        delta_delay_s=0.0,
        arm_a_decision_log_hash=arm_a.decision_log_hash,
        arm_b_decision_log_hash=arm_b.decision_log_hash,
        arm_a_summary_hash=arm_a.summary_hash,
        arm_b_summary_hash=arm_b.summary_hash,
        mobility_digest_match_status="match",
        traffic_digest_match_status="match",
        mac_drift_init_match_status="match",
        arm_a_mobility_digest=arm_a.mobility_digest,
        arm_b_mobility_digest=arm_b.mobility_digest,
        arm_a_traffic_digest=arm_a.traffic_digest,
        arm_b_traffic_digest=arm_b.traffic_digest,
        arm_a_mac_drift_hash=arm_a.mac_drift_hash,
        arm_b_mac_drift_hash=arm_b.mac_drift_hash,
        notes_limitations=G4C_LIMITATIONS,
        arm_a=arm_a,
        arm_b=arm_b,
    )


def _summary(
    *,
    scenario_name: str,
    pdr_lower: float,
    tx_upper: float,
    gap_lower: float,
) -> G4dScenarioSummary:
    pdr_ok = pdr_lower >= -EPSILON_PDR
    tx_ok = tx_upper <= -EPSILON_TX_REDUCTION
    gap_ok = gap_lower > EPSILON_GAP
    return G4dScenarioSummary(
        scenario_name=scenario_name,
        n_seeds=10,
        mean_pdr_flood=0.9,
        mean_pdr_controlled=0.9,
        mean_delta_pdr=0.0,
        ci95_delta_pdr_lower=pdr_lower,
        ci95_delta_pdr_upper=0.01,
        mean_tx_attempts_flood=100.0,
        mean_tx_attempts_controlled=80.0,
        mean_relative_delta_tx_attempts=-0.2,
        ci95_relative_delta_tx_attempts_lower=-0.3,
        ci95_relative_delta_tx_attempts_upper=tx_upper,
        oracle_gap_score=1.0,
        ci95_oracle_gap_score_lower=gap_lower,
        ci95_oracle_gap_score_upper=1.0,
        oracle_gap_successes=10,
        mobility_pairing_status="all_match",
        traffic_pairing_status="all_match",
        mac_drift_init_pairing_status="all_match",
        pairing_checks_passed=True,
        pdr_ci_condition_passed=pdr_ok,
        tx_ci_condition_passed=tx_ok,
        oracle_gap_ci_condition_passed=gap_ok,
        same_scenario_positive=pdr_ok and tx_ok and gap_ok,
    )


def test_g4d_uses_mode_level_comparison_only() -> None:
    calls: List[tuple[str, int]] = []

    def fake_compare(_cfg: Any, seed: int, *, scenario_name: str) -> ModeComparisonResult:
        calls.append((scenario_name, seed))
        return _mode_result(
            scenario_name=scenario_name,
            seed=seed,
            pdr_flood=1.0,
            pdr_controlled=1.0,
            tx_flood=100,
            tx_controlled=80,
        )

    rows, summaries, verdict = run_g4d_probe(
        {"easy": make_easy_cfg()},
        seeds=(1, 2),
        comparison_func=fake_compare,
    )

    assert calls == [("easy", 1), ("easy", 2)]
    assert {row.comparison_level for row in rows} == {COMPARISON_LEVEL}
    assert {row.mode_comparison_result_type for row in rows} == {"ModeComparisonResult"}
    assert all(row.oracle_expert == CONTROLLED_ORACLE_EXPERT for row in rows)
    assert len(summaries) == 1
    assert verdict.verdict == "NULL/NEGATIVE"


def test_g4d_same_scenario_positive_conjunction() -> None:
    summaries = [
        _summary(scenario_name="safe_only", pdr_lower=-0.01, tx_upper=0.10, gap_lower=0.20),
        _summary(scenario_name="cost_gap_only", pdr_lower=-0.03, tx_upper=-0.10, gap_lower=0.20),
    ]

    verdict = evaluate_g4d_verdict(summaries)

    assert verdict.verdict == "NULL/NEGATIVE"
    assert not verdict.g5_authorized

    positive = _summary(scenario_name="same_scenario", pdr_lower=-0.01, tx_upper=-0.10, gap_lower=0.20)
    positive_verdict = evaluate_g4d_verdict([positive])

    assert positive_verdict.verdict == "POSITIVE"
    assert positive_verdict.g5_authorized
    assert positive_verdict.positive_scenarios == ("same_scenario",)


def test_g4d_ci_based_not_point_estimate() -> None:
    point_estimates_look_good = _summary(
        scenario_name="wide_ci",
        pdr_lower=-0.021,
        tx_upper=-0.10,
        gap_lower=0.20,
    )
    assert point_estimates_look_good.mean_delta_pdr >= -EPSILON_PDR
    assert point_estimates_look_good.mean_relative_delta_tx_attempts <= -EPSILON_TX_REDUCTION
    assert point_estimates_look_good.oracle_gap_score > EPSILON_GAP

    verdict = evaluate_g4d_verdict([point_estimates_look_good])

    assert verdict.verdict == "NULL/NEGATIVE"
    assert not verdict.g5_authorized


def test_g4d_no_selector_artifacts() -> None:
    module_source = Path("src/counterfactual/g4d_oracle_gap.py").read_text(encoding="utf-8")
    forbidden_identifiers = (
        "class ExpertSelector",
        "def train_",
        "def fit_",
        "ml_model",
        "rl_policy",
        "drl_policy",
        "gnn_model",
    )

    for identifier in forbidden_identifiers:
        assert identifier not in module_source


def test_g4d_no_loro_or_forced_action() -> None:
    module_source = Path("src/counterfactual/g4d_oracle_gap.py").read_text(encoding="utf-8").lower()

    assert "forced_action" not in module_source
    assert "forcedoverride" not in module_source
    assert "single-decision" not in module_source
    assert "loro" not in module_source


def test_g4d_outputs_required_tables(tmp_path: Path) -> None:
    synthetic_rows = []
    for seed in DEFAULT_SEEDS:
        synthetic_rows.append(
            _mode_result(
                scenario_name="easy",
                seed=seed,
                pdr_flood=1.0,
                pdr_controlled=1.0,
                tx_flood=100,
                tx_controlled=80,
            )
        )
    rows = [
        row
        for row in run_g4d_probe(
            {"easy": make_easy_cfg()},
            seeds=DEFAULT_SEEDS,
            comparison_func=lambda _cfg, seed, *, scenario_name: synthetic_rows[seed - 1],
        )[0]
    ]
    summaries = summarize_by_scenario(rows)
    verdict = evaluate_g4d_verdict(summaries)

    report_path, results_path, summary_path = write_g4d_outputs(
        rows,
        summaries,
        verdict,
        docs_dir=tmp_path / "docs",
        outputs_dir=tmp_path / "outputs",
    )

    assert report_path.name == "g4d_oracle_gap_report.md"
    assert results_path.name == "g4d_mode_comparison_results.csv"
    assert summary_path.name == "g4d_oracle_gap_summary.csv"
    assert report_path.exists()
    assert results_path.exists()
    assert summary_path.exists()
    assert "Verdict:" in report_path.read_text(encoding="utf-8")
    assert "relative_delta_tx_attempts" in results_path.read_text(encoding="utf-8").splitlines()[0]
    assert "same_scenario_positive" in summary_path.read_text(encoding="utf-8").splitlines()[0]
