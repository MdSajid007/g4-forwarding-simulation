"""G4d mode-level oracle-gap probe.

This module intentionally sits above the G4c paired-mode harness. It consumes
``ModeComparisonResult`` values, adds the preregistered G4d CI logic, and writes
the report/tables requested for the oracle-gap gate. It does not implement any
selector, per-decision intervention, or policy override path.
"""
from __future__ import annotations

import csv
import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import SimCfg, make_easy_cfg
from .mode_comparison import ModeComparisonResult, compare_flood_vs_controlled_modes


EPSILON_PDR = 0.02
EPSILON_TX_REDUCTION = 0.05
EPSILON_GAP = 0.05

DEFAULT_SEEDS: Tuple[int, ...] = tuple(range(1, 11))
SCENARIO_NAMES: Tuple[str, ...] = (
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

FLOOD_ORACLE_EXPERT = "flood"
CONTROLLED_ORACLE_EXPERT = "controlled_retransmission"
COMPARISON_LEVEL = "mode"
MODE_COMPARISON_RESULT_NAME = "ModeComparisonResult"

RESULT_COLUMNS: Tuple[str, ...] = (
    "scenario_name",
    "seed",
    "cfg_hash",
    "comparison_level",
    "mode_comparison_result_type",
    "arm_a_policy",
    "arm_b_policy",
    "pdr_flood",
    "pdr_controlled",
    "delta_pdr",
    "tx_attempts_flood",
    "tx_attempts_controlled",
    "delta_tx_attempts",
    "relative_delta_tx_attempts",
    "airtime_proxy_s_flood",
    "airtime_proxy_s_controlled",
    "delta_airtime_proxy_s",
    "mean_delay_s_flood",
    "mean_delay_s_controlled",
    "delta_delay_s",
    "jointly_safe",
    "jointly_cheaper",
    "oracle_expert",
    "oracle_gap_score",
    "mobility_digest_match_status",
    "traffic_digest_match_status",
    "mac_drift_init_match_status",
    "pairing_checks_passed",
    "mobility_digest",
    "traffic_digest",
    "mac_drift_hash",
    "arm_a_decision_log_hash",
    "arm_b_decision_log_hash",
    "arm_a_summary_hash",
    "arm_b_summary_hash",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "scenario_name",
    "n_seeds",
    "mean_pdr_flood",
    "mean_pdr_controlled",
    "mean_delta_pdr",
    "ci95_delta_pdr_lower",
    "ci95_delta_pdr_upper",
    "mean_tx_attempts_flood",
    "mean_tx_attempts_controlled",
    "mean_relative_delta_tx_attempts",
    "ci95_relative_delta_tx_attempts_lower",
    "ci95_relative_delta_tx_attempts_upper",
    "oracle_gap_score",
    "ci95_oracle_gap_score_lower",
    "ci95_oracle_gap_score_upper",
    "oracle_gap_successes",
    "mobility_pairing_status",
    "traffic_pairing_status",
    "mac_drift_init_pairing_status",
    "pairing_checks_passed",
    "pdr_ci_condition_passed",
    "tx_ci_condition_passed",
    "oracle_gap_ci_condition_passed",
    "same_scenario_positive",
)


@dataclass(frozen=True)
class MeanCi:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class G4dPairedResult:
    """One G4d paired seed row derived from a G4c ModeComparisonResult."""

    scenario_name: str
    seed: int
    cfg_hash: str
    comparison_level: str
    mode_comparison_result_type: str
    arm_a_policy: str
    arm_b_policy: str
    pdr_flood: float
    pdr_controlled: float
    delta_pdr: float
    tx_attempts_flood: int
    tx_attempts_controlled: int
    delta_tx_attempts: int
    relative_delta_tx_attempts: float
    airtime_proxy_s_flood: Optional[float]
    airtime_proxy_s_controlled: Optional[float]
    delta_airtime_proxy_s: Optional[float]
    mean_delay_s_flood: Optional[float]
    mean_delay_s_controlled: Optional[float]
    delta_delay_s: Optional[float]
    jointly_safe: bool
    jointly_cheaper: bool
    oracle_expert: str
    oracle_gap_score: float
    mobility_digest_match_status: str
    traffic_digest_match_status: str
    mac_drift_init_match_status: str
    pairing_checks_passed: bool
    mobility_digest: str
    traffic_digest: str
    mac_drift_hash: str
    arm_a_decision_log_hash: str
    arm_b_decision_log_hash: str
    arm_a_summary_hash: str
    arm_b_summary_hash: str

    @classmethod
    def from_mode_comparison(cls, result: ModeComparisonResult) -> "G4dPairedResult":
        tx_flood = int(result.arm_a_tx_attempts)
        tx_controlled = int(result.arm_b_tx_attempts)
        relative_delta_tx = float(tx_controlled - tx_flood) / float(max(tx_flood, 1))
        jointly_safe = float(result.delta_pdr) >= -EPSILON_PDR
        jointly_cheaper = relative_delta_tx <= -EPSILON_TX_REDUCTION

        controlled_utility = -math.inf
        if jointly_safe:
            controlled_utility = -relative_delta_tx
        flood_utility = 0.0
        oracle_expert = (
            CONTROLLED_ORACLE_EXPERT
            if controlled_utility > flood_utility
            else FLOOD_ORACLE_EXPERT
        )
        oracle_gap_score = 1.0 if oracle_expert == CONTROLLED_ORACLE_EXPERT else 0.0

        pairing_checks_passed = (
            result.mobility_digest_match_status == "match"
            and result.traffic_digest_match_status == "match"
            and result.mac_drift_init_match_status == "match"
        )

        return cls(
            scenario_name=str(result.scenario_name),
            seed=int(result.seed),
            cfg_hash=str(result.arm_a.cfg_hash),
            comparison_level=COMPARISON_LEVEL,
            mode_comparison_result_type=type(result).__name__,
            arm_a_policy=str(result.arm_a_policy),
            arm_b_policy=str(result.arm_b_policy),
            pdr_flood=float(result.arm_a_pdr),
            pdr_controlled=float(result.arm_b_pdr),
            delta_pdr=float(result.delta_pdr),
            tx_attempts_flood=tx_flood,
            tx_attempts_controlled=tx_controlled,
            delta_tx_attempts=int(result.delta_tx_attempts),
            relative_delta_tx_attempts=float(relative_delta_tx),
            airtime_proxy_s_flood=result.arm_a_airtime_proxy_s,
            airtime_proxy_s_controlled=result.arm_b_airtime_proxy_s,
            delta_airtime_proxy_s=result.delta_airtime_proxy_s,
            mean_delay_s_flood=result.arm_a_mean_delay_s,
            mean_delay_s_controlled=result.arm_b_mean_delay_s,
            delta_delay_s=result.delta_delay_s,
            jointly_safe=bool(jointly_safe),
            jointly_cheaper=bool(jointly_cheaper),
            oracle_expert=oracle_expert,
            oracle_gap_score=float(oracle_gap_score),
            mobility_digest_match_status=str(result.mobility_digest_match_status),
            traffic_digest_match_status=str(result.traffic_digest_match_status),
            mac_drift_init_match_status=str(result.mac_drift_init_match_status),
            pairing_checks_passed=bool(pairing_checks_passed),
            mobility_digest=str(result.arm_a_mobility_digest),
            traffic_digest=str(result.arm_a_traffic_digest),
            mac_drift_hash=str(result.arm_a_mac_drift_hash),
            arm_a_decision_log_hash=str(result.arm_a_decision_log_hash),
            arm_b_decision_log_hash=str(result.arm_b_decision_log_hash),
            arm_a_summary_hash=str(result.arm_a_summary_hash),
            arm_b_summary_hash=str(result.arm_b_summary_hash),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class G4dScenarioSummary:
    """Per-scenario G4d CI summary and same-scenario gate decision."""

    scenario_name: str
    n_seeds: int
    mean_pdr_flood: float
    mean_pdr_controlled: float
    mean_delta_pdr: float
    ci95_delta_pdr_lower: float
    ci95_delta_pdr_upper: float
    mean_tx_attempts_flood: float
    mean_tx_attempts_controlled: float
    mean_relative_delta_tx_attempts: float
    ci95_relative_delta_tx_attempts_lower: float
    ci95_relative_delta_tx_attempts_upper: float
    oracle_gap_score: float
    ci95_oracle_gap_score_lower: float
    ci95_oracle_gap_score_upper: float
    oracle_gap_successes: int
    mobility_pairing_status: str
    traffic_pairing_status: str
    mac_drift_init_pairing_status: str
    pairing_checks_passed: bool
    pdr_ci_condition_passed: bool
    tx_ci_condition_passed: bool
    oracle_gap_ci_condition_passed: bool
    same_scenario_positive: bool

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class G4dVerdict:
    verdict: str
    g5_authorized: bool
    positive_scenarios: Tuple[str, ...]

    @property
    def g5_status(self) -> str:
        return "AUTHORIZED" if self.g5_authorized else "BLOCKED"


def load_g4d_scenarios() -> Dict[str, SimCfg]:
    """Load the canonical G4d fixtures in the requested order."""
    project_root = Path(__file__).resolve().parents[2]
    experiments_dir = project_root / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))

    from scenario_configs.phase5_scenarios import get_all_scenarios

    available: Dict[str, SimCfg] = {"easy": make_easy_cfg()}
    available.update(get_all_scenarios())
    missing = [name for name in SCENARIO_NAMES if name not in available]
    if missing:
        raise KeyError(f"Missing G4d scenario fixtures: {missing}")
    return {name: available[name] for name in SCENARIO_NAMES}


def _t_critical_95(df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, int(df)))
    except Exception:
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
            25: 2.060,
            30: 2.042,
        }
        if df in table:
            return table[df]
        if df < 30:
            nearest = min((k for k in table if k >= df), default=30)
            return table[nearest]
        return 1.96


def mean_ci_t(values: Sequence[float]) -> MeanCi:
    """Two-sided 95% CI for a paired-sample mean using the t distribution."""
    samples = [float(v) for v in values]
    if len(samples) < 2:
        raise ValueError("At least two paired samples are required for a t CI")
    sample_mean = float(mean(samples))
    sample_sd = float(stdev(samples))
    if sample_sd == 0.0:
        return MeanCi(mean=sample_mean, lower=sample_mean, upper=sample_mean)
    margin = _t_critical_95(len(samples) - 1) * sample_sd / math.sqrt(float(len(samples)))
    return MeanCi(mean=sample_mean, lower=sample_mean - margin, upper=sample_mean + margin)


def binomial_proportion_ci(successes: int, n: int) -> MeanCi:
    """Clopper-Pearson two-sided 95% binomial-proportion CI."""
    if n <= 0:
        raise ValueError("n must be positive for a binomial proportion CI")
    k = int(successes)
    if not (0 <= k <= int(n)):
        raise ValueError("successes must be in [0, n]")
    p_hat = float(k) / float(n)
    alpha = 0.05
    try:
        from scipy import stats

        lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
        upper = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
        return MeanCi(mean=p_hat, lower=lower, upper=upper)
    except Exception:
        if k == 0:
            return MeanCi(mean=p_hat, lower=0.0, upper=1.0 - alpha ** (1.0 / float(n)))
        if k == n:
            return MeanCi(mean=p_hat, lower=alpha ** (1.0 / float(n)), upper=1.0)
        z = 1.959963984540054
        denom = 1.0 + z * z / float(n)
        center = (p_hat + z * z / (2.0 * n)) / denom
        margin = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z / (4.0 * n * n))) / denom
        return MeanCi(mean=p_hat, lower=max(0.0, center - margin), upper=min(1.0, center + margin))


def _merged_status(rows: Sequence[G4dPairedResult], field_name: str) -> str:
    values = {str(getattr(row, field_name)) for row in rows}
    if values == {"match"}:
        return "all_match"
    if "mismatch" in values:
        return "mismatch"
    if "unavailable" in values:
        return "unavailable"
    return ",".join(sorted(values))


def summarize_scenario(rows: Sequence[G4dPairedResult]) -> G4dScenarioSummary:
    if not rows:
        raise ValueError("Cannot summarize an empty scenario row set")
    scenario_names = {row.scenario_name for row in rows}
    if len(scenario_names) != 1:
        raise ValueError(f"Expected one scenario per summary, got {sorted(scenario_names)}")

    n = len(rows)
    pdr_ci = mean_ci_t([row.delta_pdr for row in rows])
    tx_ci = mean_ci_t([row.relative_delta_tx_attempts for row in rows])
    gap_successes = int(sum(1 for row in rows if row.oracle_gap_score == 1.0))
    gap_ci = binomial_proportion_ci(gap_successes, n)

    pairing_checks_passed = all(row.pairing_checks_passed for row in rows)
    pdr_condition = bool(n >= 10 and pdr_ci.lower >= -EPSILON_PDR)
    tx_condition = bool(n >= 10 and tx_ci.upper <= -EPSILON_TX_REDUCTION)
    gap_condition = bool(n >= 10 and gap_ci.lower > EPSILON_GAP)
    same_scenario_positive = bool(
        pairing_checks_passed and pdr_condition and tx_condition and gap_condition
    )

    return G4dScenarioSummary(
        scenario_name=rows[0].scenario_name,
        n_seeds=n,
        mean_pdr_flood=float(mean(row.pdr_flood for row in rows)),
        mean_pdr_controlled=float(mean(row.pdr_controlled for row in rows)),
        mean_delta_pdr=float(pdr_ci.mean),
        ci95_delta_pdr_lower=float(pdr_ci.lower),
        ci95_delta_pdr_upper=float(pdr_ci.upper),
        mean_tx_attempts_flood=float(mean(row.tx_attempts_flood for row in rows)),
        mean_tx_attempts_controlled=float(mean(row.tx_attempts_controlled for row in rows)),
        mean_relative_delta_tx_attempts=float(tx_ci.mean),
        ci95_relative_delta_tx_attempts_lower=float(tx_ci.lower),
        ci95_relative_delta_tx_attempts_upper=float(tx_ci.upper),
        oracle_gap_score=float(gap_ci.mean),
        ci95_oracle_gap_score_lower=float(gap_ci.lower),
        ci95_oracle_gap_score_upper=float(gap_ci.upper),
        oracle_gap_successes=gap_successes,
        mobility_pairing_status=_merged_status(rows, "mobility_digest_match_status"),
        traffic_pairing_status=_merged_status(rows, "traffic_digest_match_status"),
        mac_drift_init_pairing_status=_merged_status(rows, "mac_drift_init_match_status"),
        pairing_checks_passed=bool(pairing_checks_passed),
        pdr_ci_condition_passed=pdr_condition,
        tx_ci_condition_passed=tx_condition,
        oracle_gap_ci_condition_passed=gap_condition,
        same_scenario_positive=same_scenario_positive,
    )


def summarize_by_scenario(rows: Sequence[G4dPairedResult]) -> List[G4dScenarioSummary]:
    grouped: Dict[str, List[G4dPairedResult]] = {name: [] for name in SCENARIO_NAMES}
    extras: Dict[str, List[G4dPairedResult]] = {}
    for row in rows:
        grouped.setdefault(row.scenario_name, extras.setdefault(row.scenario_name, []))
        grouped[row.scenario_name].append(row)
    ordered_names = [name for name in SCENARIO_NAMES if grouped.get(name)]
    ordered_names.extend(name for name in sorted(grouped) if name not in SCENARIO_NAMES and grouped[name])
    return [summarize_scenario(grouped[name]) for name in ordered_names]


def evaluate_g4d_verdict(summaries: Sequence[G4dScenarioSummary]) -> G4dVerdict:
    positive = tuple(summary.scenario_name for summary in summaries if summary.same_scenario_positive)
    if positive:
        return G4dVerdict(verdict="POSITIVE", g5_authorized=True, positive_scenarios=positive)
    return G4dVerdict(verdict="NULL/NEGATIVE", g5_authorized=False, positive_scenarios=tuple())


def run_g4d_probe(
    scenarios: Optional[Mapping[str, SimCfg]] = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    comparison_func: Callable[[SimCfg, int], ModeComparisonResult] | None = None,
) -> Tuple[List[G4dPairedResult], List[G4dScenarioSummary], G4dVerdict]:
    """Run G4d paired mode comparisons and return rows, summaries, and verdict."""
    scenario_map = dict(scenarios) if scenarios is not None else load_g4d_scenarios()
    if comparison_func is None:
        def comparison_func(cfg: SimCfg, seed: int, *, scenario_name: str) -> ModeComparisonResult:
            return compare_flood_vs_controlled_modes(cfg, seed, scenario_name=scenario_name)

    rows: List[G4dPairedResult] = []
    for scenario_name, cfg in scenario_map.items():
        for seed in seeds:
            result = comparison_func(cfg, int(seed), scenario_name=scenario_name)  # type: ignore[misc]
            if type(result).__name__ != MODE_COMPARISON_RESULT_NAME:
                raise TypeError(
                    "G4d requires G4c ModeComparisonResult artifacts, got "
                    f"{type(result).__name__}"
                )
            rows.append(G4dPairedResult.from_mode_comparison(result))

    summaries = summarize_by_scenario(rows)
    verdict = evaluate_g4d_verdict(summaries)
    return rows, summaries, verdict


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def render_report(
    summaries: Sequence[G4dScenarioSummary],
    verdict: G4dVerdict,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> str:
    lines: List[str] = [
        "# G4d Oracle-Gap Report",
        "",
        f"Verdict: **{verdict.verdict}**.",
        f"G5 selector status: **{verdict.g5_status}**.",
    ]
    if not verdict.g5_authorized:
        lines.append("Because the G4d result is null/negative, G5 remains blocked.")
    else:
        joined = ", ".join(verdict.positive_scenarios)
        lines.append(f"G5 is authorized by same-scenario positive evidence in: {joined}.")

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "G4d ran paired complete-mode comparisons only: Arm A was "
            "FloodExpertStateMachine and Arm B was ControlledRetransmissionExpert. "
            "Each arm used the same scenario config, seed, mobility plan, traffic schedule, "
            "and initialization conditions. Runtime byte identity after policy divergence was "
            "not required.",
            "",
            f"Seeds: {min(seeds)} through {max(seeds)} (n={len(seeds)} per scenario).",
            "",
            "Thresholds: A-042 epsilon_pdr=0.02, A-043 epsilon_tx_reduction=0.05, "
            "A-044 epsilon_gap=0.05. A-045 is reserved and inactive; A-046 is Plan A only.",
            "",
            "The relative tx-attempts delta is "
            "`(tx_attempts_controlled - tx_attempts_flood) / max(tx_attempts_flood, 1)`. "
            "`tx_attempts` is a proxy, not hardware energy or validated airtime.",
            "",
            "For PDR and relative tx-attempts deltas, CIs are paired-seed t-distribution "
            "intervals. For oracle_gap_score, the CI is a Clopper-Pearson binomial "
            "proportion interval over seeds where the mode-level oracle selects controlled "
            "retransmission.",
            "",
            "## Per-Scenario CI Summary",
            "",
            "| Scenario | PDR flood | PDR controlled | Delta PDR CI | Tx flood | Tx controlled | Rel tx CI | Oracle gap CI | Pairing | Positive |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for summary in summaries:
        pairing = "pass" if summary.pairing_checks_passed else "fail"
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.scenario_name,
                    _fmt(summary.mean_pdr_flood),
                    _fmt(summary.mean_pdr_controlled),
                    f"[{_fmt(summary.ci95_delta_pdr_lower)}, {_fmt(summary.ci95_delta_pdr_upper)}]",
                    _fmt(summary.mean_tx_attempts_flood, digits=2),
                    _fmt(summary.mean_tx_attempts_controlled, digits=2),
                    (
                        f"[{_fmt(summary.ci95_relative_delta_tx_attempts_lower)}, "
                        f"{_fmt(summary.ci95_relative_delta_tx_attempts_upper)}]"
                    ),
                    (
                        f"{_fmt(summary.oracle_gap_score)} "
                        f"[{_fmt(summary.ci95_oracle_gap_score_lower)}, "
                        f"{_fmt(summary.ci95_oracle_gap_score_upper)}]"
                    ),
                    pairing,
                    "yes" if summary.same_scenario_positive else "no",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Same-Scenario Gate",
            "",
            "A positive verdict requires one scenario to satisfy all three CI conditions "
            "together: PDR safety lower bound, tx-attempt reduction upper bound, and "
            "oracle-gap lower bound. Evidence is not mixed across scenarios and point "
            "estimates are not used as the final gate.",
            "",
            "## Output Tables",
            "",
            "- `outputs/g4d_mode_comparison_results.csv`",
            "- `outputs/g4d_oracle_gap_summary.csv`",
            "",
            "## Limitations",
            "",
            "- Results are within-simulator mode-level counterfactual characterizations.",
            "- Controlled retransmission is the pre-registered immediate controlled-redundancy expert; no tuning was applied for G4d.",
            "- `tx_attempts` is only a cost proxy.",
            "- No selector, ML, RL, DRL, GNN, route-discovery protocol, timer hook, or per-decision intervention artifact is authorized by this report.",
            "- No compliance or hardware-transfer claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_g4d_outputs(
    rows: Sequence[G4dPairedResult],
    summaries: Sequence[G4dScenarioSummary],
    verdict: G4dVerdict,
    *,
    docs_dir: Path = Path("docs"),
    outputs_dir: Path = Path("outputs"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> Tuple[Path, Path, Path]:
    """Write the required G4d report and CSV tables."""
    results_path = outputs_dir / "g4d_mode_comparison_results.csv"
    summary_path = outputs_dir / "g4d_oracle_gap_summary.csv"
    report_path = docs_dir / "g4d_oracle_gap_report.md"

    _write_csv(results_path, (row.to_dict() for row in rows), RESULT_COLUMNS)
    _write_csv(summary_path, (summary.to_dict() for summary in summaries), SUMMARY_COLUMNS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summaries, verdict, seeds=seeds), encoding="utf-8")
    return report_path, results_path, summary_path


def main() -> None:
    rows, summaries, verdict = run_g4d_probe()
    report_path, results_path, summary_path = write_g4d_outputs(
        rows,
        summaries,
        verdict,
        docs_dir=Path("docs"),
        outputs_dir=Path("outputs"),
        seeds=DEFAULT_SEEDS,
    )
    print(f"G4d verdict: {verdict.verdict}")
    print(f"G5 selector status: {verdict.g5_status}")
    print(f"Wrote {report_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
