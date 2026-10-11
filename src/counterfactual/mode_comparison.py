"""G4c mode-level paired comparison artifact.

This module runs two complete policy modes on a matched scenario and seed:
FloodExpertStateMachine versus ControlledRetransmissionExpert. It records
whole-run outcomes and pairing checks only; it does not force individual
decisions or produce any G4d oracle-gap decision.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..airtime import build_airtime_buckets
from ..config import SimCfg
from ..dataset_io import _inj_hash_cell11, _mobility_digest_hash_cell11
from ..decision_logger import DecisionLogger
from ..experts.controlled_retransmission import (
    ControlledRetransmissionExpert,
    SCHEDULER_HOOK_LIMITATION,
)
from ..experts.flood import FloodExpertStateMachine
from ..hashing import cfg_hash
from ..mac_engine import SimEngine
from ..mobility import build_plan
from ..seeds import SeedManager
from ..traffic import AppInjection, build_injections
from .noop_isolation import stable_sha256


ARM_A_POLICY_NAME = "flood"
ARM_B_POLICY_NAME = "controlled_retransmission_expert"

G4C_LIMITATIONS: Tuple[str, ...] = (
    "Compares complete mode runs only; no single-decision intervention is forced.",
    "Runtime MAC/PHY byte identity is not required after policies diverge.",
    "ControlledRetransmissionExpert emits immediate controlled redundancy under the current SimEngine policy interface.",
    SCHEDULER_HOOK_LIMITATION,
    "No G4d oracle-gap success/failure decision is produced.",
)


def _status_for_match(left: Optional[Any], right: Optional[Any]) -> str:
    if left is None or right is None:
        return "unavailable"
    return "match" if left == right else "mismatch"


def _metric_int(summary: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(summary.get(key, default))
    except Exception:
        return int(default)


def _metric_float(summary: Dict[str, Any], key: str) -> Optional[float]:
    if key not in summary:
        return None
    try:
        return float(summary[key])
    except Exception:
        return None


def _mean_delivery_delay_s(rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    delay_by_packet: Dict[str, float] = {}
    for row in rows:
        if row.get("eventual_delivered") is not True:
            continue
        delay = row.get("delivery_delay_s")
        if delay is None:
            continue
        packet_id = str(row.get("packet_id", ""))
        if packet_id and packet_id not in delay_by_packet:
            delay_by_packet[packet_id] = float(delay)
    if not delay_by_packet:
        return None
    return float(sum(delay_by_packet.values()) / len(delay_by_packet))


def _drift_factors(engine: SimEngine) -> Tuple[Tuple[int, float], ...]:
    factors = getattr(engine, "_drift_factor", None)
    if factors is None:
        return tuple()
    return tuple((int(node), float(value)) for node, value in sorted(factors.items()))


@dataclass(frozen=True)
class ModeArmRun:
    """Observable artifacts for one complete G4c mode arm."""

    cfg_hash: str
    seed: int
    policy_name: str
    summary: Dict[str, Any]
    summary_hash: str
    decision_log_hash: str
    decision_rows_hash: str
    injected_count: int
    delivered_count: int
    pdr: float
    tx_attempts: int
    airtime_proxy_s: Optional[float]
    mean_delay_s: Optional[float]
    mobility_digest: str
    traffic_digest: str
    mac_drift_factors: Tuple[Tuple[int, float], ...]
    mac_drift_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ModeComparisonResult:
    """Mode-level paired comparison for one scenario and seed."""

    scenario_name: str
    seed: int
    arm_a_policy: str
    arm_b_policy: str
    arm_a_injected_count: int
    arm_b_injected_count: int
    arm_a_delivered_count: int
    arm_b_delivered_count: int
    arm_a_pdr: float
    arm_b_pdr: float
    delta_pdr: float
    arm_a_tx_attempts: int
    arm_b_tx_attempts: int
    delta_tx_attempts: int
    arm_a_airtime_proxy_s: Optional[float]
    arm_b_airtime_proxy_s: Optional[float]
    delta_airtime_proxy_s: Optional[float]
    arm_a_mean_delay_s: Optional[float]
    arm_b_mean_delay_s: Optional[float]
    delta_delay_s: Optional[float]
    arm_a_decision_log_hash: str
    arm_b_decision_log_hash: str
    arm_a_summary_hash: str
    arm_b_summary_hash: str
    mobility_digest_match_status: str
    traffic_digest_match_status: str
    mac_drift_init_match_status: str
    arm_a_mobility_digest: str
    arm_b_mobility_digest: str
    arm_a_traffic_digest: str
    arm_b_traffic_digest: str
    arm_a_mac_drift_hash: str
    arm_b_mac_drift_hash: str
    notes_limitations: Tuple[str, ...]
    arm_a: ModeArmRun
    arm_b: ModeArmRun

    @property
    def pairing_checks(self) -> Dict[str, str]:
        return {
            "mobility_digest_match_status": self.mobility_digest_match_status,
            "traffic_digest_match_status": self.traffic_digest_match_status,
            "mac_drift_init_match_status": self.mac_drift_init_match_status,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return the serializable G4c artifact without adding any verdict field."""
        return dataclasses.asdict(self)


def _build_pairing_inputs(cfg: SimCfg, seed: int) -> Tuple[str, Any, List[AppInjection], str, str]:
    cfg_h = str(cfg_hash(cfg))
    sm = SeedManager(cfg_hash=cfg_h, seed=int(seed))
    plan = build_plan(cfg, sm)
    injections = build_injections(cfg, sm)
    mobility_digest = str(
        _mobility_digest_hash_cell11(
            plan,
            n_nodes=int(cfg.n_nodes),
            times=(0.0, float(cfg.sim_duration_s) / 2.0, float(cfg.sim_duration_s)),
        )
    )
    traffic_digest = str(_inj_hash_cell11(injections))
    return cfg_h, plan, injections, mobility_digest, traffic_digest


def _run_arm(
    *,
    cfg: SimCfg,
    seed: int,
    cfg_h: str,
    scenario_name: str,
    arm_label: str,
    policy: Any,
    plan: Any,
    injections: List[AppInjection],
    mobility_digest: str,
    traffic_digest: str,
) -> ModeArmRun:
    if hasattr(policy, "reset_for_run"):
        try:
            policy.reset_for_run(cfg, cfg_hash_for_rng=cfg_h, seed=int(seed))
        except TypeError:
            policy.reset_for_run(cfg)

    logger = DecisionLogger(
        run_id=f"g4c_{scenario_name}_{arm_label}_{cfg_h}_{int(seed)}",
        cfg_hash=cfg_h,
        seed=int(seed),
    )
    engine = SimEngine(
        cfg,
        seed=int(seed),
        plan=plan,
        policy=policy,
        cfg_hash_for_rng=cfg_h,
        airtime_buckets=build_airtime_buckets(cfg),
        decision_logger=logger,
    )
    mac_drift_factors = _drift_factors(engine)
    out = engine.run(injections)
    summary = dict(out["summary"])
    decision_log = [dict(row) for row in out.get("decision_log", [])]

    return ModeArmRun(
        cfg_hash=cfg_h,
        seed=int(seed),
        policy_name=str(getattr(policy, "name", type(policy).__name__)),
        summary=summary,
        summary_hash=stable_sha256(summary),
        decision_log_hash=str(out.get("decision_log_hash", "")),
        decision_rows_hash=stable_sha256(decision_log),
        injected_count=_metric_int(summary, "app_injected"),
        delivered_count=_metric_int(summary, "delivered"),
        pdr=float(_metric_float(summary, "pdr") or 0.0),
        tx_attempts=_metric_int(summary, "tx_attempts"),
        airtime_proxy_s=_metric_float(summary, "airtime_s"),
        mean_delay_s=_mean_delivery_delay_s(decision_log),
        mobility_digest=str(mobility_digest),
        traffic_digest=str(traffic_digest),
        mac_drift_factors=mac_drift_factors,
        mac_drift_hash=stable_sha256(mac_drift_factors),
    )


def compare_flood_vs_controlled_modes(
    cfg: SimCfg,
    seed: int,
    *,
    scenario_name: str,
) -> ModeComparisonResult:
    """Run the G4c paired mode comparison and return the artifact."""
    cfg_h, plan, injections, mobility_digest, traffic_digest = _build_pairing_inputs(cfg, int(seed))

    arm_a = _run_arm(
        cfg=cfg,
        seed=int(seed),
        cfg_h=cfg_h,
        scenario_name=scenario_name,
        arm_label="arm_a_flood",
        policy=FloodExpertStateMachine(),
        plan=plan,
        injections=injections,
        mobility_digest=mobility_digest,
        traffic_digest=traffic_digest,
    )
    arm_b = _run_arm(
        cfg=cfg,
        seed=int(seed),
        cfg_h=cfg_h,
        scenario_name=scenario_name,
        arm_label="arm_b_controlled",
        policy=ControlledRetransmissionExpert(),
        plan=plan,
        injections=injections,
        mobility_digest=mobility_digest,
        traffic_digest=traffic_digest,
    )

    delta_airtime: Optional[float] = None
    if arm_a.airtime_proxy_s is not None and arm_b.airtime_proxy_s is not None:
        delta_airtime = float(arm_b.airtime_proxy_s - arm_a.airtime_proxy_s)

    delta_delay: Optional[float] = None
    if arm_a.mean_delay_s is not None and arm_b.mean_delay_s is not None:
        delta_delay = float(arm_b.mean_delay_s - arm_a.mean_delay_s)

    return ModeComparisonResult(
        scenario_name=str(scenario_name),
        seed=int(seed),
        arm_a_policy=arm_a.policy_name,
        arm_b_policy=arm_b.policy_name,
        arm_a_injected_count=arm_a.injected_count,
        arm_b_injected_count=arm_b.injected_count,
        arm_a_delivered_count=arm_a.delivered_count,
        arm_b_delivered_count=arm_b.delivered_count,
        arm_a_pdr=float(arm_a.pdr),
        arm_b_pdr=float(arm_b.pdr),
        delta_pdr=float(arm_b.pdr - arm_a.pdr),
        arm_a_tx_attempts=int(arm_a.tx_attempts),
        arm_b_tx_attempts=int(arm_b.tx_attempts),
        delta_tx_attempts=int(arm_b.tx_attempts - arm_a.tx_attempts),
        arm_a_airtime_proxy_s=arm_a.airtime_proxy_s,
        arm_b_airtime_proxy_s=arm_b.airtime_proxy_s,
        delta_airtime_proxy_s=delta_airtime,
        arm_a_mean_delay_s=arm_a.mean_delay_s,
        arm_b_mean_delay_s=arm_b.mean_delay_s,
        delta_delay_s=delta_delay,
        arm_a_decision_log_hash=arm_a.decision_log_hash,
        arm_b_decision_log_hash=arm_b.decision_log_hash,
        arm_a_summary_hash=arm_a.summary_hash,
        arm_b_summary_hash=arm_b.summary_hash,
        mobility_digest_match_status=_status_for_match(arm_a.mobility_digest, arm_b.mobility_digest),
        traffic_digest_match_status=_status_for_match(arm_a.traffic_digest, arm_b.traffic_digest),
        mac_drift_init_match_status=_status_for_match(arm_a.mac_drift_hash, arm_b.mac_drift_hash),
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
