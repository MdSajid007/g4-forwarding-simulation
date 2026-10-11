"""G3a no-op isolation harness for FloodExpertStateMachine.

This module is intentionally additive. It does not wrap or alter SimEngine, does
not call policy_factory, and does not implement forced-action overrides. The only
accepted intervention value is ``forced_action=None``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..airtime import build_airtime_buckets
from ..config import SimCfg
from ..dataset_io import (
    _inj_hash_cell11,
    _mobility_digest_hash_cell11,
    _trace_hash_cell11,
)
from ..decision_logger import DecisionLogger
from ..experts import FloodExpertStateMachine
from ..hashing import cfg_hash
from ..mac_engine import SimEngine
from ..mobility import build_plan
from ..seeds import SeedManager
from ..traffic import build_injections


DecisionOrderRow = Tuple[float, str, int, int, int, int]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted((_jsonable(v) for v in value), key=repr)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return float(value)
    return value


def _stable_bytes(value: Any) -> bytes:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return payload.encode("utf-8")


def stable_sha256(value: Any) -> str:
    """Return a deterministic SHA256 for JSON-serializable simulator artifacts."""
    return hashlib.sha256(_stable_bytes(value)).hexdigest()


def _decision_order_proxy(rows: Iterable[Dict[str, Any]]) -> Tuple[DecisionOrderRow, ...]:
    out: List[DecisionOrderRow] = []
    for row in rows:
        out.append(
            (
                float(row.get("time_s", 0.0)),
                str(row.get("packet_id", "")),
                int(row.get("current_node", -1)),
                int(row.get("previous_node", -1)),
                int(row.get("destination_node", -1)),
                int(row.get("decision_taken", 0)),
            )
        )
    return tuple(out)


def _drift_factors(engine: SimEngine) -> Tuple[Tuple[int, float], ...]:
    factors = getattr(engine, "_drift_factor", {})
    return tuple((int(node), float(value)) for node, value in sorted(factors.items()))


@dataclass
class NoOpRunArtifacts:
    """Observable artifacts from one FloodExpertStateMachine run."""

    cfg_hash: str
    seed: int
    label: str
    policy_name: str
    runstats_snapshot: Dict[str, Any]
    runstats_hash: str
    summary: Dict[str, Any]
    summary_hash: str
    decision_log: List[Dict[str, Any]]
    decision_log_hash: str
    decision_rows_hash: str
    decision_order_proxy: Tuple[DecisionOrderRow, ...]
    decision_order_proxy_hash: str
    trace_head: Tuple[Tuple[float, int, str, int], ...]
    trace_head_hash: str
    trace_hash: str
    mobility_plan_digest: str
    traffic_schedule_digest: str
    mac_drift_factors: Tuple[Tuple[int, float], ...]
    mac_drift_hash: str

    def identity_fingerprint(self) -> Dict[str, str]:
        return {
            "runstats_hash": self.runstats_hash,
            "summary_hash": self.summary_hash,
            "decision_log_hash": self.decision_log_hash,
            "decision_rows_hash": self.decision_rows_hash,
            "decision_order_proxy_hash": self.decision_order_proxy_hash,
            "trace_head_hash": self.trace_head_hash,
            "trace_hash": self.trace_hash,
            "mobility_plan_digest": self.mobility_plan_digest,
            "traffic_schedule_digest": self.traffic_schedule_digest,
            "mac_drift_hash": self.mac_drift_hash,
        }


@dataclass
class NoOpIsolationComparison:
    """Baseline-vs-no-op comparison for one scenario and seed."""

    scenario_name: str
    seed: int
    forced_action: Optional[Any]
    baseline: NoOpRunArtifacts
    noop: NoOpRunArtifacts

    @property
    def runstats_identity(self) -> bool:
        return (
            self.baseline.runstats_hash == self.noop.runstats_hash
            and self.baseline.runstats_snapshot == self.noop.runstats_snapshot
        )

    @property
    def summary_identity(self) -> bool:
        return self.baseline.summary_hash == self.noop.summary_hash and self.baseline.summary == self.noop.summary

    @property
    def decision_log_identity(self) -> bool:
        return (
            self.baseline.decision_log_hash == self.noop.decision_log_hash
            and self.baseline.decision_rows_hash == self.noop.decision_rows_hash
            and self.baseline.decision_log == self.noop.decision_log
        )

    @property
    def scoped_rng_identity(self) -> bool:
        return (
            self.baseline.mobility_plan_digest == self.noop.mobility_plan_digest
            and self.baseline.traffic_schedule_digest == self.noop.traffic_schedule_digest
            and self.baseline.mac_drift_hash == self.noop.mac_drift_hash
            and self.baseline.mac_drift_factors == self.noop.mac_drift_factors
        )

    @property
    def event_order_proxy_identity(self) -> bool:
        return (
            self.baseline.decision_order_proxy_hash == self.noop.decision_order_proxy_hash
            and self.baseline.decision_order_proxy == self.noop.decision_order_proxy
            and self.baseline.trace_head_hash == self.noop.trace_head_hash
            and self.baseline.trace_head == self.noop.trace_head
        )

    @property
    def r_iso_1_passed(self) -> bool:
        return self.runstats_identity and self.summary_identity and self.decision_log_identity

    @property
    def r_iso_3_scoped_passed(self) -> bool:
        return self.scoped_rng_identity

    @property
    def r_iso_4_proxy_passed(self) -> bool:
        return self.event_order_proxy_identity


def _run_flood_expert(cfg: SimCfg, seed: int, *, label: str) -> NoOpRunArtifacts:
    cfg_h = str(cfg_hash(cfg))
    sd = int(seed)
    sm = SeedManager(cfg_hash=cfg_h, seed=sd)
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

    policy = FloodExpertStateMachine()
    logger = DecisionLogger(run_id=f"{cfg_h}_{sd}", cfg_hash=cfg_h, seed=sd)
    engine = SimEngine(
        cfg,
        seed=sd,
        plan=plan,
        policy=policy,
        cfg_hash_for_rng=cfg_h,
        airtime_buckets=build_airtime_buckets(cfg),
        decision_logger=logger,
    )
    mac_drift_factors = _drift_factors(engine)
    out = engine.run(injections)

    runstats_snapshot = dataclasses.asdict(engine.stats)
    summary = dict(out["summary"])
    decision_log = [dict(row) for row in out.get("decision_log", [])]
    decision_order = _decision_order_proxy(decision_log)
    trace_head = tuple(
        (float(t), int(prio), str(etype), int(src))
        for t, prio, etype, src in out.get("trace_head", [])
    )

    return NoOpRunArtifacts(
        cfg_hash=cfg_h,
        seed=sd,
        label=str(label),
        policy_name=str(getattr(policy, "name", "")),
        runstats_snapshot=runstats_snapshot,
        runstats_hash=stable_sha256(runstats_snapshot),
        summary=summary,
        summary_hash=stable_sha256(summary),
        decision_log=decision_log,
        decision_log_hash=str(out.get("decision_log_hash", "")),
        decision_rows_hash=stable_sha256(decision_log),
        decision_order_proxy=decision_order,
        decision_order_proxy_hash=stable_sha256(decision_order),
        trace_head=trace_head,
        trace_head_hash=stable_sha256(trace_head),
        trace_hash=str(_trace_hash_cell11(trace_head)),
        mobility_plan_digest=mobility_digest,
        traffic_schedule_digest=traffic_digest,
        mac_drift_factors=mac_drift_factors,
        mac_drift_hash=stable_sha256(mac_drift_factors),
    )


def run_baseline_flood_expert(cfg: SimCfg, seed: int) -> NoOpRunArtifacts:
    """Run the baseline FloodExpertStateMachine arm."""
    return _run_flood_expert(cfg, seed, label="baseline")


def run_noop_flood_expert(
    cfg: SimCfg,
    seed: int,
    *,
    forced_action: Optional[Any] = None,
) -> NoOpRunArtifacts:
    """Run the no-op intervention arm.

    ``forced_action`` is accepted only so the G3a harness can prove the no-op
    path. Non-``None`` values are rejected; forced-action override is not
    implemented in this gate.
    """
    if forced_action is not None:
        raise NotImplementedError("G3a implements only forced_action=None no-op isolation.")
    return _run_flood_expert(cfg, seed, label="noop_forced_action_none")


def compare_noop_isolation(
    cfg: SimCfg,
    seed: int,
    *,
    scenario_name: str,
    forced_action: Optional[Any] = None,
) -> NoOpIsolationComparison:
    """Run baseline and no-op FloodExpertStateMachine arms and compare artifacts."""
    if forced_action is not None:
        raise NotImplementedError("G3a implements only forced_action=None no-op isolation.")
    baseline = run_baseline_flood_expert(cfg, seed)
    noop = run_noop_flood_expert(cfg, seed, forced_action=forced_action)
    return NoOpIsolationComparison(
        scenario_name=str(scenario_name),
        seed=int(seed),
        forced_action=forced_action,
        baseline=baseline,
        noop=noop,
    )
