"""G3a no-op intervention isolation tests."""
from __future__ import annotations

import dataclasses
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

from src.airtime import build_airtime_buckets
from src.config import SimCfg, make_easy_cfg
from src.counterfactual.noop_isolation import (
    NoOpIsolationComparison,
    compare_noop_isolation,
    stable_sha256,
)
from src.decision_logger import DecisionLogger
from src.hashing import cfg_hash
from src.mac_engine import SimEngine
from src.mobility import build_plan
from src.policies import FloodPolicy
from src.seeds import SeedManager
from src.traffic import build_injections


G3A_SEED = 1

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


def _scenario_cases() -> List[Tuple[str, SimCfg]]:
    cases: List[Tuple[str, SimCfg]] = [("easy", make_easy_cfg())]
    cases.extend(get_all_scenarios().items())
    return cases


SCENARIO_CASES = _scenario_cases()
SCENARIO_BY_NAME: Dict[str, SimCfg] = dict(SCENARIO_CASES)
SCENARIO_NAMES = tuple(name for name, _ in SCENARIO_CASES)

assert SCENARIO_NAMES == EXPECTED_SCENARIOS


@lru_cache(maxsize=None)
def _comparison_for(scenario_name: str) -> NoOpIsolationComparison:
    cfg = SCENARIO_BY_NAME[scenario_name]
    return compare_noop_isolation(cfg, G3A_SEED, scenario_name=scenario_name, forced_action=None)


def _run_flood_policy_artifacts(cfg: SimCfg, seed: int) -> Dict[str, Any]:
    h = str(cfg_hash(cfg))
    sd = int(seed)
    sm = SeedManager(cfg_hash=h, seed=sd)
    plan = build_plan(cfg, sm)
    injections = build_injections(cfg, sm)
    logger = DecisionLogger(run_id=f"{h}_{sd}", cfg_hash=h, seed=sd)
    engine = SimEngine(
        cfg,
        seed=sd,
        plan=plan,
        policy=FloodPolicy(),
        cfg_hash_for_rng=h,
        airtime_buckets=build_airtime_buckets(cfg),
        decision_logger=logger,
    )
    out = engine.run(injections)
    decision_log = [dict(row) for row in out.get("decision_log", [])]
    trace_head = [
        (float(t), int(prio), str(etype), int(src))
        for t, prio, etype, src in out.get("trace_head", [])
    ]
    runstats_snapshot = dataclasses.asdict(engine.stats)
    return {
        "cfg_hash": h,
        "seed": sd,
        "runstats_hash": stable_sha256(runstats_snapshot),
        "runstats_snapshot": runstats_snapshot,
        "summary": dict(out["summary"]),
        "decision_log_hash": str(out.get("decision_log_hash", "")),
        "decision_log": decision_log,
        "trace_head": trace_head,
    }


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g3a_noop_runstats_identity(scenario_name: str) -> None:
    comparison = _comparison_for(scenario_name)

    assert comparison.forced_action is None
    assert comparison.baseline.runstats_hash == comparison.noop.runstats_hash
    assert comparison.baseline.runstats_snapshot == comparison.noop.runstats_snapshot
    assert comparison.runstats_identity


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g3a_noop_decision_log_identity(scenario_name: str) -> None:
    comparison = _comparison_for(scenario_name)

    assert comparison.baseline.decision_log_hash == comparison.noop.decision_log_hash
    assert comparison.baseline.decision_rows_hash == comparison.noop.decision_rows_hash
    assert comparison.baseline.decision_log == comparison.noop.decision_log
    assert comparison.decision_log_identity


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g3a_noop_summary_identity(scenario_name: str) -> None:
    comparison = _comparison_for(scenario_name)

    assert comparison.baseline.summary_hash == comparison.noop.summary_hash
    assert comparison.baseline.summary == comparison.noop.summary
    assert comparison.summary_identity


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g3a_noop_event_order_proxy_identity(scenario_name: str) -> None:
    comparison = _comparison_for(scenario_name)

    assert comparison.baseline.decision_order_proxy_hash == comparison.noop.decision_order_proxy_hash
    assert comparison.baseline.decision_order_proxy == comparison.noop.decision_order_proxy
    assert comparison.baseline.trace_head_hash == comparison.noop.trace_head_hash
    assert comparison.baseline.trace_head == comparison.noop.trace_head
    assert comparison.event_order_proxy_identity


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g3a_noop_determinism_across_repeats(scenario_name: str) -> None:
    cfg = SCENARIO_BY_NAME[scenario_name]
    first = _comparison_for(scenario_name)
    second = compare_noop_isolation(cfg, G3A_SEED, scenario_name=scenario_name, forced_action=None)

    assert first.baseline.identity_fingerprint() == second.baseline.identity_fingerprint()
    assert first.noop.identity_fingerprint() == second.noop.identity_fingerprint()
    assert first.baseline.summary == second.baseline.summary
    assert first.noop.summary == second.noop.summary
    assert first.baseline.decision_log == second.baseline.decision_log
    assert first.noop.decision_log == second.noop.decision_log
    assert second.r_iso_1_passed
    assert second.r_iso_3_scoped_passed
    assert second.r_iso_4_proxy_passed


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_g3a_noop_does_not_touch_existing_policy_behavior(scenario_name: str) -> None:
    cfg = SCENARIO_BY_NAME[scenario_name]
    flood_before = _run_flood_policy_artifacts(cfg, G3A_SEED)
    _ = _comparison_for(scenario_name)
    flood_after = _run_flood_policy_artifacts(cfg, G3A_SEED)

    assert flood_before == flood_after
