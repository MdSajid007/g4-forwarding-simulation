"""G4a tests for FloodExpertStateMachine live-policy equivalence."""
from __future__ import annotations

import hashlib
import json
import sys
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
from src.decision_logger import DecisionLogger
from src.entities import Packet, PacketId
from src.experts import FloodExpertState, FloodExpertStateMachine
from src.hashing import cfg_hash
from src.mac_engine import SimEngine
from src.mobility import build_plan
from src.policies import FloodPolicy
from src.seeds import SeedManager
from src.traffic import build_injections


EQUIVALENCE_SEED = 1


def _packet(*, flow_id: int = 7, seq: int = 11, dst: int = 9, ttl: int = 3) -> Packet:
    return Packet(
        pid=PacketId(flow_id=int(flow_id), seq=int(seq)),
        src=0,
        dst=int(dst),
        created_t=1.25,
        ttl_hops=int(ttl),
        payload_bytes=80,
        is_control=False,
        meta=tuple(),
    )


def _init(policy: Any, cfg: SimCfg) -> Any:
    policy.on_init(cfg)
    return policy


def _run_with_policy(cfg: SimCfg, seed: int, policy: Any) -> Dict[str, Any]:
    h = cfg_hash(cfg)
    sm = SeedManager(cfg_hash=h, seed=int(seed))
    plan = build_plan(cfg, sm)
    injs = build_injections(cfg, sm)
    buckets = build_airtime_buckets(cfg)
    logger = DecisionLogger(run_id=f"{h}_{int(seed)}", cfg_hash=h, seed=int(seed))
    eng = SimEngine(
        cfg,
        seed=int(seed),
        plan=plan,
        policy=policy,
        cfg_hash_for_rng=h,
        airtime_buckets=buckets,
        decision_logger=logger,
    )
    return eng.run(injs)


def _summary_hash(summary: Dict[str, Any]) -> str:
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_based_golden_artifacts() -> List[Path]:
    root = PROJECT_ROOT / "outputs" / "golden"
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep")


def _equivalence_cases() -> List[Tuple[str, SimCfg]]:
    cases: List[Tuple[str, SimCfg]] = [("easy", make_easy_cfg())]
    cases.extend(get_all_scenarios().items())
    return cases


def test_flood_sm_duplicate_suppression() -> None:
    cfg = make_easy_cfg()
    pol = _init(FloodExpertStateMachine(), cfg)
    pkt = _packet(dst=9, ttl=3)

    first = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)
    second = pol.on_rx(cfg, 1.1, rx_node=2, pkt=pkt, prev_hop=0)

    assert len(first) == 1
    assert second == []
    assert pol.has_seen(2, pkt)
    assert pol.last_event_by_node[2] == "EVT_DUPLICATE_RX"
    assert pol.last_reason_by_node[2] == "FLOOD_DUPLICATE"
    assert pol.state_by_node[2] == FloodExpertState.SUPPRESSED_DUPLICATE


def test_flood_sm_destination_silence() -> None:
    cfg = make_easy_cfg()
    pol = _init(FloodExpertStateMachine(), cfg)
    pkt = _packet(dst=2, ttl=3)

    out = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)

    assert out == []
    assert pol.has_seen(2, pkt)
    assert pol.last_event_by_node[2] == "EVT_NOVEL_RX"
    assert pol.last_reason_by_node[2] == "FLOOD_DESTINATION"
    assert pol.state_by_node[2] == FloodExpertState.SUPPRESSED_DESTINATION


def test_flood_sm_ttl_zero_suppression() -> None:
    cfg = make_easy_cfg()
    pol = _init(FloodExpertStateMachine(), cfg)
    pkt = _packet(dst=9, ttl=0)

    out = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)

    assert out == []
    assert pol.has_seen(2, pkt)
    assert pol.last_event_by_node[2] == "EVT_NOVEL_RX"
    assert pol.last_reason_by_node[2] == "FLOOD_TTL_ZERO"
    assert pol.state_by_node[2] == FloodExpertState.SUPPRESSED_TTL_ZERO


def test_flood_sm_ttl_decrement() -> None:
    cfg = make_easy_cfg()
    expert = _init(FloodExpertStateMachine(), cfg)
    flood = _init(FloodPolicy(), cfg)
    pkt = _packet(dst=9, ttl=4)

    expert_out = expert.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=None)
    flood_out = flood.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=None)

    assert expert_out == flood_out
    assert len(expert_out) == 1
    tx_node, fwd = expert_out[0]
    assert tx_node == 2
    assert fwd.ttl_hops == 3
    assert fwd.meta == (
        ("fwd", "flood"),
        ("policy", "flood"),
        ("prev_hop", -1),
        ("ttl_dec", True),
    )
    assert expert.pending_forward_by_node[2] is True
    assert expert.last_reason_by_node[2] == "FLOOD_RELAY"
    assert expert.state_by_node[2] == FloodExpertState.NOVEL_RECEIVED


def test_flood_sm_novelty_from_own_cache() -> None:
    cfg = make_easy_cfg()
    pol = _init(FloodExpertStateMachine(), cfg)
    pkt_marked_duplicate = _packet(dst=9, ttl=3).with_meta(is_duplicate=True)

    first = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt_marked_duplicate, prev_hop=0)
    assert len(first) == 1
    assert pol.last_event_by_node[2] == "EVT_NOVEL_RX"

    pkt_marked_novel = pkt_marked_duplicate.with_meta(is_duplicate=False)
    second_same_node = pol.on_rx(cfg, 1.1, rx_node=2, pkt=pkt_marked_novel, prev_hop=0)
    first_other_node = pol.on_rx(cfg, 1.2, rx_node=3, pkt=pkt_marked_novel, prev_hop=0)

    assert second_same_node == []
    assert pol.last_event_by_node[2] == "EVT_DUPLICATE_RX"
    assert len(first_other_node) == 1
    assert pol.last_event_by_node[3] == "EVT_NOVEL_RX"


def test_flood_sm_determinism() -> None:
    cfg = make_easy_cfg()
    out1 = _run_with_policy(cfg, EQUIVALENCE_SEED, FloodExpertStateMachine())
    out2 = _run_with_policy(cfg, EQUIVALENCE_SEED, FloodExpertStateMachine())

    assert _summary_hash(out1["summary"]) == _summary_hash(out2["summary"])
    assert out1["decision_log_hash"] == out2["decision_log_hash"]
    assert out1["decision_log"] == out2["decision_log"]


@pytest.mark.parametrize("scenario_name,cfg", _equivalence_cases(), ids=[c[0] for c in _equivalence_cases()])
def test_flood_sm_behavior_equivalence(scenario_name: str, cfg: SimCfg) -> None:
    golden_artifacts = _file_based_golden_artifacts()
    assert not golden_artifacts, (
        "File-based golden scenario artifacts are present, but no loader for that "
        "harness was discoverable in existing tests. Add a G4a loader before using "
        "the Phase 5 fallback fixture. Artifacts: "
        f"{[str(p.relative_to(PROJECT_ROOT)) for p in golden_artifacts]}"
    )

    flood_out = _run_with_policy(cfg, EQUIVALENCE_SEED, FloodPolicy())
    expert_out = _run_with_policy(cfg, EQUIVALENCE_SEED, FloodExpertStateMachine())

    assert _summary_hash(expert_out["summary"]) == _summary_hash(flood_out["summary"]), scenario_name
    assert expert_out["summary"] == flood_out["summary"], scenario_name
    assert expert_out["decision_log_hash"] == flood_out["decision_log_hash"], scenario_name
    assert expert_out["decision_log"] == flood_out["decision_log"], scenario_name
