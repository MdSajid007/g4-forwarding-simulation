"""G4b tests for ControlledRetransmissionExpert as an additive live policy."""
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
from src.entities import Packet, PacketId
from src.experts.controlled_retransmission import (
    JITTER_STREAM_NAME,
    ControlledRetransmissionExpert,
    ControlledRetxState,
)
from src.hashing import cfg_hash
from src.mac_engine import SimEngine
from src.mobility import build_plan
from src.policies import FloodPolicy
from src.seeds import SeedManager
from src.traffic import build_injections


G4B_SEED = 1
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


def _packet(*, flow_id: int = 7, seq: int = 11, dst: int = 9, ttl: int = 4) -> Packet:
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


def _scenario_cases() -> List[Tuple[str, SimCfg]]:
    cases: List[Tuple[str, SimCfg]] = [("easy", make_easy_cfg())]
    cases.extend(get_all_scenarios().items())
    return cases


SCENARIO_CASES = _scenario_cases()
SCENARIO_NAMES = tuple(name for name, _ in SCENARIO_CASES)
assert SCENARIO_NAMES == EXPECTED_SCENARIOS


def _init_ctrl(cfg: SimCfg, *, seed: int = G4B_SEED, **kwargs: Any) -> ControlledRetransmissionExpert:
    h = str(cfg_hash(cfg))
    pol = ControlledRetransmissionExpert(**kwargs)
    pol.reset_for_run(cfg, cfg_hash_for_rng=h, seed=int(seed))
    return pol


def _run_policy(cfg: SimCfg, seed: int, policy: Any) -> Dict[str, Any]:
    h = str(cfg_hash(cfg))
    sm = SeedManager(cfg_hash=h, seed=int(seed))
    plan = build_plan(cfg, sm)
    injs = build_injections(cfg, sm)
    buckets = build_airtime_buckets(cfg)
    eng = SimEngine(
        cfg,
        seed=int(seed),
        plan=plan,
        policy=policy,
        cfg_hash_for_rng=h,
        airtime_buckets=buckets,
    )
    return eng.run(injs)


def _run_controlled(cfg: SimCfg, seed: int = G4B_SEED) -> Tuple[Dict[str, Any], ControlledRetransmissionExpert]:
    pol = _init_ctrl(cfg, seed=seed)
    return _run_policy(cfg, seed, pol), pol


def _summary_hash(summary: Dict[str, Any]) -> str:
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_ctrlretx_duplicate_suppression() -> None:
    cfg = make_easy_cfg()
    pol = _init_ctrl(cfg)
    pkt = _packet(dst=9, ttl=4)

    first = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)
    entry_before = pol.relay_entry(2, pkt)
    second = pol.on_rx(cfg, 1.1, rx_node=2, pkt=pkt, prev_hop=0)

    assert len(first) == 3
    assert second == []
    assert pol.has_seen(2, pkt)
    assert pol.relay_entry(2, pkt) == entry_before
    assert pol.last_event_by_node[2] == "EVT_DUPLICATE_RX"
    assert pol.last_reason_by_node[2] == "CTRLRETX_RELAY_HANDLED"
    assert pol.state_by_node[2] == ControlledRetxState.SUPPRESSED_DUPLICATE


def test_ctrlretx_destination_silence() -> None:
    cfg = make_easy_cfg()
    pol = _init_ctrl(cfg)
    pkt = _packet(dst=2, ttl=4)

    out = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)

    assert out == []
    assert pol.has_seen(2, pkt)
    assert pol.relay_entry(2, pkt) is None
    assert pol.last_event_by_node[2] == "EVT_NOVEL_RX"
    assert pol.last_reason_by_node[2] == "CTRLRETX_DESTINATION"
    assert pol.state_by_node[2] == ControlledRetxState.SUPPRESSED_DESTINATION


def test_ctrlretx_ttl_threshold_suppression() -> None:
    cfg = make_easy_cfg()
    pol = _init_ctrl(cfg, ttl_relay_threshold=1)

    low = _packet(seq=1, dst=9, ttl=1)
    boundary_forward = _packet(seq=2, dst=9, ttl=2)

    low_out = pol.on_rx(cfg, 1.0, rx_node=2, pkt=low, prev_hop=0)
    low_reason = pol.last_reason_by_node[2]
    low_state = pol.state_by_node[2]
    forward_out = pol.on_rx(cfg, 1.1, rx_node=2, pkt=boundary_forward, prev_hop=0)

    assert low_out == []
    assert low_reason == "CTRLRETX_TTL_LOW"
    assert low_state == ControlledRetxState.SUPPRESSED_TTL_LOW
    assert pol.relay_entry(2, low) is None
    assert len(forward_out) == 3
    assert all(pkt.ttl_hops == 1 for _, pkt in forward_out)
    assert pol.relay_entry(2, boundary_forward).origin_ttl_use == 1  # type: ignore[union-attr]


def test_ctrlretx_retransmit_count_limit() -> None:
    cfg = make_easy_cfg()
    pol = _init_ctrl(cfg, relay_retransmit_count=2, relay_jitter_max_ms=0.0)
    pkt = _packet(dst=9, ttl=5)

    out = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=None)
    entry = pol.relay_entry(2, pkt)
    duplicate = pol.on_rx(cfg, 1.5, rx_node=2, pkt=pkt, prev_hop=3)

    assert len(out) == 3
    assert [dict(fwd.meta)["ctrlretx_copy_index"] for _, fwd in out] == [0, 1, 2]
    assert all(fwd.ttl_hops == 4 for _, fwd in out)
    assert entry is not None
    assert entry.copies_emitted == 3
    assert entry.retx_remaining == 0
    assert entry.state == ControlledRetxState.RELAY_COMPLETE
    assert duplicate == []

    one_shot = _init_ctrl(cfg, relay_retransmit_count=0, relay_jitter_max_ms=0.0)
    assert len(one_shot.on_rx(cfg, 2.0, rx_node=2, pkt=_packet(seq=12, ttl=5), prev_hop=0)) == 1


def test_ctrlretx_retransmit_interval_or_hook_limitation() -> None:
    cfg = make_easy_cfg()
    pol = _init_ctrl(
        cfg,
        relay_retransmit_count=2,
        relay_retransmit_interval_ms=10.0,
        relay_jitter_max_ms=0.0,
    )
    pkt = _packet(dst=9, ttl=5)

    out = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)
    entry = pol.relay_entry(2, pkt)

    assert len(out) == 3
    assert entry is not None
    assert entry.desired_delay_s_by_copy == pytest.approx((0.0, 0.010, 0.020))
    assert entry.next_retx_time_s == pytest.approx(1.010)
    assert entry.scheduler_hook_active is False
    assert pol.timer_integration_supported is False
    assert pol.retransmit_interval_applied_to_engine is False
    assert "does not dispatch policy on_timer" in pol.scheduler_hook_limitation


def test_ctrlretx_jitter_determinism_or_hook_limitation() -> None:
    cfg = make_easy_cfg()
    pkt = _packet(dst=9, ttl=5)
    pol1 = _init_ctrl(cfg, seed=3, relay_retransmit_count=2, relay_jitter_max_ms=5.0)
    pol2 = _init_ctrl(cfg, seed=3, relay_retransmit_count=2, relay_jitter_max_ms=5.0)

    out1 = pol1.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)
    out2 = pol2.on_rx(cfg, 1.0, rx_node=2, pkt=pkt, prev_hop=0)
    jitters = [j for _, _, _, j in pol1.jitter_log]

    assert JITTER_STREAM_NAME == pol1.jitter_stream_name
    assert pol1.rng_stream_is_named_substream
    assert pol1.jitter_log == pol2.jitter_log
    assert [fwd.meta for _, fwd in out1] == [fwd.meta for _, fwd in out2]
    assert len(jitters) == 3
    assert all(0.0 <= j <= 0.005 for j in jitters)
    assert pol1.jitter_applied_to_engine is False


def test_ctrlretx_novelty_from_own_cache() -> None:
    cfg = make_easy_cfg()
    pol = _init_ctrl(cfg)
    pkt_marked_duplicate = _packet(dst=9, ttl=4).with_meta(is_duplicate=True)

    first = pol.on_rx(cfg, 1.0, rx_node=2, pkt=pkt_marked_duplicate, prev_hop=0)
    pkt_marked_novel = pkt_marked_duplicate.with_meta(is_duplicate=False)
    second_same_node = pol.on_rx(cfg, 1.1, rx_node=2, pkt=pkt_marked_novel, prev_hop=0)
    first_other_node = pol.on_rx(cfg, 1.2, rx_node=3, pkt=pkt_marked_novel, prev_hop=0)

    assert len(first) == 3
    assert pol.last_event_by_node[2] == "EVT_DUPLICATE_RX"
    assert second_same_node == []
    assert len(first_other_node) == 3
    assert pol.last_event_by_node[3] == "EVT_NOVEL_RX"


def test_ctrlretx_determinism() -> None:
    cfg = make_easy_cfg()
    out1, pol1 = _run_controlled(cfg, G4B_SEED)
    out2, pol2 = _run_controlled(cfg, G4B_SEED)

    assert _summary_hash(out1["summary"]) == _summary_hash(out2["summary"])
    assert out1["trace_head"] == out2["trace_head"]
    assert pol1.jitter_log == pol2.jitter_log


def test_ctrlretx_no_existing_policy_behavior_change() -> None:
    cfg = make_easy_cfg()
    before = _run_policy(cfg, G4B_SEED, FloodPolicy())
    _ctrl, _pol = _run_controlled(cfg, G4B_SEED)
    after = _run_policy(cfg, G4B_SEED, FloodPolicy())

    assert before["summary"] == after["summary"]
    assert before["trace_head"] == after["trace_head"]


@pytest.mark.parametrize("scenario_name,cfg", SCENARIO_CASES, ids=SCENARIO_NAMES)
def test_ctrlretx_scenario_smoke(scenario_name: str, cfg: SimCfg) -> None:
    out, pol = _run_controlled(cfg, G4B_SEED)

    assert out["summary"]["app_injected"] > 0, scenario_name
    assert out["summary"]["tx_attempts"] >= 0, scenario_name
    assert isinstance(pol.jitter_log, list)
