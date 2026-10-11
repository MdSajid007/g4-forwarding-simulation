"""Controlled-retransmission expert implemented as an additive live policy.

This module stays inside the existing SimEngine policy surface: app injection
and reception return ``(node_id, Packet)`` actions. The current engine has no
policy-owned delayed timer hook, so retransmit interval and jitter are recorded
as deterministic intended timing metadata while emitted packets are queued
through the ordinary policy return path.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import SimCfg
from ..entities import Packet
from ..hashing import cfg_hash
from ..seeds import SeedManager


PacketKey = Tuple[int, int]

JITTER_STREAM_NAME = "policy.controlled_retransmission_expert.jitter"
SCHEDULER_HOOK_LIMITATION = (
    "SimEngine policy actions are (node_id, Packet) pairs without per-action "
    "delay, and SimEngine does not dispatch policy on_timer() or "
    "on_transmission_result() hooks."
)


class ControlledRetxState(str, Enum):
    IDLE = "IDLE"
    NOVEL_RECEIVED = "NOVEL_RECEIVED"
    RELAY_TIMER_WAITING = "RELAY_TIMER_WAITING"
    RETRANSMIT_ACTIVE = "RETRANSMIT_ACTIVE"
    RELAY_COMPLETE = "RELAY_COMPLETE"
    SUPPRESSED_DUPLICATE = "SUPPRESSED_DUPLICATE"
    SUPPRESSED_TTL_LOW = "SUPPRESSED_TTL_LOW"
    SUPPRESSED_DESTINATION = "SUPPRESSED_DESTINATION"


@dataclass(frozen=True)
class ControlledRelayEntry:
    state: ControlledRetxState
    retx_remaining: int
    origin_ttl_use: int
    copies_emitted: int
    first_rx_time_s: float
    desired_delay_s_by_copy: Tuple[float, ...] = tuple()
    jitter_s_by_copy: Tuple[float, ...] = tuple()
    next_retx_time_s: Optional[float] = None
    scheduler_hook_active: bool = False


@dataclass
class ControlledRetransmissionExpert:
    """Researcher-defined controlled-retransmission live policy."""

    name: str = "controlled_retransmission_expert"
    relay_retransmit_count: int = 2
    relay_retransmit_interval_ms: float = 10.0
    relay_jitter_max_ms: float = 5.0
    ttl_relay_threshold: int = 1
    jitter_stream_name: str = JITTER_STREAM_NAME

    seen_by_node: Dict[int, Set[PacketKey]] = field(default_factory=dict)
    relay_state_by_node: Dict[int, Dict[PacketKey, ControlledRelayEntry]] = field(default_factory=dict)
    state_by_node: Dict[int, ControlledRetxState] = field(default_factory=dict)
    last_event_by_node: Dict[int, str] = field(default_factory=dict)
    last_reason_by_node: Dict[int, str] = field(default_factory=dict)
    last_schedule_by_node: Dict[int, Dict[PacketKey, Tuple[float, ...]]] = field(default_factory=dict)
    jitter_log: List[Tuple[int, PacketKey, int, float]] = field(default_factory=list)

    timer_integration_supported: bool = False
    retransmit_interval_applied_to_engine: bool = False
    jitter_applied_to_engine: bool = False
    scheduler_hook_limitation: str = SCHEDULER_HOOK_LIMITATION

    _rng_stream: Optional[Any] = field(default=None, init=False, repr=False)
    _rng_stream_is_named_substream: bool = field(default=False, init=False, repr=False)

    def on_init(self, cfg: SimCfg) -> None:
        self._validate_parameters()
        n = int(cfg.n_nodes)
        self.seen_by_node = {i: set() for i in range(n)}
        self.relay_state_by_node = {i: {} for i in range(n)}
        self.state_by_node = {i: ControlledRetxState.IDLE for i in range(n)}
        self.last_event_by_node = {i: "" for i in range(n)}
        self.last_reason_by_node = {i: "" for i in range(n)}
        self.last_schedule_by_node = {i: {} for i in range(n)}
        self.jitter_log = []
        self._ensure_rng_stream(cfg)

    def reset_for_run(
        self,
        cfg: SimCfg,
        rng_stream: Optional[Any] = None,
        *,
        cfg_hash_for_rng: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Clear per-run state and optionally bind the named jitter substream."""
        if rng_stream is None and cfg_hash_for_rng is not None and seed is not None:
            rng_stream = self.make_jitter_rng(cfg_hash_for_rng, int(seed))
        self._rng_stream = rng_stream
        self._rng_stream_is_named_substream = rng_stream is not None
        self.on_init(cfg)

    @classmethod
    def make_jitter_rng(cls, cfg_hash_for_rng: str, seed: int) -> Any:
        return SeedManager(cfg_hash=str(cfg_hash_for_rng), seed=int(seed)).rng_cached(JITTER_STREAM_NAME)

    @property
    def rng_stream_is_named_substream(self) -> bool:
        return bool(self._rng_stream_is_named_substream)

    def has_seen(self, node_id: int, pkt: Packet) -> bool:
        return self._key(pkt) in self.seen_by_node.get(int(node_id), set())

    def relay_entry(self, node_id: int, pkt: Packet) -> Optional[ControlledRelayEntry]:
        return self.relay_state_by_node.get(int(node_id), {}).get(self._key(pkt))

    @staticmethod
    def _key(pkt: Packet) -> PacketKey:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def _validate_parameters(self) -> None:
        if int(self.relay_retransmit_count) < 0:
            raise ValueError("relay_retransmit_count must be >= 0")
        if float(self.relay_retransmit_interval_ms) < 0.0:
            raise ValueError("relay_retransmit_interval_ms must be >= 0")
        if float(self.relay_jitter_max_ms) < 0.0:
            raise ValueError("relay_jitter_max_ms must be >= 0")
        if int(self.ttl_relay_threshold) < 0:
            raise ValueError("ttl_relay_threshold must be >= 0")
        self.relay_retransmit_count = int(self.relay_retransmit_count)
        self.relay_retransmit_interval_ms = float(self.relay_retransmit_interval_ms)
        self.relay_jitter_max_ms = float(self.relay_jitter_max_ms)
        self.ttl_relay_threshold = int(self.ttl_relay_threshold)

    def _ensure_rng_stream(self, cfg: SimCfg) -> None:
        if self._rng_stream is not None:
            return
        fallback_seed = max(0, int(getattr(cfg, "seed_base", 0)))
        self._rng_stream = self.make_jitter_rng(str(cfg_hash(cfg)), fallback_seed)
        self._rng_stream_is_named_substream = False

    def _draw_jitter_s(self, cfg: SimCfg, node_id: int, key: PacketKey, copy_index: int) -> float:
        max_s = float(self.relay_jitter_max_ms) / 1000.0
        if max_s <= 0.0:
            jitter_s = 0.0
        else:
            self._ensure_rng_stream(cfg)
            jitter_s = float(self._rng_stream.uniform(0.0, max_s))
            jitter_s = min(max(jitter_s, 0.0), max_s)
        self.jitter_log.append((int(node_id), key, int(copy_index), float(jitter_s)))
        return float(jitter_s)

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        _ = cfg, now
        return [(int(pkt.src), pkt.with_meta(policy=self.name))]

    def on_rx(
        self,
        cfg: SimCfg,
        now: float,
        rx_node: int,
        pkt: Packet,
        prev_hop: Optional[int],
    ) -> List[Tuple[int, Packet]]:
        rx = int(rx_node)
        key = self._key(pkt)
        seen = self.seen_by_node.setdefault(rx, set())

        if key in seen:
            self.last_event_by_node[rx] = "EVT_DUPLICATE_RX"
            entry = self.relay_state_by_node.get(rx, {}).get(key)
            self.last_reason_by_node[rx] = (
                "CTRLRETX_RELAY_HANDLED" if entry is not None else "CTRLRETX_DUPLICATE"
            )
            self.state_by_node[rx] = ControlledRetxState.SUPPRESSED_DUPLICATE
            return []

        seen.add(key)
        self.last_event_by_node[rx] = "EVT_NOVEL_RX"

        if rx == int(pkt.dst):
            self.last_reason_by_node[rx] = "CTRLRETX_DESTINATION"
            self.state_by_node[rx] = ControlledRetxState.SUPPRESSED_DESTINATION
            return []

        if int(pkt.ttl_hops) <= int(self.ttl_relay_threshold):
            self.last_reason_by_node[rx] = "CTRLRETX_TTL_LOW"
            self.state_by_node[rx] = ControlledRetxState.SUPPRESSED_TTL_LOW
            return []

        total_copies = int(self.relay_retransmit_count) + 1
        origin_ttl_use = int(pkt.ttl_hops) - 1
        interval_s = float(self.relay_retransmit_interval_ms) / 1000.0
        prev_hop_value = int(prev_hop) if prev_hop is not None else -1

        actions: List[Tuple[int, Packet]] = []
        jitters: List[float] = []
        delays: List[float] = []

        for copy_index in range(total_copies):
            jitter_s = self._draw_jitter_s(cfg, rx, key, copy_index)
            desired_delay_s = float(copy_index) * interval_s + float(jitter_s)
            jitters.append(float(jitter_s))
            delays.append(float(desired_delay_s))
            fwd = dataclasses.replace(pkt, ttl_hops=origin_ttl_use).with_meta(
                policy=self.name,
                fwd=self.name,
                prev_hop=prev_hop_value,
                ttl_dec=True,
                ctrlretx_copy_index=int(copy_index),
                ctrlretx_total_copies=int(total_copies),
                ctrlretx_desired_delay_s=round(float(desired_delay_s), 9),
                ctrlretx_jitter_s=round(float(jitter_s), 9),
                ctrlretx_timer_hook_applied=False,
            )
            actions.append((rx, fwd))

        next_retx_time_s: Optional[float] = None
        if total_copies > 1:
            next_retx_time_s = float(now) + float(delays[1])

        entry = ControlledRelayEntry(
            state=ControlledRetxState.RELAY_COMPLETE,
            retx_remaining=0,
            origin_ttl_use=int(origin_ttl_use),
            copies_emitted=int(total_copies),
            first_rx_time_s=float(now),
            desired_delay_s_by_copy=tuple(float(x) for x in delays),
            jitter_s_by_copy=tuple(float(x) for x in jitters),
            next_retx_time_s=next_retx_time_s,
            scheduler_hook_active=False,
        )
        self.relay_state_by_node.setdefault(rx, {})[key] = entry
        self.last_schedule_by_node.setdefault(rx, {})[key] = entry.desired_delay_s_by_copy
        self.last_reason_by_node[rx] = "CTRLRETX_INITIAL_RELAY"
        self.state_by_node[rx] = ControlledRetxState.RELAY_COMPLETE
        return actions

    def on_transmission_result(self, node_id: int, succeeded: bool) -> None:
        node = int(node_id)
        self.last_event_by_node[node] = "EVT_TX_OK" if bool(succeeded) else "EVT_TX_FAIL"

    def on_timer(self, context: Optional[Any] = None) -> List[Tuple[int, Packet]]:
        _ = context
        return []
