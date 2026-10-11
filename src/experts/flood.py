"""Flood expert state machine implemented as a live SimEngine policy.

The live forwarding behavior intentionally matches ``src.policies.FloodPolicy``.
The class keeps explicit state-machine observability around that behavior without
changing packet metadata, RNG consumption, or simulator hooks.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import SimCfg
from ..entities import Packet


PacketKey = Tuple[int, int]


class FloodExpertState(str, Enum):
    IDLE = "IDLE"
    NOVEL_RECEIVED = "NOVEL_RECEIVED"
    RELAYED = "RELAYED"
    SUPPRESSED_DESTINATION = "SUPPRESSED_DESTINATION"
    SUPPRESSED_DUPLICATE = "SUPPRESSED_DUPLICATE"
    SUPPRESSED_TTL_ZERO = "SUPPRESSED_TTL_ZERO"


@dataclass
class FloodExpertStateMachine:
    """High-redundancy flood expert, byte-compatible with ``FloodPolicy``."""

    # Must remain "flood" so Packet.meta is byte-identical to FloodPolicy output.
    name: str = "flood"
    seen_by_node: Dict[int, Set[PacketKey]] = field(default_factory=dict)
    state_by_node: Dict[int, FloodExpertState] = field(default_factory=dict)
    pending_forward_by_node: Dict[int, bool] = field(default_factory=dict)
    last_event_by_node: Dict[int, str] = field(default_factory=dict)
    last_reason_by_node: Dict[int, str] = field(default_factory=dict)

    def on_init(self, cfg: SimCfg) -> None:
        n = int(cfg.n_nodes)
        self.seen_by_node = {i: set() for i in range(n)}
        self.state_by_node = {i: FloodExpertState.IDLE for i in range(n)}
        self.pending_forward_by_node = {i: False for i in range(n)}
        self.last_event_by_node = {i: "" for i in range(n)}
        self.last_reason_by_node = {i: "" for i in range(n)}

    def reset_for_run(self, cfg: SimCfg, rng_stream: Optional[Any] = None) -> None:
        """Shared expert interface hook; FloodExpert ignores RNG streams."""
        _ = rng_stream
        self.on_init(cfg)

    @staticmethod
    def _key(pkt: Packet) -> PacketKey:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def has_seen(self, node_id: int, pkt: Packet) -> bool:
        return self._key(pkt) in self.seen_by_node.get(int(node_id), set())

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
        _ = cfg, now
        rx = int(rx_node)
        key = self._key(pkt)

        if key in self.seen_by_node.get(rx, set()):
            self.last_event_by_node[rx] = "EVT_DUPLICATE_RX"
            self.last_reason_by_node[rx] = "FLOOD_DUPLICATE"
            self.state_by_node[rx] = FloodExpertState.SUPPRESSED_DUPLICATE
            return []
        self.seen_by_node[rx].add(key)
        self.last_event_by_node[rx] = "EVT_NOVEL_RX"

        if rx == int(pkt.dst):
            self.pending_forward_by_node[rx] = False
            self.last_reason_by_node[rx] = "FLOOD_DESTINATION"
            self.state_by_node[rx] = FloodExpertState.SUPPRESSED_DESTINATION
            return []
        if int(pkt.ttl_hops) <= 0:
            self.pending_forward_by_node[rx] = False
            self.last_reason_by_node[rx] = "FLOOD_TTL_ZERO"
            self.state_by_node[rx] = FloodExpertState.SUPPRESSED_TTL_ZERO
            return []

        self.pending_forward_by_node[rx] = True
        self.last_reason_by_node[rx] = "FLOOD_RELAY"
        self.state_by_node[rx] = FloodExpertState.NOVEL_RECEIVED
        fwd = dataclasses.replace(pkt, ttl_hops=int(pkt.ttl_hops) - 1).with_meta(
            policy=self.name,
            fwd=self.name,
            prev_hop=int(prev_hop) if prev_hop is not None else -1,
            ttl_dec=True,
        )
        return [(rx, fwd)]

    def on_transmission_result(self, node_id: int, succeeded: bool) -> None:
        node = int(node_id)
        if self.state_by_node.get(node) != FloodExpertState.NOVEL_RECEIVED:
            return
        if bool(succeeded):
            self.pending_forward_by_node[node] = False
            self.last_event_by_node[node] = "EVT_TX_OK"
            self.state_by_node[node] = FloodExpertState.RELAYED
        else:
            self.pending_forward_by_node[node] = True
            self.last_event_by_node[node] = "EVT_TX_FAIL"

    def on_timer(self, context: Optional[Any] = None) -> None:
        _ = context
        return None
