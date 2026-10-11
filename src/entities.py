"""
Core entities: DropReason, PacketId, Packet, NodeState, RunStats, PacketIdGen.

Extracted from notebook Cell 4, Phase 3.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


# ----------------------------
# Drop reasons (frozen contract)
# ----------------------------

class DropReason(str, Enum):
    TTL_EXPIRED = "TTL_EXPIRED"
    NO_ROUTE = "NO_ROUTE"
    QUEUE_FULL = "QUEUE_FULL"

    NOT_SCANNING = "NOT_SCANNING"
    CCA_BUSY = "CCA_BUSY"
    COLLISION = "COLLISION"
    PHY_ERROR = "PHY_ERROR"
    DUPLICATE = "DUPLICATE"
    POLICY_SUPPRESSED = "POLICY_SUPPRESSED"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    BUDGET = "BUDGET"


# ----------------------------
# Packet and IDs
# ----------------------------

@dataclass(frozen=True)
class PacketId:
    flow_id: int
    seq: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.flow_id, self.seq)


@dataclass(frozen=True)
class Packet:
    """
    Core packet representation.
    Immutable for deterministic logging.
    """
    pid: PacketId
    src: int
    dst: int
    created_t: float
    ttl_hops: int
    payload_bytes: int
    is_control: bool = False
    meta: Tuple[Tuple[str, Any], ...] = tuple()

    def with_meta(self, **kwargs: Any) -> "Packet":
        items = dict(self.meta)
        items.update(kwargs)
        return dataclasses.replace(self, meta=tuple(sorted(items.items(), key=lambda kv: kv[0])))


# ----------------------------
# Node state
# ----------------------------

@dataclass
class NodeState:
    node_id: int

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    tx_queue: List[Packet] = field(default_factory=list)

    sent_data: int = 0
    sent_ctrl: int = 0
    rx_data: int = 0
    rx_ctrl: int = 0
    dropped: Dict[DropReason, int] = field(default_factory=dict)

    def drop(self, reason: DropReason, n: int = 1) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + n


# ----------------------------
# Per-run results container (SCHEMA v4)
# ----------------------------

@dataclass
class RunStats:
    """
    Aggregated statistics for a single run.

    Rules:
    - dropped / dropped_by_reason: FINAL packet drops only.
    - Event counters are not drops.
    - RX counters are receiver-side PHY decode outcomes among scanning receivers only.
    """
    cfg_hash: str
    seed: int

    sim_duration_s: float = 0.0
    n_nodes: int = 0

    app_injected: int = 0
    delivered: int = 0
    dropped: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)

    tx_attempts: int = 0
    tx_success: int = 0

    rx_total: int = 0
    rx_ok: int = 0
    rx_fail: int = 0

    overlap_events: int = 0
    overlap_waveforms: int = 0
    overlap_rx_evals: int = 0
    overlap_interferer_pairs: int = 0

    rx_opportunities: int = 0
    rx_scans: int = 0

    cca_busy_events: int = 0

    not_scanning_events: int = 0
    not_scanning_rx_opportunities: int = 0
    not_scanning_waveforms: int = 0

    not_scanning_dst_events: int = 0

    airtime_s: float = 0.0
    energy_j: float = 0.0

    airtime_data_s: float = 0.0
    airtime_ctrl_s: float = 0.0

    budget_deferrals: int = 0

    def summary(self) -> Dict[str, Any]:
        pdr = (self.delivered / self.app_injected) if self.app_injected > 0 else 0.0

        return {
            "cfg_hash": self.cfg_hash,
            "seed": self.seed,
            "n_nodes": self.n_nodes,
            "duration_s": self.sim_duration_s,
            "app_injected": self.app_injected,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "pdr": round(pdr, 6),

            "airtime_s": round(self.airtime_s, 9),
            "energy_j": round(self.energy_j, 9),
            "airtime_data_s": round(self.airtime_data_s, 9),
            "airtime_ctrl_s": round(self.airtime_ctrl_s, 9),

            "dropped_by_reason": dict(self.dropped_by_reason),

            "cca_busy_events": int(self.cca_busy_events),

            "not_scanning_events": int(self.not_scanning_events),
            "not_scanning_rx_opportunities": int(self.not_scanning_rx_opportunities),
            "not_scanning_waveforms": int(self.not_scanning_waveforms),
            "not_scanning_dst_events": int(self.not_scanning_dst_events),

            "budget_deferrals": int(self.budget_deferrals),

            "overlap_events": int(self.overlap_events),
            "overlap_waveforms": int(self.overlap_waveforms),
            "overlap_rx_evals": int(self.overlap_rx_evals),
            "overlap_interferer_pairs": int(self.overlap_interferer_pairs),

            "tx_attempts": int(self.tx_attempts),
            "tx_success": int(self.tx_success),
            "rx_total": int(self.rx_total),
            "rx_ok": int(self.rx_ok),
            "rx_fail": int(self.rx_fail),

            "rx_opportunities": int(self.rx_opportunities),
            "rx_scans": int(self.rx_scans),
        }

    def add_drop(self, reason: DropReason, n: int = 1) -> None:
        """
        Record a FINAL packet drop.
        Do not use this for deferrals/retries/event observations.
        """
        key = reason.value
        self.dropped_by_reason[key] = self.dropped_by_reason.get(key, 0) + int(n)
        self.dropped += int(n)


# ----------------------------
# Deterministic packet id generator (local, no globals)
# ----------------------------

@dataclass
class PacketIdGen:
    flow_id: int
    _next_seq: int = 0

    def next(self) -> PacketId:
        pid = PacketId(flow_id=int(self.flow_id), seq=int(self._next_seq))
        self._next_seq += 1
        return pid
