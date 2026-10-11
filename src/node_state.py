"""
NodeStateTracker: manages per-node NodeReliabilityState for all nodes in a run.

Phase 6 passive state tracking. Forwarding decisions are NOT changed here.

NodeStateTracker is created once per SimEngine run (inside SimEngine.__init__).
It receives event-driven update calls from the engine during the run loop.
At the end of the run, snapshot_all() exports all node-level reliability features.

Assumptions:
  A-012 — Window size W = 50 events (experimental design choice).
  A-013 — Neighbor density max_age_s = 30.0 s (experimental design choice).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .airtime import AirtimeTokenBucket
from .entities import NodeState
from .reliability_state import NodeReliabilityState


# Per-packet features (not stored in NodeReliabilityState; documented here for Phase 7)
PACKET_FEATURES = (
    "packet_ttl_remaining",   # pkt.ttl_hops at decision time
    "packet_age_s",           # t_now - pkt.created_t
    "packet_priority",        # int(pkt.is_control)
)

# Fields that must NEVER appear in node-local state snapshots (leakage guard)
FORBIDDEN_SNAPSHOT_FIELDS = frozenset(NodeReliabilityState.FORBIDDEN_FIELDS)


class NodeStateTracker:
    """
    Manages per-node NodeReliabilityState objects for all nodes in a run.

    Usage:
        tracker = NodeStateTracker(n_nodes=cfg.n_nodes)
        # Engine calls: tracker.on_cca_busy(node_id), tracker.on_rx_ok(...), etc.
        # At end of run:
        snapshots = tracker.snapshot_all(t_now, nodes, max_queue, airtime_buckets)

    No RNG. Deterministic given the same event call sequence.
    """

    def __init__(
        self,
        n_nodes: int,
        *,
        window_size: int = 50,
        density_max_age_s: float = 30.0,
    ) -> None:
        if int(n_nodes) < 1:
            raise ValueError("n_nodes must be >= 1")
        if int(window_size) < 1:
            raise ValueError("window_size must be >= 1")
        if float(density_max_age_s) <= 0.0:
            raise ValueError("density_max_age_s must be > 0")

        self.n_nodes: int = int(n_nodes)
        self.window_size: int = int(window_size)
        self.density_max_age_s: float = float(density_max_age_s)

        self._states: Dict[int, NodeReliabilityState] = {
            i: NodeReliabilityState(
                node_id=i,
                window_size=self.window_size,
                density_max_age_s=self.density_max_age_s,
            )
            for i in range(self.n_nodes)
        }

    def state(self, node_id: int) -> NodeReliabilityState:
        """Return the NodeReliabilityState for the given node."""
        return self._states[int(node_id)]

    # -------- TX-side hooks --------

    def on_cca_check(self, node_id: int, busy: bool) -> None:
        """
        Called for each MAC_TX_CH event at node_id.
        busy=True when CCA found channel occupied; False when TX proceeds.
        """
        self._states[int(node_id)].observe_cca_check(busy)

    def on_tx_attempt(self, node_id: int) -> None:
        """Called when node_id starts a TX on a channel (CCA clear)."""
        self._states[int(node_id)].observe_tx_attempt()

    def on_tx_burst_outcome(self, node_id: int, success: bool, had_interference: bool) -> None:
        """
        Called at the last channel of each TX burst for node_id.
        success: >=1 eligible receiver decoded OK.
        had_interference: any channel in this burst had overlapping waveforms.
        """
        self._states[int(node_id)].observe_tx_burst_outcome(success, had_interference)

    def on_budget_deferral(self, node_id: int) -> None:
        """Called when node_id's TX is deferred by airtime budget."""
        self._states[int(node_id)].observe_budget_deferral()

    def on_airtime(self, node_id: int, dt_s: float) -> None:
        """Called when node_id transmits a waveform of duration dt_s."""
        self._states[int(node_id)].observe_airtime(float(dt_s))

    # -------- RX-side hooks --------

    def on_rx_ok(self, node_id: int, t_now: float, from_node: int, rssi_dbm: float) -> None:
        """Called when node_id decodes a packet successfully from from_node."""
        self._states[int(node_id)].observe_rx_ok(float(t_now), int(from_node), float(rssi_dbm))

    def on_rx_fail(self, node_id: int) -> None:
        """Called when node_id evaluates a waveform but PHY decode fails."""
        self._states[int(node_id)].observe_rx_fail()

    def on_not_scanning(self, node_id: int) -> None:
        """Called when a waveform is present but node_id was not scanning."""
        self._states[int(node_id)].observe_not_scanning()

    def on_scanning_rx_opportunity(self, node_id: int) -> None:
        """Called when a waveform is present and node_id WAS scanning."""
        self._states[int(node_id)].observe_scanning_rx_opportunity()

    def on_overlap_rx(self, node_id: int) -> None:
        """Called when node_id evaluates a waveform under interference."""
        self._states[int(node_id)].observe_overlap_rx()

    # -------- Snapshot --------

    def snapshot_all(
        self,
        t_now: float,
        nodes: Dict[int, NodeState],
        max_queue: int,
        airtime_buckets: Optional[Dict[int, AirtimeTokenBucket]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Export decision-time features for all nodes at t_now.

        Returns:
            Dict mapping node_id (int) → feature dict.

        Per-packet features are NOT included (see PACKET_FEATURES module constant).
        """
        result: Dict[int, Dict[str, Any]] = {}
        for nid, rs in self._states.items():
            ns = nodes.get(nid)
            ql = int(len(ns.tx_queue)) if ns is not None else 0

            bucket = None
            if airtime_buckets is not None:
                bucket = airtime_buckets.get(nid)
            bucket_snap = bucket.snapshot(float(t_now)) if bucket is not None else None

            result[nid] = rs.snapshot(
                t_now=float(t_now),
                queue_len=ql,
                max_queue=int(max_queue),
                airtime_bucket_snapshot=bucket_snap,
            )
        return result

    def packet_features(self, t_now: float, pkt: Any) -> Dict[str, Any]:
        """
        Extract per-packet decision-time features from a packet at t_now.

        These are NOT stored in NodeReliabilityState (they are per-decision,
        not per-node). Call this at decision time alongside snapshot() for
        Phase 7 decision-level logging.

        Leakage note: all values are available at decision time from the packet
        and the simulation clock. None of these are post-run labels.
        """
        try:
            ttl = int(pkt.ttl_hops)
        except Exception:
            ttl = -1
        try:
            created = float(pkt.created_t)
            age_s = max(0.0, float(t_now) - created)
        except Exception:
            age_s = -1.0
        try:
            priority = int(bool(pkt.is_control))
        except Exception:
            priority = 0
        return {
            "packet_ttl_remaining": ttl,
            "packet_age_s": round(age_s, 6),
            "packet_priority": priority,
        }
