"""
NodeReliabilityState: per-node decision-time reliability context for Phase 6.

Phase 6 passive state tracking. Forwarding decisions are NOT changed here.

All features are:
- Directly local (queue state, airtime budget)
- Locally estimable from rolling history (rolling-window rates)
- Locally observable from received signals (neighbor table, RSSI)

No future-outcome features. No post-run labels. No final TTL, delivery outcome,
or future delay. All values are available at the time a forwarding decision
would be made during simulation.

Per-packet features (packet_ttl_remaining, packet_age_s, packet_priority) are
NOT stored here — they are per-decision and must be extracted from the packet
at decision time.

Assumptions:
  A-012 — Rolling window size W = 50 events (experimental design choice).
  A-013 — Neighbor density max age = 30.0 s (experimental design choice).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


# ---------------------------------------------------------------------------
# Rolling binary window
# ---------------------------------------------------------------------------

class RollingBinaryWindow:
    """
    Fixed-size rolling window of binary (0/1) outcomes.

    Semantics:
    - Push 1 for "success / event occurred".
    - Push 0 for "failure / event did not occur".
    - rate = sum / len when non-empty; 0.0 when empty.
    - When full, oldest entry is evicted (FIFO).

    Deterministic: given the same push() call sequence, snapshot() is identical.
    No RNG.
    """

    def __init__(self, maxlen: int = 50) -> None:
        if int(maxlen) < 1:
            raise ValueError("maxlen must be >= 1")
        self._maxlen: int = int(maxlen)
        self._buf: Deque[int] = deque(maxlen=self._maxlen)

    def push(self, outcome: int) -> None:
        """Push 0 (failure/absence) or 1 (success/presence)."""
        self._buf.append(int(bool(outcome)))

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def count(self) -> int:
        return len(self._buf)

    @property
    def successes(self) -> int:
        return int(sum(self._buf))

    @property
    def rate(self) -> float:
        """Fraction of entries that are 1. Returns 0.0 if empty."""
        n = len(self._buf)
        if n == 0:
            return 0.0
        return float(sum(self._buf)) / float(n)

    def reset(self) -> None:
        self._buf.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "maxlen": self._maxlen,
            "count": self.count,
            "successes": self.successes,
            "rate": round(self.rate, 6),
        }


# ---------------------------------------------------------------------------
# Neighbor observation and table
# ---------------------------------------------------------------------------

@dataclass
class NeighborObservation:
    """
    Single record of a locally observed neighbor.

    t_first_seen: simulation time of the FIRST observation (preserved on updates).
    t_last_seen:  simulation time of the MOST RECENT observation.

    Tracking both times is required for a non-degenerate stability_estimate().
    """
    neighbor_id: int
    t_first_seen: float
    t_last_seen: float
    rssi_dbm: float


class NeighborTable:
    """
    Tracks recently observed neighbors via received signals.

    A neighbor is recorded when this node successfully decodes a packet from it
    (rx_ok event). This is locally observable: real BLE devices can measure RSSI
    and detect packet reception (A-008).

    Density estimate: count of distinct neighbors heard within max_age_s.

    Stability estimate: fraction of neighbors seen in the recent half of the
    age window that were also seen in the older half. Returns 0.0 if the recent
    half is empty (insufficient history).

    Assumption A-013: max_age_s = 30.0 s (experimental design choice).
    Limitation: short simulations (<= 30 s) treat all time as "recent".

    No RNG. Deterministic given the same observe() call sequence.
    """

    def __init__(self, max_age_s: float = 30.0) -> None:
        if float(max_age_s) <= 0.0:
            raise ValueError("max_age_s must be > 0")
        self._max_age_s: float = float(max_age_s)
        self._table: Dict[int, NeighborObservation] = {}

    @property
    def max_age_s(self) -> float:
        return self._max_age_s

    def observe(self, neighbor_id: int, t_now: float, rssi_dbm: float) -> None:
        """
        Record or update an observation of neighbor_id at t_now.

        On the FIRST observation, t_first_seen = t_now.
        On subsequent updates, t_first_seen is PRESERVED (only t_last_seen and rssi_dbm change).
        Preserving t_first_seen is required for a non-degenerate stability_estimate().
        """
        nid = int(neighbor_id)
        t = float(t_now)
        existing = self._table.get(nid)
        if existing is not None:
            self._table[nid] = NeighborObservation(
                neighbor_id=nid,
                t_first_seen=existing.t_first_seen,  # preserved
                t_last_seen=t,
                rssi_dbm=float(rssi_dbm),
            )
        else:
            self._table[nid] = NeighborObservation(
                neighbor_id=nid,
                t_first_seen=t,
                t_last_seen=t,
                rssi_dbm=float(rssi_dbm),
            )

    def _active_entries(self, t_now: float) -> Dict[int, NeighborObservation]:
        cutoff = float(t_now) - self._max_age_s
        return {nid: obs for nid, obs in self._table.items() if obs.t_last_seen >= cutoff}

    def density_estimate(self, t_now: float) -> int:
        """Count of distinct neighbors heard within max_age_s."""
        return len(self._active_entries(float(t_now)))

    def stability_estimate(self, t_now: float) -> float:
        """
        Fraction of recently-heard neighbors that have been known for at least
        half the age window (max_age_s / 2).

        A neighbor is "recently heard" if t_last_seen >= t_now - max_age_s/2.
        A recently-heard neighbor is "stable" if t_first_seen <= t_now - max_age_s/2
        (i.e., it was first observed BEFORE the midpoint of the age window).

        Returns 0.0 when no neighbors have been heard recently.

        This is non-degenerate when t_first_seen is tracked independently of
        t_last_seen (requires the two-field NeighborObservation). Assumes A-013.
        """
        t = float(t_now)
        half = self._max_age_s / 2.0
        recent_cutoff = t - half          # boundary: recent half vs older half
        max_age_cutoff = t - self._max_age_s   # boundary: active vs expired

        recent_set: set = set()
        stable_set: set = set()

        for nid, obs in self._table.items():
            if obs.t_last_seen < max_age_cutoff:
                continue  # neighbor has aged out entirely
            if obs.t_last_seen >= recent_cutoff:
                recent_set.add(nid)
                # Stable = first observed before the midpoint (known for >= half the window)
                if obs.t_first_seen <= recent_cutoff:
                    stable_set.add(nid)

        if not recent_set:
            return 0.0
        return float(len(stable_set)) / float(len(recent_set))

    def rssi_by_neighbor(self, t_now: float) -> Dict[int, float]:
        """Last observed RSSI (dBm) per active neighbor. Locally observable (A-008)."""
        return {
            int(nid): float(obs.rssi_dbm)
            for nid, obs in self._active_entries(float(t_now)).items()
        }

    def snapshot(self, t_now: float) -> Dict[str, Any]:
        active = self._active_entries(float(t_now))
        return {
            "density_estimate": int(len(active)),
            "stability_estimate": round(self.stability_estimate(float(t_now)), 6),
            "rssi_by_neighbor": {
                int(nid): round(float(obs.rssi_dbm), 3)
                for nid, obs in active.items()
            },
        }


# ---------------------------------------------------------------------------
# Per-node reliability state
# ---------------------------------------------------------------------------

class NodeReliabilityState:
    """
    Decision-time reliability context for one node.

    Rolling windows (assumption A-012, W=50):
    - cca_busy_window:      per MAC_TX_CH event — 1 if CCA busy, 0 if TX proceeds
    - tx_success_window:    per TX burst (last channel) — 1 if >=1 eligible rx decoded, 0 if not
    - collision_window:     per TX burst (last channel) — 1 if any channel had interference, 0 if not
    - rx_success_window:    per scanning RX opportunity — 1 if PHY decode OK, 0 if fail
    - not_scanning_window:  per waveform opportunity — 1 if receiver was not scanning, 0 if scanning

    Neighbor table (assumption A-013, max_age_s=30.0s):
    - Tracks recently heard neighbors with RSSI (locally observable).

    Feature leakage notes:
    - tx_success_window (recent_forwarding_success_rate): In simulation, the TX node
      observes whether >=1 eligible receiver decoded, via the engine's global visibility.
      On real BLE ADV hardware, there are no ACKs — this is NOT directly observable.
      Must be flagged as "simulation-estimable only" during Phase 9 feature leakage audit.
    - All other features are locally observable or locally estimable on real hardware.
    - energy_proxy_consumed_j: airtime × 1.0 mW proxy. NOT A-005 device supply energy.
      Do not use in quantitative battery or device-lifetime claims.

    No future-outcome features: none of the rolling windows record post-run outcomes.
    All push() calls happen at the time the event occurs during simulation.
    """

    # Forbidden fields: these must NEVER appear in a snapshot. Used by leakage tests.
    FORBIDDEN_FIELDS = frozenset({
        "delivered",
        "final_delivered",
        "delivery_success",
        "final_ttl",
        "future_delay",
        "post_run_label",
        "was_delivered",
        "eventual_delivery",
        "final_hop_count",
    })

    def __init__(
        self,
        node_id: int,
        window_size: int = 50,
        density_max_age_s: float = 30.0,
    ) -> None:
        self.node_id: int = int(node_id)
        self.window_size: int = int(window_size)

        # TX-side rolling windows
        self.cca_busy_window = RollingBinaryWindow(maxlen=window_size)
        self.tx_success_window = RollingBinaryWindow(maxlen=window_size)
        self.collision_window = RollingBinaryWindow(maxlen=window_size)

        # RX-side rolling windows
        self.rx_success_window = RollingBinaryWindow(maxlen=window_size)
        self.not_scanning_window = RollingBinaryWindow(maxlen=window_size)

        # Neighbor table (locally observable from received signals)
        self.neighbor_table = NeighborTable(max_age_s=float(density_max_age_s))

        # Cumulative counters (full-run context; not rolling)
        self.total_tx_attempts: int = 0
        self.total_tx_success: int = 0
        self.total_cca_busy: int = 0
        self.total_rx_ok: int = 0
        self.total_rx_fail: int = 0
        self.total_not_scanning: int = 0
        self.total_budget_deferrals: int = 0
        self.total_overlap_rx: int = 0

        # Energy proxy (airtime × 1.0 mW; NOT A-005 device supply energy)
        self.energy_proxy_consumed_j: float = 0.0

    # -------- TX-side update methods --------

    def observe_cca_check(self, busy: bool) -> None:
        """
        Called for each MAC_TX_CH event at this node.
        busy=True: CCA detected channel occupied (push 1).
        busy=False: CCA clear, TX proceeds (push 0).
        """
        if busy:
            self.total_cca_busy += 1
            self.cca_busy_window.push(1)
        else:
            self.cca_busy_window.push(0)

    def observe_tx_attempt(self) -> None:
        """Called when this node starts a TX on a channel (CCA clear)."""
        self.total_tx_attempts += 1

    def observe_tx_burst_outcome(self, success: bool, had_interference: bool) -> None:
        """
        Called at the last channel of each TX burst attempt.
        success: whether >=1 eligible receiver decoded OK.
        had_interference: whether any channel had overlapping waveforms.
        """
        if success:
            self.total_tx_success += 1
            self.tx_success_window.push(1)
        else:
            self.tx_success_window.push(0)

        self.collision_window.push(1 if had_interference else 0)

    def observe_budget_deferral(self) -> None:
        """Called when this node's TX is deferred by the airtime budget."""
        self.total_budget_deferrals += 1

    def observe_airtime(self, dt_s: float) -> None:
        """
        Called when this node transmits a waveform.
        Accumulates airtime-proportional energy proxy (NOT A-005 device energy).
        """
        self.energy_proxy_consumed_j += float(dt_s)

    # -------- RX-side update methods --------

    def observe_rx_ok(self, t_now: float, from_node: int, rssi_dbm: float) -> None:
        """
        Called when this node decodes a packet successfully (PHY OK).
        Records the neighbor observation (RSSI locally observable per A-008).
        """
        self.total_rx_ok += 1
        self.rx_success_window.push(1)
        self.neighbor_table.observe(int(from_node), float(t_now), float(rssi_dbm))

    def observe_rx_fail(self) -> None:
        """Called when this node evaluates a waveform and PHY decode fails."""
        self.total_rx_fail += 1
        self.rx_success_window.push(0)

    def observe_not_scanning(self) -> None:
        """
        Called when a waveform is present but this node was not scanning.
        Pushes 1 to not_scanning_window (missed opportunity).
        """
        self.total_not_scanning += 1
        self.not_scanning_window.push(1)

    def observe_scanning_rx_opportunity(self) -> None:
        """
        Called when a waveform is present and this node WAS scanning
        (regardless of PHY outcome). Pushes 0 to not_scanning_window.
        """
        self.not_scanning_window.push(0)

    def observe_overlap_rx(self) -> None:
        """Called when this node evaluates a waveform under interference."""
        self.total_overlap_rx += 1

    # -------- Snapshot --------

    def snapshot(
        self,
        t_now: float,
        queue_len: int,
        max_queue: int,
        airtime_bucket_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export all node-level decision-time features at t_now.

        Per-packet features (packet_ttl_remaining, packet_age_s, packet_priority)
        are NOT included — they are per-decision and must be extracted from the
        packet at decision time.

        All features in this snapshot are available at decision time during
        simulation. See module docstring for leakage notes on individual features.
        """
        ql = int(queue_len)
        mq = max(1, int(max_queue))
        queue_pressure = min(1.0, float(ql) / float(mq))

        if airtime_bucket_snapshot is not None:
            tokens = float(airtime_bucket_snapshot.get("tokens_s", 0.0))
            capacity = float(airtime_bucket_snapshot.get("capacity_s", 1.0))
            airtime_budget_remaining_s: Optional[float] = round(tokens, 9)
            cap_safe = capacity if capacity > 0.0 else 1.0
            airtime_budget_pressure = max(0.0, min(1.0, 1.0 - tokens / cap_safe))
        else:
            airtime_budget_remaining_s = None
            airtime_budget_pressure = 0.0

        neighbor_snap = self.neighbor_table.snapshot(float(t_now))

        return {
            "node_id": int(self.node_id),
            # Queue (directly local)
            "queue_length": ql,
            "queue_pressure": round(queue_pressure, 6),
            # TX-side rolling windows
            "recent_forwarding_success_rate": round(self.tx_success_window.rate, 6),
            "recent_cca_busy_rate": round(self.cca_busy_window.rate, 6),
            "recent_collision_or_overlap_rate": round(self.collision_window.rate, 6),
            # RX-side rolling windows
            "recent_receive_success_rate": round(self.rx_success_window.rate, 6),
            "recent_not_scanning_rate": round(self.not_scanning_window.rate, 6),
            # Airtime budget (directly local if enabled)
            "airtime_budget_remaining_s": airtime_budget_remaining_s,
            "airtime_budget_pressure": round(airtime_budget_pressure, 6),
            # Energy proxy (airtime-proportional; NOT A-005 device energy)
            "energy_proxy_consumed_j": round(self.energy_proxy_consumed_j, 9),
            # Neighbor / density (locally observable)
            "local_density_estimate": int(neighbor_snap["density_estimate"]),
            "neighbor_stability_estimate": float(neighbor_snap["stability_estimate"]),
            "link_rssi_by_neighbor": neighbor_snap["rssi_by_neighbor"],
            # Cumulative counters (verification / logging; not primary rolling features)
            "total_tx_attempts": int(self.total_tx_attempts),
            "total_tx_success": int(self.total_tx_success),
            "total_cca_busy": int(self.total_cca_busy),
            "total_rx_ok": int(self.total_rx_ok),
            "total_rx_fail": int(self.total_rx_fail),
            "total_not_scanning": int(self.total_not_scanning),
            "total_budget_deferrals": int(self.total_budget_deferrals),
        }
