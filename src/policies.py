"""
Routing policies: pluggable Protocol interface; Flood, NoFwd, BatmanLite variants.

Extracted from notebook Cell 10.5, Phase 3.
Phase 8: added ControlledFloodPolicy and EnergyAwareHeuristicPolicy.

Safe naming rules (Phase 8):
- BatmanLitePolicy is a "BatmanLite-style route-quality fallback," NOT full BATMAN-ADV.
- Do NOT claim AODV unless full route discovery/reply/error/sequence-numbers are implemented.
- ControlledFloodPolicy is a "deterministic-fraction flood," NOT the Ni et al. (2000)
  probabilistic rebroadcast protocol.
- EnergyAwareHeuristicPolicy is a "rate-limiting relay heuristic," NOT a
  hardware-validated energy protocol.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .config import SimCfg
from .entities import Packet


# ----------------------------
# Canonical Policy protocol
# ----------------------------

class Policy(Protocol):
    """
    Canonical policy interface (duck-typed by SimEngine).
    Must be deterministic given (cfg, now, pkt, prev_hop) + its internal state.
    """
    name: str

    def on_init(self, cfg: SimCfg) -> None:
        ...

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        ...

    def on_rx(
        self,
        cfg: SimCfg,
        now: float,
        rx_node: int,
        pkt: Packet,
        prev_hop: Optional[int],
    ) -> List[Tuple[int, Packet]]:
        ...


# ----------------------------
# Phase D+ policies
# ----------------------------

@dataclass
class FloodPolicy:
    """
    Deterministic flood:
    - duplicate suppression per node by (flow_id, seq)
    - forward from any non-dst node while ttl>0
    """
    name: str = "flood"
    seen_by_node: Dict[int, set] = dataclasses.field(default_factory=dict)

    def on_init(self, cfg: SimCfg) -> None:
        self.seen_by_node = {i: set() for i in range(int(cfg.n_nodes))}

    @staticmethod
    def _key(pkt: Packet) -> Tuple[int, int]:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
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

        if key in self.seen_by_node.get(rx, set()):
            return []
        self.seen_by_node[rx].add(key)

        if rx == int(pkt.dst):
            return []
        if int(pkt.ttl_hops) <= 0:
            return []

        fwd = dataclasses.replace(pkt, ttl_hops=int(pkt.ttl_hops) - 1).with_meta(
            policy=self.name,
            fwd=self.name,
            prev_hop=int(prev_hop) if prev_hop is not None else -1,
            ttl_dec=True,
        )
        return [(rx, fwd)]


@dataclass
class NoFwdPolicy:
    """
    No-forward (contention / baseline MAC testing):
    - only sources transmit
    - receivers never forward
    """
    name: str = "no_fwd"

    def on_init(self, cfg: SimCfg) -> None:
        return

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        return [(int(pkt.src), pkt.with_meta(policy=self.name))]

    def on_rx(
        self,
        cfg: SimCfg,
        now: float,
        rx_node: int,
        pkt: Packet,
        prev_hop: Optional[int],
    ) -> List[Tuple[int, Packet]]:
        return []


@dataclass
class BatmanLitePolicy:
    """
    BatmanLite-style route-quality fallback baseline. Phase 3+.

    SAFE CLAIM ONLY: This is a simplified first-path heuristic. It is NOT BATMAN-ADV,
    NOT full BATMAN protocol, and NOT AODV. No route discovery, route reply, route error,
    sequence numbers, or route maintenance protocol is implemented. Do not call this
    BATMAN or AODV in any paper, report, or claim.

    Mechanism: records the first observed prev_hop as the next_hop target for forwarding.
    Unicasts subsequent hops toward that upstream neighbor.

    State per node:
    - seen[(flow,seq)] : bool duplicate suppression
    - parent[(flow,seq)] : first prev_hop observed when receiving this pid

    Known limitation: forwards "toward" the upstream neighbor (prev_hop), which is
    topologically backward in many network configurations. seen_by_node dedup prevents
    forwarding loops. Preserved intentionally as a simplified heuristic baseline.

    candidate_neighbor: populated from next_hop in forwarded packet meta (= prev_hop).
    """
    name: str = "batman_lite"
    seen_by_node: Dict[int, set] = dataclasses.field(default_factory=dict)
    parent_by_node: Dict[int, Dict[Tuple[int, int], int]] = dataclasses.field(default_factory=dict)

    def on_init(self, cfg: SimCfg) -> None:
        n = int(cfg.n_nodes)
        self.seen_by_node = {i: set() for i in range(n)}
        self.parent_by_node = {i: {} for i in range(n)}

    @staticmethod
    def _key(pkt: Packet) -> Tuple[int, int]:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
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

        if key in self.seen_by_node.get(rx, set()):
            return []
        self.seen_by_node[rx].add(key)

        if rx == int(pkt.dst):
            return []

        if int(pkt.ttl_hops) <= 0:
            return []

        if prev_hop is None:
            return []
        ph = int(prev_hop)
        self.parent_by_node[rx][key] = ph

        fwd = dataclasses.replace(pkt, ttl_hops=int(pkt.ttl_hops) - 1).with_meta(
            policy=self.name,
            fwd=self.name,
            prev_hop=ph,
            ttl_dec=True,
            next_hop=ph,
        )
        return [(rx, fwd)]


@dataclass
class ControlledFloodPolicy:
    """
    Deterministic controlled flooding baseline. Phase 8.

    Extends Flood with a hash-based relay-fraction gate: only a deterministic
    subset of nodes relay each packet. Reduces airtime and duplicate transmissions
    compared to full Flood while preserving duplicate suppression and TTL control.

    Relay decision: hash(rx_node, flow_id, seq) % 100 < relay_fraction_pct
      - Fully deterministic: same (node, packet) pair → same relay/suppress decision
      - relay_fraction_pct=100 → equivalent to Flood
      - relay_fraction_pct=0  → no relay (equivalent to NoForward)
      - Default: 50 (approximately 50% of nodes relay at each hop)

    Hash uses multiplicative constants (Knuth-style) to spread bits across the low
    100-bucket modulus. No RNG is used.

    Information used: pkt.pid (duplicate suppression), pkt.ttl_hops, pkt.dst,
    rx_node ID, relay_fraction_pct (static config). No future information.

    Claim level: simulation-based deterministic-fraction flood.
    NOT the probabilistic rebroadcast in Ni et al. (2000).
    NOT Bluetooth Mesh Managed Flood.
    candidate_neighbor: -1 (broadcast).
    """
    name: str = "controlled_flood"
    relay_fraction_pct: int = 50
    seen_by_node: Dict[int, set] = dataclasses.field(default_factory=dict)

    def on_init(self, cfg: SimCfg) -> None:
        self.seen_by_node = {i: set() for i in range(int(cfg.n_nodes))}
        self.relay_fraction_pct = int(self.relay_fraction_pct)

    @staticmethod
    def _key(pkt: Packet) -> Tuple[int, int]:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def _should_relay(self, rx_node: int, pkt: Packet) -> bool:
        """Deterministic relay gate: hash of (node, packet_id) % 100 < relay_fraction_pct."""
        h = (
            int(rx_node) * 2654435761
            ^ int(pkt.pid.flow_id) * 2246822519
            ^ int(pkt.pid.seq) * 3266489917
        ) & 0xFFFFFFFF
        return (int(h) % 100) < int(self.relay_fraction_pct)

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
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

        if key in self.seen_by_node.get(rx, set()):
            return []
        self.seen_by_node[rx].add(key)

        if rx == int(pkt.dst):
            return []
        if int(pkt.ttl_hops) <= 0:
            return []

        if not self._should_relay(rx, pkt):
            return []

        fwd = dataclasses.replace(pkt, ttl_hops=int(pkt.ttl_hops) - 1).with_meta(
            policy=self.name,
            fwd=self.name,
            prev_hop=int(prev_hop) if prev_hop is not None else -1,
            ttl_dec=True,
        )
        return [(rx, fwd)]


@dataclass
class EnergyAwareHeuristicPolicy:
    """
    Relay-rate-limiting resource heuristic baseline. Phase 8.

    MECHANISM: Suppresses low-priority (data) relay at nodes that have already
    relayed >= max_relays_per_window data packets in the last window_s seconds.
    Control packets bypass the suppression gate unconditionally.

    RESOURCE SIGNAL USED: Per-node relay-count rate (policy-internal relay timestamps).
    This is a relay-count proxy for resource pressure, NOT the actual energy or
    airtime budget signals from the simulator's NodeReliabilityState.

    IMPORTANT LIMITATIONS AND SAFE CLAIMS:
    - Does NOT use energy_proxy_consumed_j from NodeReliabilityState.
    - Does NOT use airtime_budget_remaining from AirtimeTokenBucket.
    - Does NOT use nRF52840 hardware current values (A-005 is not wired in).
    - Makes no hardware-energy or quantitative power-consumption claims.
    - The relay-count window is an experimental design choice, not calibrated to
      real BLE hardware or measured energy behavior.
    - This baseline is named "energy-aware" only in the sense that relay-rate pressure
      is a coarse proxy for resource consumption. The Phase 9 ML advisory will use
      actual node-local reliability state (Phase 6 features: energy_proxy_consumed_j,
      airtime_budget_remaining, queue_pressure) as input features.

    Safe paper description: "relay-rate-limiting resource heuristic baseline
    (uses relay-count rate as a resource-pressure proxy; does not use hardware
    energy measurements or airtime budget state)."

    Parameters:
      max_relays_per_window: maximum data relays per node per window (default 5)
      window_s: sliding time window in seconds (default 1.0)

    Information used: pkt.is_control, pkt.pid (dedup), pkt.ttl_hops, pkt.dst,
    now (simulation time), relay_times (policy-internal). No RNG.

    Determinism: fully deterministic given the same event sequence and now values.

    Claim level: relay-rate-limiting resource heuristic. NOT hardware-validated.
    NOT battery-aware. NOT a Bluetooth-standard energy management protocol.
    candidate_neighbor: -1 (broadcast when relaying).
    """
    name: str = "energy_aware"
    max_relays_per_window: int = 5
    window_s: float = 1.0
    seen_by_node: Dict[int, set] = dataclasses.field(default_factory=dict)
    relay_times: Dict[int, List[float]] = dataclasses.field(default_factory=dict)

    def on_init(self, cfg: SimCfg) -> None:
        n = int(cfg.n_nodes)
        self.seen_by_node = {i: set() for i in range(n)}
        self.relay_times = {i: [] for i in range(n)}
        self.max_relays_per_window = int(self.max_relays_per_window)
        self.window_s = float(self.window_s)

    @staticmethod
    def _key(pkt: Packet) -> Tuple[int, int]:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def _recent_relay_count(self, node: int, now: float) -> int:
        """Count recent relays by this node; prune expired entries in-place."""
        cutoff = float(now) - float(self.window_s)
        times = self.relay_times.get(node, [])
        pruned = [t for t in times if t >= cutoff]
        self.relay_times[node] = pruned
        return len(pruned)

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
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

        if key in self.seen_by_node.get(rx, set()):
            return []
        self.seen_by_node[rx].add(key)

        if rx == int(pkt.dst):
            return []
        if int(pkt.ttl_hops) <= 0:
            return []

        # Relay-rate suppression: only for low-priority (data) packets
        if not bool(pkt.is_control):
            recent_count = self._recent_relay_count(rx, float(now))
            if recent_count >= int(self.max_relays_per_window):
                return []

        # Record this relay event
        if rx not in self.relay_times:
            self.relay_times[rx] = []
        self.relay_times[rx].append(float(now))

        fwd = dataclasses.replace(pkt, ttl_hops=int(pkt.ttl_hops) - 1).with_meta(
            policy=self.name,
            fwd=self.name,
            prev_hop=int(prev_hop) if prev_hop is not None else -1,
            ttl_dec=True,
        )
        return [(rx, fwd)]


# ----------------------------
# Stable policy factory
# ----------------------------

def policy_factory(name: str, **kwargs: Any) -> Any:
    """
    Canonical policy factory.

    Supported:
    - "flood"
    - "no_fwd"
    - "batman_lite" / "batman" / "batman-lite"
    - "controlled_flood" / "cflood"   [Phase 8]
    - "energy_aware"                   [Phase 8]
    - "drf_ble" / "reliability_gated" [Phase 10R]
    - "fr_drf" / "frdrf"               [Phase 12D headline name] FR-DRF: Frontier-aware
      Redundancy-gated Decision-Regime Forwarding with TTL Rescue (BLE ADV-inspired,
      deterministic, node-local, ML-free; NOT a routing-protocol claim). Aliases:
      "bridge_drf" / "bridge" (backward-compat; FORWARD_BRIDGE_PROXY default-off, diagnostic-only)
    - "cgf_experimental"               [Phase 13B diagnostic only] conservative CGF policy.
    - "cgf_r_experimental"             [Phase 13B-R diagnostic only] default-off CGF-R
      path-depth suppression veto enabled, width-4 only. The bare "cgf" production name
      is deliberately NOT registered.

    kwargs are forwarded to the policy constructor for parameterised policies
    (e.g., relay_fraction_pct for ControlledFloodPolicy).

    NOTE: "drf_ble_advisor_forced" is NOT registered here. That negative-control
    subclass must be imported directly and requires allow_negative_control=True.
    """
    n = str(name).strip().lower()
    if n == "flood":
        return FloodPolicy()
    if n == "no_fwd":
        return NoFwdPolicy()
    if n in ("batman", "batman-lite", "batman_lite"):
        return BatmanLitePolicy()
    if n in ("controlled_flood", "cflood"):
        return ControlledFloodPolicy(**kwargs)
    if n in ("energy_aware", "energy_aware_heuristic"):
        return EnergyAwareHeuristicPolicy(**kwargs)
    if n in ("drf_ble", "reliability_gated"):
        from .reliability_gated_policy import ReliabilityGatedPolicy
        return ReliabilityGatedPolicy(**kwargs)
    if n in ("fr_drf", "frdrf", "bridge_drf", "bridge"):
        # FR-DRF (Frontier-aware Redundancy-gated Decision-Regime Forwarding with TTL Rescue) is
        # the headline name (Phase 12D / Option D). "bridge_drf"/"bridge" are backward-compat
        # aliases for the same class; FORWARD_BRIDGE_PROXY is default-off (diagnostic-only).
        from .bridge_drf_policy import BridgeDrfPolicy
        return BridgeDrfPolicy(**kwargs)
    if n == "cgf_experimental":
        from .cgf_policy import CgfPolicy
        return CgfPolicy(**kwargs)
    if n == "cgf_r_experimental":
        from .cgf_policy import CgfPolicy
        params = {
            "name": "cgf_r_experimental",
            "hint_width_bytes": 4,
            "gossip_hint_bytes": 4,
            "charge_overhead": True,
            "enable_path_depth_veto": True,
            "path_depth_max": 2,
        }
        params.update(kwargs)
        return CgfPolicy(**params)
    if "advisor_forced" in n or "negative_control" in n:
        raise ValueError(
            f"Policy {name!r} is a negative-control diagnostic instrument and is NOT "
            "registered in policy_factory(). Import ReliabilityGatedPolicyAdvisorForced "
            "directly and pass allow_negative_control=True."
        )
    raise ValueError(f"Unsupported policy: {name!r}")
