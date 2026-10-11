"""
Phase 7: Decision-level logging for BLE forwarding simulation.

One row = one forwarding decision opportunity.

A forwarding decision opportunity occurs when a relay node successfully decodes
a packet and the policy decides whether to forward it. State is captured
immediately BEFORE policy.on_rx() is called; the decision outcome is recorded
immediately after.

Feature/label separation:
- INPUT_FEATURE_COLUMNS (whitelist): approved decision-time ML input candidates.
- LABEL_COLUMNS: post-run outcomes (eventual_delivered, delivery_delay_s,
  final_ttl_remaining). These are None when the row is first logged and filled
  after the run completes via fill_labels(). They must NEVER be used as inputs.

Excluded simulation-estimable features (NOT in INPUT_FEATURE_COLUMNS):
- recent_forwarding_success_rate: requires global TX-RX visibility; no BLE ADV ACKs.
- recent_collision_or_overlap_rate: requires global overlap ground truth.

neighbor_stability feature note (P6-A fixed in pre-Phase-9 cleanup):
- neighbor_stability: NeighborTable now tracks t_first_seen per neighbor.
  stability_estimate() is non-degenerate for established neighbors.
  Still excluded from INPUT_FEATURE_COLUMNS until full Phase 9 ML audit.

Do NOT import or call any ML training or inference code from this module.
Do NOT use end-of-run snapshot_all() as a decision-time snapshot.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Collection, Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Schema version constants (E1/E4 — Phase 10R)
# ---------------------------------------------------------------------------

# v1: 30-column schema (Phase 7/8/9).
#   - Original Phase 7 hash (28-col, before N2 context columns were added):
#       af180263faf56abeb369c7859d862b8afa37724088c7bd28ddab83fded51c951
#     This is a HISTORICAL artifact, not the current v1 hash.
#   - Confirmed v1 (30-col, flood, easy_cfg, seed=1):
#       bf77934bffbc35d632520d2e666ed06bed786b96a2ebb04249eac7dab0ddfe03
DECISION_LOG_SCHEMA_VERSION_V1: str = "v1"
DECISION_LOG_SCHEMA_VERSION_V1_28COL_HISTORICAL_HASH: str = (
    "af180263faf56abeb369c7859d862b8afa37724088c7bd28ddab83fded51c951"
)
DECISION_LOG_SCHEMA_V1_REF_HASH: str = (
    "bf77934bffbc35d632520d2e666ed06bed786b96a2ebb04249eac7dab0ddfe03"
)

# v2: 31-column schema (Phase 10+). Adds `reason_code` after `suppression_used`.
#   v2 reference hash (flood, easy_cfg, seed=1) is locked in Phase 10R report.
DECISION_LOG_SCHEMA_VERSION: str = "v2"


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

DECISION_ROW_COLUMNS: Tuple[str, ...] = (
    # Identity / context
    "run_id",
    "cfg_hash",
    "seed",
    "time_s",
    "packet_id",
    "current_node",
    "previous_node",
    "destination_node",
    "candidate_neighbor",
    # Per-packet features (decision-time)
    "ttl_remaining",
    "hop_count",
    "packet_age_s",
    "packet_priority",
    # Node-level features (decision-time; from NodeReliabilityState)
    "queue_length",
    "queue_pressure",
    "airtime_budget_remaining",
    "energy_proxy_consumed_j",
    "recent_receive_success_rate",
    "recent_cca_busy_rate",
    "recent_not_scanning_rate",
    "neighbor_stability",   # logged; EXCLUDED from INPUT_FEATURE_COLUMNS (pending Phase 9 re-evaluation)
    "local_density",
    # Decision metadata (decision-time)
    "decision_taken",
    "fallback_used",
    "suppression_used",
    # Phase 10R: reason code — which DRF-BLE rule produced this decision.
    # "" for non-DRF-BLE policies. In FORBIDDEN_INPUT_FIELDS (decision action, not input feature).
    "reason_code",
    # Decision context — diagnostic (N2 resolution; not ML input features)
    "is_first_rx_for_node",   # 1 if first eligible decode of this (node, flow_id, seq) pair
    "is_destination_rx",      # 1 if current_node == destination_node (destination-side row)
    # Labels (post-run; None at decision time; filled by fill_labels())
    "eventual_delivered",
    "delivery_delay_s",
    "final_ttl_remaining",
)

# Approved ML input features — whitelist.
# Any column not in this set must NOT be used as a model input in Phase 9.
INPUT_FEATURE_COLUMNS: FrozenSet[str] = frozenset({
    "ttl_remaining",
    "hop_count",
    "packet_age_s",
    "packet_priority",
    "queue_length",
    "queue_pressure",
    "airtime_budget_remaining",
    "energy_proxy_consumed_j",
    "recent_receive_success_rate",
    "recent_cca_busy_rate",
    "recent_not_scanning_rate",
    # "neighbor_stability" is EXCLUDED until the degeneracy is fixed (Phase 6 report).
    "local_density",
})

# Post-run outcome labels. Must never appear in INPUT_FEATURE_COLUMNS.
LABEL_COLUMNS: FrozenSet[str] = frozenset({
    "eventual_delivered",
    "delivery_delay_s",
    "final_ttl_remaining",
})

# Simulation-estimable features: logged for reference but excluded from whitelist.
# These require global simulator visibility not available on real BLE hardware.
SIMULATION_ESTIMABLE_FEATURES: FrozenSet[str] = frozenset({
    "recent_forwarding_success_rate",    # requires global TX-RX visibility; no ACKs
    "recent_collision_or_overlap_rate",  # requires global overlap ground truth
})

# Fields that must NEVER appear as ML input features in any context.
# This set is used by the forbidden-field guard in assert_valid_input_features().
FORBIDDEN_INPUT_FIELDS: FrozenSet[str] = frozenset(
    LABEL_COLUMNS
    | SIMULATION_ESTIMABLE_FEATURES
    | {
        # Names from NodeReliabilityState.FORBIDDEN_FIELDS
        "delivered",
        "final_delivered",
        "delivery_success",
        "final_ttl",
        "future_delay",
        "post_run_label",
        "was_delivered",
        "eventual_delivery",
        "final_hop_count",
        # N5: action/decision columns — must never be circular ML inputs
        "decision_taken",    # the action being predicted; circular if used as input
        "fallback_used",     # Phase 10 policy action; not decision-time input
        "suppression_used",  # Phase 10 policy action; not decision-time input
        # Phase 10R: reason_code is a decision action, not a decision-time input feature
        "reason_code",
        # N2: context/diagnostic columns — not ML features
        "is_first_rx_for_node",  # logging artifact; not directly node-observable
        "is_destination_rx",     # context only; correlated with decision_taken=0
    }
)

# Verify internal invariants at module load time
assert INPUT_FEATURE_COLUMNS.isdisjoint(LABEL_COLUMNS), \
    "BUG: input features overlap with labels"
assert INPUT_FEATURE_COLUMNS.isdisjoint(FORBIDDEN_INPUT_FIELDS), \
    "BUG: input features overlap with forbidden fields"
assert LABEL_COLUMNS.issubset(FORBIDDEN_INPUT_FIELDS), \
    "BUG: labels are not all forbidden as input"


# ---------------------------------------------------------------------------
# Whitelist / blacklist guard
# ---------------------------------------------------------------------------

def assert_valid_input_features(feature_names: Collection[str]) -> None:
    """
    Assert that every name in feature_names is a safe, approved ML input feature.

    Raises ValueError if:
    1. Any name is in FORBIDDEN_INPUT_FIELDS (explicit prohibition).
    2. Any name is not in INPUT_FEATURE_COLUMNS (not in whitelist).

    Call this before training any ML model to prevent accidental label leakage
    or use of simulation-estimable-only features.
    """
    names = set(feature_names)
    forbidden = names & FORBIDDEN_INPUT_FIELDS
    if forbidden:
        raise ValueError(
            f"Forbidden fields used as ML input features: {sorted(forbidden)}. "
            f"These are future-outcome labels or simulation-estimable-only features."
        )
    unknown = names - INPUT_FEATURE_COLUMNS
    if unknown:
        raise ValueError(
            f"Unknown or unapproved fields used as ML input features: {sorted(unknown)}. "
            f"Only columns in INPUT_FEATURE_COLUMNS are approved. "
            f"Approved: {sorted(INPUT_FEATURE_COLUMNS)}"
        )


# ---------------------------------------------------------------------------
# Canonical row serialization
# ---------------------------------------------------------------------------

def _normalize_value(v: Any) -> Any:
    """Normalize a value for stable JSON serialization."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return round(v, 9)
    return v


def canonical_row_hash(row: Dict[str, Any]) -> str:
    """
    Compute a deterministic sha256 hash of a decision row.

    Only DECISION_ROW_COLUMNS are included, in column order.
    Values are normalized for stable serialization (bool→int, float→rounded).
    """
    ordered = [
        [k, _normalize_value(row.get(k))]
        for k in DECISION_ROW_COLUMNS
    ]
    serialized = json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# DecisionLogger
# ---------------------------------------------------------------------------

class DecisionLogger:
    """
    Passive decision-level logger for one simulation run.

    Usage in SimEngine:
        logger = DecisionLogger(run_id=f"{cfg_hash}_{seed}", cfg_hash=h, seed=seed)
        eng = SimEngine(..., decision_logger=logger)
        result = eng.run(injections)
        # After run(): result["decision_log"] contains the filled rows
        # logger.rows() also returns the same filled rows

    Thread safety: not thread-safe (single-threaded simulation assumed).
    """

    def __init__(self, run_id: str, cfg_hash: str, seed: int) -> None:
        self.run_id: str = str(run_id)
        self.cfg_hash: str = str(cfg_hash)
        self.seed: int = int(seed)

        self._rows: List[Dict[str, Any]] = []
        self._row_pid_keys: List[Tuple[int, int]] = []  # (flow_id, seq) per row
        self._row_created_t: List[float] = []           # pkt.created_t per row

        # Map packet_id → original TTL (for hop_count computation).
        # Populated by on_packet_created() at packet injection time.
        self._orig_ttl: Dict[Tuple[int, int], int] = {}

        # N2: track (node_id, flow_id, seq) tuples already logged.
        # Used to compute is_first_rx_for_node (1 if first eligible decode for this pair).
        self._seen_node_pkt: set = set()  # Set[Tuple[int, int, int]]

    # -------- Packet lifecycle --------

    def on_packet_created(self, pid_key: Tuple[int, int], orig_ttl: int) -> None:
        """
        Register the original TTL for a newly injected packet.

        Called from SimEngine._enqueue_app_injection() immediately after
        the Packet object is created. Must be called before log_decision()
        is called for this packet.

        pid_key: (flow_id, seq)
        orig_ttl: pkt.ttl_hops at creation time (before any relay decrements)
        """
        self._orig_ttl[(int(pid_key[0]), int(pid_key[1]))] = int(orig_ttl)

    # -------- Decision logging --------

    def log_decision(
        self,
        *,
        t_now: float,
        packet: Any,
        current_node: int,
        previous_node: int,
        node_snap: Dict[str, Any],
        pkt_feats: Dict[str, Any],
        decision_taken: bool,
        fallback_used: bool = False,
        suppression_used: bool = False,
        candidate_neighbor: int = -1,
        reason_code: str = "",
    ) -> None:
        """
        Log one forwarding decision opportunity.

        Call this AFTER capturing node_snap and pkt_feats from Phase 6 helpers,
        but AFTER policy.on_rx() has returned (to record the decision outcome).
        All state in node_snap and pkt_feats must have been captured BEFORE
        policy.on_rx() was called.

        Labels (eventual_delivered, delivery_delay_s, final_ttl_remaining) are
        initialized to None and filled later by fill_labels().

        Args:
            t_now: simulation time of the decision (seconds)
            packet: the Packet being forwarded (pkt.ttl_hops, pkt.dst, etc.)
            current_node: relay node making the forwarding decision
            previous_node: node that transmitted this packet to current_node
            node_snap: output of NodeReliabilityState.snapshot() for current_node
            pkt_feats: output of NodeStateTracker.packet_features() for this packet
            decision_taken: True if policy.on_rx() returned non-empty actions
            fallback_used: True if deterministic fallback was invoked (Phase 10)
            suppression_used: True if forwarding was suppressed by policy (Phase 10)
            candidate_neighbor: the intended next-hop node ID when the policy selects a
                specific neighbor (e.g. BatmanLite sets next_hop in packet meta);
                -1 for broadcast policies (flood, controlled_flood, energy_aware).
                Extracted from the forwarded packet's next_hop meta field by mac_engine.
        """
        try:
            flow_id = int(packet.pid.flow_id)
            seq = int(packet.pid.seq)
        except Exception:
            flow_id = -1
            seq = -1

        pid_key = (flow_id, seq)
        packet_id = f"{flow_id}:{seq}"

        try:
            dst = int(packet.dst)
        except Exception:
            dst = -1

        try:
            created_t = float(packet.created_t)
        except Exception:
            created_t = 0.0

        # Compute hop_count from tracked orig_ttl
        orig_ttl = self._orig_ttl.get(pid_key, -1)
        ttl_remaining = int(pkt_feats.get("packet_ttl_remaining", -1))
        if orig_ttl >= 0 and ttl_remaining >= 0:
            hop_count = int(orig_ttl) - int(ttl_remaining)
        else:
            hop_count = -1

        # N2: context columns — is this the first eligible decode for (node, packet)?
        node_pkt_key = (int(current_node), flow_id, seq)
        is_first_rx = node_pkt_key not in self._seen_node_pkt
        self._seen_node_pkt.add(node_pkt_key)
        is_dst_rx = (int(current_node) == int(dst))

        # Extract node-level features from snapshot
        ql = int(node_snap.get("queue_length", 0))
        qp = float(node_snap.get("queue_pressure", 0.0))
        budget = node_snap.get("airtime_budget_remaining_s")  # None if disabled
        energy = float(node_snap.get("energy_proxy_consumed_j", 0.0))
        rx_succ = float(node_snap.get("recent_receive_success_rate", 0.0))
        cca_busy = float(node_snap.get("recent_cca_busy_rate", 0.0))
        not_scan = float(node_snap.get("recent_not_scanning_rate", 0.0))
        stability = float(node_snap.get("neighbor_stability_estimate", 0.0))
        density = int(node_snap.get("local_density_estimate", 0))

        row: Dict[str, Any] = {
            # Identity / context
            "run_id": self.run_id,
            "cfg_hash": self.cfg_hash,
            "seed": self.seed,
            "time_s": round(float(t_now), 9),
            "packet_id": packet_id,
            "current_node": int(current_node),
            "previous_node": int(previous_node),
            "destination_node": dst,
            "candidate_neighbor": int(candidate_neighbor),  # -1 for broadcast; populated for next-hop policies
            # Per-packet features (decision-time)
            "ttl_remaining": ttl_remaining,
            "hop_count": hop_count,
            "packet_age_s": round(float(pkt_feats.get("packet_age_s", 0.0)), 6),
            "packet_priority": int(pkt_feats.get("packet_priority", 0)),
            # Node-level features (decision-time)
            "queue_length": ql,
            "queue_pressure": round(qp, 6),
            "airtime_budget_remaining": round(float(budget), 9) if budget is not None else None,
            "energy_proxy_consumed_j": round(energy, 9),
            "recent_receive_success_rate": round(rx_succ, 6),
            "recent_cca_busy_rate": round(cca_busy, 6),
            "recent_not_scanning_rate": round(not_scan, 6),
            "neighbor_stability": round(stability, 6),  # degenerate; not in whitelist
            "local_density": density,
            # Decision metadata (decision-time)
            "decision_taken": int(bool(decision_taken)),
            "fallback_used": int(bool(fallback_used)),
            "suppression_used": int(bool(suppression_used)),
            # Phase 10R: reason code (empty string for non-DRF-BLE policies)
            "reason_code": str(reason_code),
            # Decision context — diagnostic (N2 resolution; not ML input features)
            "is_first_rx_for_node": int(is_first_rx),   # 1 = first channel decode for this (node, pkt)
            "is_destination_rx": int(is_dst_rx),         # 1 = current_node is the packet destination
            # Labels — None until fill_labels() is called
            "eventual_delivered": None,
            "delivery_delay_s": None,
            "final_ttl_remaining": None,
        }

        self._rows.append(row)
        self._row_pid_keys.append(pid_key)
        self._row_created_t.append(created_t)

    # -------- Post-run label fill --------

    def fill_labels(
        self,
        delivery_outcomes: Dict[Tuple[int, int], Dict[str, Any]],
    ) -> None:
        """
        Fill post-run label columns for all logged rows.

        delivery_outcomes: dict mapping (flow_id, seq) → {
            "delivery_time": float,
            "final_ttl_remaining": int,
        }

        Rows for packets that were NOT delivered get:
            eventual_delivered = False
            delivery_delay_s = None
            final_ttl_remaining = None

        Rows for packets that WERE delivered get:
            eventual_delivered = True
            delivery_delay_s = delivery_time - pkt.created_t
            final_ttl_remaining = int

        This must be called AFTER the simulation run completes.
        These are post-run labels, not decision-time features.
        """
        for i, row in enumerate(self._rows):
            pid_key = self._row_pid_keys[i]
            created_t = self._row_created_t[i]
            outcome = delivery_outcomes.get(pid_key)
            if outcome is not None:
                row["eventual_delivered"] = True
                delivery_t = float(outcome["delivery_time"])
                row["delivery_delay_s"] = round(delivery_t - float(created_t), 6)
                row["final_ttl_remaining"] = int(outcome["final_ttl_remaining"])
            else:
                row["eventual_delivered"] = False
                row["delivery_delay_s"] = None
                row["final_ttl_remaining"] = None

    # -------- Output --------

    def rows(self) -> List[Dict[str, Any]]:
        """Return a copy of all logged decision rows."""
        return [dict(r) for r in self._rows]

    def canonical_row_hash(self, row: Dict[str, Any]) -> str:
        """Deterministic sha256 hash of a single decision row."""
        return canonical_row_hash(row)

    def log_hash(self) -> str:
        """
        Deterministic sha256 hash over all logged rows.

        Identical for repeated runs with the same config + seed.
        """
        row_hashes = [canonical_row_hash(r) for r in self._rows]
        combined = repr(row_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
