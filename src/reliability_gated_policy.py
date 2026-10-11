"""
Phase 10R: Reliability-Gated DRF-BLE Forwarding Policy.

Protocol identity:
- BLE ADV-inspired, simulation-based, infrastructure-disrupted mesh conditions.
- Deterministic decision authority — no ML drives forwarding decisions.
- Learned advisor is inert by construction: advisor_authorized is a non-configurable
  False field (init=False, re-asserted in __post_init__); cannot be enabled via
  constructor, policy_factory kwargs, config, CLI, or environment variables.
- NOT Bluetooth Mesh compliant; NOT BATMAN-ADV; NOT AODV; NOT guaranteed delivery.
- Energy/airtime proxy only (NOT A-005 hardware current).

Safe claim vocabulary: BLE ADV-inspired, standards-aligned (BT-MESH-PROFILE-11 TTL),
  deterministic decision authority, advisory ML inert unless authorized,
  node-local reliability state, controlled stress scenario, simulation-based evidence.

Decision rule order (strict priority):
  Step 0: Eligibility / duplicate-suppression checks (no reason code — not a decision)
  Step 1: TTL-critical deterministic forwarding (A-014; best-effort at queue layer)
  Step 2: Resource-pressure suppression (A-015; all data packets, including emergency)
  Step 3: Route-quality directed-relay fallback — computes optional action (Option C; does not return)
  Step 4: ML authorization gate — inert for normal policy (advisor_authorized=False returns
           route-quality candidate byte-identical); forced subclass may suppress Step 3 candidate
  Step 5: Default deterministic fallback relay (only when Step 3 produced no candidate)

P3 NOTE: All application data packets (including the p5_priority_emergency "emergency"
  flow) have is_control=False → packet_priority=0. Step 2 suppresses ALL data packets
  equally when queue is full. There is no per-flow priority distinction at relay nodes.
  is_control=True is reserved for MAC-level control frames only.

Phase 11 ablation parameters (Tier 4 controlled stress design choices):
  ttl_critical_threshold: int = TTL_CRITICAL_THRESHOLD   (A-014 default = 1)
    - ttl_critical_threshold=0 → Step 1 never fires (no TTL-critical protection)
    - ttl_critical_threshold=2 → Step 1 fires when ttl_hops <= 2
  resource_pressure_threshold: float = RESOURCE_PRESSURE_THRESHOLD  (A-015 default = 0.8)
    - resource_pressure_threshold=1.1 → Step 2 never fires (no resource suppression)
    - resource_pressure_threshold ∈ {0.6, 0.7, 0.9} → sensitivity sweep
  use_route_quality: bool = True
    - use_route_quality=False → Step 3 always skips (falls to Step 4/5)
  Defaults preserve Phase 10R behavior exactly. These are NOT advisor_authorized.

Claim boundaries (Research_Contract_v2.md):
  - "energy proxy only; NOT A-005 nRF52840 supply energy"
  - "TTL-critical protection is best-effort at the queue layer; engine may still drop
    when tx_queue is full"
  - "resource suppression applies to all data packets (packet_priority=0) equally"
  - "no ML improvement claim; advisor_authorized=False"
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import SimCfg
from .entities import Packet
from .decision_logger import assert_valid_input_features, INPUT_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Advisor feature assembly
# ---------------------------------------------------------------------------

def build_advisor_input_features(
    node_snap: Dict[str, Any],
    pkt: Any,
    now: float,
) -> Dict[str, Any]:
    """
    Build the full 12-feature advisor input vector matching Phase 9B feature assembly.

    Sources (must match decision_logger.log_decision() + NodeStateTracker.packet_features()):

        node_snap keys (direct name matches):
            queue_length, queue_pressure, recent_cca_busy_rate,
            recent_not_scanning_rate, recent_receive_success_rate,
            energy_proxy_consumed_j

        node_snap keys (name remapping):
            local_density_estimate      → 'local_density'
            airtime_budget_remaining_s  → 'airtime_budget_remaining'
                                          (None when budget disabled → 0.0,
                                           matching Phase 9B training-data behavior)

        pkt fields:
            pkt.ttl_hops                → 'ttl_remaining'
            pkt.meta['orig_ttl']
              - pkt.ttl_hops            → 'hop_count'
              (orig_ttl stored by on_app_inject; matches decision_logger hop_count)
            now - pkt.created_t         → 'packet_age_s'
            pkt.is_control              → 'packet_priority' (int(bool(...)))

    Raises ValueError if any required source is unavailable (no silent defaults).
    """
    features: Dict[str, Any] = {}

    # --- Node-snapshot features: direct name matches ---
    _DIRECT = (
        "queue_length",
        "queue_pressure",
        "recent_cca_busy_rate",
        "recent_not_scanning_rate",
        "recent_receive_success_rate",
        "energy_proxy_consumed_j",
    )
    _snap_keys_printable = [
        k for k in node_snap if not isinstance(node_snap.get(k), dict)
    ]
    for col in _DIRECT:
        if col not in node_snap:
            raise ValueError(
                f"Required advisor feature '{col}' missing from node_snap. "
                f"Available scalar keys: {sorted(_snap_keys_printable)}"
            )
        val = node_snap[col]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            raise ValueError(
                f"Required advisor feature '{col}' is None or NaN in node_snap."
            )
        features[col] = val

    # --- Node-snapshot: local_density ← local_density_estimate ---
    if "local_density_estimate" not in node_snap:
        raise ValueError(
            "Required advisor feature source 'local_density_estimate' missing from "
            f"node_snap (maps to 'local_density'). "
            f"Available scalar keys: {sorted(_snap_keys_printable)}"
        )
    ld = node_snap["local_density_estimate"]
    if ld is None or (isinstance(ld, float) and math.isnan(ld)):
        raise ValueError("Required source 'local_density_estimate' is None or NaN.")
    features["local_density"] = ld

    # --- Node-snapshot: airtime_budget_remaining ← airtime_budget_remaining_s ---
    # None when airtime budgeting is disabled; convert to -1.0 to match Phase 9B training
    # (train_phase9b_loso_artifacts.py uses fillna(-1.0) for this column).
    if "airtime_budget_remaining_s" not in node_snap:
        raise ValueError(
            "Required advisor feature source 'airtime_budget_remaining_s' missing from "
            f"node_snap (maps to 'airtime_budget_remaining'). "
            f"Available scalar keys: {sorted(_snap_keys_printable)}"
        )
    ab = node_snap["airtime_budget_remaining_s"]
    features["airtime_budget_remaining"] = float(ab) if ab is not None else -1.0

    # --- Packet features ---
    # ttl_remaining ← pkt.ttl_hops (matches decision_logger: packet_ttl_remaining = int(pkt.ttl_hops))
    try:
        ttl_remaining = int(pkt.ttl_hops)
    except Exception as exc:
        raise ValueError(
            f"Cannot read pkt.ttl_hops for required feature 'ttl_remaining': {exc}"
        ) from exc
    features["ttl_remaining"] = ttl_remaining

    # hop_count ← orig_ttl - ttl_remaining (matches decision_logger hop_count computation).
    # orig_ttl is stored in pkt.meta by on_app_inject() at injection time.
    pkt_meta: Dict[str, Any] = dict(pkt.meta) if hasattr(pkt, "meta") else {}
    orig_ttl = pkt_meta.get("orig_ttl")
    if orig_ttl is None:
        raise ValueError(
            "Required advisor feature 'hop_count' cannot be assembled: "
            "'orig_ttl' is absent from pkt.meta. "
            "on_app_inject() must store orig_ttl=pkt.ttl_hops in packet metadata "
            "before any relay hops occur."
        )
    features["hop_count"] = int(orig_ttl) - int(ttl_remaining)

    # packet_age_s ← max(0.0, now - pkt.created_t) (matches NodeStateTracker.packet_features())
    try:
        packet_age_s = max(0.0, float(now) - float(pkt.created_t))
    except Exception as exc:
        raise ValueError(
            f"Cannot compute required feature 'packet_age_s' from pkt.created_t: {exc}"
        ) from exc
    features["packet_age_s"] = round(packet_age_s, 6)

    # packet_priority ← int(bool(pkt.is_control)) (matches NodeStateTracker.packet_features())
    try:
        packet_priority = int(bool(pkt.is_control))
    except Exception as exc:
        raise ValueError(
            f"Cannot read pkt.is_control for required feature 'packet_priority': {exc}"
        ) from exc
    features["packet_priority"] = packet_priority

    return features


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

RC_TTL_CRITICAL_FORWARD: str = "TTL_CRITICAL_FORWARD"
RC_RESOURCE_PRESSURE_SUPPRESS: str = "RESOURCE_PRESSURE_SUPPRESS"
RC_ROUTE_QUALITY_FORWARD: str = "ROUTE_QUALITY_FORWARD"
RC_ROUTE_QUALITY_SUPPRESS: str = "ROUTE_QUALITY_SUPPRESS"   # reserved
RC_ADVISOR_INERT_NOT_AUTHORIZED: str = "ADVISOR_INERT_NOT_AUTHORIZED"
RC_ADVISOR_OPTIONAL_NARROWING: str = "ADVISOR_OPTIONAL_NARROWING"
RC_DEFAULT_FALLBACK: str = "DEFAULT_FALLBACK"
RC_NO_TRACKER_FALLBACK: str = "NO_TRACKER_FALLBACK"

KNOWN_REASON_CODES: frozenset = frozenset({
    RC_TTL_CRITICAL_FORWARD,
    RC_RESOURCE_PRESSURE_SUPPRESS,
    RC_ROUTE_QUALITY_FORWARD,
    RC_ROUTE_QUALITY_SUPPRESS,
    RC_ADVISOR_INERT_NOT_AUTHORIZED,
    RC_ADVISOR_OPTIONAL_NARROWING,
    RC_DEFAULT_FALLBACK,
    RC_NO_TRACKER_FALLBACK,
})

# Thresholds (Tier 4 controlled stress assumptions, per assumption_ledger.md)
TTL_CRITICAL_THRESHOLD: int = 1      # A-014
RESOURCE_PRESSURE_THRESHOLD: float = 0.8  # A-015


# ---------------------------------------------------------------------------
# ReliabilityGatedPolicy
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityGatedPolicy:
    """
    Reliability-Gated DRF-BLE forwarding policy. Phase 10R.

    SAFE CLAIM ONLY:
    - BLE ADV-inspired; NOT Bluetooth Mesh compliant.
    - Deterministic decision authority; NO ML drives forwarding.
    - advisor_authorized is a non-configurable False field (init=False). It is
      re-asserted to False in __post_init__ and cannot be enabled via constructor
      arguments, policy_factory kwargs, config fields, CLI flags, or environment
      variables. The only deliberate override path is the guarded
      ReliabilityGatedPolicyAdvisorForced subclass with allow_negative_control=True.
    - Learned advisor (if attached) is INERT: stored but never consulted.
    - attach_advisor() is a STORE-ONLY operation for this class.
    - NOT BATMAN-ADV; NOT AODV; NOT guaranteed delivery.
    - Energy/airtime proxy only; NOT hardware-validated.

    Tracker modes:
    - Mode A (default, allow_no_tracker=False): raises RuntimeError if on_rx()
      reaches a state-dependent step without a tracker attached. Use for all
      research runs.
    - Mode B (allow_no_tracker=True): emits NO_TRACKER_FALLBACK reason code when
      tracker is absent. DO NOT use for Phase 11 evaluation results.
    """

    # Non-configurable False field (init=False): re-asserted in __post_init__.
    # Cannot be set via constructor, policy_factory, config, CLI, or env.
    # Alterable only via deliberate subclass (ReliabilityGatedPolicyAdvisorForced)
    # under allow_negative_control=True and explicit user approval.
    advisor_authorized: bool = dataclasses.field(default=False, init=False)

    name: str = "drf_ble"
    allow_no_tracker: bool = False

    # Phase 11 ablation parameters (Tier 4 controlled stress design choices).
    # Defaults MUST preserve Phase 10R behavior exactly (all 66 tests must still pass).
    # These are NOT advisor_authorized; they do not create any path to enable the advisor.
    ttl_critical_threshold: int = TTL_CRITICAL_THRESHOLD         # A-014 default = 1
    resource_pressure_threshold: float = RESOURCE_PRESSURE_THRESHOLD  # A-015 default = 0.8
    use_route_quality: bool = True                                # True = Step 3 active

    # Duplicate-suppression state (per-node, per (flow_id, seq))
    seen_by_node: Dict[int, set] = dataclasses.field(default_factory=dict)

    # Tracker references (set via attach_tracker())
    _nst: Any = dataclasses.field(default=None, init=False, repr=False, compare=False)
    _nodes: Any = dataclasses.field(default=None, init=False, repr=False, compare=False)
    _airtime_buckets: Any = dataclasses.field(default=None, init=False, repr=False, compare=False)

    # Advisory model store (set via attach_advisor(); never consulted for this class)
    _advisor: Any = dataclasses.field(default=None, init=False, repr=False, compare=False)

    # Per-call scratch: reason code and metadata (set on every on_rx() return path)
    _last_reason_code: str = dataclasses.field(default="", init=False, repr=False, compare=False)
    _last_decision_metadata: Dict[str, Any] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Enforce: advisor_authorized cannot be overridden via __init__ args.
        # The field is defined with init=False so it cannot be set by dataclass
        # __init__. This check is defense-in-depth.
        object.__setattr__(self, "advisor_authorized", False)

    def on_init(self, cfg: SimCfg) -> None:
        self.seen_by_node = {i: set() for i in range(int(cfg.n_nodes))}

    def attach_tracker(self, nst: Any, nodes: Any, airtime_buckets: Any = None) -> None:
        """
        Attach NodeStateTracker and node/bucket references.
        Must be called AFTER on_init() and BEFORE any on_rx() calls.
        This is the only way to give the policy access to node-local state.
        """
        self._nst = nst
        self._nodes = nodes
        self._airtime_buckets = airtime_buckets

    def attach_advisor(self, model: Any) -> None:
        """
        Store an advisory model. STORE-ONLY for this class.
        The normal ReliabilityGatedPolicy NEVER consults self._advisor because
        advisor_authorized = False. Attaching any model must not affect decisions.
        """
        self._advisor = model

    @staticmethod
    def _key(pkt: Packet) -> Tuple[int, int]:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def _get_queue_pressure(self, cfg: SimCfg, rx: int, now: float) -> float:
        """
        Read queue_pressure from NodeStateTracker using a side-effect-free peek.
        Raises RuntimeError (Mode A) or returns 0.0 (Mode B).
        """
        if self._nst is None or self._nodes is None:
            if not self.allow_no_tracker:
                raise RuntimeError(
                    "ReliabilityGatedPolicy requires attach_tracker() before on_rx() "
                    "reaches a state-dependent step. Call attach_tracker() on the policy "
                    "before running the simulation, or set allow_no_tracker=True for "
                    "legacy/test-only use (produces NO_TRACKER_FALLBACK reason code)."
                )
            return 0.0  # Mode B fallback — caller checks allow_no_tracker for reason code

        ns = self._nodes.get(int(rx))
        queue_len = int(len(ns.tx_queue)) if ns is not None else 0

        # E6: use peek() (read-only) instead of mutating snapshot()
        bucket = None
        if self._airtime_buckets is not None:
            bucket = self._airtime_buckets.get(int(rx))
        bucket_snap = bucket.peek(float(now)) if bucket is not None else None

        snap = self._nst.state(int(rx)).snapshot(
            t_now=float(now),
            queue_len=queue_len,
            max_queue=256,  # matches SimEngine._max_queue (hardcoded in engine)
            airtime_bucket_snapshot=bucket_snap,
        )
        return float(snap.get("queue_pressure", 0.0))

    def _set_result(
        self,
        reason: str,
        *,
        decision_taken: bool,
        fallback_used: bool = False,
        suppression_used: bool = False,
    ) -> None:
        self._last_reason_code = reason
        self._last_decision_metadata = {
            "reason_code": reason,
            "decision_taken": bool(decision_taken),
            "fallback_used": bool(fallback_used),
            "suppression_used": bool(suppression_used),
        }

    def _make_fwd(
        self, pkt: Packet, rx: int, prev_hop: Optional[int], *, next_hop: Optional[int] = None
    ) -> List[Tuple[int, Packet]]:
        fwd = dataclasses.replace(pkt, ttl_hops=int(pkt.ttl_hops) - 1).with_meta(
            policy=self.name,
            fwd=self.name,
            prev_hop=int(prev_hop) if prev_hop is not None else -1,
            ttl_dec=True,
            **({"next_hop": int(next_hop)} if next_hop is not None else {}),
        )
        return [(int(rx), fwd)]

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        # Store orig_ttl so _run_authorized_advisor can compute hop_count
        # (matching decision_logger: hop_count = orig_ttl - ttl_remaining).
        return [(int(pkt.src), pkt.with_meta(policy=self.name, orig_ttl=int(pkt.ttl_hops)))]

    def on_rx(
        self,
        cfg: SimCfg,
        now: float,
        rx_node: int,
        pkt: Packet,
        prev_hop: Optional[int],
    ) -> List[Tuple[int, Packet]]:
        """
        Priority-ordered deterministic decision authority.

        Step 0: Eligibility / duplicate-suppression (early return; no reason code logged)
        Step 1: TTL-critical deterministic forward
        Step 2: Resource-pressure suppression (ALL data packets, priority=0)
        Step 3: Route-quality directed-relay fallback — computes optional action (does not return)
        Step 4: ML authorization gate
                  advisor_authorized=False → return route-quality candidate (ROUTE_QUALITY_FORWARD)
                                            byte-identical to pre-Option-C Step 3 early-return
                  advisor_authorized=True (forced subclass only) → advisor may suppress candidate
        Step 5: Default deterministic fallback relay (only when Step 3 produced no candidate)
        """
        rx = int(rx_node)
        key = self._key(pkt)

        # Step 0 — eligibility checks (no reason code; not a forwarding decision)
        if key in self.seen_by_node.get(rx, set()):
            self._last_reason_code = ""
            self._last_decision_metadata = {}
            return []
        self.seen_by_node[rx].add(key)

        if rx == int(pkt.dst):
            self._last_reason_code = ""
            self._last_decision_metadata = {}
            return []

        if int(pkt.ttl_hops) <= 0:
            self._last_reason_code = ""
            self._last_decision_metadata = {}
            return []

        # Step 1 — TTL-critical deterministic forwarding protection (A-014)
        # Highest priority. Cannot be overridden by resource pressure or advisor.
        # NOTE: this is best-effort; engine may still drop if tx_queue is full.
        # Uses self.ttl_critical_threshold (Phase 11 ablation; default=TTL_CRITICAL_THRESHOLD=1).
        if int(pkt.ttl_hops) <= self.ttl_critical_threshold:
            self._set_result(RC_TTL_CRITICAL_FORWARD, decision_taken=True, fallback_used=True)
            return self._make_fwd(pkt, rx, prev_hop)

        # Step 2 — Resource-pressure suppression (A-015)
        # Suppresses ALL data packets (packet_priority=0) when queue is full.
        # P3 NOTE: is_control=True bypasses; application data (including "emergency"
        # flows) have is_control=False and are subject to suppression.
        # Uses self.resource_pressure_threshold (Phase 11 ablation; default=0.8).
        packet_priority = int(bool(pkt.is_control))
        if packet_priority == 0:
            if self._nst is None and self.allow_no_tracker:
                # Mode B: no tracker → emit NO_TRACKER_FALLBACK, relay by default
                self._set_result(
                    RC_NO_TRACKER_FALLBACK, decision_taken=True, fallback_used=True
                )
                return self._make_fwd(pkt, rx, prev_hop)

            queue_pressure = self._get_queue_pressure(cfg, rx, now)
            if queue_pressure >= self.resource_pressure_threshold:
                self._set_result(
                    RC_RESOURCE_PRESSURE_SUPPRESS,
                    decision_taken=False,
                    suppression_used=True,
                )
                return []

        # Step 3 — Route-quality directed-relay fallback (BatmanLite-style first-path)
        # NOT BATMAN-ADV; NOT AODV. No route discovery/reply/error/sequence numbers.
        # Option C: computes optional directed-relay action but does not return immediately.
        # This makes Step 4 structurally reachable for the forced-advisor diagnostic path.
        # next_hop=prev_hop (directed relay); not a DEFAULT_FALLBACK broadcast relay.
        # Uses self.use_route_quality (Phase 11 ablation; default=True).
        optional_action: Optional[List[Tuple[int, Packet]]] = None
        if self.use_route_quality and prev_hop is not None:
            optional_action = self._make_fwd(pkt, rx, prev_hop, next_hop=int(prev_hop))

        # Step 4 — ML authorization gate
        # When optional_action is set (Step 3 has a route-quality candidate):
        #   advisor_authorized=False → return route-quality candidate unchanged.
        #     reason_code=ROUTE_QUALITY_FORWARD; byte-identical to pre-Option-C Step 3 behavior.
        #   advisor_authorized=True (ReliabilityGatedPolicyAdvisorForced only) → advisor may suppress.
        # When optional_action is None (Step 3 produced no candidate) → fall through to Step 5.
        # The advisor cannot synthesize a relay where Step 3 produced none.
        if optional_action is not None:
            if self.advisor_authorized:
                # Forced-advisor path: advisor may suppress the Step 3 route-quality candidate.
                # _run_authorized_advisor sets RC_ADVISOR_OPTIONAL_NARROWING and returns [] if it suppresses,
                # or returns None if it does not suppress (caller returns the route-quality candidate).
                result = self._run_authorized_advisor(cfg, now, rx, pkt, prev_hop)
                if result is not None:
                    return result
            # Normal path (advisor_authorized=False) or advisor did not suppress:
            # return route-quality candidate — ROUTE_QUALITY_FORWARD, byte-identical to pre-Option-C.
            self._set_result(RC_ROUTE_QUALITY_FORWARD, decision_taken=True)
            return optional_action

        # Step 5 — Default deterministic fallback relay
        # Only reached when Step 3 produced no route-quality candidate
        # (use_route_quality=False or prev_hop is None).
        self._set_result(RC_DEFAULT_FALLBACK, decision_taken=True, fallback_used=True)
        return self._make_fwd(pkt, rx, prev_hop)

    def _run_authorized_advisor(
        self,
        cfg: SimCfg,
        now: float,
        rx: int,
        pkt: Packet,
        prev_hop: Optional[int],
    ) -> Optional[List[Tuple[int, Packet]]]:
        """
        Authorized advisor inference path (only reachable in ReliabilityGatedPolicyAdvisorForced).
        May only NARROW the Step 3 route-quality relay candidate (suppress it).
        Cannot force a relay where Step 3 produced none, cannot override Step 1 TTL-critical,
        and cannot bypass Step 2 resource-pressure suppression.
        Returns [] with RC_ADVISOR_OPTIONAL_NARROWING set if suppressing;
        returns None if not suppressing (caller returns the route-quality candidate unchanged).
        """
        if self._advisor is None:
            # No model attached — do not suppress; caller returns route-quality candidate
            return None

        # Gather whitelisted features for inference
        if self._nst is not None and self._nodes is not None:
            ns = self._nodes.get(int(rx))
            queue_len = int(len(ns.tx_queue)) if ns is not None else 0
            bucket = None
            if self._airtime_buckets is not None:
                bucket = self._airtime_buckets.get(int(rx))
            bucket_snap = bucket.peek(float(now)) if bucket is not None else None
            node_snap = self._nst.state(int(rx)).snapshot(
                t_now=float(now),
                queue_len=queue_len,
                max_queue=256,
                airtime_bucket_snapshot=bucket_snap,
            )
        else:
            node_snap = {}

        # Build full 12-feature advisor vector matching Phase 9B decision-log assembly.
        # Raises ValueError if any required source is unavailable (no silent defaults).
        features = build_advisor_input_features(node_snap, pkt, float(now))

        # Hard feature guard: all 12 INPUT_FEATURE_COLUMNS must be present,
        # no extra forbidden/leaky fields, and all keys must be whitelisted.
        _missing = set(INPUT_FEATURE_COLUMNS) - set(features.keys())
        if _missing:
            raise ValueError(
                f"Advisor feature assembly incomplete: missing {sorted(_missing)} "
                f"from the required INPUT_FEATURE_COLUMNS. "
                "Bug in build_advisor_input_features()."
            )
        _extra = set(features.keys()) - set(INPUT_FEATURE_COLUMNS)
        if _extra:
            raise ValueError(
                f"Advisor feature assembly produced forbidden/extra fields: "
                f"{sorted(_extra)}. Only INPUT_FEATURE_COLUMNS are allowed."
            )
        # Whitelist check (also catches any forbidden fields)
        assert_valid_input_features(list(features.keys()))

        # Advisory inference: may only suppress the Step 3 route-quality relay candidate
        try:
            suppress = bool(self._advisor.predict_suppress(features, pkt))
        except Exception:
            suppress = False

        if suppress:
            # Advisor suppresses the Step 3 route-quality relay candidate
            self._set_result(
                RC_ADVISOR_OPTIONAL_NARROWING, decision_taken=False, suppression_used=True
            )
            return []
        # Advisor does not suppress — caller returns the route-quality candidate unchanged
        return None


# ---------------------------------------------------------------------------
# ReliabilityGatedPolicyAdvisorForced (negative-control subclass)
# ---------------------------------------------------------------------------

__negative_control_instrument__: bool = True  # module-level marker


@dataclass
class ReliabilityGatedPolicyAdvisorForced(ReliabilityGatedPolicy):
    """
    NEGATIVE-CONTROL DIAGNOSTIC INSTRUMENT — NOT A PROPOSED FORWARDING IMPROVEMENT.

    Phase 11 ablation only. Forces advisor_authorized=True to demonstrate that an
    unauthorized advisor degrades or equals the deterministic core on held-out scenarios
    — empirically validating the LOSO authorization gate.

    Access restrictions:
    - Requires allow_negative_control=True at construction time.
    - NOT registered in policy_factory() — must be imported directly.
    - Carries module-level __negative_control_instrument__ = True marker.
    - Every Phase 11 table/figure must label results as
      "negative-control ablation (unauthorized advisor force-enabled)".

    Safety preserved:
    - Step 1 (TTL-critical) still cannot be overridden by the advisor.
    - Step 2 (resource suppression) still fires before Step 4.
    - Advisor may only NARROW the Step 3 route-quality relay candidate (suppress it), not force relays.
    """

    # Override the parent's non-configurable False field to True for the
    # guarded negative-control path. Intentional; enforced by allow_negative_control.
    advisor_authorized: bool = dataclasses.field(default=True, init=False)

    allow_negative_control: bool = False
    name: str = "drf_ble_advisor_forced"

    def __post_init__(self) -> None:
        if not self.allow_negative_control:
            raise RuntimeError(
                "ReliabilityGatedPolicyAdvisorForced is a Phase 11 negative-control "
                "diagnostic instrument and must not be used in normal forwarding. "
                "To instantiate, pass allow_negative_control=True explicitly. "
                "This subclass is NOT registered in policy_factory()."
            )
        object.__setattr__(self, "advisor_authorized", True)

    @property
    def __negative_control_instrument__(self) -> bool:
        return True
