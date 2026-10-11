"""
FR-DRF — Frontier-aware Redundancy-gated Decision-Regime Forwarding with TTL Rescue.
(Phase 12D / Option D consolidation. Code identifier `bridge_drf` / `BridgeDrfPolicy` and
factory name "bridge_drf" are RETAINED for backward compatibility; the headline / paper-facing
method name is **FR-DRF**. Factory aliases "fr_drf"/"frdrf" and class alias `FrDrfPolicy` are
provided.)

HEADLINE METHOD (FR-DRF): frontier-aware coverage forwarding + redundancy-gated suppression +
TTL rescue, with resource/channel guards and a route-quality fallback. Active default regimes:
FORWARD_TTL_RESCUE, FORWARD_FRONTIER, SUPPRESS_BUDGET_LIMIT, DEFER_CCA_BUSY, SUPPRESS_DUP_DENSE,
FALLBACK_ROUTE_QUALITY, DEFAULT_FORWARD.

RETIRED FROM HEADLINE (Phase 12B-2a, Option D): FORWARD_BRIDGE_PROXY. The bridge-proxy score was
structurally dominated and semantically misdirected (it fired on dense cluster cores, not bridges).
It is now DEFAULT-OFF (`enable_bridge_proxy=False`) and kept only as a backward-compatible
diagnostic regime. **No paper-facing claim of bridge preservation, bridge detection, cut-vertex
detection, CDS, or MPR may be made.** The word "Bridge" survives only in the legacy code name.

Identity (claim-bounded):
- BLE ADV-inspired, fully deterministic, node-local forwarding decision mechanism.
- Classifies each relay opportunity into reason-coded decision regimes and acts in a fixed
  priority order (protect TTL/frontier BEFORE budget/CCA/dense suppression).
- FULLY ML-FREE: no advisor is attached, consulted, or authorized.

Claim boundaries (Research_Contract_v2.md, docs/claim_boundaries.md):
- NOT Bluetooth Mesh compliant; NOT IEEE-standard compliant.
- NOT realistic BLE hardware simulation; NOT real disaster/emergency validation.
- NOT AODV / BATMAN-ADV / OLSR / CDS / MPR / PRoPHET / MaxProp / Spray-and-Wait /
  RAPID / RPL / Babel (cited only for contrast in docs).
- Energy is an airtime-proportional proxy only (A-005 not wired).

Decision order (strict; see docs/phase12b_bridge_drf_plan.md §5):
  Step 0  eligibility/cache + recent_duplicate_rate update (no reason code)
  Step 1  FORWARD_TTL_RESCUE      (A-018)
  Step 2  FORWARD_BRIDGE_PROXY    (A-019)
  Step 3  FORWARD_FRONTIER        (A-020)
  Step 4  SUPPRESS_BUDGET_LIMIT   (A-021; data packets only)
  Step 5  DEFER_CCA_BUSY          (A-022; data packets only)
  Step 6  SUPPRESS_DUP_DENSE      (A-023)
  Step 7  FALLBACK_ROUTE_QUALITY  (use_route_quality and prev_hop is not None)
  Step 8  DEFAULT_FORWARD

DEFAULT_FORWARD structural-dormancy caveat: with use_route_quality=True (default) and
prev_hop always concrete on the engine relay path (mac_engine passes prev_hop=int(tx)),
Step 7 always fires before Step 8, so DEFAULT_FORWARD is structurally unreachable under
the default config. This is ACCEPTABLE only because Steps 7 and 8 are substitutable
catch-alls (exactly one fires) and DEFAULT_FORWARD is exercised ONLY in the
use_route_quality=False ablation arm. This is documented, not a defect to rediscover.

All thresholds are Tier-4 controlled design assumptions (A-018..A-024). Defaults preserve
the intended Phase 12B-1 behavior.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import SimCfg
from .entities import Packet
from .reliability_state import RollingBinaryWindow


# ---------------------------------------------------------------------------
# Reason codes (exactly 8 decision regimes; "" is a non-decision, not a regime)
# ---------------------------------------------------------------------------

RC_FORWARD_TTL_RESCUE: str = "FORWARD_TTL_RESCUE"
RC_FORWARD_BRIDGE_PROXY: str = "FORWARD_BRIDGE_PROXY"
RC_FORWARD_FRONTIER: str = "FORWARD_FRONTIER"
RC_SUPPRESS_BUDGET_LIMIT: str = "SUPPRESS_BUDGET_LIMIT"
RC_DEFER_CCA_BUSY: str = "DEFER_CCA_BUSY"
RC_SUPPRESS_DUP_DENSE: str = "SUPPRESS_DUP_DENSE"
RC_FALLBACK_ROUTE_QUALITY: str = "FALLBACK_ROUTE_QUALITY"
RC_DEFAULT_FORWARD: str = "DEFAULT_FORWARD"

# The 8 decision regimes. "" (Step-0 eligibility) is intentionally NOT in this set;
# it is a non-decision and must not be counted as a regime in any novelty claim.
BRIDGE_DRF_REASON_CODES: Tuple[str, ...] = (
    RC_FORWARD_TTL_RESCUE,
    RC_FORWARD_BRIDGE_PROXY,
    RC_FORWARD_FRONTIER,
    RC_SUPPRESS_BUDGET_LIMIT,
    RC_DEFER_CCA_BUSY,
    RC_SUPPRESS_DUP_DENSE,
    RC_FALLBACK_ROUTE_QUALITY,
    RC_DEFAULT_FORWARD,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


# ---------------------------------------------------------------------------
# BRIDGE-DRF policy
# ---------------------------------------------------------------------------

@dataclass
class BridgeDrfPolicy:
    """
    BRIDGE-DRF forwarding policy (Phase 12B-1). Deterministic, node-local, ML-free.

    Determinism: no RNG anywhere; all thresholds are constants; all state is
    event-order deterministic. Same seed + config => identical decisions and
    identical decision_log_hash.
    """

    name: str = "bridge_drf"

    # --- Tier-4 thresholds (A-018..A-024) ---
    ttl_rescue_threshold: int = 2            # A-018
    bridge_proxy_threshold: float = 0.4      # A-019 (standardized name per the plan; older spelling retired)
    frontier_density_max: int = 2            # A-020
    frontier_threshold: float = 0.6          # A-020
    density_ref: float = 5.0                 # A-020 (calibrate from p5_baseline_normal before sweep)
    budget_pressure_max: float = 0.7         # A-021
    cca_busy_max: float = 0.3                # A-022
    dense_density_min: int = 4               # A-023
    dense_dup_rate_min: float = 0.6          # A-023
    regime_dominance_max: float = 0.90       # A-024 (fixed verification gate; not used at decision time)

    # --- regime toggles (ablation) ---
    use_route_quality: bool = True
    # Phase 12D (Option D): FORWARD_BRIDGE_PROXY is RETIRED from the headline method and is
    # DEFAULT-OFF. Phase 12B-2a showed the bridge-proxy score is structurally dominated and
    # semantically misdirected (fires on dense cluster cores, not bridges). The regime is kept
    # in code for backward compatibility / diagnostics only and must be explicitly opted into
    # (enable_bridge_proxy=True). No paper-facing claim may use it.
    enable_bridge_proxy: bool = False
    enable_frontier: bool = True
    enable_budget_guard: bool = True
    enable_cca_guard: bool = True
    enable_dense_suppression: bool = True

    # --- config ---
    window_size: int = 50                    # A-012 (recent_duplicate_rate window)
    allow_no_tracker: bool = False           # Mode A default: raise if state needed but no tracker

    # --- internal state (not constructor args) ---
    seen_by_node: Dict[int, set] = field(default_factory=dict)
    dup_window: Dict[int, RollingBinaryWindow] = field(default_factory=dict)
    _nst: Any = field(default=None, init=False, repr=False, compare=False)
    _nodes: Any = field(default=None, init=False, repr=False, compare=False)
    _airtime_buckets: Any = field(default=None, init=False, repr=False, compare=False)
    _last_reason_code: str = field(default="", init=False, repr=False, compare=False)
    _last_decision_metadata: Dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not (float(self.density_ref) > 0.0):
            raise ValueError("BridgeDrfPolicy.density_ref must be > 0")
        if int(self.window_size) < 1:
            raise ValueError("BridgeDrfPolicy.window_size must be >= 1")

    # -------- lifecycle --------

    def on_init(self, cfg: SimCfg) -> None:
        n = int(cfg.n_nodes)
        self.seen_by_node = {i: set() for i in range(n)}
        self.dup_window = {i: RollingBinaryWindow(maxlen=int(self.window_size)) for i in range(n)}

    def attach_tracker(self, nst: Any, nodes: Any, airtime_buckets: Any = None) -> None:
        """Give the policy read access to node-local reliability state (SimEngine calls this)."""
        self._nst = nst
        self._nodes = nodes
        self._airtime_buckets = airtime_buckets

    def attach_advisor(self, model: Any) -> None:  # noqa: D401 - explicit no-op
        """BRIDGE-DRF is ML-free. Any advisor is ignored (never stored, never consulted)."""
        return

    # -------- helpers --------

    @staticmethod
    def _key(pkt: Packet) -> Tuple[int, int]:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def _set_reason(
        self,
        reason: str,
        *,
        taken: bool,
        fallback: bool = False,
        suppression: bool = False,
    ) -> None:
        self._last_reason_code = reason
        if reason == "":
            # Step-0 non-decision: empty metadata so the engine logs reason_code="".
            self._last_decision_metadata = {}
            return
        self._last_decision_metadata = {
            "reason_code": reason,
            "decision_taken": bool(taken),
            "fallback_used": bool(fallback),
            "suppression_used": bool(suppression),
        }

    def _peek_snapshot(self, rx: int, now: float) -> Dict[str, Any]:
        """
        Side-effect-free read of node-local decision-time state (mirrors the peek used
        by ReliabilityGatedPolicy). Raises in Mode A if no tracker is attached.
        """
        if self._nst is None or self._nodes is None:
            if not self.allow_no_tracker:
                raise RuntimeError(
                    "BridgeDrfPolicy requires attach_tracker() before on_rx() reaches a "
                    "state-dependent step. Attach a NodeStateTracker, or set "
                    "allow_no_tracker=True for test-only use."
                )
            # Mode B: degenerate zero-state (test-only).
            return {
                "local_density_estimate": 0,
                "recent_cca_busy_rate": 0.0,
                "airtime_budget_pressure": 0.0,
            }
        ns = self._nodes.get(int(rx))
        queue_len = int(len(ns.tx_queue)) if ns is not None else 0
        bucket = None
        if self._airtime_buckets is not None:
            bucket = self._airtime_buckets.get(int(rx))
        bucket_snap = bucket.peek(float(now)) if bucket is not None else None
        return self._nst.state(int(rx)).snapshot(
            t_now=float(now),
            queue_len=queue_len,
            max_queue=256,  # matches SimEngine._max_queue
            airtime_bucket_snapshot=bucket_snap,
        )

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

    def _broadcast(self, pkt: Packet, rx: int, prev_hop: Optional[int]) -> List[Tuple[int, Packet]]:
        return self._make_fwd(pkt, rx, prev_hop)

    def _directed(self, pkt: Packet, rx: int, prev_hop: Optional[int]) -> List[Tuple[int, Packet]]:
        # Route-quality fallback: directed relay toward prev_hop (next_hop = prev_hop).
        return self._make_fwd(pkt, rx, prev_hop, next_hop=int(prev_hop))

    # -------- packet lifecycle --------

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        return [(int(pkt.src), pkt.with_meta(policy=self.name))]

    # -------- forwarding decision --------

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

        # ===== Step 0: eligibility + deterministic recent_duplicate_rate update =====
        # recent_duplicate_rate is a LOCAL REDUNDANCY PROXY: the fraction of recent
        # receptions at this node that were duplicates of already-seen packets. It is
        # NOT proof that other relays delivered the packet; it only reflects locally
        # observed repetition. No future observation and no global success label is used.
        if key in self.seen_by_node.get(rx, set()):
            self.dup_window[rx].push(1)         # duplicate reception observed
            self._set_reason("", taken=False)
            return []
        self.seen_by_node[rx].add(key)
        self.dup_window[rx].push(0)             # novel reception (current decision included in rate)

        if rx == int(pkt.dst):
            self._set_reason("", taken=False)
            return []
        if int(pkt.ttl_hops) <= 0:
            self._set_reason("", taken=False)
            return []

        # ===== read node-local decision-time state (side-effect-free) =====
        snap = self._peek_snapshot(rx, now)
        density = float(snap.get("local_density_estimate", 0))
        cca_busy = float(snap.get("recent_cca_busy_rate", 0.0))
        budget_p = float(snap.get("airtime_budget_pressure", 0.0))
        dup_rate = float(self.dup_window[rx].rate)
        ttl = int(pkt.ttl_hops)
        is_data = (int(bool(pkt.is_control)) == 0)

        frontier_score = _clamp(1.0 - density / float(self.density_ref), 0.0, 1.0)
        bridge_proxy_score = _clamp(
            (1.0 - dup_rate) * min(1.0, density / float(self.density_ref)), 0.0, 1.0
        )

        # ===== PROTECT regimes (forward value BEFORE any suppression) =====

        # Step 1 — TTL rescue (urgency; highest priority)
        if ttl <= int(self.ttl_rescue_threshold):
            self._set_reason(RC_FORWARD_TTL_RESCUE, taken=True, fallback=True)
            return self._broadcast(pkt, rx, prev_hop)

        # Step 2 — Bridge-proxy relay (low-redundancy relay value; NOT connectivity guarantee)
        if self.enable_bridge_proxy and bridge_proxy_score >= float(self.bridge_proxy_threshold):
            self._set_reason(RC_FORWARD_BRIDGE_PROXY, taken=True)
            return self._broadcast(pkt, rx, prev_hop)

        # Step 3 — Frontier extension (coverage value)
        if self.enable_frontier and (
            density <= float(self.frontier_density_max)
            or frontier_score >= float(self.frontier_threshold)
        ):
            self._set_reason(RC_FORWARD_FRONTIER, taken=True)
            return self._broadcast(pkt, rx, prev_hop)

        # ===== GUARD regimes (data packets only; control bypasses) =====

        # Step 4 — Airtime-budget suppression (resource guard)
        if self.enable_budget_guard and is_data and budget_p >= float(self.budget_pressure_max):
            self._set_reason(RC_SUPPRESS_BUDGET_LIMIT, taken=False, suppression=True)
            return []

        # Step 5 — CCA-busy defer (channel guard; suppress this opportunity)
        if self.enable_cca_guard and is_data and cca_busy >= float(self.cca_busy_max):
            self._set_reason(RC_DEFER_CCA_BUSY, taken=False, suppression=True)
            return []

        # ===== SUPPRESS dense redundancy =====

        # Step 6 — Dense duplicate suppression
        if (
            self.enable_dense_suppression
            and density >= float(self.dense_density_min)
            and dup_rate >= float(self.dense_dup_rate_min)
        ):
            self._set_reason(RC_SUPPRESS_DUP_DENSE, taken=False, suppression=True)
            return []

        # ===== FALLBACK + DEFAULT =====

        # Step 7 — Route-quality fallback (directed relay toward prev_hop)
        if self.use_route_quality and prev_hop is not None:
            self._set_reason(RC_FALLBACK_ROUTE_QUALITY, taken=True, fallback=True)
            return self._directed(pkt, rx, prev_hop)

        # Step 8 — Default forward (broadcast). Structurally reached only when Step 7 is
        # skipped (use_route_quality=False or prev_hop is None). See module dormancy caveat.
        self._set_reason(RC_DEFAULT_FORWARD, taken=True)
        return self._broadcast(pkt, rx, prev_hop)


# ---------------------------------------------------------------------------
# Phase 12D (Option D): claim-safe headline alias. FR-DRF is the paper-facing name.
# BridgeDrfPolicy is retained as the backward-compatible code identifier.
# ---------------------------------------------------------------------------
FrDrfPolicy = BridgeDrfPolicy
