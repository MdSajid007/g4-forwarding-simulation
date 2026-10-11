"""
Phase 13B conservative Coverage-Gossip Forwarding policy.

Registered only as "cgf_experimental". This policy is deterministic, ML-free,
node-local, and forwards by default under uncertainty. CGF-specific hint and guard
metrics are exposed through ``diagnostic_log`` for separate artifacts; the decision
log schema remains v2 and receives only reason_code string values.
"""
from __future__ import annotations

import dataclasses
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import SimCfg
from .coverage_hint import (
    COV_HINT_META_KEY,
    add_relayer_to_hint,
    cov_hint_from_meta,
    cov_hint_present_and_valid,
    coverage_estimate,
    mask_for_width,
    merge_cov_hints,
)
from .entities import Packet
from .reliability_state import RollingBinaryWindow


RC_FORWARD_TTL_RESCUE: str = "FORWARD_TTL_RESCUE"
RC_FORWARD_COVERAGE_GAP: str = "FORWARD_COVERAGE_GAP"
RC_FORWARD_HINT_UNCERTAIN: str = "FORWARD_HINT_UNCERTAIN"
RC_SUPPRESS_COVERAGE_PROVEN: str = "SUPPRESS_COVERAGE_PROVEN"
RC_SUPPRESS_BUDGET_LIMIT: str = "SUPPRESS_BUDGET_LIMIT"
RC_DEFER_CCA_BUSY: str = "DEFER_CCA_BUSY"
RC_FALLBACK_ROUTE_QUALITY: str = "FALLBACK_ROUTE_QUALITY"
RC_DEFAULT_FORWARD: str = "DEFAULT_FORWARD"

CGF_REASON_CODES: Tuple[str, ...] = (
    RC_FORWARD_TTL_RESCUE,
    RC_FORWARD_COVERAGE_GAP,
    RC_FORWARD_HINT_UNCERTAIN,
    RC_SUPPRESS_COVERAGE_PROVEN,
    RC_SUPPRESS_BUDGET_LIMIT,
    RC_DEFER_CCA_BUSY,
    RC_FALLBACK_ROUTE_QUALITY,
    RC_DEFAULT_FORWARD,
)


@dataclass
class CgfPolicy:
    """
    Conservative experimental CGF policy.

    The only coverage-based suppression reason is SUPPRESS_COVERAGE_PROVEN, and it
    can fire only when propagated coverage, duplicate-rate corroboration, density
    corroboration, TTL, suppression budget, uncertainty, and runtime collapse guards
    all pass.
    """

    name: str = "cgf_experimental"

    # Hint and overhead configuration (A-027/A-028).
    hint_width_bytes: int = 4
    gossip_hint_bytes: Optional[int] = None
    charge_overhead: bool = True
    max_accum_entries: int = 1000

    # Conservative CGF thresholds (A-018/A-030).
    ttl_rescue_threshold: int = 2
    coverage_low_water: int = 1
    coverage_high_water: int = 3
    corrob_dup_min: float = 0.60
    corrob_density_min: int = 4

    # A-033/A-035 guard windows.
    window_size: int = 50
    suppress_budget_frac: float = 0.25
    suppress_budget_window_size: int = 20
    collapse_window_size: int = 10
    collapse_ceiling: float = 0.60

    # Optional reused resource/channel guards. Disabled by default so CGF's
    # conservative behavior is governed by propagated coverage, not resource pressure.
    enable_budget_guard: bool = False
    enable_cca_guard: bool = False
    budget_pressure_max: float = 0.70
    cca_busy_max: float = 0.30

    # Fallback is available but not the default CGF arm; this keeps uncertainty
    # behavior close to broadcast flood unless explicitly enabled.
    use_route_quality: bool = False
    allow_no_tracker: bool = False

    # Phase 13B-R CGF-R experimental guard (A-036). Default-off; when enabled,
    # coverage suppression is permitted only while the packet is still shallow.
    enable_path_depth_veto: bool = False
    path_depth_max: int = 2

    seen_by_node: Dict[int, set] = field(default_factory=dict)
    dup_window: Dict[int, RollingBinaryWindow] = field(default_factory=dict)
    suppression_budget_window: Dict[int, RollingBinaryWindow] = field(default_factory=dict)
    collapse_window: Dict[int, RollingBinaryWindow] = field(default_factory=dict)
    accum_hints: "OrderedDict[Tuple[int, int, int], int]" = field(default_factory=OrderedDict)
    observed_orig_ttl: Dict[Tuple[int, int], int] = field(default_factory=dict)
    evicted_accum_keys: set = field(default_factory=set)
    diagnostic_log: List[Dict[str, Any]] = field(default_factory=list)
    eviction_count: int = 0
    max_accum_size_seen: int = 0
    runtime_guard_activations: int = 0

    _nst: Any = field(default=None, init=False, repr=False, compare=False)
    _nodes: Any = field(default=None, init=False, repr=False, compare=False)
    _airtime_buckets: Any = field(default=None, init=False, repr=False, compare=False)
    _last_reason_code: str = field(default="", init=False, repr=False, compare=False)
    _last_decision_metadata: Dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _ = mask_for_width(int(self.hint_width_bytes))
        if self.gossip_hint_bytes is None:
            self.gossip_hint_bytes = int(self.hint_width_bytes)
        if int(self.gossip_hint_bytes) < 0:
            raise ValueError("gossip_hint_bytes must be >= 0")
        if int(self.max_accum_entries) < 1:
            raise ValueError("max_accum_entries must be >= 1")
        if int(self.window_size) < 1:
            raise ValueError("window_size must be >= 1")
        if int(self.suppress_budget_window_size) < 1:
            raise ValueError("suppress_budget_window_size must be >= 1")
        if int(self.collapse_window_size) < 1:
            raise ValueError("collapse_window_size must be >= 1")
        if not (0.0 <= float(self.suppress_budget_frac) <= 1.0):
            raise ValueError("suppress_budget_frac must be in [0, 1]")
        if not (0.0 <= float(self.collapse_ceiling) <= 1.0):
            raise ValueError("collapse_ceiling must be in [0, 1]")
        if bool(self.enable_path_depth_veto) and int(self.hint_width_bytes) != 4:
            raise ValueError("CGF-R path-depth veto is width-4 only")
        if int(self.path_depth_max) != 2:
            raise ValueError("CGF-R path_depth_max is pre-registered at 2")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_init(self, cfg: SimCfg) -> None:
        n = int(cfg.n_nodes)
        self.seen_by_node = {i: set() for i in range(n)}
        self.dup_window = {i: RollingBinaryWindow(maxlen=int(self.window_size)) for i in range(n)}
        self.suppression_budget_window = {
            i: RollingBinaryWindow(maxlen=int(self.suppress_budget_window_size)) for i in range(n)
        }
        self.collapse_window = {
            i: RollingBinaryWindow(maxlen=int(self.collapse_window_size)) for i in range(n)
        }
        self.accum_hints = OrderedDict()
        self.observed_orig_ttl = {}
        self.evicted_accum_keys = set()
        self.diagnostic_log = []
        self.eviction_count = 0
        self.max_accum_size_seen = 0
        self.runtime_guard_activations = 0
        self._set_reason("", taken=False)

    def attach_tracker(self, nst: Any, nodes: Any, airtime_buckets: Any = None) -> None:
        self._nst = nst
        self._nodes = nodes
        self._airtime_buckets = airtime_buckets

    def attach_advisor(self, model: Any) -> None:
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        self._last_reason_code = str(reason)
        if reason == "":
            self._last_decision_metadata = {}
            return
        self._last_decision_metadata = {
            "reason_code": str(reason),
            "decision_taken": bool(taken),
            "fallback_used": bool(fallback),
            "suppression_used": bool(suppression),
        }

    def _overhead_bytes(self) -> int:
        return int(self.gossip_hint_bytes) if bool(self.charge_overhead) else 0

    def _remember_accum(self, key: Tuple[int, int, int], value: int) -> None:
        if key in self.accum_hints:
            self.accum_hints.move_to_end(key)
        self.accum_hints[key] = int(value)
        while len(self.accum_hints) > int(self.max_accum_entries):
            old_key, _old_value = self.accum_hints.popitem(last=False)
            self.evicted_accum_keys.add(old_key)
            self.eviction_count += 1
        self.max_accum_size_seen = max(self.max_accum_size_seen, len(self.accum_hints))

    def _remember_observed_ttl(self, pkt: Packet) -> None:
        key = self._key(pkt)
        ttl = int(pkt.ttl_hops)
        previous = self.observed_orig_ttl.get(key)
        if previous is None or ttl > int(previous):
            self.observed_orig_ttl[key] = ttl

    def _path_depth_state(self, pkt: Packet) -> Tuple[bool, str, bool]:
        if not bool(self.enable_path_depth_veto):
            return True, "", True
        key = self._key(pkt)
        ttl = int(pkt.ttl_hops)
        previous = self.observed_orig_ttl.get(key)
        if previous is None:
            self.observed_orig_ttl[key] = ttl
            return False, "", False
        if ttl > int(previous):
            self.observed_orig_ttl[key] = ttl
            previous = ttl
        hops_traveled = max(0, int(previous) - ttl)
        return True, str(int(hops_traveled)), hops_traveled < int(self.path_depth_max)

    def _peek_snapshot(self, rx: int, now: float) -> Dict[str, Any]:
        if self._nst is None or self._nodes is None:
            if not self.allow_no_tracker:
                raise RuntimeError(
                    "CgfPolicy requires attach_tracker() before state-dependent on_rx() "
                    "decisions. Attach a NodeStateTracker, or set allow_no_tracker=True "
                    "for controlled tests."
                )
            return {
                "local_density_estimate": 0,
                "recent_cca_busy_rate": 0.0,
                "airtime_budget_pressure": 0.0,
            }
        ns = self._nodes.get(int(rx))
        queue_len = int(len(ns.tx_queue)) if ns is not None else 0
        bucket_snap = None
        if self._airtime_buckets is not None:
            bucket = self._airtime_buckets.get(int(rx))
            bucket_snap = bucket.peek(float(now)) if bucket is not None else None
        return self._nst.state(int(rx)).snapshot(
            t_now=float(now),
            queue_len=queue_len,
            max_queue=256,
            airtime_bucket_snapshot=bucket_snap,
        )

    def _budget_token_available(self, rx: int) -> bool:
        if float(self.suppress_budget_frac) <= 0.0:
            return False
        if float(self.suppress_budget_frac) >= 1.0:
            return True
        win = self.suppression_budget_window[int(rx)]
        allowance = max(1, int(float(self.suppress_budget_frac) * int(win.maxlen)))
        return int(win.successes) < int(allowance)

    def _collapse_guard_permits(self, rx: int) -> bool:
        win = self.collapse_window[int(rx)]
        return float(win.rate) < float(self.collapse_ceiling)

    def _record_guard_outcome(self, rx: int, *, suppressed: bool, coverage_suppressed: bool) -> None:
        self.collapse_window[int(rx)].push(1 if suppressed else 0)
        self.suppression_budget_window[int(rx)].push(1 if coverage_suppressed else 0)

    def _make_fwd(
        self,
        pkt: Packet,
        rx: int,
        prev_hop: Optional[int],
        h_acc: int,
        *,
        next_hop: Optional[int] = None,
    ) -> List[Tuple[int, Packet]]:
        h_out = add_relayer_to_hint(h_acc, int(rx), int(self.hint_width_bytes))
        overhead = self._overhead_bytes()
        fwd = dataclasses.replace(
            pkt,
            ttl_hops=int(pkt.ttl_hops) - 1,
            payload_bytes=int(pkt.payload_bytes) + int(overhead),
        ).with_meta(
            policy=self.name,
            fwd=self.name,
            prev_hop=int(prev_hop) if prev_hop is not None else -1,
            ttl_dec=True,
            **({COV_HINT_META_KEY: int(h_out)}),
            **({"next_hop": int(next_hop)} if next_hop is not None else {}),
        )
        return [(int(rx), fwd)]

    def _broadcast(
        self, pkt: Packet, rx: int, prev_hop: Optional[int], h_acc: int
    ) -> List[Tuple[int, Packet]]:
        return self._make_fwd(pkt, rx, prev_hop, h_acc)

    def _directed(
        self, pkt: Packet, rx: int, prev_hop: Optional[int], h_acc: int
    ) -> List[Tuple[int, Packet]]:
        return self._make_fwd(pkt, rx, prev_hop, h_acc, next_hop=int(prev_hop))

    def _append_diag(
        self,
        *,
        now: float,
        rx: int,
        pkt: Packet,
        prev_hop: Optional[int],
        reason: str,
        forwarded: bool,
        suppressed: bool,
        h_in: int,
        h_acc: int,
        coverage_est: int,
        hint_present_valid: bool,
        stale_accum: bool,
        uncertainty_flag: bool,
        dup_rate: float,
        density: float,
        cca_busy: float,
        budget_pressure: float,
        budget_token_available: bool,
        collapse_permits: bool,
        runtime_guard_blocked: bool,
        overhead_charged_bytes: int,
        path_depth_known: bool,
        hops_traveled: str,
        path_depth_permits: bool,
        path_depth_vetoed: bool,
    ) -> None:
        flow_id, seq = self._key(pkt)
        row = {
            "time_s": round(float(now), 9),
            "rx_node": int(rx),
            "prev_hop": int(prev_hop) if prev_hop is not None else -1,
            "flow_id": int(flow_id),
            "seq": int(seq),
            "src": int(pkt.src),
            "dst": int(pkt.dst),
            "ttl_in": int(pkt.ttl_hops),
            "hint_width_bytes": int(self.hint_width_bytes),
            "gossip_hint_bytes": int(self.gossip_hint_bytes),
            "charge_overhead": int(bool(self.charge_overhead)),
            "overhead_charged_bytes": int(overhead_charged_bytes),
            "payload_in_bytes": int(pkt.payload_bytes),
            "payload_out_bytes": (
                int(pkt.payload_bytes) + int(overhead_charged_bytes) if forwarded else ""
            ),
            "reason_code": str(reason),
            "forwarded": int(bool(forwarded)),
            "suppressed": int(bool(suppressed)),
            "cov_hint_in": int(h_in),
            "cov_hint_accum": int(h_acc),
            "coverage_est": int(coverage_est),
            "hint_present_valid": int(bool(hint_present_valid)),
            "stale_accum": int(bool(stale_accum)),
            "uncertainty_flag": int(bool(uncertainty_flag)),
            "recent_duplicate_rate": round(float(dup_rate), 6),
            "local_density": round(float(density), 6),
            "recent_cca_busy_rate": round(float(cca_busy), 6),
            "airtime_budget_pressure": round(float(budget_pressure), 6),
            "coverage_low_water": int(self.coverage_low_water),
            "coverage_high_water": int(self.coverage_high_water),
            "corrob_dup_min": round(float(self.corrob_dup_min), 6),
            "corrob_density_min": int(self.corrob_density_min),
            "budget_token_available": int(bool(budget_token_available)),
            "suppression_budget_rate": round(
                float(self.suppression_budget_window[int(rx)].rate), 6
            ),
            "collapse_guard_permits": int(bool(collapse_permits)),
            "collapse_suppression_rate": round(float(self.collapse_window[int(rx)].rate), 6),
            "runtime_guard_blocked": int(bool(runtime_guard_blocked)),
            "runtime_guard_activations_total": int(self.runtime_guard_activations),
            "accum_entries": int(len(self.accum_hints)),
            "eviction_count": int(self.eviction_count),
        }
        if bool(self.enable_path_depth_veto):
            row.update(
                {
                    "enable_path_depth_veto": 1,
                    "path_depth_max": int(self.path_depth_max),
                    "path_depth_known": int(bool(path_depth_known)),
                    "hops_traveled": str(hops_traveled),
                    "path_depth_permits": int(bool(path_depth_permits)),
                    "path_depth_vetoed": int(bool(path_depth_vetoed)),
                }
            )
        self.diagnostic_log.append(row)

    # ------------------------------------------------------------------
    # Packet lifecycle
    # ------------------------------------------------------------------

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        self._remember_observed_ttl(pkt)
        seeded = pkt.with_meta(policy=self.name, **{COV_HINT_META_KEY: 0})
        return [(int(pkt.src), seeded)]

    # ------------------------------------------------------------------
    # Forwarding decision
    # ------------------------------------------------------------------

    def on_rx(
        self,
        cfg: SimCfg,
        now: float,
        rx_node: int,
        pkt: Packet,
        prev_hop: Optional[int],
    ) -> List[Tuple[int, Packet]]:
        rx = int(rx_node)
        flow_id, seq = self._key(pkt)
        pkt_key = (flow_id, seq)
        accum_key = (rx, flow_id, seq)
        width = int(self.hint_width_bytes)

        h_in = cov_hint_from_meta(pkt.meta, width)
        hint_present_valid = cov_hint_present_and_valid(pkt.meta, width)
        stale_accum = accum_key in self.evicted_accum_keys
        h_prev = self.accum_hints.get(accum_key, 0)
        h_acc = merge_cov_hints(h_prev, h_in, width)
        self._remember_accum(accum_key, h_acc)

        if pkt_key in self.seen_by_node.get(rx, set()):
            self.dup_window[rx].push(1)
            self._set_reason("", taken=False)
            return []

        self.seen_by_node[rx].add(pkt_key)
        self.dup_window[rx].push(0)

        if rx == int(pkt.dst):
            self._set_reason("", taken=False)
            return []
        if int(pkt.ttl_hops) <= 0:
            self._set_reason("", taken=False)
            return []

        snap = self._peek_snapshot(rx, float(now))
        density = float(snap.get("local_density_estimate", 0))
        cca_busy = float(snap.get("recent_cca_busy_rate", 0.0))
        budget_pressure = float(snap.get("airtime_budget_pressure", 0.0))
        dup_rate = float(self.dup_window[rx].rate)
        ttl = int(pkt.ttl_hops)
        is_data = (int(bool(pkt.is_control)) == 0)
        coverage_est = int(coverage_estimate(h_acc, width))
        low_confidence = int(self.coverage_low_water) < coverage_est < int(self.coverage_high_water)
        uncertainty_flag = (
            (not bool(hint_present_valid))
            or bool(stale_accum)
            or bool(low_confidence)
        )
        path_depth_known, hops_traveled, path_depth_permits = self._path_depth_state(pkt)

        budget_token = self._budget_token_available(rx)
        collapse_permits = self._collapse_guard_permits(rx)
        all_except_collapse = (
            coverage_est >= int(self.coverage_high_water)
            and dup_rate >= float(self.corrob_dup_min)
            and density >= float(self.corrob_density_min)
            and ttl > int(self.ttl_rescue_threshold)
            and bool(budget_token)
            and not bool(uncertainty_flag)
        )
        runtime_guard_blocked = bool(all_except_collapse and not collapse_permits)
        original_suppression_ready = bool(all_except_collapse and collapse_permits)
        path_depth_vetoed = bool(
            self.enable_path_depth_veto
            and original_suppression_ready
            and not bool(path_depth_permits)
        )

        def finish(
            reason: str,
            actions: List[Tuple[int, Packet]],
            *,
            fallback: bool = False,
            suppressed: bool = False,
            coverage_suppressed: bool = False,
            runtime_guard_hit: bool = False,
        ) -> List[Tuple[int, Packet]]:
            if runtime_guard_hit:
                self.runtime_guard_activations += 1
            self._record_guard_outcome(
                rx,
                suppressed=bool(suppressed),
                coverage_suppressed=bool(coverage_suppressed),
            )
            self._set_reason(
                reason,
                taken=bool(actions),
                fallback=bool(fallback),
                suppression=bool(suppressed),
            )
            self._append_diag(
                now=float(now),
                rx=rx,
                pkt=pkt,
                prev_hop=prev_hop,
                reason=reason,
                forwarded=bool(actions),
                suppressed=bool(suppressed),
                h_in=h_in,
                h_acc=h_acc,
                coverage_est=coverage_est,
                hint_present_valid=hint_present_valid,
                stale_accum=stale_accum,
                uncertainty_flag=uncertainty_flag,
                dup_rate=dup_rate,
                density=density,
                cca_busy=cca_busy,
                budget_pressure=budget_pressure,
                budget_token_available=budget_token,
                collapse_permits=collapse_permits,
                runtime_guard_blocked=runtime_guard_hit,
                overhead_charged_bytes=self._overhead_bytes() if actions else 0,
                path_depth_known=path_depth_known,
                hops_traveled=hops_traveled,
                path_depth_permits=path_depth_permits,
                path_depth_vetoed=path_depth_vetoed,
            )
            if stale_accum:
                self.evicted_accum_keys.discard(accum_key)
            return actions

        # Step 1: TTL rescue overrides all suppression.
        if ttl <= int(self.ttl_rescue_threshold):
            return finish(
                RC_FORWARD_TTL_RESCUE,
                self._broadcast(pkt, rx, prev_hop, h_acc),
                fallback=True,
            )

        # Step 2: low propagated coverage protects frontier/load-bearing relays.
        if coverage_est <= int(self.coverage_low_water):
            return finish(
                RC_FORWARD_COVERAGE_GAP,
                self._broadcast(pkt, rx, prev_hop, h_acc),
            )

        # Step 3: the only coverage suppression; every guard must pass.
        if original_suppression_ready and not path_depth_vetoed:
            return finish(
                RC_SUPPRESS_COVERAGE_PROVEN,
                [],
                suppressed=True,
                coverage_suppressed=True,
            )

        # Phase 13B-R: deep or unknown path depth conservatively vetoes suppression.
        if path_depth_vetoed:
            return finish(
                RC_FORWARD_HINT_UNCERTAIN,
                self._broadcast(pkt, rx, prev_hop, h_acc),
            )

        # Runtime anti-collapse guard blocks otherwise-eligible coverage suppression.
        if runtime_guard_blocked:
            return finish(
                RC_FORWARD_HINT_UNCERTAIN,
                self._broadcast(pkt, rx, prev_hop, h_acc),
                runtime_guard_hit=True,
            )

        # Step 4: optional resource guard, data packets only.
        if (
            self.enable_budget_guard
            and is_data
            and budget_pressure >= float(self.budget_pressure_max)
        ):
            return finish(RC_SUPPRESS_BUDGET_LIMIT, [], suppressed=True)

        # Step 5: optional CCA guard, data packets only.
        if self.enable_cca_guard and is_data and cca_busy >= float(self.cca_busy_max):
            return finish(RC_DEFER_CCA_BUSY, [], suppressed=True)

        # Step 6: route-quality fallback, only when explicitly enabled and concrete.
        if self.use_route_quality and prev_hop is not None:
            return finish(
                RC_FALLBACK_ROUTE_QUALITY,
                self._directed(pkt, rx, prev_hop, h_acc),
                fallback=True,
            )

        # Step 7: deterministic forward under uncertainty, else structural default.
        if uncertainty_flag:
            return finish(
                RC_FORWARD_HINT_UNCERTAIN,
                self._broadcast(pkt, rx, prev_hop, h_acc),
            )
        return finish(
            RC_DEFAULT_FORWARD,
            self._broadcast(pkt, rx, prev_hop, h_acc),
        )
