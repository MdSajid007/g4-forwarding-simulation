"""
SimEngine: hop-aware TX node + observability v4 counters.

Extracted from notebook Cell 8, Phase 3.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .airtime import AirtimeTokenBucket, airtime_cost_s_for_adv_burst
from .config import SimCfg, validate_cfg
from .decision_logger import DecisionLogger
from .entities import (
    DropReason, Packet, PacketId, PacketIdGen, NodeState, RunStats,
)
from .mobility import MobilityPlan
from .node_state import NodeStateTracker
from .phy import PhyModel, TxWaveform, dbm_to_mw
from .policies import policy_factory
from .scheduler import EventPrio, EventScheduler
from .seeds import SeedManager
from .traffic import AppInjection


# ----------------------------
# BLE-ish on-air timing (simple, stable)
# ----------------------------

def adv_tx_duration_s(payload_bytes: int) -> float:
    """
    Approximate BLE advertising PDU airtime at 1 Mbps.
    bits = preamble(8) + access(32) + header(16) + payload(8*B) + crc(24)
    duration = bits / 1e6
    """
    b = int(payload_bytes)
    bits = 8 + 32 + 16 + (8 * b) + 24
    return float(bits) / 1_000_000.0


def _airtime_cost_adv_burst_local(payload_bytes: int, n_channels: int) -> float:
    return float(int(n_channels)) * float(adv_tx_duration_s(int(payload_bytes)))


ADV_CH_SPACING_S = 0.000150  # 150 us

ADV_DELAY_MIN_S = 0.0
ADV_DELAY_MAX_S = 0.010

KICK_EPS_BASE_S = 1e-6
KICK_EPS_PER_SRC_S = 1e-7
KICK_EPS_PER_KICK_S = 1e-9


# ----------------------------
# Local (Cell8-only) policy stubs (legacy fallback — kept for NameError safety)
# ----------------------------

class _PolicyCell8:
    name: str = "UNKNOWN"

    def on_init(self, cfg: SimCfg) -> None:
        return

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        return [(int(pkt.src), pkt)]

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
class _FloodCell8(_PolicyCell8):
    name: str = "flood"
    seen_by_node: Dict[int, Set[Tuple[int, int]]] = dataclasses.field(default_factory=dict)

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
class _NoFwdCell8(_PolicyCell8):
    name: str = "no_fwd"

    def on_app_inject(self, cfg: SimCfg, now: float, pkt: Packet) -> List[Tuple[int, Packet]]:
        return [(int(pkt.src), pkt.with_meta(policy=self.name))]


def _policy_factory_cell8(name: str) -> _PolicyCell8:
    n = str(name).strip().lower()
    if n == "flood":
        return _FloodCell8()
    if n == "no_fwd":
        return _NoFwdCell8()
    raise ValueError(f"Unsupported policy in Cell 8: {name!r}")


# ----------------------------
# MAC helpers
# ----------------------------

@dataclass(frozen=True)
class AttemptState:
    """
    One TX attempt state.

    IMPORTANT:
    - tx_node is the actual transmitter (queue owner).
    - pkt.src remains the application source endpoint; do NOT use it for MAC scheduling.
    """
    pkt: Packet
    tx_node: int
    dst: int
    retries_left: int
    last_fail_reason: Optional[DropReason] = None


def is_scanning(cfg: SimCfg, t: float) -> bool:
    si = float(cfg.mac.scan_interval_s)
    sw = float(cfg.mac.scan_window_s)
    if si <= 0.0 or sw <= 0.0:
        return False
    phase = float(t) % si
    return phase < sw + 1e-15


def _meta_dict(pkt: Packet) -> Dict[str, Any]:
    try:
        return dict(pkt.meta)
    except Exception:
        return {}


def _intended_next_hop(pkt: Packet) -> Optional[int]:
    if bool(getattr(pkt, "is_control", False)):
        return None
    m = _meta_dict(pkt)
    if "next_hop" not in m:
        return None
    try:
        return int(m["next_hop"])
    except Exception:
        return None


def _rx_eligible(pkt: Packet, rx: int) -> bool:
    nh = _intended_next_hop(pkt)
    if nh is None:
        return True
    dst = int(pkt.dst)
    r = int(rx)
    return (r == dst) or (r == int(nh))


# ----------------------------
# Airtime budget helper
# ----------------------------

def _airtime_cost_adv_burst(payload_bytes: int, n_ch: int) -> float:
    return float(airtime_cost_s_for_adv_burst(int(payload_bytes), int(n_ch)))


# ----------------------------
# Canonical engine
# ----------------------------

class SimEngine:
    def __init__(
        self,
        cfg: SimCfg,
        *,
        seed: int,
        plan: MobilityPlan,
        policy: Any,
        cfg_hash_for_rng: str,
        airtime_buckets: Optional[Dict[int, AirtimeTokenBucket]] = None,
        decision_logger: Optional[DecisionLogger] = None,
    ) -> None:
        validate_cfg(cfg)
        self.cfg = cfg
        self.seed = int(seed)
        self.plan = plan
        self.policy = policy
        self.policy.on_init(cfg)
        self.cfg_h = str(cfg_hash_for_rng)

        # Phase 12C-0: opt-in pre-transmit re-check capability. Dual-gated and cached once.
        # Existing policies define neither attribute, so this is False for all of them and
        # the on_tx_ready branch in _start_attempt is never entered (byte-identical behavior).
        self._tx_ready_hook_on: bool = bool(
            hasattr(self.policy, "on_tx_ready")
            and getattr(self.policy, "uses_tx_ready_hook", False) is True
        )

        self.sm = SeedManager(cfg_hash=self.cfg_h, seed=self.seed)
        self.rng_mac = self.sm.rng_cached("mac")
        self.rng_phy = self.sm.rng_cached("phy")
        self.rng_drift = self.sm.rng_cached("mac_drift")

        self.phy = PhyModel(cfg)
        self.sch = EventScheduler()

        self.nodes: Dict[int, NodeState] = {i: NodeState(node_id=i) for i in range(int(cfg.n_nodes))}
        self._max_queue = 256

        self._pid_gens: Dict[int, PacketIdGen] = {}

        self._attempts: Dict[int, AttemptState] = {}
        self._next_attempt_id: int = 1
        self._attempt_any_eligible_ok: Dict[int, bool] = {}

        self._tx_active: List[TxWaveform] = []
        self._tx_prune_eps: float = 1e-12

        self._kick_seq_by_src: Dict[int, int] = {i: 0 for i in range(int(cfg.n_nodes))}

        self.stats = RunStats(
            cfg_hash=str(self.cfg_h),
            seed=self.seed,
            sim_duration_s=float(cfg.sim_duration_s),
            n_nodes=int(cfg.n_nodes),
        )

        # Defensive init for v3+v4 fields
        for name, default in (
            ("overlap_events", 0),
            ("overlap_waveforms", 0),
            ("overlap_rx_evals", 0),
            ("overlap_interferer_pairs", 0),
            ("rx_ok", 0),
            ("rx_fail", 0),
            ("not_scanning_dst_events", 0),
            ("not_scanning_events", 0),
            ("not_scanning_rx_opportunities", 0),
            ("not_scanning_waveforms", 0),
            ("budget_deferrals", 0),
            ("airtime_data_s", 0.0),
            ("airtime_ctrl_s", 0.0),
        ):
            if not hasattr(self.stats, name):
                setattr(self.stats, name, default)

        self._delivered_keys: Set[Tuple[int, int]] = set()
        self._final_dropped_keys: Set[Tuple[int, int]] = set()

        self.airtime_buckets = airtime_buckets

        # Phase 6: passive per-node reliability state (no forwarding decision changes)
        self.nst = NodeStateTracker(n_nodes=int(cfg.n_nodes))
        self._attempt_had_interference: Dict[int, bool] = {}

        # Phase 7: passive decision-level logger (no forwarding decision changes)
        self.decision_logger: Optional[DecisionLogger] = decision_logger
        # Delivery outcomes tracked for post-run label fill (only when logger set)
        self._delivery_outcomes: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Phase 10R: wire attach_tracker() for ReliabilityGatedPolicy (and any future
        # policy that exposes this interface). Backward-compatible: guarded by hasattr.
        if hasattr(self.policy, "attach_tracker"):
            self.policy.attach_tracker(self.nst, self.nodes, self.airtime_buckets)

        p = cfg.phy
        fm = str(getattr(p, "fading_model", "none")).strip().lower()
        ss = bool(getattr(p, "small_scale_fading", False))
        sh = float(getattr(p, "shadow_sigma_db", 0.0))
        self._phy_fading_enabled = (fm not in ("none", "", "off", "disabled")) or ss or (sh > 0.0)

        self._rssi_cache: Dict[Tuple[int, int, int, int, str], float] = {}

        ppm = float(getattr(cfg.mac, "drift_ppm", 0.0))
        if ppm <= 0.0:
            self._drift_factor: Dict[int, float] = {i: 1.0 for i in range(int(cfg.n_nodes))}
        else:
            self._drift_factor = {}
            for i in range(int(cfg.n_nodes)):
                u = float(self.rng_drift.uniform(-ppm, +ppm))
                self._drift_factor[i] = 1.0 + (u * 1e-6)

    def _df(self, tx_node: int) -> float:
        return float(self._drift_factor.get(int(tx_node), 1.0))

    def _drift_delay(self, tx_node: int, dt: float) -> float:
        d = float(dt)
        if d <= 0.0:
            return d
        return d * self._df(int(tx_node))

    def _pos(self, nid: int, t: float) -> Tuple[float, float]:
        return self.plan.pos_at(int(nid), float(t))

    def _retire_attempt(self, attempt_id: int) -> None:
        aid = int(attempt_id)
        if aid in self._attempts:
            del self._attempts[aid]
        if aid in self._attempt_any_eligible_ok:
            del self._attempt_any_eligible_ok[aid]
        self._attempt_had_interference.pop(aid, None)

    def _prune_tx_active(self, t_now: float) -> None:
        tt = float(t_now)
        if not self._tx_active:
            return
        eps = float(self._tx_prune_eps)
        self._tx_active = [w for w in self._tx_active if float(w.t_end) > (tt + eps)]

    @staticmethod
    def _interval_overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
        return (float(a0) < float(b1)) and (float(b0) < float(a1))

    def _active_overlapping_interval(self, ch: int, t0: float, t1: float, exclude_tx: int) -> List[TxWaveform]:
        self._prune_tx_active(float(t0))
        out: List[TxWaveform] = []
        for w in self._tx_active:
            if int(w.ch) != int(ch):
                continue
            if int(w.tx_node) == int(exclude_tx):
                continue
            if self._interval_overlaps(float(t0), float(t1), float(w.t_start), float(w.t_end)):
                out.append(w)
        return out

    def _cca_busy(self, tx_node: int, ch: int, t: float) -> bool:
        th = float(self.cfg.mac.cca_threshold_dbm)
        tx_pos = self._pos(tx_node, t)
        overlaps = self._active_overlapping_interval(int(ch), float(t), float(t) + 1e-12, exclude_tx=int(tx_node))
        for w in overlaps:
            other_pos = self._pos(int(w.tx_node), float(t))
            rssi = self.phy.rssi_dbm(float(w.tx_power_dbm), other_pos, tx_pos, rng=None)
            if rssi >= th:
                return True
        return False

    def _backoff_delay(self, tx_node: int) -> float:
        a = float(self.cfg.mac.backoff_min_s)
        b = float(self.cfg.mac.backoff_max_s)
        if b <= a:
            return max(a, 0.0)
        return self._drift_delay(int(tx_node), float(self.rng_mac.uniform(a, b)))

    def _adv_delay(self, tx_node: int) -> float:
        a = float(ADV_DELAY_MIN_S)
        b = float(ADV_DELAY_MAX_S)
        if b <= a:
            return max(a, 0.0)
        return self._drift_delay(int(tx_node), float(self.rng_mac.uniform(a, b)))

    def _new_attempt(
        self,
        *,
        pkt: Packet,
        tx_node: int,
        retries_left: int,
        last_fail_reason: Optional[DropReason] = None,
    ) -> int:
        aid = self._next_attempt_id
        self._next_attempt_id += 1
        self._attempts[aid] = AttemptState(
            pkt=pkt,
            tx_node=int(tx_node),
            dst=int(pkt.dst),
            retries_left=int(retries_left),
            last_fail_reason=last_fail_reason,
        )
        self._attempt_any_eligible_ok[aid] = False
        self._attempt_had_interference[aid] = False
        return aid

    def _schedule_adv_channels(self, attempt_id: int, t0: float) -> None:
        att = self._attempts[attempt_id]
        chs = tuple(self.cfg.mac.adv_channels)
        dur = adv_tx_duration_s(att.pkt.payload_bytes)
        base = float(t0) + float(self._adv_delay(int(att.tx_node)))
        for i, ch in enumerate(chs):
            ts = float(base) + float(i) * float(self._drift_delay(int(att.tx_node), dur + float(ADV_CH_SPACING_S)))
            te = ts + float(self._drift_delay(int(att.tx_node), dur))
            self.sch.schedule(
                ts,
                "MAC_TX_CH",
                prio=int(EventPrio.MAC),
                data={
                    "attempt_id": int(attempt_id),
                    "src": int(att.tx_node),
                    "ch": int(ch),
                    "t_end": float(te),
                },
                stream="mac",
            )

    def _maybe_budget_defer(self, tx_node: int, t_now: float, pkt: Packet) -> Optional[float]:
        if self.airtime_buckets is None:
            return None
        b = self.airtime_buckets.get(int(tx_node))
        if b is None:
            return None
        n_ch = int(len(self.cfg.mac.adv_channels))
        cost = float(_airtime_cost_adv_burst(int(pkt.payload_bytes), n_ch))
        if b.can_consume(float(t_now), float(cost)):
            _ = b.consume(float(t_now), float(cost))
            return None
        need = float(cost) - float(b.snapshot(float(t_now))["tokens_s"])
        r = float(b.refill_rate_sps)
        if r <= 0.0:
            return float("inf")
        dt = max(need / r, 0.0)
        self.stats.budget_deferrals += 1
        self.nst.on_budget_deferral(int(tx_node))
        return float(t_now) + dt

    def _pid_key(self, pkt: Packet) -> Tuple[int, int]:
        return (int(pkt.pid.flow_id), int(pkt.pid.seq))

    def _mark_final_drop_if_data(self, pkt: Packet, reason: DropReason, n: int = 1) -> None:
        if bool(getattr(pkt, "is_control", False)):
            return
        key = self._pid_key(pkt)
        if key in self._delivered_keys:
            return
        if key in self._final_dropped_keys:
            return
        self._final_dropped_keys.add(key)
        self.stats.add_drop(reason, int(n))

    def _kick_try_tx(self, tx_node: int, t: float) -> None:
        s = int(tx_node)
        base_t = float(t)
        k = int(self._kick_seq_by_src.get(s, 0))
        self._kick_seq_by_src[s] = k + 1
        eps = float(KICK_EPS_BASE_S) + float(s) * float(KICK_EPS_PER_SRC_S) + float(k) * float(KICK_EPS_PER_KICK_S)
        self.sch.schedule(
            float(base_t + eps),
            "MAC_TRY_TX",
            prio=int(EventPrio.MAC),
            data={"src": s},
            stream=f"mac_try_{s}",
        )

    def enqueue_packet(self, node_id: int, pkt: Packet, now: float) -> None:
        nid = int(node_id)
        ns = self.nodes[nid]
        if len(ns.tx_queue) >= self._max_queue:
            self._mark_final_drop_if_data(pkt, DropReason.QUEUE_FULL, 1)
            return
        ns.tx_queue.append(pkt)
        self._kick_try_tx(nid, float(now))

    def schedule_packet_inject(self, t: float, pkt: Packet) -> None:
        self.sch.schedule(float(t), "PKT_INJECT", prio=int(EventPrio.APP), data={"pkt": pkt}, stream="ctrl")

    def _enqueue_app_injection(self, inj: AppInjection) -> None:
        flow_id = int(inj.flow_id)
        g = self._pid_gens.get(flow_id)
        if g is None:
            g = PacketIdGen(flow_id=flow_id)
            self._pid_gens[flow_id] = g
        pid = g.next()
        pkt = Packet(
            pid=pid,
            src=int(inj.src),
            dst=int(inj.dst),
            created_t=float(inj.t),
            ttl_hops=int(inj.ttl_hops),
            payload_bytes=int(inj.payload_bytes),
            is_control=False,
            meta=tuple(),
        )
        # Phase 7: register original TTL for hop_count computation at decision time
        if self.decision_logger is not None:
            self.decision_logger.on_packet_created(
                (int(pkt.pid.flow_id), int(pkt.pid.seq)),
                int(inj.ttl_hops),
            )
        self.stats.app_injected += 1
        actions = self.policy.on_app_inject(self.cfg, float(inj.t), pkt)
        for nid, pkt2 in actions:
            self.enqueue_packet(int(nid), pkt2, now=float(inj.t))

    def _rssi_link_dbm(
        self,
        *,
        attempt_id: int,
        ch: int,
        tx: int,
        rx: int,
        t_sample: float,
        tx_power_dbm: float,
        tag: str,
    ) -> float:
        key = (int(attempt_id), int(ch), int(tx), int(rx), str(tag))
        if key in self._rssi_cache:
            return float(self._rssi_cache[key])
        tx_pos = self._pos(int(tx), float(t_sample))
        rx_pos = self._pos(int(rx), float(t_sample))
        if not self._phy_fading_enabled:
            r = self.phy.rssi_dbm(float(tx_power_dbm), tx_pos, rx_pos, rng=None)
            self._rssi_cache[key] = float(r)
            return float(r)
        r = self.phy.rssi_dbm(float(tx_power_dbm), tx_pos, rx_pos, rng=self.rng_phy)
        self._rssi_cache[key] = float(r)
        return float(r)

    def _eval_rx_for_waveform(self, attempt_id: int, w: TxWaveform) -> None:
        t_end = float(w.t_end)
        tx = int(w.tx_node)
        overlaps = self._active_overlapping_interval(int(w.ch), float(w.t_start), float(w.t_end), exclude_tx=int(tx))
        had_int = len(overlaps) > 0
        dst = int(w.packet.dst)

        scanning_now = bool(is_scanning(self.cfg, t_end))

        if had_int:
            self.stats.overlap_waveforms += 1
            # Phase 6: mark this attempt as having encountered interference
            self._attempt_had_interference[attempt_id] = True

        if had_int and scanning_now:
            self.stats.overlap_events += 1

        if not scanning_now:
            self.stats.not_scanning_waveforms += 1

        any_eligible_ok_this_waveform = False

        for rx in range(int(self.cfg.n_nodes)):
            if rx == tx:
                continue

            self.stats.rx_opportunities += 1

            if not scanning_now:
                self.stats.not_scanning_rx_opportunities += 1
                self.stats.not_scanning_events += 1
                if rx == dst:
                    self.stats.not_scanning_dst_events += 1
                self.nst.on_not_scanning(rx)
                continue

            self.stats.rx_scans += 1
            self.nst.on_scanning_rx_opportunity(rx)

            if had_int:
                self.stats.overlap_rx_evals += 1
                self.stats.overlap_interferer_pairs += int(len(overlaps))
                self.nst.on_overlap_rx(rx)

            sig_dbm = self._rssi_link_dbm(
                attempt_id=int(attempt_id),
                ch=int(w.ch),
                tx=int(tx),
                rx=int(rx),
                t_sample=float(t_end),
                tx_power_dbm=float(w.tx_power_dbm),
                tag="sig",
            )

            interf_dbm: List[float] = []
            for o in overlaps:
                interf_dbm.append(
                    self._rssi_link_dbm(
                        attempt_id=int(attempt_id),
                        ch=int(w.ch),
                        tx=int(o.tx_node),
                        rx=int(rx),
                        t_sample=float(t_end),
                        tx_power_dbm=float(o.tx_power_dbm),
                        tag="int",
                    )
                )

            sinr = self.phy.sinr_db(sig_dbm, interf_dbm, float(self.cfg.phy.noise_floor_dbm))
            per = self.phy.per_from_sinr(sinr)
            u = float(self.rng_phy.random())
            ok = (u >= per)

            self.stats.rx_total += 1
            if ok:
                self.stats.rx_ok += 1
                self.nst.on_rx_ok(rx, t_end, tx, sig_dbm)
            else:
                self.stats.rx_fail += 1
                self.nst.on_rx_fail(rx)
                continue

            if not _rx_eligible(w.packet, rx):
                continue

            any_eligible_ok_this_waveform = True
            self._attempt_any_eligible_ok[attempt_id] = True

            # Phase 7: capture decision-time state BEFORE policy decision
            _dl_snap: Optional[Dict[str, Any]] = None
            _dl_pkt: Optional[Dict[str, Any]] = None
            if self.decision_logger is not None:
                _bucket_snap: Optional[Dict[str, Any]] = None
                if self.airtime_buckets is not None and rx in self.airtime_buckets:
                    _bucket_snap = self.airtime_buckets[rx].snapshot(float(t_end))
                _dl_snap = self.nst.state(int(rx)).snapshot(
                    t_now=float(t_end),
                    queue_len=int(len(self.nodes[int(rx)].tx_queue)),
                    max_queue=int(self._max_queue),
                    airtime_bucket_snapshot=_bucket_snap,
                )
                _dl_pkt = self.nst.packet_features(float(t_end), w.packet)

            actions = self.policy.on_rx(self.cfg, t_end, rx_node=int(rx), pkt=w.packet, prev_hop=int(tx))

            # Phase 7/8/10R: log the forwarding decision immediately after policy call
            if self.decision_logger is not None and _dl_snap is not None and _dl_pkt is not None:
                # Phase 8: extract candidate_neighbor from forwarded packet meta
                # For broadcast policies: next_hop not in meta → -1
                # For next-hop policies (e.g. BatmanLite, drf_ble route-quality): use it
                _candidate_neighbor = -1
                if actions:
                    try:
                        _fwd_meta = dict(actions[0][1].meta)
                        if "next_hop" in _fwd_meta:
                            _candidate_neighbor = int(_fwd_meta["next_hop"])
                    except Exception:
                        _candidate_neighbor = -1

                # Phase 10R: read reason_code, fallback_used, suppression_used from policy
                # if the policy exposes _last_decision_metadata (ReliabilityGatedPolicy).
                # Falls back to Phase 7/8 defaults (empty reason, False flags) for other policies.
                _reason_code = ""
                _fallback_used = False
                _suppression_used = False
                if hasattr(self.policy, "_last_decision_metadata"):
                    _md = self.policy._last_decision_metadata
                    if _md:  # non-empty (populated after an eligible decision)
                        _reason_code = str(_md.get("reason_code", ""))
                        _fallback_used = bool(_md.get("fallback_used", False))
                        _suppression_used = bool(_md.get("suppression_used", False))

                self.decision_logger.log_decision(
                    t_now=float(t_end),
                    packet=w.packet,
                    current_node=int(rx),
                    previous_node=int(tx),
                    node_snap=_dl_snap,
                    pkt_feats=_dl_pkt,
                    decision_taken=bool(len(actions) > 0),
                    fallback_used=_fallback_used,
                    suppression_used=_suppression_used,
                    candidate_neighbor=_candidate_neighbor,
                    reason_code=_reason_code,
                )

            for nid, pkt2 in actions:
                self.enqueue_packet(int(nid), pkt2, now=t_end)

            if (not bool(w.packet.is_control)) and rx == dst:
                key2 = self._pid_key(w.packet)
                if key2 not in self._delivered_keys:
                    self._delivered_keys.add(key2)
                    self.stats.delivered += 1
                    # Phase 7: record delivery outcome for post-run label fill
                    if self.decision_logger is not None:
                        self._delivery_outcomes[key2] = {
                            "delivery_time": float(t_end),
                            "final_ttl_remaining": int(w.packet.ttl_hops),
                        }

        if any_eligible_ok_this_waveform:
            self.stats.tx_success += 1

    def _start_attempt(self, tx_node: int) -> None:
        ns = self.nodes[int(tx_node)]
        if not ns.tx_queue:
            return
        pkt = ns.tx_queue[0]

        if not bool(getattr(pkt, "is_control", False)):
            key = self._pid_key(pkt)
            if key in self._delivered_keys or key in self._final_dropped_keys:
                ns.tx_queue.pop(0)
                self._kick_try_tx(int(tx_node), float(self.sch.now))
                return

        t_now = float(self.sch.now)
        t_resume = self._maybe_budget_defer(int(tx_node), t_now, pkt)
        if t_resume is not None:
            if math.isinf(t_resume):
                self._mark_final_drop_if_data(pkt, DropReason.BUDGET, 1)
                if ns.tx_queue and ns.tx_queue[0].pid == pkt.pid:
                    ns.tx_queue.pop(0)
                self._kick_try_tx(int(tx_node), t_now)
                return
            self._kick_try_tx(int(tx_node), float(t_resume))
            return

        # Phase 12C-0: opt-in pre-transmit re-check (additive; existing policies never enter).
        # Runs only at the airtime-commit point, after delivered/dropped + budget-defer checks
        # and before any attempt/airtime is scheduled. A False return is VOLUNTARY suppression:
        # drop the queued relay without a final-drop counter (other nodes may still relay), then
        # resume scheduling via the same pop+kick path used by the delivered/dropped case.
        if self._tx_ready_hook_on:
            if self.policy.on_tx_ready(self.cfg, t_now, tx_node=int(tx_node), pkt=pkt) is False:
                if ns.tx_queue and ns.tx_queue[0].pid == pkt.pid:
                    ns.tx_queue.pop(0)
                self._kick_try_tx(int(tx_node), t_now)
                return

        aid = self._new_attempt(pkt=pkt, tx_node=int(tx_node), retries_left=int(self.cfg.mac.max_retries))
        self._schedule_adv_channels(aid, t0=t_now)

    def run(self, injections: List[AppInjection], duration_s: Optional[float] = None) -> Dict[str, Any]:
        dur = float(self.cfg.sim_duration_s) if duration_s is None else float(duration_s)
        dur = max(0.0, dur)
        self.stats.sim_duration_s = dur

        for inj in injections:
            self.sch.schedule(
                t=float(inj.t),
                etype="APP_INJECT",
                prio=int(EventPrio.APP),
                data={"inj": inj},
                stream="traffic",
            )

        trace_head: List[Tuple[float, int, str, int]] = []

        while True:
            try:
                ev = self.sch.pop()
            except IndexError:
                break

            now = float(ev.t)
            if now >= dur - 1e-12:
                break

            if len(trace_head) < 30:
                src_val = -1
                if isinstance(ev.data, dict):
                    if ev.etype == "APP_INJECT":
                        try:
                            src_val = int(ev.data["inj"].src)
                        except Exception:
                            src_val = -1
                    elif ev.etype == "MAC_TRY_TX" and "src" in ev.data:
                        try:
                            src_val = int(ev.data["src"])
                        except Exception:
                            src_val = -1
                    elif ev.etype == "MAC_TX_CH":
                        try:
                            aid0 = int(ev.data.get("attempt_id", -1))
                            att0 = self._attempts.get(aid0)
                            if att0 is not None:
                                src_val = int(att0.tx_node)
                        except Exception:
                            src_val = -1
                        if src_val == -1 and "src" in ev.data:
                            try:
                                src_val = int(ev.data["src"])
                            except Exception:
                                src_val = -1
                trace_head.append((float(ev.t), int(ev.prio), str(ev.etype), int(src_val)))

            if ev.etype == "APP_INJECT":
                self._enqueue_app_injection(ev.data["inj"])

            elif ev.etype == "PKT_INJECT":
                pkt = ev.data["pkt"]
                # Phase 7: register original TTL for control/external packets
                if self.decision_logger is not None:
                    self.decision_logger.on_packet_created(
                        (int(pkt.pid.flow_id), int(pkt.pid.seq)),
                        int(pkt.ttl_hops),
                    )
                actions = self.policy.on_app_inject(self.cfg, now, pkt)
                for nid, pkt2 in actions:
                    self.enqueue_packet(int(nid), pkt2, now=now)

            elif ev.etype == "MAC_TRY_TX":
                self._start_attempt(int(ev.data["src"]))

            elif ev.etype == "MAC_TX_CH":
                attempt_id = int(ev.data["attempt_id"])
                ch = int(ev.data["ch"])
                t_end = float(ev.data["t_end"])

                att = self._attempts.get(attempt_id)
                if att is None:
                    continue

                tx_node = int(att.tx_node)
                tx_ns = self.nodes[tx_node]
                if not tx_ns.tx_queue or tx_ns.tx_queue[0].pid != att.pkt.pid:
                    self._retire_attempt(attempt_id)
                    continue

                self._prune_tx_active(now)

                if self._cca_busy(int(tx_node), ch, now):
                    self.stats.cca_busy_events += 1
                    self.nst.on_cca_check(int(tx_node), busy=True)
                    self._retire_attempt(attempt_id)
                    if att.retries_left > 0:
                        new_aid = self._new_attempt(
                            pkt=att.pkt,
                            tx_node=int(tx_node),
                            retries_left=int(att.retries_left) - 1,
                            last_fail_reason=DropReason.CCA_BUSY,
                        )
                        t_retry = now + self._backoff_delay(int(tx_node))
                        self._schedule_adv_channels(new_aid, t0=t_retry)
                    else:
                        self._mark_final_drop_if_data(att.pkt, DropReason.CCA_BUSY, 1)
                        if tx_ns.tx_queue and tx_ns.tx_queue[0].pid == att.pkt.pid:
                            tx_ns.tx_queue.pop(0)
                            self._kick_try_tx(int(tx_node), now)
                    continue

                w = TxWaveform(
                    tx_node=int(tx_node),
                    packet=att.pkt,
                    ch=int(ch),
                    t_start=now,
                    t_end=float(t_end),
                    tx_power_dbm=float(self.cfg.phy.tx_power_dbm),
                )
                self._tx_active.append(w)

                self.stats.tx_attempts += 1
                self.nst.on_cca_check(int(tx_node), busy=False)
                self.nst.on_tx_attempt(int(tx_node))

                dt = float(w.t_end - w.t_start)
                self.stats.airtime_s += dt
                if bool(att.pkt.is_control):
                    self.stats.airtime_ctrl_s += dt
                else:
                    self.stats.airtime_data_s += dt
                self.stats.energy_j += dt * dbm_to_mw(float(self.cfg.phy.tx_power_dbm))  # airtime-proportional radiated-power proxy; NOT A-005 device supply energy
                self.nst.on_airtime(int(tx_node), dt)

                self._eval_rx_for_waveform(attempt_id, w)

                chs = tuple(self.cfg.mac.adv_channels)
                if int(ch) == int(chs[-1]):
                    if not (tx_ns.tx_queue and tx_ns.tx_queue[0].pid == att.pkt.pid):
                        self._retire_attempt(attempt_id)
                        continue

                    _had_int = bool(self._attempt_had_interference.get(attempt_id, False))

                    if bool(self._attempt_any_eligible_ok.get(attempt_id, False)):
                        self.nst.on_tx_burst_outcome(int(tx_node), success=True, had_interference=_had_int)
                        tx_ns.tx_queue.pop(0)
                        self._retire_attempt(attempt_id)
                        self._kick_try_tx(int(tx_node), float(t_end))
                        continue

                    if att.retries_left > 0:
                        self.nst.on_tx_burst_outcome(int(tx_node), success=False, had_interference=_had_int)
                        self._retire_attempt(attempt_id)
                        new_aid = self._new_attempt(
                            pkt=att.pkt,
                            tx_node=int(tx_node),
                            retries_left=int(att.retries_left) - 1,
                            last_fail_reason=DropReason.PHY_ERROR,
                        )
                        t_retry = float(t_end) + self._backoff_delay(int(tx_node))
                        self._schedule_adv_channels(new_aid, t0=t_retry)
                    else:
                        self.nst.on_tx_burst_outcome(int(tx_node), success=False, had_interference=_had_int)
                        self._retire_attempt(attempt_id)
                        if not is_scanning(self.cfg, float(t_end)):
                            self._mark_final_drop_if_data(att.pkt, DropReason.NOT_SCANNING, 1)
                        else:
                            self._mark_final_drop_if_data(att.pkt, DropReason.PHY_ERROR, 1)
                        if tx_ns.tx_queue and tx_ns.tx_queue[0].pid == att.pkt.pid:
                            tx_ns.tx_queue.pop(0)
                            self._kick_try_tx(int(tx_node), float(t_end))

            else:
                raise ValueError(f"Unexpected event type: {ev.etype}")

        result: Dict[str, Any] = {
            "summary": self.stats.summary(),
            "trace_head": trace_head,
            "node_state_snapshots": self.nst.snapshot_all(
                t_now=dur,
                nodes=self.nodes,
                max_queue=self._max_queue,
                airtime_buckets=self.airtime_buckets,
            ),
        }

        # Phase 7: fill post-run labels and attach decision log
        if self.decision_logger is not None:
            self.decision_logger.fill_labels(self._delivery_outcomes)
            result["decision_log"] = self.decision_logger.rows()
            result["decision_log_hash"] = self.decision_logger.log_hash()

        return result
