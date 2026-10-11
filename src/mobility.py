"""
Random Waypoint mobility engine.

Extracted from notebook Cell 5, Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import SimCfg, validate_cfg
from .hashing import MOBILITY_DIGEST_NDIGITS
from .seeds import SeedManager


@dataclass(frozen=True)
class MotionSegment:
    """Linear motion segment from (x0,y0) at t0 to (x1,y1) at t1 inclusive."""
    t0: float
    t1: float
    x0: float
    y0: float
    x1: float
    y1: float

    def pos_at(self, t: float) -> Tuple[float, float]:
        if t <= self.t0:
            return (self.x0, self.y0)
        if t >= self.t1:
            return (self.x1, self.y1)
        dt = self.t1 - self.t0
        if dt <= 0:
            return (self.x1, self.y1)
        a = (t - self.t0) / dt
        return (self.x0 + a * (self.x1 - self.x0), self.y0 + a * (self.y1 - self.y0))


@dataclass
class MobilityPlan:
    """
    Deterministic mobility plan for all nodes.
    segments_by_node[node_id] is a time-ordered list.
    """
    segments_by_node: Dict[int, List[MotionSegment]]
    area_w: float
    area_h: float

    def pos_at(self, node_id: int, t: float) -> Tuple[float, float]:
        segs = self.segments_by_node[node_id]
        for seg in segs:
            if seg.t0 <= t <= seg.t1:
                return seg.pos_at(t)
        if t < segs[0].t0:
            return (segs[0].x0, segs[0].y0)
        return (segs[-1].x1, segs[-1].y1)

    def digest_first_segments(self, k: int = 2) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """
        Compact deterministic digest for testing: per node, first k segments as tuples
        (t0,t1,x0,y0) rounded using MOBILITY_DIGEST_NDIGITS.
        """
        nd = MOBILITY_DIGEST_NDIGITS
        out: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for nid, segs in self.segments_by_node.items():
            take = segs[:k]
            out[nid] = [(round(s.t0, nd), round(s.t1, nd), round(s.x0, nd), round(s.y0, nd)) for s in take]
        return out


def _sample_uniform_xy(rng: np.random.Generator, w: float, h: float) -> Tuple[float, float]:
    return (float(rng.uniform(0.0, w)), float(rng.uniform(0.0, h)))


def build_rwp_plan(cfg: SimCfg, sm: SeedManager) -> MobilityPlan:
    """
    Build a Random Waypoint mobility plan for all nodes up to sim_duration.
    Deterministic given cfg_hash + seed via SeedManager.

    Contract:
      - Must be called once per SeedManager instance for stream 'mobility' (enforced).
      - Uses cached stream 'mobility'.
    """
    validate_cfg(cfg)
    m = cfg.mobility
    assert m.model == "rwp"

    sm.guard_build_once("mobility", caller="build_rwp_plan")
    rng = sm.rng_cached("mobility")

    segs_by_node: Dict[int, List[MotionSegment]] = {}
    for nid in range(cfg.n_nodes):
        t = 0.0
        x, y = _sample_uniform_xy(rng, m.area_w, m.area_h)
        segs: List[MotionSegment] = []
        segs.append(MotionSegment(t0=0.0, t1=0.0, x0=x, y0=y, x1=x, y1=y))

        while t < cfg.sim_duration_s:
            if m.pause_s > 0:
                t_pause_end = min(cfg.sim_duration_s, t + float(m.pause_s))
                if t_pause_end > t:
                    segs.append(MotionSegment(t0=t, t1=t_pause_end, x0=x, y0=y, x1=x, y1=y))
                    t = t_pause_end
                    if t >= cfg.sim_duration_s:
                        break

            nx, ny = _sample_uniform_xy(rng, m.area_w, m.area_h)
            v = float(rng.uniform(m.v_min, m.v_max)) if m.v_max > 0 else 0.0
            dx = nx - x
            dy = ny - y
            dist = float(np.hypot(dx, dy))

            if dist == 0.0 or v == 0.0:
                t1 = min(cfg.sim_duration_s, t + 1e-9)
                segs.append(MotionSegment(t0=t, t1=t1, x0=x, y0=y, x1=x, y1=y))
                t = t1
                continue

            travel = dist / v
            t1 = min(cfg.sim_duration_s, t + travel)
            if t + travel > cfg.sim_duration_s:
                frac = (cfg.sim_duration_s - t) / travel
                px = x + frac * dx
                py = y + frac * dy
                segs.append(MotionSegment(t0=t, t1=cfg.sim_duration_s, x0=x, y0=y, x1=px, y1=py))
                x, y = px, py
                t = cfg.sim_duration_s
            else:
                segs.append(MotionSegment(t0=t, t1=t1, x0=x, y0=y, x1=nx, y1=ny))
                x, y = nx, ny
                t = t1

        segs_by_node[nid] = segs

    return MobilityPlan(segments_by_node=segs_by_node, area_w=m.area_w, area_h=m.area_h)


def build_static_plan(cfg: SimCfg) -> MobilityPlan:
    """
    Build a deterministic static-placement mobility plan (Phase 12B-0, A-025).

    Each node occupies a single fixed position for the whole run, expressed as one
    constant MotionSegment spanning [0, sim_duration_s].

    Properties (verified by tests T4/T5):
      - RNG-FREE: makes no random draws and takes no SeedManager.
      - SEED-INDEPENDENT: positions depend only on cfg.mobility.node_positions.
      - DETERMINISTIC: identical output for identical cfg.
      - Does NOT consume the 'mobility' seed substream, so it cannot perturb the
        independent 'phy' / 'mac' / 'traffic' substreams.

    CLAIM BOUNDARY: static placement is a controlled diagnostic topology only; see
    StaticMobilityCfg docstring and docs/assumption_ledger.md A-025.
    """
    validate_cfg(cfg)
    m = cfg.mobility
    assert m.model == "static", "build_static_plan requires mobility.model == 'static'"
    positions = getattr(m, "node_positions", ())

    dur = float(cfg.sim_duration_s)
    segs_by_node: Dict[int, List[MotionSegment]] = {}
    for nid in range(int(cfg.n_nodes)):
        x = float(positions[nid][0])
        y = float(positions[nid][1])
        segs_by_node[nid] = [
            MotionSegment(t0=0.0, t1=dur, x0=x, y0=y, x1=x, y1=y)
        ]

    return MobilityPlan(segments_by_node=segs_by_node, area_w=m.area_w, area_h=m.area_h)


def build_plan(cfg: SimCfg, sm: SeedManager) -> MobilityPlan:
    """
    Mobility-plan dispatcher (Phase 12B-0).

    Routes by cfg.mobility.model:
      - "rwp"    -> build_rwp_plan(cfg, sm)   [UNCHANGED RWP path; same RNG usage]
      - "static" -> build_static_plan(cfg)    [RNG-free; sm unused]

    The RWP branch is byte-identical to calling build_rwp_plan directly (verified by
    test T3). Existing direct callers of build_rwp_plan are intentionally left
    unchanged; only the canonical entrypoint (simulate_once) routes through here.
    """
    model = str(cfg.mobility.model)
    if model == "rwp":
        return build_rwp_plan(cfg, sm)
    if model == "static":
        return build_static_plan(cfg)
    raise ValueError(f"Unsupported mobility model: {model!r}")
