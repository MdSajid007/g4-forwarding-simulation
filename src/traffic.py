"""
Traffic injection: CBR and Poisson APP_INJECT events; stable injection hash for manifests.

Extracted from notebook Cell 6, Phase 3.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import CfgError, SimCfg, validate_cfg
from .hashing import INJECT_T_NDIGITS, canonical_json
from .scheduler import EventPrio, EventScheduler
from .seeds import SeedManager


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AppInjection:
    t: float
    flow_id: int
    src: int
    dst: int
    payload_bytes: int
    ttl_hops: int


DEFAULT_TTL_HOPS = 16


def injections_hash(injections: List[AppInjection], t_ndigits: int = INJECT_T_NDIGITS) -> str:
    """
    Stable hash of the injection list for manifests/gates.
    Depends on canonical_json().
    """
    rows: List[Tuple[float, int, int, int, int, int]] = []
    for inj in injections:
        rows.append(
            (
                round(float(inj.t), t_ndigits),
                int(inj.flow_id),
                int(inj.src),
                int(inj.dst),
                int(inj.payload_bytes),
                int(inj.ttl_hops),
            )
        )
    return _sha256_hex(canonical_json(rows))


def build_injections(cfg: SimCfg, sm: SeedManager, ttl_hops: int = DEFAULT_TTL_HOPS) -> List[AppInjection]:
    """
    Build deterministic application injection times per flow.

    Contract:
      - Must be called once per SeedManager instance for stream 'traffic' (enforced).
      - Uses cached stream 'traffic' (rng_cached).
    """
    validate_cfg(cfg)
    if not isinstance(ttl_hops, int) or ttl_hops <= 0:
        raise ValueError("ttl_hops must be positive int")

    sm.guard_build_once("traffic", caller="build_injections")
    rng = sm.rng_cached("traffic")

    injections: List[AppInjection] = []
    for flow_id, f in enumerate(cfg.traffic.flows):
        t = float(f.start_s)

        if f.model == "cbr":
            step = 1.0 / float(f.rate_pps)
            while t < float(f.stop_s) - 1e-12:
                injections.append(
                    AppInjection(
                        t=t,
                        flow_id=flow_id,
                        src=f.src,
                        dst=f.dst,
                        payload_bytes=f.payload_bytes,
                        ttl_hops=ttl_hops,
                    )
                )
                t += step

        elif f.model == "poisson":
            while t < float(f.stop_s) - 1e-12:
                injections.append(
                    AppInjection(
                        t=t,
                        flow_id=flow_id,
                        src=f.src,
                        dst=f.dst,
                        payload_bytes=f.payload_bytes,
                        ttl_hops=ttl_hops,
                    )
                )
                dt = float(rng.exponential(1.0 / float(f.rate_pps)))
                if dt <= 0.0:
                    dt = 1e-9
                t += dt

        else:
            raise CfgError(f"Unsupported traffic model: {f.model}")

    injections.sort(key=lambda a: (a.t, a.flow_id, a.src, a.dst))
    return injections


def schedule_injections(sch: EventScheduler, injections: List[AppInjection]) -> None:
    for inj in injections:
        sch.schedule(
            t=inj.t,
            etype="APP_INJECT",
            prio=int(EventPrio.APP),
            data={
                "flow_id": inj.flow_id,
                "src": inj.src,
                "dst": inj.dst,
                "payload_bytes": inj.payload_bytes,
                "ttl_hops": inj.ttl_hops,
            },
            stream="traffic",
        )
