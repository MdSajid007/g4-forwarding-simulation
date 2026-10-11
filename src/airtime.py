"""
Airtime token bucket: per-node MAC airtime budget with refill rate; gating TX attempts.

Extracted from notebook Cell 10, Phase 3.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import SimCfg, validate_cfg, CfgError


# ---- internal cost model ----

def _adv_tx_duration_s_airtime(payload_bytes: int) -> float:
    """
    Approximate BLE advertising PDU airtime at 1 Mbps.
    bits = preamble(8) + access(32) + header(16) + payload(8*B) + crc(24)
    duration = bits / 1e6
    """
    b = int(payload_bytes)
    bits = 8 + 32 + 16 + (8 * b) + 24
    return float(bits) / 1_000_000.0


def airtime_cost_s_for_adv_burst(payload_bytes: int, n_channels: int) -> float:
    """
    Airtime "cost" for one ADV burst across n_channels.
    Stable API: do not rename.
    """
    return float(int(n_channels)) * float(_adv_tx_duration_s_airtime(int(payload_bytes)))


# ---- token bucket ----

@dataclass
class AirtimeTokenBucket:
    """
    Tokens measured in seconds-of-airtime.
    Refill is linear: tokens += refill_rate_sps * dt, capped at capacity_s.

    Determinism:
    - Deterministic given call sequence (t_now, cost_s).
    - No RNG. No wall-clock.
    """
    capacity_s: float
    refill_rate_sps: float
    init_tokens_s: float = 0.0

    _tokens_s: float = 0.0
    _t_last: Optional[float] = None

    def __post_init__(self) -> None:
        self.capacity_s = float(self.capacity_s)
        self.refill_rate_sps = float(self.refill_rate_sps)
        self.init_tokens_s = float(self.init_tokens_s)

        if self.capacity_s < 0.0:
            raise ValueError("capacity_s must be >= 0")
        if self.refill_rate_sps < 0.0:
            raise ValueError("refill_rate_sps must be >= 0")
        if self.init_tokens_s < 0.0:
            raise ValueError("init_tokens_s must be >= 0")

        init = self.init_tokens_s
        if init > self.capacity_s:
            init = self.capacity_s
        self._tokens_s = float(init)
        self._t_last = None

    def _refill_to(self, t_now: float) -> None:
        t = float(t_now)
        if self._t_last is None:
            self._t_last = t
            return

        dt = t - float(self._t_last)
        self._t_last = t
        if dt <= 0.0:
            return

        if self.capacity_s <= 0.0 or self.refill_rate_sps <= 0.0:
            return

        self._tokens_s = min(
            self.capacity_s,
            float(self._tokens_s) + float(self.refill_rate_sps) * float(dt),
        )

    def peek(self, t_now: float) -> Dict[str, Any]:
        """
        Read-only snapshot: returns token state as of t_now WITHOUT mutating
        _t_last or _tokens_s. Safe for decision-time hot paths (E6 / Phase 7 N1).
        """
        t = float(t_now)
        tokens = float(self._tokens_s)
        if (
            self._t_last is not None
            and self.capacity_s > 0.0
            and self.refill_rate_sps > 0.0
        ):
            dt = t - float(self._t_last)
            if dt > 0.0:
                tokens = min(
                    float(self.capacity_s),
                    tokens + float(self.refill_rate_sps) * dt,
                )
        return {
            "t_last": None if self._t_last is None else float(self._t_last),
            "tokens_s": float(tokens),
            "capacity_s": float(self.capacity_s),
            "refill_rate_sps": float(self.refill_rate_sps),
        }

    def snapshot(self, t_now: float) -> Dict[str, Any]:
        self._refill_to(float(t_now))
        return {
            "t_last": None if self._t_last is None else float(self._t_last),
            "tokens_s": float(self._tokens_s),
            "capacity_s": float(self.capacity_s),
            "refill_rate_sps": float(self.refill_rate_sps),
        }

    def can_consume(self, t_now: float, cost_s: float) -> bool:
        self._refill_to(float(t_now))
        c = float(cost_s)
        if c <= 0.0:
            return True
        return float(self._tokens_s) + 1e-18 >= c

    def consume(self, t_now: float, cost_s: float) -> bool:
        self._refill_to(float(t_now))
        c = float(cost_s)
        if c <= 0.0:
            return True
        if float(self._tokens_s) + 1e-18 < c:
            return False
        self._tokens_s = float(self._tokens_s) - c
        if self._tokens_s < 0.0:
            self._tokens_s = 0.0
        return True


def build_airtime_buckets(cfg: SimCfg) -> Optional[Dict[int, AirtimeTokenBucket]]:
    """
    Canonical per-node bucket builder from cfg.mac airtime fields.

    Gating (must match engine expectations):
    - If budget disabled => None
    - If capacity_s <= 0 or refill_rate_sps <= 0 => None
    - init_tokens_s is clamped to [0, capacity_s]
    """
    validate_cfg(cfg)
    mac = cfg.mac

    enabled = bool(getattr(mac, "airtime_budget_enabled", False))
    if not enabled:
        return None

    cap = float(getattr(mac, "airtime_budget_capacity_s", 0.0))
    refill = float(getattr(mac, "airtime_budget_refill_sps", 0.0))
    if cap <= 0.0 or refill <= 0.0:
        return None

    init_raw = float(getattr(mac, "airtime_budget_init_s", cap))
    init = max(0.0, min(cap, init_raw))

    out: Dict[int, AirtimeTokenBucket] = {}
    for i in range(int(cfg.n_nodes)):
        out[int(i)] = AirtimeTokenBucket(capacity_s=cap, refill_rate_sps=refill, init_tokens_s=init)
    return out
