"""
PHY/channel model: log-distance path loss, noise, Rayleigh/Rician fading, shadowing,
logistic PER curve, SINR, interference.

Extracted from notebook Cell 7, Phase 3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .config import SimCfg, validate_cfg
from .entities import Packet


# ----------------------------
# Power helpers (stable math)
# ----------------------------

def dbm_to_mw(p_dbm: float) -> float:
    """Convert dBm to mW."""
    return 10.0 ** (float(p_dbm) / 10.0)


def mw_to_dbm(p_mw: float) -> float:
    """Convert mW to dBm. Returns -inf for non-positive."""
    p = float(p_mw)
    if p <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(p)


def sum_dbm(p_dbm_list: Iterable[float]) -> float:
    """Stable sum in power domain; returns dBm."""
    total_mw = 0.0
    for p_dbm in p_dbm_list:
        if math.isfinite(float(p_dbm)):
            total_mw += dbm_to_mw(float(p_dbm))
    return mw_to_dbm(total_mw)


# ----------------------------
# PHY model dataclasses
# ----------------------------

@dataclass(frozen=True)
class TxWaveform:
    """A single transmission instance on one advertising channel."""
    tx_node: int
    packet: Packet
    ch: int  # 37/38/39
    t_start: float
    t_end: float
    tx_power_dbm: float


@dataclass(frozen=True)
class PhyObs:
    """Per-receiver observation outcome for one TxWaveform."""
    rx_node: int
    ch: int
    t_sample: float
    rssi_dbm: float
    sinr_db: float
    per: float
    ok: bool
    had_interference: bool


# ----------------------------
# PHY model
# ----------------------------

class PhyModel:
    """
    BLE-ish PHY abstraction:

    - RSSI = tx_power - pathloss(d) + optional shadowing + optional small-scale fading.
    - Interference from overlapping transmissions on same channel (provided by MAC engine).
    - SINR->PER via logistic curve:
        per = 1 / (1 + exp(per_k*(sinr_db - sinr0_db)))
      (per ~ 0.5 at sinr0_db, -> 0 at high SINR, -> 1 at low SINR)

    Determinism:
    - No internal RNG state. Caller passes rng when it wants fading/shadowing.
    - If rng is None => legacy pathloss-only behavior.
    """

    def __init__(self, cfg: SimCfg) -> None:
        validate_cfg(cfg)
        self.cfg = cfg
        self.p = cfg.phy

        self.fading_model = str(getattr(self.p, "fading_model", "none")).strip().lower()
        self.rician_k_db = float(getattr(self.p, "rician_k_db", 6.0))
        self.shadow_sigma_db = float(getattr(self.p, "shadow_sigma_db", 0.0))

        # Legacy compatibility: small_scale_fading True implies Rayleigh if fading_model is "none"
        if bool(getattr(self.p, "small_scale_fading", False)) and self.fading_model in ("none", "", "off", "disabled"):
            self.fading_model = "rayleigh"

    def pathloss_db(self, d_m: float) -> float:
        """Log-distance pathloss (d>=ref_dist)."""
        d = max(float(d_m), 1e-9)
        d0 = float(self.p.ref_dist_m)
        n = float(self.p.pathloss_exp)
        pl0 = float(self.p.ref_pl_db)
        if d <= d0:
            return pl0
        return pl0 + 10.0 * n * math.log10(d / d0)

    @staticmethod
    def _rayleigh_power_gain(rng: np.random.Generator) -> float:
        """Rayleigh fading power gain ~ Exponential(mean=1)."""
        g = float(rng.exponential(1.0))
        return max(g, 1e-30)

    @staticmethod
    def _rician_power_gain(rng: np.random.Generator, k_db: float) -> float:
        """Rician fading power gain (normalized to mean ~1)."""
        K = 10.0 ** (float(k_db) / 10.0)
        los_amp = math.sqrt(K / (K + 1.0))
        sigma = math.sqrt(1.0 / (2.0 * (K + 1.0)))
        i = float(rng.normal(0.0, sigma))
        q = float(rng.normal(0.0, sigma))
        ci = los_amp + i
        cq = q
        power = (ci * ci) + (cq * cq)
        return max(float(power), 1e-30)

    def _small_scale_fading_db(self, rng: np.random.Generator) -> float:
        m = self.fading_model
        if m in ("none", "", "off", "disabled"):
            return 0.0
        if m in ("rayleigh", "ray"):
            return 10.0 * math.log10(self._rayleigh_power_gain(rng))
        if m in ("rician", "rice"):
            return 10.0 * math.log10(self._rician_power_gain(rng, self.rician_k_db))
        return 0.0

    def _shadowing_db(self, rng: np.random.Generator) -> float:
        s = float(self.shadow_sigma_db)
        if s <= 0.0:
            return 0.0
        return float(rng.normal(0.0, s))

    def rssi_dbm(
        self,
        tx_power_dbm: float,
        tx_pos: Tuple[float, float],
        rx_pos: Tuple[float, float],
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        dx = float(tx_pos[0]) - float(rx_pos[0])
        dy = float(tx_pos[1]) - float(rx_pos[1])
        d = math.hypot(dx, dy)
        base = float(tx_power_dbm) - self.pathloss_db(d)
        if rng is None:
            return base
        return float(base + self._shadowing_db(rng) + self._small_scale_fading_db(rng))

    def sinr_db(self, sig_dbm: float, interf_dbm_list: List[float], noise_floor_dbm: float) -> float:
        s_mw = dbm_to_mw(sig_dbm)
        i_mw = 0.0
        for p_dbm in interf_dbm_list:
            if math.isfinite(float(p_dbm)):
                i_mw += dbm_to_mw(float(p_dbm))
        n_mw = dbm_to_mw(float(noise_floor_dbm))
        denom = i_mw + n_mw
        if denom <= 0.0:
            return float("inf")
        return 10.0 * math.log10(s_mw / denom)

    def per_from_sinr(self, sinr_db: float) -> float:
        x = float(sinr_db) - float(self.p.sinr0_db)
        k = float(self.p.per_k)
        z = max(min(k * x, 60.0), -60.0)
        per = 1.0 / (1.0 + math.exp(z))
        if per < 0.0:
            return 0.0
        if per > 1.0:
            return 1.0
        return per
