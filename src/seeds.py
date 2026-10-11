"""
SeedManager: deterministic per-stream RNG derivation from config hash and master seed.

Extracted from notebook Cell 3, Phase 3.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np


def _u64_from_bytes(b: bytes) -> int:
    """Interpret first 8 bytes as uint64 (little-endian) deterministically."""
    return int.from_bytes(b[:8], byteorder="little", signed=False)


def _derive_seed_u64(*parts: str) -> int:
    """
    Derive a uint64 seed from arbitrary string parts using SHA256.
    Stable across platforms.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")  # separator
    return _u64_from_bytes(h.digest())


@dataclass(frozen=True)
class SeedManager:
    """
    Deterministic RNG substream manager.

    Inputs:
    - cfg_hash: stable hash from Cell 1
    - seed: per-run seed integer

    Provides:
    - rng(name): NEW numpy Generator for a named substream (stateless constructor)
    - rng_cached(name): PERSISTENT numpy Generator per stream_name
    - guard_build_once(stream_name, caller): enforce "build once per run" for timeline builders
    """
    cfg_hash: str
    seed: int
    algorithm: str = "PCG64DXSM"

    _cache: Dict[str, np.random.Generator] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _built: Dict[str, bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def seed_u64(self, stream_name: str) -> int:
        if not isinstance(stream_name, str) or not stream_name:
            raise ValueError("stream_name must be non-empty string")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative int")
        return _derive_seed_u64(self.cfg_hash, str(self.seed), self.algorithm, stream_name)

    def _make_rng(self, stream_name: str) -> np.random.Generator:
        u64 = self.seed_u64(stream_name)
        ss = np.random.SeedSequence([u64 & 0xFFFFFFFF, (u64 >> 32) & 0xFFFFFFFF])
        bitgen = np.random.PCG64DXSM(ss)
        return np.random.Generator(bitgen)

    def rng(self, stream_name: str) -> np.random.Generator:
        """Return a NEW Generator for this substream (restarts at origin each call)."""
        return self._make_rng(stream_name)

    def rng_cached(self, stream_name: str) -> np.random.Generator:
        """Return a PERSISTENT Generator for this substream (continues across calls)."""
        g = self._cache.get(stream_name)
        if g is None:
            g = self._make_rng(stream_name)
            self._cache[stream_name] = g
        return g

    def spawn(self, stream_names: Iterable[str]) -> Dict[str, np.random.Generator]:
        """Return NEW Generators for each requested stream name."""
        return {name: self.rng(name) for name in stream_names}

    def clear_cache(self) -> None:
        """Clear cached generators (debug/testing only)."""
        self._cache.clear()

    def guard_build_once(self, stream_name: str, caller: str) -> None:
        """
        Enforce: a builder that consumes a stream (e.g., mobility/traffic timeline) is called once per run.
        Does NOT depend on RNG cache structure.
        """
        if not isinstance(stream_name, str) or not stream_name:
            raise ValueError("stream_name must be non-empty string")
        if self._built.get(stream_name, False):
            raise RuntimeError(
                f"{caller} called more than once for build stream '{stream_name}' in the same run. "
                f"Build once and reuse the returned object."
            )
        self._built[stream_name] = True


STD_STREAMS = (
    "core",
    "scheduler",
    "mobility",
    "traffic",
    "phy",
    "mac",
    "policy",
    "logging",
)
