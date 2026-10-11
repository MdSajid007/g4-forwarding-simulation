"""
Pure Phase 13 CGF coverage-hint codec helpers.

The functions here mirror the proven Phase 13A probe helpers: deterministic,
fixed-width integer hints, no RNG, no topology state, and no simulator side effects.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Tuple


COV_HINT_META_KEY = "cov_hint"
SUPPORTED_HINT_WIDTHS = (1, 2, 4)


def mask_for_width(width_bytes: int) -> int:
    width = int(width_bytes)
    if width not in SUPPORTED_HINT_WIDTHS:
        raise ValueError(f"unsupported CGF hint width: {width_bytes!r}")
    return (1 << (8 * width)) - 1


def normalize_cov_hint(value: Any, width_bytes: int) -> int:
    """
    Decode a hint-like value into the valid integer range for width_bytes.

    Invalid, missing, negative, NaN-like, or otherwise undecodable values map to 0.
    Over-wide integers are clamped to the fixed-width mask. This preserves the
    Phase 13A probe behavior.
    """
    mask = mask_for_width(int(width_bytes))
    if value is None:
        return 0
    try:
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return 0
            raw = int(s, 16) if s.lower().startswith("0x") else int(s)
        else:
            raw = int(value)
    except Exception:
        return 0
    if raw < 0:
        return 0
    if raw > mask:
        return int(mask)
    return int(raw)


def cov_hint_from_meta(meta: Iterable[Tuple[str, Any]], width_bytes: int) -> int:
    """Extract and normalize Packet.meta['cov_hint']; absent/invalid hints become 0."""
    try:
        items = dict(meta)
    except Exception:
        items = {}
    return normalize_cov_hint(items.get(COV_HINT_META_KEY, 0), int(width_bytes))


def cov_hint_present_and_valid(meta: Iterable[Tuple[str, Any]], width_bytes: int) -> bool:
    """
    Return True only when a cov_hint key is present and exactly decodable in range.

    This is stricter than normalize_cov_hint: over-wide values are safe to clamp for
    measurement, but they are not valid evidence for CGF suppression.
    """
    try:
        items = dict(meta)
    except Exception:
        return False
    if COV_HINT_META_KEY not in items:
        return False
    value = items.get(COV_HINT_META_KEY)
    mask = mask_for_width(int(width_bytes))
    if value is None:
        return False
    try:
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return False
            raw = int(s, 16) if s.lower().startswith("0x") else int(s)
        else:
            raw = int(value)
    except Exception:
        return False
    return 0 <= int(raw) <= int(mask)


def encode_cov_hint(value: Any, width_bytes: int) -> int:
    """Return the fixed-width integer representation used in Packet.meta."""
    return normalize_cov_hint(value, int(width_bytes))


def merge_cov_hints(a: Any, b: Any, width_bytes: int) -> int:
    """
    Merge two decoded hint states without adding the current relayer.

    Width 1 is a saturating counter arm, so merge is max. Bloom-like widths use OR.
    """
    width = int(width_bytes)
    aa = normalize_cov_hint(a, width)
    bb = normalize_cov_hint(b, width)
    if width == 1:
        return max(aa, bb)
    _ = mask_for_width(width)
    return int(aa | bb)


def _stable_relayer_bit(relayer_id: int, width_bytes: int) -> int:
    """Map a relayer id to one deterministic bit in a fixed-width Bloom-like hint."""
    width = int(width_bytes)
    bit_count = 8 * width
    x = int(relayer_id) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return 1 << (int(x) % bit_count)


def add_relayer_to_hint(hint: Any, relayer_id: int, width_bytes: int) -> int:
    """
    Add this relayer's contribution to a hint state.

    Width 1 increments a saturating counter. Widths 2/4 set one deterministic
    Bloom-like bit, preserving idempotent OR-merge semantics.
    """
    width = int(width_bytes)
    h = normalize_cov_hint(hint, width)
    if width == 1:
        return min(255, int(h) + 1)
    mask = mask_for_width(width)
    return int((h | _stable_relayer_bit(int(relayer_id), width)) & mask)


def coverage_estimate(hint: Any, width_bytes: int) -> int:
    """Quantized coverage estimate: counter value for width 1; popcount otherwise."""
    width = int(width_bytes)
    h = normalize_cov_hint(hint, width)
    if width == 1:
        return int(h)
    return int(h.bit_count())


def canonical_records_hash(records: Iterable[Dict[str, Any]]) -> str:
    """Deterministic artifact hash for CGF diagnostic records."""
    payload = json.dumps(
        list(records),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
