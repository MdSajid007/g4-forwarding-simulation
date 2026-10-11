"""
Hashing utilities and frozen rounding constants.

Extracted from notebook Cell 2 (Config + hashing), Phase 3.
No project-level imports — stdlib only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

# ----------------------------
# Frozen rounding constants (global contract)
# ----------------------------

CFG_FLOAT_NDIGITS: int = 12        # canonical_json / cfg_hash float rounding
MOBILITY_DIGEST_NDIGITS: int = 6   # mobility digest rounding
INJECT_T_NDIGITS: int = 12         # injections_hash time rounding


# ----------------------------
# Utility: canonicalization for stable hashing
# ----------------------------

def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _normalize(obj: Any, float_ndigits: int = CFG_FLOAT_NDIGITS) -> Any:
    """
    Normalize nested structures into JSON-serializable canonical form.

    Supported:
    - dataclasses -> dict
    - dict -> dict[str, normalized]
    - list/tuple -> list[normalized]
    - float -> rounded
    - int/bool/str/None
    - pathlib.Path -> posix string

    Unsupported types raise TypeError (fail fast; prevents silent hash drift).
    """
    if dataclasses.is_dataclass(obj):
        return _normalize(asdict(obj), float_ndigits=float_ndigits)

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _normalize(v, float_ndigits=float_ndigits)
        return out

    if isinstance(obj, (list, tuple)):
        return [_normalize(v, float_ndigits=float_ndigits) for v in obj]

    if _is_finite_number(obj):
        if isinstance(obj, int):
            return int(obj)
        x = float(obj)
        if x == 0.0:
            x = 0.0  # normalize -0.0 to 0.0
        return round(x, float_ndigits)

    if obj is None or isinstance(obj, (bool, str, int)):
        return obj

    if isinstance(obj, Path):
        return obj.as_posix()

    raise TypeError(f"Unsupported type in canonical_json: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Return canonical JSON string for stable hashing."""
    norm = _normalize(obj, float_ndigits=CFG_FLOAT_NDIGITS)
    return json.dumps(norm, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cfg_hash(cfg: Any) -> str:
    """Stable SHA256 hash of config."""
    s = canonical_json(cfg).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
