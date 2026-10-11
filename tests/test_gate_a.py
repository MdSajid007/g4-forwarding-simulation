"""
Gate A regression test — reconstructed from src/ modules (no notebook imports).

The Gate A harness (Cell 6 in the notebook) runs a scheduler + RunStats snapshot
with no MAC/PHY/mobility, using a deterministic sequence of 32 NOOP events driven
by SeedManager("scheduler" stream).  It is a purity check: if any of SeedManager,
EventScheduler, RunStats.summary(), canonical_json, or cfg_hash change in a
non-backward-compatible way, this hash breaks.

Phase 2 golden value (GATEA_LOCK_VERSION=6, easy_cfg, seed=1):
  c9077676352c5eda984ba9e3dd8a666d7d3738b01eacf9e437ab36b3be085070
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from src.config import make_easy_cfg, validate_cfg
from src.entities import RunStats
from src.hashing import CFG_FLOAT_NDIGITS, canonical_json, cfg_hash
from src.scheduler import EventPrio, EventScheduler
from src.seeds import SeedManager

PHASE2_GATE_A_SUMMARY_HASH = (
    "c9077676352c5eda984ba9e3dd8a666d7d3738b01eacf9e437ab36b3be085070"
)
GATEA_NOOP_EVENTS: int = 32


def _run_gate_a_harness(cfg: Any, seed: int) -> str:
    """
    Reconstruct the Cell 6 Gate A harness from src/ components.
    Returns summary_hash (SHA256 hex of canonical_json(summary)).
    """
    validate_cfg(cfg)
    h = cfg_hash(cfg)

    sm = SeedManager(cfg_hash=h, seed=int(seed))
    sch = EventScheduler()

    sch.schedule(0.0, "BOOT", prio=int(EventPrio.SYSTEM), stream="core")
    sch.schedule(0.0, "INIT", prio=int(EventPrio.SYSTEM), stream="core")

    rng = sm.rng("scheduler")

    ts = rng.uniform(0.0, float(cfg.sim_duration_s), size=GATEA_NOOP_EVENTS)
    prios = rng.integers(0, 3, size=GATEA_NOOP_EVENTS)
    prio_map = [int(EventPrio.SYSTEM), int(EventPrio.MAC), int(EventPrio.APP)]

    for i in range(GATEA_NOOP_EVENTS):
        sch.schedule(
            float(ts[i]),
            "NOOP",
            prio=prio_map[int(prios[i])],
            data={"i": int(i)},
            stream="scheduler",
        )

    sch.schedule(
        float(cfg.sim_duration_s),
        "FINALIZE",
        prio=int(EventPrio.SYSTEM),
        stream="core",
    )

    trace: List[Tuple[float, int, str, str]] = []
    while True:
        try:
            ev = sch.pop()
        except IndexError:
            break
        trace.append(
            (
                round(float(ev.t), CFG_FLOAT_NDIGITS),
                int(ev.prio),
                str(ev.etype),
                str(ev.stream),
            )
        )

    rs = RunStats(
        cfg_hash=h,
        seed=int(seed),
        sim_duration_s=float(cfg.sim_duration_s),
        n_nodes=int(cfg.n_nodes),
    )
    summary: Dict[str, Any] = rs.summary()
    summary["cells_covered"] = "0-4.5"
    summary["n_noop_events"] = int(GATEA_NOOP_EVENTS)
    summary["trace_len"] = len(trace)
    summary["trace_head"] = trace[:10]

    return hashlib.sha256(canonical_json(summary).encode("utf-8")).hexdigest()


def test_gate_a_summary_hash_regression():
    """Phase 2 regression: Gate A summary hash for easy_cfg seed=1 must match locked value."""
    cfg = make_easy_cfg()
    computed = _run_gate_a_harness(cfg, seed=1)
    assert computed == PHASE2_GATE_A_SUMMARY_HASH, (
        f"Gate A summary hash regression failed.\n"
        f"  Expected: {PHASE2_GATE_A_SUMMARY_HASH}\n"
        f"  Got:      {computed}"
    )


def test_gate_a_deterministic():
    """Two identical Gate A runs must produce the same summary hash."""
    cfg = make_easy_cfg()
    h1 = _run_gate_a_harness(cfg, seed=1)
    h2 = _run_gate_a_harness(cfg, seed=1)
    assert h1 == h2, "Gate A: summary hash not deterministic across identical runs"


def test_gate_a_seed_sensitivity():
    """Different seeds must produce different Gate A summary hashes."""
    cfg = make_easy_cfg()
    h1 = _run_gate_a_harness(cfg, seed=1)
    h2 = _run_gate_a_harness(cfg, seed=2)
    assert h1 != h2, "Gate A: different seeds unexpectedly produced identical summary hash"
