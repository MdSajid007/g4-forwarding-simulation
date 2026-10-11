"""
Phase 5 Infrastructure-Disrupted Stress Scenario Configs.

Each function returns a fully specified, validated SimCfg for one of the eight
Phase 5 scenario families. All configs use dataclasses.replace() on a shared
Phase 5 base config — no simulator behavior is changed.

Claim safety:
- These are controlled simulation stress tests, NOT real disaster simulations.
- energy_j is a radiated-power proxy (airtime × dbm_to_mw(tx_power_dbm)).
  A-005 hardware current values are NOT wired in; see docs/claim_boundaries.md.
- All scenario parameters are researcher-defined stress levers unless noted.

Source IDs: see docs/literature_map.md
Assumption IDs: see docs/assumption_ledger.md
Scenario design: see docs/phase5_stress_scenario_plan.md
"""
from __future__ import annotations

import dataclasses
from typing import Dict

from src.config import (
    MacCfg,
    MobilityCfg,
    PhyCfg,
    SimCfg,
    TrafficCfg,
    TrafficFlowCfg,
    validate_cfg,
)


# ---------------------------------------------------------------------------
# Phase 5 base config (all scenario variants derive from this)
# ---------------------------------------------------------------------------

def _make_p5_base() -> SimCfg:
    """
    Phase 5 base: moderate 50x50m indoor-like environment, n_nodes=20,
    single CBR flow (src=0, dst=19), flood policy.

    Sources: BT-CORE-54, BT-MESH-PROFILE-11, PHY-RAPPAPORT-2002, PHY-ITU-P1238,
             MOB-CAMP-2002, VV-LAW-2007, HW-NRF52840 (A-005, not wired)
    Assumptions: A-001, A-002, A-003, A-004, A-005 (future), A-007, A-008, A-009, A-010
    """
    flow = TrafficFlowCfg(
        src=0,
        dst=19,
        model="cbr",
        rate_pps=1.0,
        start_s=2.0,
        stop_s=55.0,
        payload_bytes=80,
    )
    return SimCfg(
        sim_duration_s=60.0,
        n_nodes=20,
        seed_base=0,
        mobility=MobilityCfg(
            model="rwp",
            area_w=50.0,
            area_h=50.0,
            v_min=0.5,
            v_max=2.0,
            pause_s=1.0,
        ),
        traffic=TrafficCfg(flows=(flow,)),
        phy=PhyCfg(
            freq_mhz=2402.0,
            tx_power_dbm=0.0,
            noise_floor_dbm=-95.0,
            pathloss_exp=3.0,
            ref_dist_m=1.0,
            ref_pl_db=40.0,
            sinr0_db=6.0,
            per_k=1.5,
        ),
        mac=MacCfg(
            scan_interval_s=0.21,
            scan_window_s=0.105,   # 50% duty
            adv_interval_s=0.25,
            adv_channels=(37, 38, 39),
            max_retries=2,
            cca_threshold_dbm=-78.0,
            backoff_min_s=0.001,
            backoff_max_s=0.02,
        ),
        policy="flood",
        run_tag="p5_base",
        notes="Phase 5 base: literature-calibrated moderate indoor-like BLE ADV stress scenario.",
    )


# ---------------------------------------------------------------------------
# Scenario 1 — Normal Baseline
# ---------------------------------------------------------------------------

def make_baseline_normal() -> SimCfg:
    """
    Scenario 1: Normal Baseline.
    All parameters at Phase 5 base values. Control condition.
    Sources: BT-CORE-54, BT-MESH-PROFILE-11, BLE-MESH-PERF-001
    Assumptions: A-001, A-002, A-003, A-004, A-009
    """
    cfg = _make_p5_base()
    return dataclasses.replace(
        cfg,
        run_tag="p5_baseline_normal",
        notes=(
            "Phase 5 Scenario 1: Normal baseline. No stress applied. "
            "Infrastructure-disrupted stress scenario reference condition. "
            "BLE ADV-inspired simulation under literature-calibrated parameters."
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Congestion Burst
# ---------------------------------------------------------------------------

def make_congestion_burst() -> SimCfg:
    """
    Scenario 2: Congestion Burst.
    Adds a 6x-rate secondary burst flow (start=25s, stop=45s) to stress channel
    capacity mid-simulation. Burst rate is a controlled stress lever (A-006).
    Sources: BLE-MESH-PERF-001, GOMEZ-2012, BLE-BCAST-STORM-NI-1999
    Assumptions: A-006, A-010
    """
    base = _make_p5_base()
    normal_flow = base.traffic.flows[0]
    burst_flow = TrafficFlowCfg(
        src=2,
        dst=17,
        model="cbr",
        rate_pps=6.0,      # 6x normal rate — controlled stress lever (A-006)
        start_s=25.0,
        stop_s=45.0,
        payload_bytes=80,
    )
    return dataclasses.replace(
        base,
        traffic=TrafficCfg(flows=(normal_flow, burst_flow)),
        run_tag="p5_congestion_burst",
        notes=(
            "Phase 5 Scenario 2: Congestion burst stress scenario. "
            "Secondary flow at 6x rate (A-006 controlled stress lever) mid-simulation. "
            "Not an empirical disaster traffic measurement."
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 3 — Low-Power Survival
# ---------------------------------------------------------------------------

def make_low_power_survival() -> SimCfg:
    """
    Scenario 3: Low-Power Survival.
    Airtime budget enabled with tight constraints. Reduced scan duty (20%).
    ENERGY MODEL CAUTION: energy_j is radiated-power proxy only.
    A-005 hardware current values are NOT wired in (see claim_boundaries.md).
    Sources: HW-NRF52840 (A-005, future reference), BT-CORE-54
    Assumptions: A-005 (future target), A-002, A-007
    """
    base = _make_p5_base()
    return dataclasses.replace(
        base,
        mac=dataclasses.replace(
            base.mac,
            scan_window_s=0.042,           # 20% scan duty (reduced from 50%)
            airtime_budget_enabled=True,
            # Tight budget: each TX burst costs ~2.16ms (80-byte payload, 3 channels).
            # refill_rate=1ms/s < consumption_rate=2.16ms/pps -> budget depletes over time.
            airtime_budget_capacity_s=0.01,    # 10ms capacity per node
            airtime_budget_refill_sps=0.001,   # 1ms/s refill — below consumption rate
            airtime_budget_init_s=0.005,       # starts at half capacity
        ),
        run_tag="p5_low_power_survival",
        notes=(
            "Phase 5 Scenario 3: Low-power survival stress scenario. "
            "Airtime budget gating simulates resource-constrained operation. "
            "energy_j is a radiated-power proxy ONLY (A-005 not yet wired in). "
            "No device energy or operating-duration claim is made. Budget parameters are stress levers."
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 4 — Interference Zone
# ---------------------------------------------------------------------------

def make_interference_zone() -> SimCfg:
    """
    Scenario 4: Interference Zone.
    Elevated noise floor models additive interference (A-011 controlled stress).
    Sources: BLE-MESH-PERF-001 (primary), COEX-IEEE-802-15-2 (conceptual)
    Assumptions: A-011, A-004, A-008
    """
    base = _make_p5_base()
    return dataclasses.replace(
        base,
        phy=dataclasses.replace(
            base.phy,
            noise_floor_dbm=-78.0,   # severe interference zone (A-011 additive noise proxy)
        ),
        run_tag="p5_interference_zone",
        notes=(
            "Phase 5 Scenario 4: Interference zone stress scenario. "
            "Elevated noise floor (A-011 additive noise model) degrades SINR. "
            "Not a full RF coexistence simulator. Literature trend from BLE-MESH-PERF-001. "
            "COEX-IEEE-802-15-2 for conceptual framing only (covers BR/EDR)."
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 5 — Partition and Reconnection
# ---------------------------------------------------------------------------

def make_partition_reconnect() -> SimCfg:
    """
    Scenario 5: Partition and Reconnection.
    Large 200x200m area with limited PHY range (~40m) creates partition-like
    conditions for most seeds. Fast RWP mobility allows transient reconnection.
    Sources: BLE-MESH-PERF-001, EMERG-RFC4838, EMERG-FALL-2003
    Assumptions: A-003, A-004, A-009
    """
    base = _make_p5_base()
    flow = TrafficFlowCfg(
        src=0,
        dst=24,     # adjusted for n_nodes=25
        model="cbr",
        rate_pps=1.0,
        start_s=2.0,
        stop_s=55.0,
        payload_bytes=80,
    )
    return dataclasses.replace(
        base,
        n_nodes=25,
        mobility=dataclasses.replace(
            base.mobility,
            area_w=200.0,
            area_h=200.0,
            v_min=0.5,
            v_max=3.0,      # faster mobility for reconnection opportunity
            pause_s=0.5,
        ),
        traffic=TrafficCfg(flows=(flow,)),
        run_tag="p5_partition_reconnect",
        notes=(
            "Phase 5 Scenario 5: Partition-like stress scenario. "
            "Large area with limited PHY range (~40m) creates statistical partition. "
            "RWP mobility (A-009) provides transient reconnection. "
            "Not a real partition-reconnection protocol. DTN not implemented."
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 6 — Node Failure
# ---------------------------------------------------------------------------

def make_node_failure() -> SimCfg:
    """
    Scenario 6: Node Failure.
    Reduced to 8 nodes (60% failure from 20-node baseline) in same 50x50m area.
    Models post-failure relay sparsity, not dynamic failure transition.
    Sources: BLE-MESH-PERF-001, EMERG-FALL-2003
    Assumptions: A-003, A-009
    """
    base = _make_p5_base()
    flow = TrafficFlowCfg(
        src=0,
        dst=7,      # adjusted for n_nodes=8
        model="cbr",
        rate_pps=1.0,
        start_s=2.0,
        stop_s=55.0,
        payload_bytes=80,
    )
    return dataclasses.replace(
        base,
        n_nodes=8,    # 60% failure from 20-node baseline
        traffic=TrafficCfg(flows=(flow,)),
        run_tag="p5_node_failure",
        notes=(
            "Phase 5 Scenario 6: Node failure stress scenario. "
            "n_nodes=8 models 60% failure from 20-node baseline. "
            "Models post-failure steady state only; no dynamic failure injection. "
            "Survivor topology is random (RWP-based)."
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 7 — Group Mobility
# ---------------------------------------------------------------------------

def make_group_mobility() -> SimCfg:
    """
    Scenario 7: Group Mobility.
    Narrow corridor (200x10m) with fast RWP creates linear chain topology,
    a controlled proxy for correlated group movement.
    Sources: MOB-CAMP-2002 (A-009, RWP basis)
    Assumptions: A-009, A-003, A-004
    """
    base = _make_p5_base()
    return dataclasses.replace(
        base,
        mobility=dataclasses.replace(
            base.mobility,
            area_w=200.0,
            area_h=10.0,    # narrow corridor
            v_min=0.5,
            v_max=3.0,
            pause_s=0.3,
        ),
        run_tag="p5_group_mobility",
        notes=(
            "Phase 5 Scenario 7: Corridor-constrained mobility stress scenario. "
            "Narrow 200x10m area creates linear chain relay patterns as a controlled "
            "proxy for correlated group movement. RWP (A-009) within corridor; "
            "not a full group mobility model."
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 8 — Priority Emergency Traffic
# ---------------------------------------------------------------------------

def make_priority_emergency() -> SimCfg:
    """
    Scenario 8: Priority Emergency Traffic.
    Sustained high-rate secondary 'emergency' flow (5x normal) competes with
    normal flow throughout simulation. Models channel contention stress.
    Sources: BLE-MESH-PERF-001, GOMEZ-2012, EMERG-RFC4838
    Assumptions: A-006, A-010
    """
    base = _make_p5_base()
    normal_flow = base.traffic.flows[0]
    emergency_flow = TrafficFlowCfg(
        src=3,
        dst=16,
        model="cbr",
        rate_pps=5.0,       # 5x normal — controlled stress lever (A-006)
        start_s=2.0,
        stop_s=55.0,
        payload_bytes=80,
    )
    return dataclasses.replace(
        base,
        traffic=TrafficCfg(flows=(normal_flow, emergency_flow)),
        run_tag="p5_priority_emergency",
        notes=(
            "Phase 5 Scenario 8: Priority emergency traffic stress scenario. "
            "High-rate secondary flow (A-006 stress lever) models emergency message "
            "contention. No priority queuing implemented; models channel-level "
            "contention only. Not a real emergency communication protocol."
        ),
    )


# ---------------------------------------------------------------------------
# Registry: all scenarios in canonical order
# ---------------------------------------------------------------------------

def get_all_scenarios() -> Dict[str, SimCfg]:
    """
    Return all Phase 5 scenario configs keyed by config name.
    All configs are validated before returning.
    Order is deterministic (insertion order, Python 3.7+).
    """
    scenarios = {
        "p5_baseline_normal":    make_baseline_normal(),
        "p5_congestion_burst":   make_congestion_burst(),
        "p5_low_power_survival": make_low_power_survival(),
        "p5_interference_zone":  make_interference_zone(),
        "p5_partition_reconnect": make_partition_reconnect(),
        "p5_node_failure":       make_node_failure(),
        "p5_group_mobility":     make_group_mobility(),
        "p5_priority_emergency": make_priority_emergency(),
    }
    for name, cfg in scenarios.items():
        validate_cfg(cfg)   # fail fast if any config is malformed
    return scenarios
