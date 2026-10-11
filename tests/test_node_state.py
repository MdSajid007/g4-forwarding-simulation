"""
Phase 6 tests: NodeStateTracker and SimEngine integration.

Validates:
- NodeStateTracker initialization (deterministic zero state for all nodes)
- Rolling window updates via tracker hooks
- Queue pressure updates in snapshot_all
- CCA / not-scanning / PHY failure counters update correctly
- Airtime budget/proxy updates correctly
- Neighbor density is deterministic
- No future-outcome fields in node-local state
- Repeated same seed/config gives identical node-state summaries
- Existing golden hash tests are unaffected (backward compatibility)
- Per-packet features available at decision time
"""
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.node_state import NodeStateTracker, PACKET_FEATURES, FORBIDDEN_SNAPSHOT_FIELDS
from src.reliability_state import NodeReliabilityState
from src.entities import NodeState, Packet, PacketId
from src.airtime import AirtimeTokenBucket


# ---------------------------------------------------------------------------
# NodeStateTracker initialization tests
# ---------------------------------------------------------------------------

class TestNodeStateTrackerInit:

    def test_init_creates_all_nodes(self):
        tracker = NodeStateTracker(n_nodes=10)
        assert tracker.n_nodes == 10
        for i in range(10):
            assert tracker.state(i) is not None

    def test_init_deterministic(self):
        t1 = NodeStateTracker(n_nodes=5, window_size=50, density_max_age_s=30.0)
        t2 = NodeStateTracker(n_nodes=5, window_size=50, density_max_age_s=30.0)
        nodes = {i: NodeState(node_id=i) for i in range(5)}
        snap1 = t1.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256)
        snap2 = t2.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256)
        assert snap1 == snap2

    def test_init_zero_state_all_nodes(self):
        tracker = NodeStateTracker(n_nodes=3)
        nodes = {i: NodeState(node_id=i) for i in range(3)}
        snap = tracker.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256)
        for nid in range(3):
            assert snap[nid]["queue_length"] == 0
            assert snap[nid]["queue_pressure"] == 0.0
            assert snap[nid]["recent_forwarding_success_rate"] == 0.0
            assert snap[nid]["recent_cca_busy_rate"] == 0.0
            assert snap[nid]["recent_not_scanning_rate"] == 0.0
            assert snap[nid]["energy_proxy_consumed_j"] == 0.0
            assert snap[nid]["local_density_estimate"] == 0

    def test_invalid_n_nodes(self):
        with pytest.raises(ValueError):
            NodeStateTracker(n_nodes=0)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            NodeStateTracker(n_nodes=5, window_size=0)

    def test_invalid_density_max_age(self):
        with pytest.raises(ValueError):
            NodeStateTracker(n_nodes=5, density_max_age_s=0.0)


# ---------------------------------------------------------------------------
# Rolling window update tests via tracker hooks
# ---------------------------------------------------------------------------

class TestNodeStateTrackerCCAUpdates:

    def test_cca_busy_updates_window(self):
        tracker = NodeStateTracker(n_nodes=5)
        # Node 2: 2 CCA busy, 1 clear
        tracker.on_cca_check(node_id=2, busy=True)
        tracker.on_cca_check(node_id=2, busy=True)
        tracker.on_cca_check(node_id=2, busy=False)
        rs = tracker.state(2)
        assert rs.total_cca_busy == 2
        assert abs(rs.cca_busy_window.rate - 2.0 / 3.0) < 1e-9

    def test_cca_updates_only_target_node(self):
        tracker = NodeStateTracker(n_nodes=5)
        tracker.on_cca_check(node_id=0, busy=True)
        for nid in range(1, 5):
            assert tracker.state(nid).total_cca_busy == 0

    def test_tx_burst_outcome_success_rate(self):
        tracker = NodeStateTracker(n_nodes=3)
        # Node 1: 3 successes, 2 failures
        for _ in range(3):
            tracker.on_tx_burst_outcome(1, success=True, had_interference=False)
        for _ in range(2):
            tracker.on_tx_burst_outcome(1, success=False, had_interference=False)
        rs = tracker.state(1)
        assert rs.total_tx_success == 3
        assert abs(rs.tx_success_window.rate - 3.0 / 5.0) < 1e-9

    def test_collision_window_updates(self):
        tracker = NodeStateTracker(n_nodes=3)
        tracker.on_tx_burst_outcome(0, success=True, had_interference=True)
        tracker.on_tx_burst_outcome(0, success=True, had_interference=False)
        rs = tracker.state(0)
        assert abs(rs.collision_window.rate - 0.5) < 1e-9


class TestNodeStateTrackerRXUpdates:

    def test_rx_ok_updates_window_and_neighbors(self):
        tracker = NodeStateTracker(n_nodes=5)
        tracker.on_rx_ok(node_id=1, t_now=5.0, from_node=0, rssi_dbm=-70.0)
        tracker.on_rx_ok(node_id=1, t_now=6.0, from_node=2, rssi_dbm=-75.0)
        rs = tracker.state(1)
        assert rs.total_rx_ok == 2
        assert rs.rx_success_window.rate == 1.0
        assert rs.neighbor_table.density_estimate(t_now=6.0) == 2

    def test_rx_fail_updates_window(self):
        tracker = NodeStateTracker(n_nodes=3)
        tracker.on_rx_ok(node_id=0, t_now=1.0, from_node=1, rssi_dbm=-70.0)
        tracker.on_rx_fail(node_id=0)
        rs = tracker.state(0)
        assert abs(rs.rx_success_window.rate - 0.5) < 1e-9

    def test_not_scanning_updates(self):
        tracker = NodeStateTracker(n_nodes=3)
        tracker.on_not_scanning(0)
        tracker.on_not_scanning(0)
        tracker.on_scanning_rx_opportunity(0)
        rs = tracker.state(0)
        assert rs.total_not_scanning == 2
        assert abs(rs.not_scanning_window.rate - 2.0 / 3.0) < 1e-9

    def test_overlap_rx_counter(self):
        tracker = NodeStateTracker(n_nodes=3)
        tracker.on_overlap_rx(1)
        tracker.on_overlap_rx(1)
        rs = tracker.state(1)
        assert rs.total_overlap_rx == 2


class TestNodeStateTrackerAirtimeAndBudget:

    def test_airtime_accumulation(self):
        tracker = NodeStateTracker(n_nodes=3)
        tracker.on_airtime(node_id=0, dt_s=0.001)
        tracker.on_airtime(node_id=0, dt_s=0.002)
        rs = tracker.state(0)
        assert abs(rs.energy_proxy_consumed_j - 0.003) < 1e-12

    def test_budget_deferral_counter(self):
        tracker = NodeStateTracker(n_nodes=3)
        tracker.on_budget_deferral(0)
        tracker.on_budget_deferral(0)
        rs = tracker.state(0)
        assert rs.total_budget_deferrals == 2

    def test_airtime_budget_in_snapshot(self):
        tracker = NodeStateTracker(n_nodes=3)
        bucket = AirtimeTokenBucket(capacity_s=0.01, refill_rate_sps=0.001, init_tokens_s=0.005)
        buckets = {0: bucket, 1: bucket, 2: bucket}
        nodes = {i: NodeState(node_id=i) for i in range(3)}
        snap = tracker.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256, airtime_buckets=buckets)
        assert snap[0]["airtime_budget_remaining_s"] is not None
        assert snap[0]["airtime_budget_pressure"] > 0.0

    def test_no_budget_gives_none_remaining(self):
        tracker = NodeStateTracker(n_nodes=3)
        nodes = {i: NodeState(node_id=i) for i in range(3)}
        snap = tracker.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256, airtime_buckets=None)
        for nid in range(3):
            assert snap[nid]["airtime_budget_remaining_s"] is None
            assert snap[nid]["airtime_budget_pressure"] == 0.0


# ---------------------------------------------------------------------------
# Queue pressure tests
# ---------------------------------------------------------------------------

class TestQueuePressure:

    def test_queue_pressure_empty(self):
        tracker = NodeStateTracker(n_nodes=3)
        nodes = {i: NodeState(node_id=i) for i in range(3)}
        snap = tracker.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256)
        assert snap[0]["queue_length"] == 0
        assert snap[0]["queue_pressure"] == 0.0

    def test_queue_pressure_with_packets(self):
        tracker = NodeStateTracker(n_nodes=3)
        ns0 = NodeState(node_id=0)
        # Fake a queue of 10 packets
        for i in range(10):
            pkt = Packet(
                pid=PacketId(flow_id=0, seq=i),
                src=0, dst=1,
                created_t=0.0, ttl_hops=3,
                payload_bytes=50,
            )
            ns0.tx_queue.append(pkt)
        nodes = {0: ns0, 1: NodeState(node_id=1), 2: NodeState(node_id=2)}
        snap = tracker.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256)
        assert snap[0]["queue_length"] == 10
        assert abs(snap[0]["queue_pressure"] - 10.0 / 256.0) < 1e-6


# ---------------------------------------------------------------------------
# Neighbor / density tests
# ---------------------------------------------------------------------------

class TestDensityEstimate:

    def test_density_increases_with_rx_ok(self):
        tracker = NodeStateTracker(n_nodes=5, density_max_age_s=30.0)
        # Node 2 hears from 3 different nodes
        tracker.on_rx_ok(2, t_now=1.0, from_node=0, rssi_dbm=-70.0)
        tracker.on_rx_ok(2, t_now=2.0, from_node=1, rssi_dbm=-75.0)
        tracker.on_rx_ok(2, t_now=3.0, from_node=3, rssi_dbm=-80.0)
        rs = tracker.state(2)
        assert rs.neighbor_table.density_estimate(t_now=5.0) == 3

    def test_density_deterministic(self):
        def build(n_nodes, seed_seq):
            t = NodeStateTracker(n_nodes=n_nodes, density_max_age_s=30.0)
            for step in seed_seq:
                t.on_rx_ok(step["rx"], t_now=step["t"], from_node=step["from"], rssi_dbm=step["rssi"])
            return t
        seq = [
            {"rx": 1, "t": 1.0, "from": 0, "rssi": -70.0},
            {"rx": 1, "t": 2.0, "from": 2, "rssi": -75.0},
        ]
        t1 = build(5, seq)
        t2 = build(5, seq)
        nodes = {i: NodeState(node_id=i) for i in range(5)}
        snap1 = t1.snapshot_all(t_now=5.0, nodes=nodes, max_queue=256)
        snap2 = t2.snapshot_all(t_now=5.0, nodes=nodes, max_queue=256)
        assert snap1[1]["local_density_estimate"] == snap2[1]["local_density_estimate"]


# ---------------------------------------------------------------------------
# Leakage safety tests
# ---------------------------------------------------------------------------

class TestLeakageSafety:

    def test_no_forbidden_fields_in_snapshot(self):
        tracker = NodeStateTracker(n_nodes=3)
        nodes = {i: NodeState(node_id=i) for i in range(3)}
        snap = tracker.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256)
        for nid, nsnap in snap.items():
            for forbidden in FORBIDDEN_SNAPSHOT_FIELDS:
                assert forbidden not in nsnap, (
                    f"Forbidden field '{forbidden}' found in node {nid} snapshot. "
                    "This is a feature leakage risk."
                )

    def test_per_packet_features_are_decision_time(self):
        tracker = NodeStateTracker(n_nodes=3)
        pkt = Packet(
            pid=PacketId(flow_id=0, seq=0),
            src=0, dst=1,
            created_t=1.0, ttl_hops=5,
            payload_bytes=50,
            is_control=False,
        )
        features = tracker.packet_features(t_now=3.0, pkt=pkt)
        # TTL at decision time (not final TTL)
        assert features["packet_ttl_remaining"] == 5
        # Age at decision time (not future delivery time)
        assert abs(features["packet_age_s"] - 2.0) < 1e-6
        # Priority from is_control flag
        assert features["packet_priority"] == 0

    def test_per_packet_features_match_expected_keys(self):
        tracker = NodeStateTracker(n_nodes=3)
        pkt = Packet(
            pid=PacketId(flow_id=0, seq=0),
            src=0, dst=1,
            created_t=0.0, ttl_hops=3,
            payload_bytes=50,
        )
        features = tracker.packet_features(t_now=5.0, pkt=pkt)
        for key in PACKET_FEATURES:
            assert key in features, f"Missing per-packet feature key: {key}"

    def test_node_state_snapshot_does_not_contain_packet_fields(self):
        tracker = NodeStateTracker(n_nodes=3)
        nodes = {i: NodeState(node_id=i) for i in range(3)}
        snap = tracker.snapshot_all(t_now=0.0, nodes=nodes, max_queue=256)
        for nsnap in snap.values():
            for key in PACKET_FEATURES:
                assert key not in nsnap, (
                    f"Per-packet feature '{key}' should not be in node-level snapshot. "
                    "It must be extracted per-decision, not stored per-node."
                )


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

class TestEngineIntegration:

    def _run_engine(self, cfg, seed, injs, policy_name):
        from src.airtime import build_airtime_buckets
        from src.hashing import cfg_hash
        from src.mac_engine import SimEngine
        from src.mobility import build_rwp_plan
        from src.policies import policy_factory
        from src.seeds import SeedManager
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=seed)
        plan = build_rwp_plan(cfg, sm)
        pol = policy_factory(policy_name)
        buckets = build_airtime_buckets(cfg)
        eng = SimEngine(cfg, seed=seed, plan=plan, policy=pol, cfg_hash_for_rng=h, airtime_buckets=buckets)
        return eng.run(injs)

    def test_run_returns_node_state_snapshots(self):
        from src.config import make_easy_cfg
        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections
        cfg = make_easy_cfg()
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=1)
        injs = build_injections(cfg, sm)
        result = self._run_engine(cfg, 1, injs, "flood")
        assert "node_state_snapshots" in result
        snaps = result["node_state_snapshots"]
        assert isinstance(snaps, dict)
        assert len(snaps) == cfg.n_nodes

    def test_node_state_snapshots_deterministic(self):
        from src.config import make_easy_cfg
        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections
        cfg = make_easy_cfg()
        h = cfg_hash(cfg)
        sm1 = SeedManager(cfg_hash=h, seed=1)
        sm2 = SeedManager(cfg_hash=h, seed=1)
        injs1 = build_injections(cfg, sm1)
        injs2 = build_injections(cfg, sm2)
        r1 = self._run_engine(cfg, 1, injs1, "flood")
        r2 = self._run_engine(cfg, 1, injs2, "flood")
        snaps1 = r1["node_state_snapshots"]
        snaps2 = r2["node_state_snapshots"]
        # Same seed/config → same node-state snapshots
        assert snaps1 == snaps2, "Node state snapshots not deterministic for same seed/config"

    def test_existing_summary_unchanged(self):
        from src.config import make_easy_cfg
        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections
        cfg = make_easy_cfg()
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=1)
        injs = build_injections(cfg, sm)
        result = self._run_engine(cfg, 1, injs, "flood")
        summary = result["summary"]
        # Core summary fields must still be present and correct
        required_summary_keys = [
            "cfg_hash", "seed", "app_injected", "delivered", "dropped", "pdr",
            "airtime_s", "energy_j", "cca_busy_events", "not_scanning_events",
            "tx_attempts", "tx_success", "rx_total", "rx_ok", "rx_fail",
        ]
        for key in required_summary_keys:
            assert key in summary, f"Summary key '{key}' missing after Phase 6 integration"

    def test_golden_hash_unchanged(self):
        """Phase 2 trace hash must remain identical after Phase 6 integration."""
        import hashlib
        from src.config import make_easy_cfg
        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections

        PHASE2_TRACE_HASH = "784e5436222fcec34bb1a9ab22435a68411fcaec935774a3040d8451d78aabf1"

        cfg = make_easy_cfg()
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=1)
        injs = build_injections(cfg, sm)
        result = self._run_engine(cfg, 1, injs, "flood")
        trace = result["trace_head"]
        rows = [(round(float(t), 12), int(p), str(e), int(src)) for (t, p, e, src) in trace]
        computed = hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()
        assert computed == PHASE2_TRACE_HASH, (
            f"Golden hash regression: Phase 6 integration changed the trace hash!\n"
            f"  Expected: {PHASE2_TRACE_HASH}\n"
            f"  Got:      {computed}\n"
            "Phase 6 passive state updates must NOT affect event ordering or RNG."
        )

    def test_no_forbidden_fields_in_engine_snapshots(self):
        from src.config import make_easy_cfg
        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections
        cfg = make_easy_cfg()
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=1)
        injs = build_injections(cfg, sm)
        result = self._run_engine(cfg, 1, injs, "flood")
        snaps = result["node_state_snapshots"]
        for nid, nsnap in snaps.items():
            for forbidden in FORBIDDEN_SNAPSHOT_FIELDS:
                assert forbidden not in nsnap, (
                    f"Forbidden field '{forbidden}' in node {nid} snapshot from engine run."
                )

    def test_cca_scenario_updates_node_state(self):
        """Congestion scenario should produce non-zero CCA busy rates in some nodes."""
        import sys
        from pathlib import Path
        EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
        if str(EXPERIMENTS_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENTS_DIR))
        try:
            from scenario_configs.phase5_scenarios import make_congestion_burst
        except ImportError:
            pytest.skip("Phase 5 scenario configs not available")

        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections

        cfg = make_congestion_burst()
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=1)
        injs = build_injections(cfg, sm)
        result = self._run_engine(cfg, 1, injs, "flood")
        snaps = result["node_state_snapshots"]

        # At least one node should have seen a CCA busy event in congestion scenario
        any_cca = any(
            snap["total_cca_busy"] > 0
            for snap in snaps.values()
        )
        assert any_cca, (
            "Expected at least one node with CCA busy events in congestion scenario. "
            "Node state CCA tracking may not be working."
        )

    def test_low_power_scenario_updates_budget(self):
        """Low-power scenario should produce non-zero budget deferrals in node state."""
        import sys
        from pathlib import Path
        EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
        if str(EXPERIMENTS_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENTS_DIR))
        try:
            from scenario_configs.phase5_scenarios import make_low_power_survival
        except ImportError:
            pytest.skip("Phase 5 scenario configs not available")

        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections

        cfg = make_low_power_survival()
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=1)
        injs = build_injections(cfg, sm)
        result = self._run_engine(cfg, 1, injs, "flood")
        snaps = result["node_state_snapshots"]

        any_deferrals = any(
            snap["total_budget_deferrals"] > 0
            for snap in snaps.values()
        )
        assert any_deferrals, (
            "Expected budget deferrals in low-power survival scenario. "
            "Node state budget tracking may not be working."
        )

    def test_node_state_energy_proxy_consistent_with_stats(self):
        """Sum of per-node energy proxies should not exceed run-level airtime_s (proxy)."""
        from src.config import make_easy_cfg
        from src.hashing import cfg_hash
        from src.seeds import SeedManager
        from src.traffic import build_injections
        cfg = make_easy_cfg()
        h = cfg_hash(cfg)
        sm = SeedManager(cfg_hash=h, seed=1)
        injs = build_injections(cfg, sm)
        result = self._run_engine(cfg, 1, injs, "flood")
        snaps = result["node_state_snapshots"]
        summary = result["summary"]
        total_node_energy = sum(snap["energy_proxy_consumed_j"] for snap in snaps.values())
        run_airtime = summary["airtime_s"]
        # energy_j = airtime_s * 1.0 mW at tx_power_dbm=0
        # Total per-node energy proxy should equal run-level airtime (within float precision)
        assert abs(total_node_energy - run_airtime) < 1e-6, (
            f"Sum of per-node energy proxies ({total_node_energy:.9f}) does not match "
            f"run-level airtime_s ({run_airtime:.9f}). "
            "Node-level energy tracking may be inconsistent."
        )
