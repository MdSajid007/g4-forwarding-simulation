"""
Phase 6 tests: RollingBinaryWindow, NeighborTable, NodeReliabilityState.

Validates:
- Rolling window rate computation, eviction, reset
- Neighbor table density, stability, RSSI tracking, age cutoff
- NodeReliabilityState initialization (deterministic zero state)
- TX-side and RX-side update methods
- Snapshot key completeness and no-leakage guarantee
- Energy proxy accumulation
- Airtime budget snapshot integration
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliability_state import (
    RollingBinaryWindow,
    NeighborTable,
    NodeReliabilityState,
)


# ---------------------------------------------------------------------------
# RollingBinaryWindow tests
# ---------------------------------------------------------------------------

class TestRollingBinaryWindow:

    def test_empty_rate_is_zero(self):
        w = RollingBinaryWindow(maxlen=10)
        assert w.rate == 0.0
        assert w.count == 0
        assert w.successes == 0

    def test_single_success(self):
        w = RollingBinaryWindow(maxlen=10)
        w.push(1)
        assert w.count == 1
        assert w.successes == 1
        assert w.rate == 1.0

    def test_single_failure(self):
        w = RollingBinaryWindow(maxlen=10)
        w.push(0)
        assert w.count == 1
        assert w.successes == 0
        assert w.rate == 0.0

    def test_mixed_rate(self):
        w = RollingBinaryWindow(maxlen=10)
        for _ in range(3):
            w.push(1)
        for _ in range(7):
            w.push(0)
        assert w.count == 10
        assert w.successes == 3
        assert abs(w.rate - 0.3) < 1e-9

    def test_maxlen_evicts_oldest(self):
        w = RollingBinaryWindow(maxlen=3)
        w.push(1)
        w.push(1)
        w.push(1)
        assert w.rate == 1.0
        # Push a 0: oldest 1 is evicted, now [1, 1, 0]
        w.push(0)
        assert w.count == 3
        assert w.successes == 2
        assert abs(w.rate - 2.0 / 3.0) < 1e-9

    def test_maxlen_enforced(self):
        w = RollingBinaryWindow(maxlen=5)
        for i in range(20):
            w.push(i % 2)
        assert w.count == 5

    def test_rate_after_all_zero(self):
        w = RollingBinaryWindow(maxlen=5)
        for _ in range(5):
            w.push(0)
        assert w.rate == 0.0
        assert w.successes == 0

    def test_reset_clears(self):
        w = RollingBinaryWindow(maxlen=10)
        for _ in range(5):
            w.push(1)
        w.reset()
        assert w.count == 0
        assert w.rate == 0.0

    def test_invalid_maxlen(self):
        with pytest.raises(ValueError):
            RollingBinaryWindow(maxlen=0)

    def test_determinism_same_sequence(self):
        w1 = RollingBinaryWindow(maxlen=10)
        w2 = RollingBinaryWindow(maxlen=10)
        seq = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
        for v in seq:
            w1.push(v)
            w2.push(v)
        assert w1.rate == w2.rate
        assert w1.successes == w2.successes

    def test_snapshot_keys(self):
        w = RollingBinaryWindow(maxlen=10)
        w.push(1)
        snap = w.snapshot()
        assert "maxlen" in snap
        assert "count" in snap
        assert "successes" in snap
        assert "rate" in snap


# ---------------------------------------------------------------------------
# NeighborTable tests
# ---------------------------------------------------------------------------

class TestNeighborTable:

    def test_empty_density(self):
        t = NeighborTable(max_age_s=30.0)
        assert t.density_estimate(t_now=0.0) == 0

    def test_observe_and_density(self):
        t = NeighborTable(max_age_s=30.0)
        t.observe(neighbor_id=1, t_now=10.0, rssi_dbm=-70.0)
        t.observe(neighbor_id=2, t_now=10.0, rssi_dbm=-80.0)
        assert t.density_estimate(t_now=10.0) == 2

    def test_age_cutoff_excludes_old_neighbors(self):
        t = NeighborTable(max_age_s=30.0)
        t.observe(neighbor_id=1, t_now=0.0, rssi_dbm=-70.0)
        t.observe(neighbor_id=2, t_now=0.0, rssi_dbm=-75.0)
        # At t=31, neighbors last seen at t=0 are older than 30s
        assert t.density_estimate(t_now=31.0) == 0

    def test_age_cutoff_boundary(self):
        t = NeighborTable(max_age_s=30.0)
        t.observe(neighbor_id=1, t_now=0.0, rssi_dbm=-70.0)
        # At t=30, neighbor is exactly at boundary (30.0 >= 0.0 + 0.0 cutoff check)
        # cutoff = 30.0 - 30.0 = 0.0; t_last_seen = 0.0 >= 0.0 → included
        assert t.density_estimate(t_now=30.0) == 1
        # At t=30.001, cutoff = 0.001; 0.0 < 0.001 → excluded
        assert t.density_estimate(t_now=30.001) == 0

    def test_update_existing_neighbor(self):
        t = NeighborTable(max_age_s=30.0)
        t.observe(neighbor_id=1, t_now=0.0, rssi_dbm=-70.0)
        t.observe(neighbor_id=1, t_now=20.0, rssi_dbm=-65.0)  # update
        assert t.density_estimate(t_now=0.0) == 1  # still 1 neighbor
        # Check RSSI was updated
        rssi = t.rssi_by_neighbor(t_now=20.0)
        assert 1 in rssi
        assert abs(rssi[1] - (-65.0)) < 0.01

    def test_rssi_by_neighbor(self):
        t = NeighborTable(max_age_s=30.0)
        t.observe(1, 0.0, -70.0)
        t.observe(2, 0.0, -80.0)
        rssi = t.rssi_by_neighbor(t_now=0.0)
        assert rssi[1] == pytest.approx(-70.0, abs=0.01)
        assert rssi[2] == pytest.approx(-80.0, abs=0.01)

    def test_rssi_excludes_aged_out(self):
        t = NeighborTable(max_age_s=30.0)
        t.observe(1, 0.0, -70.0)
        t.observe(2, 20.0, -80.0)
        rssi = t.rssi_by_neighbor(t_now=31.0)
        # neighbor 1 aged out; neighbor 2 still active
        assert 1 not in rssi
        assert 2 in rssi

    def test_stability_empty(self):
        t = NeighborTable(max_age_s=30.0)
        assert t.stability_estimate(t_now=30.0) == 0.0

    def test_stability_only_recent(self):
        t = NeighborTable(max_age_s=30.0)
        # Observe neighbor only recently (last 15s)
        t.observe(1, t_now=20.0, rssi_dbm=-70.0)
        # At t=30: recent half = [15, 30], older half = [0, 15)
        # Neighbor 1 is at t=20 → in recent half; NOT in older half
        stab = t.stability_estimate(t_now=30.0)
        assert stab == 0.0  # recent set has neighbor 1, but older set does not

    def test_stability_stable_neighbor(self):
        t = NeighborTable(max_age_s=30.0)
        # Neighbor 1: first seen at t=5 (before recent_cutoff=15), last seen at t=20 (recent)
        t.observe(1, t_now=5.0, rssi_dbm=-70.0)
        t.observe(2, t_now=10.0, rssi_dbm=-75.0)  # older half only; last_seen=10 < recent_cutoff=15
        t.observe(1, t_now=20.0, rssi_dbm=-70.0)  # update; t_first_seen=5 preserved
        # At t=30: max_age=30, half=15, recent_cutoff=15
        # Neighbor 1: last_seen=20 >= 15 (recent), first_seen=5 <= 15 (stable)
        # Neighbor 2: last_seen=10 < 15 (NOT recent → not counted)
        # recent_set = {1}, stable_set = {1} → stability = 1.0
        stab = t.stability_estimate(t_now=30.0)
        assert stab == pytest.approx(1.0)

    def test_stability_fully_stable(self):
        t = NeighborTable(max_age_s=30.0)
        # Neighbor 1: first observed at t=5, updated at t=20; t_first_seen=5 is preserved
        t.observe(1, t_now=5.0, rssi_dbm=-70.0)
        t.observe(1, t_now=20.0, rssi_dbm=-70.0)
        # At t=30: half=15, recent_cutoff=15
        # last_seen=20 >= 15 (in recent half) AND first_seen=5 <= 15 (stable)
        # stability = 1/1 = 1.0
        stab = t.stability_estimate(t_now=30.0)
        assert stab == pytest.approx(1.0)

    def test_stability_non_degenerate_mixed(self):
        """A stable neighbor (known early) and a new neighbor produce fractional stability."""
        t = NeighborTable(max_age_s=30.0)
        # Neighbor 1: first seen at t=0 (well before midpoint), still active at t=20
        t.observe(1, t_now=0.0, rssi_dbm=-70.0)
        t.observe(1, t_now=20.0, rssi_dbm=-70.0)
        # Neighbor 2: first seen at t=20 (within the recent half), not stable yet
        t.observe(2, t_now=20.0, rssi_dbm=-75.0)
        # At t=30: half=15, recent_cutoff=15
        # Neighbor 1: last=20 >= 15, first=0 <= 15 → stable
        # Neighbor 2: last=20 >= 15, first=20 <= 15? NO (20 > 15) → not stable
        # recent_set={1,2}, stable_set={1} → stability = 0.5
        stab = t.stability_estimate(t_now=30.0)
        assert stab == pytest.approx(0.5)

    def test_stability_first_seen_preserved_on_update(self):
        """t_first_seen must be preserved when observe() updates an existing neighbor."""
        t = NeighborTable(max_age_s=30.0)
        t.observe(1, t_now=0.0, rssi_dbm=-70.0)   # first_seen=0
        t.observe(1, t_now=10.0, rssi_dbm=-65.0)  # update; first_seen should still be 0
        t.observe(1, t_now=20.0, rssi_dbm=-60.0)  # another update
        obs = t._table[1]
        assert obs.t_first_seen == pytest.approx(0.0), "t_first_seen must be preserved on updates"
        assert obs.t_last_seen == pytest.approx(20.0)

    def test_invalid_max_age(self):
        with pytest.raises(ValueError):
            NeighborTable(max_age_s=0.0)

    def test_snapshot_keys(self):
        t = NeighborTable(max_age_s=30.0)
        t.observe(1, 0.0, -70.0)
        snap = t.snapshot(t_now=0.0)
        assert "density_estimate" in snap
        assert "stability_estimate" in snap
        assert "rssi_by_neighbor" in snap


# ---------------------------------------------------------------------------
# NodeReliabilityState tests
# ---------------------------------------------------------------------------

class TestNodeReliabilityState:

    def test_init_zero_state(self):
        rs = NodeReliabilityState(node_id=0)
        assert rs.node_id == 0
        assert rs.total_tx_attempts == 0
        assert rs.total_cca_busy == 0
        assert rs.total_rx_ok == 0
        assert rs.total_rx_fail == 0
        assert rs.total_not_scanning == 0
        assert rs.energy_proxy_consumed_j == 0.0
        assert rs.cca_busy_window.rate == 0.0
        assert rs.tx_success_window.rate == 0.0
        assert rs.rx_success_window.rate == 0.0

    def test_deterministic_init(self):
        rs1 = NodeReliabilityState(node_id=5, window_size=50)
        rs2 = NodeReliabilityState(node_id=5, window_size=50)
        snap1 = rs1.snapshot(t_now=0.0, queue_len=0, max_queue=256)
        snap2 = rs2.snapshot(t_now=0.0, queue_len=0, max_queue=256)
        assert snap1 == snap2

    def test_cca_check_busy(self):
        rs = NodeReliabilityState(node_id=0)
        rs.observe_cca_check(busy=True)
        rs.observe_cca_check(busy=True)
        rs.observe_cca_check(busy=False)
        assert rs.total_cca_busy == 2
        assert rs.cca_busy_window.count == 3
        assert abs(rs.cca_busy_window.rate - 2.0 / 3.0) < 1e-9

    def test_tx_burst_outcome_success(self):
        rs = NodeReliabilityState(node_id=0)
        # 3 successes, 1 fail
        rs.observe_tx_burst_outcome(success=True, had_interference=False)
        rs.observe_tx_burst_outcome(success=True, had_interference=False)
        rs.observe_tx_burst_outcome(success=True, had_interference=True)
        rs.observe_tx_burst_outcome(success=False, had_interference=False)
        assert rs.total_tx_success == 3
        assert abs(rs.tx_success_window.rate - 3.0 / 4.0) < 1e-9
        # collision: 1 out of 4 bursts had interference
        assert abs(rs.collision_window.rate - 1.0 / 4.0) < 1e-9

    def test_tx_attempt_counter(self):
        rs = NodeReliabilityState(node_id=0)
        rs.observe_tx_attempt()
        rs.observe_tx_attempt()
        assert rs.total_tx_attempts == 2

    def test_airtime_accumulation(self):
        rs = NodeReliabilityState(node_id=0)
        rs.observe_airtime(0.001)
        rs.observe_airtime(0.002)
        assert abs(rs.energy_proxy_consumed_j - 0.003) < 1e-12

    def test_rx_ok_updates(self):
        rs = NodeReliabilityState(node_id=0)
        rs.observe_rx_ok(t_now=5.0, from_node=2, rssi_dbm=-70.0)
        rs.observe_rx_ok(t_now=6.0, from_node=3, rssi_dbm=-75.0)
        assert rs.total_rx_ok == 2
        assert rs.rx_success_window.rate == 1.0
        # Neighbor 2 and 3 observed
        assert rs.neighbor_table.density_estimate(t_now=6.0) == 2

    def test_rx_fail_updates(self):
        rs = NodeReliabilityState(node_id=0)
        rs.observe_rx_ok(t_now=1.0, from_node=1, rssi_dbm=-70.0)
        rs.observe_rx_fail()
        assert rs.total_rx_fail == 1
        assert abs(rs.rx_success_window.rate - 0.5) < 1e-9

    def test_not_scanning_updates(self):
        rs = NodeReliabilityState(node_id=0)
        rs.observe_not_scanning()
        rs.observe_not_scanning()
        rs.observe_scanning_rx_opportunity()
        assert rs.total_not_scanning == 2
        assert rs.not_scanning_window.count == 3
        assert abs(rs.not_scanning_window.rate - 2.0 / 3.0) < 1e-9

    def test_budget_deferral_counter(self):
        rs = NodeReliabilityState(node_id=0)
        rs.observe_budget_deferral()
        rs.observe_budget_deferral()
        assert rs.total_budget_deferrals == 2

    def test_snapshot_has_required_keys(self):
        rs = NodeReliabilityState(node_id=7)
        snap = rs.snapshot(t_now=10.0, queue_len=3, max_queue=256)
        required_keys = {
            "node_id",
            "queue_length",
            "queue_pressure",
            "recent_forwarding_success_rate",
            "recent_cca_busy_rate",
            "recent_collision_or_overlap_rate",
            "recent_receive_success_rate",
            "recent_not_scanning_rate",
            "airtime_budget_remaining_s",
            "airtime_budget_pressure",
            "energy_proxy_consumed_j",
            "local_density_estimate",
            "neighbor_stability_estimate",
            "link_rssi_by_neighbor",
            "total_tx_attempts",
            "total_tx_success",
            "total_cca_busy",
            "total_rx_ok",
            "total_rx_fail",
            "total_not_scanning",
            "total_budget_deferrals",
        }
        for key in required_keys:
            assert key in snap, f"Missing required key: {key}"

    def test_snapshot_no_future_outcome_fields(self):
        rs = NodeReliabilityState(node_id=0)
        snap = rs.snapshot(t_now=0.0, queue_len=0, max_queue=256)
        for forbidden in NodeReliabilityState.FORBIDDEN_FIELDS:
            assert forbidden not in snap, (
                f"Forbidden (future-outcome) field '{forbidden}' found in snapshot. "
                "This would be a leakage risk."
            )

    def test_queue_pressure_calculation(self):
        rs = NodeReliabilityState(node_id=0)
        snap = rs.snapshot(t_now=0.0, queue_len=64, max_queue=256)
        assert snap["queue_length"] == 64
        assert abs(snap["queue_pressure"] - 64.0 / 256.0) < 1e-6

    def test_queue_pressure_clamped_to_one(self):
        rs = NodeReliabilityState(node_id=0)
        # Queue longer than max (shouldn't happen, but defensive)
        snap = rs.snapshot(t_now=0.0, queue_len=300, max_queue=256)
        assert snap["queue_pressure"] <= 1.0

    def test_airtime_budget_snapshot_no_bucket(self):
        rs = NodeReliabilityState(node_id=0)
        snap = rs.snapshot(t_now=0.0, queue_len=0, max_queue=256, airtime_bucket_snapshot=None)
        assert snap["airtime_budget_remaining_s"] is None
        assert snap["airtime_budget_pressure"] == 0.0

    def test_airtime_budget_snapshot_with_bucket(self):
        rs = NodeReliabilityState(node_id=0)
        bucket_snap = {"tokens_s": 5.0, "capacity_s": 10.0, "refill_rate_sps": 1.0}
        snap = rs.snapshot(t_now=0.0, queue_len=0, max_queue=256, airtime_bucket_snapshot=bucket_snap)
        assert snap["airtime_budget_remaining_s"] == pytest.approx(5.0, abs=1e-6)
        assert snap["airtime_budget_pressure"] == pytest.approx(0.5, abs=1e-6)

    def test_determinism_same_event_sequence(self):
        def run_sequence(node_id):
            rs = NodeReliabilityState(node_id=node_id, window_size=50)
            rs.observe_cca_check(busy=False)
            rs.observe_tx_attempt()
            rs.observe_tx_burst_outcome(success=True, had_interference=False)
            rs.observe_rx_ok(t_now=1.0, from_node=2, rssi_dbm=-70.0)
            rs.observe_not_scanning()
            rs.observe_airtime(0.001)
            return rs.snapshot(t_now=5.0, queue_len=2, max_queue=256)

        snap1 = run_sequence(node_id=0)
        snap2 = run_sequence(node_id=0)
        assert snap1 == snap2

    def test_node_id_in_snapshot(self):
        rs = NodeReliabilityState(node_id=42)
        snap = rs.snapshot(t_now=0.0, queue_len=0, max_queue=256)
        assert snap["node_id"] == 42
