"""
Phase 3 test: scheduler ordering.
Maps to Phase 2 requirement: Scheduler ordering (self_test_cell3, 5 assertions).
"""
import pytest

from src.scheduler import EventPrio, EventScheduler


def test_seq_tiebreak():
    sch = EventScheduler()
    sch.schedule(1.0, "A", prio=10)
    sch.schedule(1.0, "B", prio=10)
    sch.schedule(1.0, "C", prio=10)
    assert sch.drain_types() == ["A", "B", "C"], "Seq tie-break failed"


def test_priority_tiebreak():
    sch = EventScheduler()
    sch.schedule(1.0, "APP", prio=int(EventPrio.APP))
    sch.schedule(1.0, "MAC", prio=int(EventPrio.MAC))
    sch.schedule(1.0, "SYS", prio=int(EventPrio.SYSTEM))
    assert sch.drain_types() == ["SYS", "MAC", "APP"], "Priority tie-break failed"


def test_time_ordering():
    sch = EventScheduler()
    sch.schedule(2.0, "T2", prio=10)
    sch.schedule(1.0, "T1", prio=10)
    assert sch.drain_types() == ["T1", "T2"], "Time ordering failed"


def test_cancel_skips_event():
    sch = EventScheduler()
    e1 = sch.schedule(1.0, "X", assign_eid=True)
    sch.schedule(1.0, "Y", assign_eid=True)
    sch.cancel(e1.eid)
    assert sch.pop().etype == "Y", "Cancel did not skip event"
    with pytest.raises(IndexError):
        sch.pop()


def test_schedule_in_past_raises():
    sch = EventScheduler()
    sch.schedule(1.0, "OK")
    _ = sch.pop()
    with pytest.raises(ValueError):
        sch.schedule(0.5, "BAD")
