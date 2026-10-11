"""
EventScheduler: discrete-event simulation kernel with stable tie-breaking.

Extracted from notebook Cell 3, Phase 3.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


class EventPrio(IntEnum):
    """
    Lower value = higher priority (pops earlier at same time).
    """
    SYSTEM = 0
    MAC = 10
    PHY = 20
    APP = 30
    LOG = 40


@dataclass(frozen=True)
class Event:
    t: float
    prio: int
    seq: int
    etype: str
    data: Dict[str, Any]
    stream: str = "core"
    eid: Optional[int] = None


class EventScheduler:
    """
    Deterministic scheduler using heap.

    Ordering key: (t, prio, seq)
    seq is assigned monotonically on schedule() calls to break ties deterministically.
    """

    __slots__ = ("_heap", "_seq", "_now", "_canceled", "_next_eid")

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, int, Event]] = []
        self._seq: int = 0
        self._now: float = 0.0
        self._canceled: Dict[int, bool] = {}
        self._next_eid: int = 1

    @property
    def now(self) -> float:
        return self._now

    def __len__(self) -> int:
        return len(self._heap)

    def schedule(
        self,
        t: float,
        etype: str,
        data: Optional[Dict[str, Any]] = None,
        prio: int = int(EventPrio.SYSTEM),
        stream: str = "core",
        assign_eid: bool = False,
    ) -> Event:
        if t < self._now:
            raise ValueError(f"Cannot schedule in the past: t={t} < now={self._now}")
        if not isinstance(etype, str) or not etype:
            raise ValueError("etype must be non-empty string")
        if data is None:
            data = {}

        seq = self._seq
        self._seq += 1

        eid = None
        if assign_eid:
            eid = self._next_eid
            self._next_eid += 1

        ev = Event(
            t=float(t),
            prio=int(prio),
            seq=seq,
            etype=etype,
            data=dict(data),
            stream=stream,
            eid=eid,
        )
        heapq.heappush(self._heap, (ev.t, ev.prio, ev.seq, ev))
        return ev

    def cancel(self, eid: int) -> None:
        """Mark an event id canceled. No-op if unknown."""
        if not isinstance(eid, int) or eid <= 0:
            raise ValueError("eid must be positive int")
        self._canceled[eid] = True

    def pop(self) -> Event:
        """Pop next non-canceled event, advancing time deterministically."""
        while self._heap:
            _, _, _, ev = heapq.heappop(self._heap)
            if ev.eid is not None and self._canceled.get(ev.eid, False):
                continue
            self._now = ev.t
            return ev
        raise IndexError("pop from empty scheduler")

    def peek(self) -> Event:
        """Peek next non-canceled event without popping (may scan)."""
        best_idx = None
        best_key = None
        for i, (t, pr, sq, ev) in enumerate(self._heap):
            if ev.eid is not None and self._canceled.get(ev.eid, False):
                continue
            key = (t, pr, sq)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        if best_idx is None:
            raise IndexError("peek from empty scheduler")
        return self._heap[best_idx][3]

    def drain_types(self, max_events: int = 100000) -> List[str]:
        """Utility to drain and return event types (for deterministic testing)."""
        out: List[str] = []
        n = 0
        while self._heap and n < max_events:
            ev = self.pop()
            out.append(ev.etype)
            n += 1
        return out

    def snapshot(self, max_items: int = 10) -> Dict[str, Any]:
        """Return a small scheduler state snapshot for debugging."""
        items = sorted([(t, pr, sq, ev.etype) for (t, pr, sq, ev) in self._heap])[:max_items]
        return {"now": self._now, "pending": len(self._heap), "head": items}
