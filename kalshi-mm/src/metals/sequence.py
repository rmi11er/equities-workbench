"""Strict per-subscription sequence tracking."""

from __future__ import annotations

from typing import Dict, Optional

from .types import SequenceDecision, SequenceStatus


class SequenceTracker:
    """Classify duplicates and gaps without accidentally dispatching either."""

    def __init__(self) -> None:
        self._last: Dict[str, int] = {}

    def observe(self, key: str, sequence: Optional[int]) -> SequenceDecision:
        if sequence is None:
            return SequenceDecision(SequenceStatus.OK, None, None)

        previous = self._last.get(key)
        if previous is None:
            self._last[key] = sequence
            return SequenceDecision(SequenceStatus.FIRST, None, sequence)

        expected = previous + 1
        if sequence <= previous:
            return SequenceDecision(SequenceStatus.DUPLICATE, expected, sequence)

        self._last[key] = sequence
        if sequence != expected:
            return SequenceDecision(SequenceStatus.GAP, expected, sequence)
        return SequenceDecision(SequenceStatus.OK, expected, sequence)

    def reset(self, key: Optional[str] = None) -> None:
        if key is None:
            self._last.clear()
        else:
            self._last.pop(key, None)
