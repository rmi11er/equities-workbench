from src.metals.sequence import SequenceTracker
from src.metals.types import SequenceStatus


def test_sequence_tracker_classifies_first_ok_duplicate_and_gap():
    tracker = SequenceTracker()

    assert tracker.observe("connection:sid", 10).status == SequenceStatus.FIRST
    assert tracker.observe("connection:sid", 11).status == SequenceStatus.OK
    duplicate = tracker.observe("connection:sid", 11)
    assert duplicate.status == SequenceStatus.DUPLICATE
    assert duplicate.expected == 12
    gap = tracker.observe("connection:sid", 14)
    assert gap.status == SequenceStatus.GAP
    assert gap.expected == 12
    assert tracker.observe("connection:sid", 15).status == SequenceStatus.OK


def test_sequence_state_is_isolated_by_subscription():
    tracker = SequenceTracker()

    assert tracker.observe("connection:1", 3).status == SequenceStatus.FIRST
    assert tracker.observe("connection:2", 50).status == SequenceStatus.FIRST
    assert tracker.observe("connection:1", 4).status == SequenceStatus.OK
    assert tracker.observe("connection:2", 51).status == SequenceStatus.OK
