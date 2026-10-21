from concurrent.futures import ThreadPoolExecutor

import pytest

from src.metals.ingestion import (
    GlobalReceiveSequencer,
    IngestionQueueClosed,
    IngestionQueueOverflow,
    UnifiedIngestionQueue,
)


def test_global_receive_sequence_is_unique_across_concurrent_sources():
    sequencer = GlobalReceiveSequencer()

    def stamp(index):
        return sequencer.stamp(
            "pyth" if index % 2 else "cme",
            "price",
        ).ingestion_sequence

    with ThreadPoolExecutor(max_workers=8) as executor:
        observed = list(executor.map(stamp, range(200)))

    assert sorted(observed) == list(range(1, 201))
    assert sequencer.last_sequence == 200


def test_supplied_clock_regression_is_retained_and_flagged():
    sequencer = GlobalReceiveSequencer()
    first = sequencer.stamp(
        "pyth",
        "price",
        receive_ts_ns=20,
        local_monotonic_ts_ns=10,
    )
    second = sequencer.stamp(
        "cme",
        "mbo",
        receive_ts_ns=21,
        local_monotonic_ts_ns=9,
    )

    assert first.monotonic_regression is False
    assert second.ingestion_sequence == 2
    assert second.local_monotonic_ts_ns == 9
    assert second.previous_monotonic_ts_ns == 10
    assert second.monotonic_regression is True


def test_unified_queue_stamps_at_publish_and_preserves_payload():
    ingestion = UnifiedIngestionQueue(maxsize=2)

    stamp = ingestion.publish(
        "pyth",
        "price",
        "silver",
        {"price": "32.10"},
        receive_ts_ns=100,
        local_monotonic_ts_ns=50,
    )
    item = ingestion.get(timeout=0.1)

    assert item.stamp == stamp
    assert item.metal == "silver"
    assert item.payload == {"price": "32.10"}
    ingestion.task_done()
    ingestion.join()


def test_unified_queue_fails_closed_on_overflow_and_after_close():
    ingestion = UnifiedIngestionQueue(maxsize=1)
    ingestion.publish("cme", "mbo", "silver", {"order_id": 1})

    with pytest.raises(IngestionQueueOverflow, match="sequence 2"):
        ingestion.publish("pyth", "price", "silver", {"price": "32"})

    ingestion.close()
    with pytest.raises(IngestionQueueClosed):
        ingestion.publish("pyth", "price", "silver", {"price": "33"})


@pytest.mark.parametrize("field", ["source", "stream", "metal"])
def test_unified_queue_rejects_empty_partition_identity(field):
    values = {"source": "pyth", "stream": "price", "metal": "silver"}
    values[field] = ""
    ingestion = UnifiedIngestionQueue()

    with pytest.raises(ValueError):
        ingestion.publish(
            values["source"],
            values["stream"],
            values["metal"],
            {},
        )
