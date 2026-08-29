from decimal import Decimal

from src.metals.book import FixedPointOrderBook


SNAPSHOT = {
    "market_ticker": "KXSILVER15M-TEST",
    "yes_dollars_fp": [["0.4700", "10.25"], ["0.4600", "20.00"]],
    "no_dollars_fp": [["0.5200", "7.50"], ["0.5100", "30.00"]],
}


def test_fixed_point_book_preserves_fractional_values_and_complements():
    book = FixedPointOrderBook("KXSILVER15M-TEST")
    book.apply_snapshot(SNAPSHOT, "snapshot-1")

    assert book.valid
    assert book.yes_bids["0.4700"] == "10.25"
    assert book.best_yes_bid() == Decimal("0.4700")
    assert book.best_yes_ask() == Decimal("0.4800")


def test_delta_changes_quantity_and_zero_removes_level():
    book = FixedPointOrderBook("KXSILVER15M-TEST")
    book.apply_snapshot(SNAPSHOT, "snapshot-1")
    book.apply_delta(
        {"side": "yes", "price_dollars": "0.4700", "delta_fp": "-2.25"}
    )
    assert Decimal(book.yes_bids["0.4700"]) == Decimal("8.00")

    book.apply_delta(
        {"side": "yes", "price_dollars": "0.4700", "delta_fp": "-8.00"}
    )
    assert "0.4700" not in book.yes_bids


def test_invalidated_book_does_not_apply_deltas_until_snapshot():
    book = FixedPointOrderBook("KXSILVER15M-TEST")
    book.apply_snapshot(SNAPSHOT, "snapshot-1")
    original_hash = book.state_hash()
    book.invalidate()
    book.apply_delta(
        {"side": "yes", "price_dollars": "0.4700", "delta_fp": "100.00"}
    )

    assert book.valid is False
    assert book.yes_bids["0.4700"] == "10.25"
    assert book.state_hash() != original_hash  # validity is part of replay state


def test_replay_is_deterministic():
    events = [
        ("snapshot", SNAPSHOT),
        ("delta", {"side": "yes", "price_dollars": "0.4600", "delta_fp": "2.50"}),
        ("delta", {"side": "no", "price_dollars": "0.5100", "delta_fp": "-5.00"}),
    ]
    books = []
    for _ in range(2):
        book = FixedPointOrderBook("KXSILVER15M-TEST")
        for kind, message in events:
            if kind == "snapshot":
                book.apply_snapshot(message, "snapshot-1")
            else:
                book.apply_delta(message)
        books.append(book)

    assert books[0].state_hash() == books[1].state_hash()
