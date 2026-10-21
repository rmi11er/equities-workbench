from pathlib import Path

import pytest

from src.metals.si_fast_feed_counterfactual import _input_base


def test_input_base_requires_exactly_one_request_prefix(tmp_path):
    parent = (
        tmp_path
        / "normalized"
        / "databento_history"
        / "date=2026-09-01"
        / "metal=silver"
    )
    (parent / "request=abcdef123456").mkdir(parents=True)

    assert _input_base(tmp_path, "abcdef", "silver") == Path(
        parent / "request=abcdef123456"
    )

    (parent / "request=abcdef999999").mkdir()
    with pytest.raises(ValueError, match="expected one"):
        _input_base(tmp_path, "abcdef", "silver")
