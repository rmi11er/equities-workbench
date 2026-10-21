import pytest

from src.metals.path_variance_signals import (
    PathVarianceSignalConfig,
    PathVarianceSignalError,
)


def test_path_variance_defaults_keep_signal_definitions_separate():
    config = PathVarianceSignalConfig(baseline_study_id="a" * 64)

    assert config.range_lookback_minutes == (3, 10)
    assert config.efficiency_lookback_minutes == (5, 10)
    assert config.embargo_windows == 1


def test_path_variance_rejects_invalid_lookback():
    with pytest.raises(PathVarianceSignalError, match="lookbacks"):
        PathVarianceSignalConfig(
            baseline_study_id="a" * 64,
            range_lookback_minutes=(1,),
        )
