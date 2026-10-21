import pytest

from src.metals.semivariance_jump_signals import (
    SemivarianceJumpSignalConfig,
    SemivarianceJumpSignalError,
)


def test_semivariance_jump_defaults_freeze_one_family_horizon():
    config = SemivarianceJumpSignalConfig(baseline_study_id="a" * 64)

    assert config.lookback_minutes == 10
    assert config.embargo_windows == 1
    assert config.jump_threshold_standard_deviations == 2.0


def test_semivariance_jump_rejects_overlong_lookback():
    with pytest.raises(SemivarianceJumpSignalError, match="lookback"):
        SemivarianceJumpSignalConfig(
            baseline_study_id="a" * 64,
            lookback_minutes=15,
        )
