import pytest

from src.metals.cross_metal_signals import (
    CrossMetalSignalConfig,
    CrossMetalSignalError,
    fit_cross_metal_normalization,
)


def test_cross_metal_defaults_freeze_one_family_horizon():
    config = CrossMetalSignalConfig(baseline_study_id="a" * 64)

    assert config.lookback_minutes == 5
    assert config.embargo_windows == 1


def test_cross_metal_rejects_overlong_lookback():
    with pytest.raises(CrossMetalSignalError, match="lookback"):
        CrossMetalSignalConfig(
            baseline_study_id="a" * 64,
            lookback_minutes=15,
        )


def test_cross_metal_normalization_uses_centered_development_beta():
    pairs = [(float(index), 2.0 * index + (-1) ** index) for index in range(20)]

    fitted = fit_cross_metal_normalization(pairs)

    assert fitted.observations == 20
    assert fitted.silver_std > 0
    assert fitted.gold_std > 0
    assert fitted.residual_std > 0
    assert fitted.silver_on_gold_beta == pytest.approx(0.5)
