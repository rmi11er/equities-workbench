import math

import pytest

from src.metals.spot_return_signals import (
    SpotReturnSignalConfig,
    SpotReturnSignalError,
    fit_offset_logistic,
)


def test_offset_logistic_recovers_positive_incremental_signal():
    rows = []
    for index in range(200):
        feature = (index - 100) / 25
        probability = 1 / (1 + math.exp(-feature))
        rows.append(
            {
                "feature": feature,
                "baseline": 0.5,
                "outcome": int((index % 100) / 100 < probability),
            }
        )

    model = fit_offset_logistic(
        rows,
        feature_field="feature",
        baseline_field="baseline",
        ridge_penalty=1.0,
        include_feature=True,
    )

    assert model.coefficient > 0
    assert model.predict(0.5, 2.0) > model.predict(0.5, -2.0)


def test_intercept_only_control_cannot_smuggle_in_the_feature():
    rows = [
        {"feature": float(index), "baseline": 0.6, "outcome": index % 2}
        for index in range(40)
    ]

    model = fit_offset_logistic(
        rows,
        feature_field="feature",
        baseline_field="baseline",
        ridge_penalty=10.0,
        include_feature=False,
    )

    assert model.coefficient == 0
    assert model.predict(0.6, -100) == model.predict(0.6, 100)


def test_signal_config_requires_independent_unique_lookbacks():
    with pytest.raises(SpotReturnSignalError, match="lookbacks"):
        SpotReturnSignalConfig(
            baseline_study_id="a" * 64,
            lookback_minutes=(1, 1),
        )
