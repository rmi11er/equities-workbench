import math

import pytest

from src.metals.natural_target_signals import (
    NaturalTargetSignalConfig,
    NaturalTargetSignalError,
    build_future_path_targets,
    chronological_window_split,
    evaluate_binary_signal,
    evaluate_continuous_signal,
    unavailable_target_registry,
    walk_forward_variance_diagnostic,
)


def test_natural_target_defaults_freeze_chronology_and_walk_forward_burnin():
    config = NaturalTargetSignalConfig(baseline_study_id="a" * 64)

    assert config.embargo_windows == 1
    assert config.walk_forward_initial_burnin_windows == 384
    assert config.walk_forward_bootstrap_samples == 2_000
    assert config.momentum_lookback_minutes == (1, 5, 10)


def test_natural_target_config_requires_complete_window_embargo():
    with pytest.raises(NaturalTargetSignalError, match="embargoed"):
        NaturalTargetSignalConfig(
            baseline_study_id="a" * 64,
            embargo_windows=0,
        )


def test_chronological_split_keeps_all_rows_from_a_window_together():
    rows = [
        {
            "market_ticker": f"window-{window:02d}",
            "window_open_ts_ns": window * 900,
        }
        for window in range(66)
        for _ in range(3)
    ]

    development, embargo, test = chronological_window_split(
        rows,
        development_fraction=0.67,
        embargo_windows=1,
        minimum_development_windows=40,
        minimum_test_windows=20,
    )

    assert len(development) == 44
    assert embargo == {"window-44"}
    assert len(test) == 21
    assert not development & embargo
    assert not development & test
    assert not embargo & test


def test_future_path_targets_begin_after_the_causal_close():
    decision = 1_000_000
    candles = {
        decision - 60: 100.0,
        decision: 101.0,
        decision + 60: 100.0,
        decision + 120: 102.0,
    }

    targets = build_future_path_targets(
        candles,
        decision_seconds=decision,
        window_close_seconds=decision + 180,
        forecast_variance_remaining=3e-4,
        jump_threshold_standard_deviations=2.0,
    )

    assert targets is not None
    moves = [math.log(101 / 100), math.log(100 / 101), math.log(102 / 100)]
    assert targets["future_signed_log_return"] == pytest.approx(math.log(1.02))
    assert targets["future_realized_variance"] == pytest.approx(
        sum(move * move for move in moves)
    )
    assert targets["future_close_range"] == pytest.approx(math.log(1.02))


def test_continuous_evaluator_uses_development_mean_as_control():
    development = [
        {
            "feature": float(value),
            "target": 2.0 * value,
            "remaining_seconds": 900,
        }
        for value in range(-30, 30)
    ]
    test = [
        {
            "feature": float(value),
            "target": 2.0 * value,
            "remaining_seconds": 120,
        }
        for value in range(30, 40)
    ]

    result = evaluate_continuous_signal(
        development,
        test,
        feature_field="feature",
        target_field="target",
        ridge_penalty=0.01,
    )

    metrics = result["incremental_test_metrics"]
    assert metrics["delta_mse_vs_mean_control"] < 0
    assert metrics["mse_skill_vs_mean_control"] > 0.99
    assert result["development_only_model"]["observations"] == 60


def test_binary_evaluator_uses_development_base_rate_control():
    development = [
        {
            "feature": float(value),
            "target": int(value > 0),
            "remaining_seconds": 600,
        }
        for value in range(-30, 31)
        if value != 0
    ]
    test = [
        {
            "feature": float(value),
            "target": int(value > 0),
            "remaining_seconds": 300,
        }
        for value in (-10, -8, -6, -4, -2, 2, 4, 6, 8, 10)
    ]

    result = evaluate_binary_signal(
        development,
        test,
        feature_field="feature",
        target_field="target",
        ridge_penalty=1.0,
    )

    metrics = result["incremental_test_metrics"]
    assert result["development_base_rate"] == pytest.approx(0.5)
    assert metrics["delta_brier_vs_base_rate"] < 0
    assert metrics["delta_log_loss_vs_base_rate"] < 0


def test_walk_forward_variance_forecasts_do_not_use_current_return():
    observations = [
        {
            "market_ticker": f"w-{index}",
            "open_ts_ns": index * 900_000_000_000,
            "utc_day": f"day-{index // 96}",
            "bucket": index % 96,
            "log_return": 0.001 + (index % 7) * 0.0001,
        }
        for index in range(200)
    ]
    changed = [dict(row) for row in observations]
    changed[-1]["log_return"] = 0.25

    original_result = walk_forward_variance_diagnostic(
        observations,
        initial_burnin_windows=96,
        tod_prior_observations=20,
        ewma_lambda=0.97,
        ewma_ratio_floor=0.25,
        ewma_ratio_ceiling=4.0,
    )
    changed_result = walk_forward_variance_diagnostic(
        changed,
        initial_burnin_windows=96,
        tod_prior_observations=20,
        ewma_lambda=0.97,
        ewma_ratio_floor=0.25,
        ewma_ratio_ceiling=4.0,
    )

    original = original_result["forecast_rows"]
    altered = changed_result["forecast_rows"]
    assert len(original) == 104
    assert original[0]["history_observations_available"] == 96
    for field in (
        "flat_variance_forecast",
        "time_of_day_variance_forecast",
        "time_of_day_slow_ewma_variance_forecast",
    ):
        assert altered[-1][field] == original[-1][field]
    assert (
        altered[-1]["realized_squared_log_return"]
        != original[-1]["realized_squared_log_return"]
    )


def test_walk_forward_reports_deterministic_day_block_intervals():
    observations = [
        {
            "market_ticker": f"w-{index}",
            "open_ts_ns": index * 900_000_000_000,
            "utc_day": f"day-{index // 96}",
            "bucket": index % 96,
            "log_return": 0.001 + (index % 9) * 0.0001,
        }
        for index in range(384)
    ]

    result = walk_forward_variance_diagnostic(
        observations,
        initial_burnin_windows=96,
        tod_prior_observations=20,
        ewma_lambda=0.97,
        ewma_ratio_floor=0.25,
        ewma_ratio_ceiling=4.0,
        bootstrap_samples=500,
        bootstrap_seed=7,
    )

    interval = result["day_block_confidence_intervals"][
        "time_of_day_vs_flat"
    ]["qlike"]
    assert interval["effective_utc_days"] == 3
    assert interval["bootstrap_samples"] == 500
    assert interval["sign_convention"] == "negative_favors_left"


def test_unavailable_registry_names_subsecond_and_fill_targets():
    registry = unavailable_target_registry()

    assert {item["research_target"] for item in registry} == {
        "subsecond_pyth_or_external_fair_movement",
        "fill_conditioned_external_fair_markout",
    }
    assert all(item["status"] == "unavailable_current_data" for item in registry)
