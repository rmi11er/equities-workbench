import json
import math
from pathlib import Path

import pytest

from src.metals.baseline_fair import (
    NSEC,
    BaselineFairConfig,
    BaselineFairError,
    ReturnObservation,
    fit_baseline_vol_model,
    run_baseline_fair_study,
)
from src.metals.research_design import materialize_research_design
from src.metals.storage import HashChainedJsonlWriter

EXAMPLE_DESIGN = (
    Path(__file__).resolve().parents[2] / "configs" / "research_design.example.json"
)
VALIDATION_START = 1_787_875_200 * NSEC


def _config(design_id="a" * 64, pilot_id="b" * 64):
    return BaselineFairConfig(
        metal="silver",
        training_end_ts_ns=VALIDATION_START,
        validation_start_ts_ns=VALIDATION_START,
        validation_end_ts_ns=VALIDATION_START + 3_600 * NSEC,
        formal_research_design_id=design_id,
        prerequisite_proxy_pilot_id=pilot_id,
    )


def _observation(index, return_value):
    target = 60.0
    open_ts = VALIDATION_START - (200 - index) * 900 * NSEC
    return ReturnObservation(
        market_ticker=f"TRAIN-{index}",
        open_ts_ns=open_ts,
        close_ts_ns=open_ts + 900 * NSEC,
        target=target,
        expiration=target * math.exp(return_value),
        outcome=int(return_value >= 0),
        round_digits=3,
    )


def _write_chain(root, relative, rows):
    path = root / relative
    with HashChainedJsonlWriter(path, fsync_every_records=100) as writer:
        for row in rows:
            writer.write(row)
    return path


def test_basic_vol_model_is_monotone_and_distribution_consistent():
    observations = [
        _observation(index, ((index % 11) - 5) * 0.0004) for index in range(150)
    ]
    model = fit_baseline_vol_model(observations, _config())

    for distribution in ("gaussian", "empirical"):
        probabilities = [
            model.probability_yes(
                spot,
                60.0,
                300,
                48,
                model.ewma_variance_at_training_end,
                3,
                distribution,
            )
            for spot in (59.9, 60.0, 60.1)
        ]
        assert probabilities == sorted(probabilities)
        assert all(0 < probability < 1 for probability in probabilities)
        assert model.probability_yes(
            60.0,
            59.9,
            300,
            48,
            model.ewma_variance_at_training_end,
            3,
            distribution,
        ) >= model.probability_yes(
            60.0,
            60.1,
            300,
            48,
            model.ewma_variance_at_training_end,
            3,
            distribution,
        )


@pytest.mark.parametrize("distribution", ["gaussian", "empirical"])
def test_manual_vol_multiplier_preserves_default_and_scales_return_dispersion(distribution):
    model = fit_baseline_vol_model(
        [_observation(index, ((index % 11) - 5) * 0.0004) for index in range(150)],
        _config(),
    )
    def probability(spot, seconds, multiplier=1.0):
        return model.probability_yes(
            spot, 60.0, seconds, 48, model.ewma_variance_at_training_end,
            3, distribution, volatility_multiplier=multiplier,
        )

    original = model.probability_yes(
        59.95, 60.0, 150, 48, model.ewma_variance_at_training_end, 3, distribution,
    )
    assert probability(59.95, 150) == original
    # Doubling sigma has the same dispersion as quadrupling the remaining horizon.
    assert probability(59.95, 150, 2.0) == pytest.approx(probability(59.95, 600))
    assert probability(59.95, 150, 2.0) > original
    assert probability(60.05, 150, 2.0) < probability(60.05, 150)


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("nan"), float("inf")])
def test_manual_vol_multiplier_rejects_invalid_values(multiplier):
    model = fit_baseline_vol_model(
        [_observation(index, ((index % 11) - 5) * 0.0004) for index in range(150)],
        _config(),
    )
    with pytest.raises(BaselineFairError, match="volatility multiplier"):
        model.probability_yes(
            59.95, 60.0, 150, 48, model.ewma_variance_at_training_end,
            3, "gaussian", volatility_multiplier=multiplier,
        )


def test_study_is_content_addressed_and_keeps_formal_holdout_sealed(tmp_path):
    design = materialize_research_design(
        tmp_path, json.loads(EXAMPLE_DESIGN.read_text(encoding="utf-8"))
    )
    pilot_id = "b" * 64
    pilot_path = (
        tmp_path
        / "derived"
        / "settlement_proxy_pilot"
        / f"run=settlement-proxy-{pilot_id[:24]}"
        / "report.json"
    )
    pilot_path.parent.mkdir(parents=True)
    pilot_path.write_text(
        json.dumps(
            {
                "pilot_id": pilot_id,
                "formal_settlement_gate_passed": False,
                "decision": {"exploratory_expectations_passed": True},
            }
        ),
        encoding="utf-8",
    )

    labels = []
    level = 60.0
    for index in range(120):
        open_ts = VALIDATION_START - (120 - index) * 900 * NSEC
        change = ((index % 9) - 4) * 0.01
        expiration = level + change
        labels.append(
            {
                "metal": "silver",
                "market_ticker": f"TRAIN-{index}",
                "open_ts_ns": open_ts,
                "close_ts_ns": open_ts + 900 * NSEC,
                "target_price": str(level),
                "expiration_value": str(expiration),
                "result": "yes" if expiration >= level else "no",
                "round_digits": 3,
            }
        )
        level = expiration
    for index in range(2):
        open_ts = VALIDATION_START + index * 900 * NSEC
        expiration = level + (0.02 if index else -0.01)
        labels.append(
            {
                "metal": "silver",
                "market_ticker": f"VALID-{index}",
                "open_ts_ns": open_ts,
                "close_ts_ns": open_ts + 900 * NSEC,
                "target_price": str(level),
                "expiration_value": str(expiration),
                "result": "yes" if expiration >= level else "no",
                "round_digits": 3,
            }
        )
        level = expiration
    labels_path = _write_chain(tmp_path, "inputs/labels.jsonl", labels)

    proxy_rows = []
    for minute in range(-1, 31):
        proxy_rows.append(
            {
                "metal": "silver",
                "timestamp_seconds": VALIDATION_START // NSEC + minute * 60,
                "close": str(60 + minute * 0.002),
                "settlement_eligible": False,
            }
        )
    proxy_path = _write_chain(tmp_path, "inputs/proxy.jsonl", proxy_rows)
    config = _config(design.design_id, pilot_id)

    first = run_baseline_fair_study(tmp_path, labels_path, proxy_path, config)
    second = run_baseline_fair_study(tmp_path, labels_path, proxy_path, config)

    assert first["study_id"] == second["study_id"]
    assert first["cohort"]["training_windows"] == 120
    assert first["cohort"]["validation_windows_with_complete_proxy"] == 1
    assert first["cohort"]["probability_rows"] == 15
    assert first["model_invariants"]["passed"] is True
    assert first["formal_final_holdout_accessed"] is False
    assert first["trading_eligible"] is False
    assert first["decision"]["eligible_for_incremental_signal_development"] is True


def test_config_requires_full_return_horizon_embargo():
    with pytest.raises(ValueError, match="embargo"):
        BaselineFairConfig(
            **{
                **_config().__dict__,
                "train_validation_embargo_seconds": 899,
            }
        )
