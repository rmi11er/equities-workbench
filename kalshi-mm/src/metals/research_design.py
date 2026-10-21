"""Immutable sample, label, leakage, and holdout policy for metals research.

This module freezes *design*, not research results.  It deliberately has no data
loader and its window classifier refuses to expose final-holdout membership at all.
Historical rows from the July/August plumbing period can therefore never become a
final holdout merely because they already exist locally.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

RESEARCH_DESIGN_SCHEMA_VERSION = "metals-research-design/v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_SPLIT_NAMES = ("research", "validation", "final_holdout")


class ResearchDesignError(RuntimeError):
    """A frozen research design is missing, malformed, tampered with, or misused."""


class FinalHoldoutSealedError(ResearchDesignError):
    """Raised before any API returns final-holdout membership or contents."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the frozen schema")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - stable public parser contract
            f"{label} must be an object"
        )
    return value


def _literal(value: Any, expected: Any, label: str) -> Any:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{label} must be exactly {expected!r}")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string without NUL")
    return value.strip()


def _reviewed_version(value: Any, label: str) -> str:
    result = _nonempty(value, label)
    if result.lower() in {"pending", "todo", "unreviewed", "replace-me"}:
        raise ValueError(f"{label} must identify a frozen version")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must be UTC")
    return parsed


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value.lower())
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class ProvisionalExchangeLabel:
    label_name: str
    status: str
    source_stream: str
    value_field: str
    availability_time_field: str
    yes_value: int
    no_value: int
    construction_uses_exact_pyth: bool
    eligible_as_training_label: bool
    permitted_use: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProvisionalExchangeLabel:
        value = _object(value, "provisional_exchange_reported")
        _exact_keys(
            value,
            {
                "label_name",
                "status",
                "source_stream",
                "value_field",
                "availability_time_field",
                "encoding",
                "construction_uses_exact_pyth",
                "eligible_as_training_label",
                "permitted_use",
            },
            "provisional_exchange_reported",
        )
        encoding = _object(value["encoding"], "provisional encoding")
        _exact_keys(encoding, {"yes", "no"}, "provisional encoding")
        return cls(
            label_name=_literal(
                value["label_name"],
                "exchange_reported_binary_settlement",
                "provisional label_name",
            ),
            status=_literal(value["status"], "provisional", "provisional status"),
            source_stream=_literal(
                value["source_stream"],
                "kalshi_settlement",
                "provisional source_stream",
            ),
            value_field=_literal(
                value["value_field"], "result", "provisional value_field"
            ),
            availability_time_field=_literal(
                value["availability_time_field"],
                "receive_ts_ns",
                "provisional availability_time_field",
            ),
            yes_value=_literal(encoding["yes"], 1, "provisional encoding.yes"),
            no_value=_literal(encoding["no"], 0, "provisional encoding.no"),
            construction_uses_exact_pyth=_literal(
                value["construction_uses_exact_pyth"],
                False,
                "provisional construction_uses_exact_pyth",
            ),
            eligible_as_training_label=_literal(
                value["eligible_as_training_label"],
                False,
                "provisional eligible_as_training_label",
            ),
            permitted_use=_literal(
                value["permitted_use"],
                "plumbing_and_discrepancy_audit_only",
                "provisional permitted_use",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_name": self.label_name,
            "status": self.status,
            "source_stream": self.source_stream,
            "value_field": self.value_field,
            "availability_time_field": self.availability_time_field,
            "encoding": {"yes": self.yes_value, "no": self.no_value},
            "construction_uses_exact_pyth": self.construction_uses_exact_pyth,
            "eligible_as_training_label": self.eligible_as_training_label,
            "permitted_use": self.permitted_use,
        }


@dataclass(frozen=True)
class ExactPythLabel:
    label_name: str
    status: str
    source_audit_stream: str
    source_audit_schema: str
    raw_price_stream: str
    raw_price_field: str
    timestamp_field: str
    decision_observation_rule: str
    final_observation_rule: str
    return_label_name: str
    return_formula: str
    opening_target_rule: str
    final_value_rule: str
    strike_source_stream: str
    strike_field: str
    comparison_operator: str
    rounding_digits_field: str
    rounding_modes_by_series: tuple[tuple[str, str], ...]
    yes_value: int
    no_value: int
    construction_uses_exchange_result: bool
    eligibility_gate: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExactPythLabel:
        value = _object(value, "independently_reconstructed_exact_pyth")
        _exact_keys(
            value,
            {
                "label_name",
                "status",
                "source_audit_stream",
                "source_audit_schema",
                "raw_price_stream",
                "raw_price_field",
                "timestamp_field",
                "decision_observation_rule",
                "final_observation_rule",
                "return_label_name",
                "return_formula",
                "opening_target_rule",
                "final_value_rule",
                "strike_source_stream",
                "strike_field",
                "comparison_operator",
                "rounding_digits_field",
                "rounding_modes_by_series",
                "settlement_encoding",
                "construction_uses_exchange_result",
                "eligibility_gate",
            },
            "independently_reconstructed_exact_pyth",
        )
        rounding = _object(value["rounding_modes_by_series"], "rounding modes")
        _exact_keys(rounding, {"KXGOLD15M", "KXSILVER15M"}, "rounding modes")
        for series in sorted(rounding):
            _literal(rounding[series], "half_up", f"rounding mode {series}")
        encoding = _object(value["settlement_encoding"], "exact encoding")
        _exact_keys(encoding, {"yes", "no"}, "exact encoding")
        literals = {
            "label_name": "independently_reconstructed_exact_pyth_settlement",
            "status": "authoritative_research_label_after_exact_audit",
            "source_audit_stream": "settlement_check",
            "source_audit_schema": "metals-settlement-audit/v2",
            "raw_price_stream": "pyth_exact_history",
            "raw_price_field": "price",
            "timestamp_field": "publish_timestamp_us",
            "decision_observation_rule": (
                "last_publication_strictly_before_decision_time"
            ),
            "final_observation_rule": ("last_publication_strictly_before_window_close"),
            "return_label_name": "pyth_raw_log_return_to_settlement",
            "return_formula": "ln(final_raw_pyth_value/decision_raw_pyth_value)",
            "opening_target_rule": (
                "round_half_up(opening_raw_pyth_value,round_digits)"
            ),
            "final_value_rule": ("round_half_up(final_raw_pyth_value,round_digits)"),
            "strike_source_stream": "market_dim",
            "strike_field": "target_price",
            "comparison_operator": "greater_or_equal",
            "rounding_digits_field": "round_digits",
            "eligibility_gate": (
                "audit_usable_and_target_value_result_agreements_all_true"
            ),
        }
        parsed = {
            field: _literal(value[field], expected, f"exact {field}")
            for field, expected in literals.items()
        }
        return cls(
            **parsed,
            rounding_modes_by_series=tuple(sorted(rounding.items())),
            yes_value=_literal(encoding["yes"], 1, "exact encoding.yes"),
            no_value=_literal(encoding["no"], 0, "exact encoding.no"),
            construction_uses_exchange_result=_literal(
                value["construction_uses_exchange_result"],
                False,
                "exact construction_uses_exchange_result",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rounding_modes_by_series"] = dict(self.rounding_modes_by_series)
        value["settlement_encoding"] = {
            "yes": value.pop("yes_value"),
            "no": value.pop("no_value"),
        }
        return value


@dataclass(frozen=True)
class WindowUnit:
    unit: str
    group_key: tuple[str, ...]
    atomic_split_assignment: bool
    required_terminal_streams: tuple[str, ...]
    completeness_rule: str
    partial_window_action: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WindowUnit:
        value = _object(value, "observation_unit")
        _exact_keys(
            value,
            {
                "unit",
                "group_key",
                "atomic_split_assignment",
                "required_terminal_streams",
                "completeness_rule",
                "partial_window_action",
            },
            "observation_unit",
        )
        group_key = value["group_key"]
        terminal = value["required_terminal_streams"]
        if group_key != ["series_ticker", "market_ticker", "open_ts_ns", "close_ts_ns"]:
            raise ValueError("observation_unit.group_key is not the frozen group key")
        if terminal != ["market_dim", "kalshi_settlement"]:
            raise ValueError("required_terminal_streams are not the frozen streams")
        return cls(
            unit=_literal(value["unit"], "complete_kalshi_window", "unit"),
            group_key=tuple(group_key),
            atomic_split_assignment=_literal(
                value["atomic_split_assignment"], True, "atomic_split_assignment"
            ),
            required_terminal_streams=tuple(terminal),
            completeness_rule=_literal(
                value["completeness_rule"],
                "one_canonical_metadata_one_final_settlement",
                "completeness_rule",
            ),
            partial_window_action=_literal(
                value["partial_window_action"], "exclude", "partial_window_action"
            ),
        )


@dataclass(frozen=True)
class TimeInterval:
    start: str
    end: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> TimeInterval:
        value = _object(value, label)
        _exact_keys(value, {"start", "end"}, label)
        start = _parse_utc(value["start"], f"{label}.start")
        end = _parse_utc(value["end"], f"{label}.end")
        if end <= start:
            raise ValueError(f"{label}.end must be after start")
        return cls(start=value["start"], end=value["end"])

    @property
    def start_datetime(self) -> datetime:
        return _parse_utc(self.start, "interval.start")

    @property
    def end_datetime(self) -> datetime:
        return _parse_utc(self.end, "interval.end")


@dataclass(frozen=True)
class SplitPlan:
    time_basis: str
    boundary_convention: str
    engineering_burn_in: TimeInterval
    minimum_complete_burn_in_trading_days: int
    engineering_burn_in_use: str
    intervals: tuple[tuple[str, TimeInterval], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SplitPlan:
        value = _object(value, "splits")
        _exact_keys(
            value,
            {
                "time_basis",
                "boundary_convention",
                "engineering_burn_in",
                "minimum_complete_burn_in_trading_days",
                "engineering_burn_in_use",
                "intervals",
            },
            "splits",
        )
        intervals = _object(value["intervals"], "splits.intervals")
        _exact_keys(intervals, set(_SPLIT_NAMES), "splits.intervals")
        parsed = tuple(
            (name, TimeInterval.from_mapping(intervals[name], f"splits.{name}"))
            for name in _SPLIT_NAMES
        )
        burn_in = TimeInterval.from_mapping(
            value["engineering_burn_in"], "engineering_burn_in"
        )
        for (_, left), (_, right) in pairwise(parsed):
            if left.end_datetime != right.start_datetime:
                raise ValueError("research splits must be chronological and contiguous")
        if burn_in.end_datetime > parsed[0][1].start_datetime:
            raise ValueError("engineering burn-in must end before research starts")
        return cls(
            time_basis=_literal(value["time_basis"], "UTC", "splits.time_basis"),
            boundary_convention=_literal(
                value["boundary_convention"], "[start,end)", "boundary_convention"
            ),
            engineering_burn_in=burn_in,
            minimum_complete_burn_in_trading_days=_positive_int(
                value["minimum_complete_burn_in_trading_days"],
                "minimum_complete_burn_in_trading_days",
            ),
            engineering_burn_in_use=_literal(
                value["engineering_burn_in_use"],
                "plumbing_only_not_model_selection",
                "engineering_burn_in_use",
            ),
            intervals=parsed,
        )

    def interval(self, name: str) -> TimeInterval:
        return dict(self.intervals)[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_basis": self.time_basis,
            "boundary_convention": self.boundary_convention,
            "engineering_burn_in": asdict(self.engineering_burn_in),
            "minimum_complete_burn_in_trading_days": (
                self.minimum_complete_burn_in_trading_days
            ),
            "engineering_burn_in_use": self.engineering_burn_in_use,
            "intervals": {name: asdict(interval) for name, interval in self.intervals},
        }


@dataclass(frozen=True)
class LeakageControl:
    forward_label_horizons_seconds: tuple[int, ...]
    max_forward_label_horizon_seconds: int
    max_order_lifetime_seconds: int
    embargo_seconds: int
    purge_rule: str
    embargo_rule: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LeakageControl:
        value = _object(value, "leakage_control")
        _exact_keys(
            value,
            {
                "forward_label_horizons_seconds",
                "max_forward_label_horizon_seconds",
                "max_order_lifetime_seconds",
                "embargo_seconds",
                "purge_rule",
                "embargo_rule",
            },
            "leakage_control",
        )
        horizons = value["forward_label_horizons_seconds"]
        if (
            not isinstance(horizons, list)
            or not horizons
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in horizons
            )
            or horizons != sorted(set(horizons))
        ):
            raise ValueError(
                "forward label horizons must be unique positive sorted integers"
            )
        maximum = _positive_int(
            value["max_forward_label_horizon_seconds"],
            "max_forward_label_horizon_seconds",
        )
        if maximum != max(horizons):
            raise ValueError("max forward label horizon must equal the largest horizon")
        order_lifetime = _positive_int(
            value["max_order_lifetime_seconds"], "max_order_lifetime_seconds"
        )
        embargo = _positive_int(value["embargo_seconds"], "embargo_seconds")
        if embargo < max(maximum, order_lifetime):
            raise ValueError(
                "embargo must be at least the maximum forward label/order lifetime"
            )
        return cls(
            forward_label_horizons_seconds=tuple(horizons),
            max_forward_label_horizon_seconds=maximum,
            max_order_lifetime_seconds=order_lifetime,
            embargo_seconds=embargo,
            purge_rule=_literal(
                value["purge_rule"],
                "exclude_window_if_any_forward_label_or_order_lifetime_crosses_split_end",
                "purge_rule",
            ),
            embargo_rule=_literal(
                value["embargo_rule"],
                "exclude_windows_opening_before_split_start_plus_embargo_except_research",
                "embargo_rule",
            ),
        )


@dataclass(frozen=True)
class ConfidenceIntervalPolicy:
    block_unit: str
    window_to_block_rule: str
    interval_level: str
    resampling_method: str
    report_effective_day_and_window_counts: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ConfidenceIntervalPolicy:
        value = _object(value, "confidence_intervals")
        _exact_keys(
            value,
            {
                "block_unit",
                "window_to_block_rule",
                "interval_level",
                "resampling_method",
                "report_effective_day_and_window_counts",
            },
            "confidence_intervals",
        )
        return cls(
            block_unit=_literal(value["block_unit"], "UTC_trading_day", "block_unit"),
            window_to_block_rule=_literal(
                value["window_to_block_rule"],
                "UTC_calendar_date_of_window_open",
                "window_to_block_rule",
            ),
            interval_level=_literal(
                value["interval_level"], "95_percent", "interval_level"
            ),
            resampling_method=_literal(
                value["resampling_method"],
                "cluster_bootstrap_complete_UTC_trading_days",
                "resampling_method",
            ),
            report_effective_day_and_window_counts=_literal(
                value["report_effective_day_and_window_counts"],
                True,
                "report_effective_day_and_window_counts",
            ),
        )


@dataclass(frozen=True)
class HoldoutPolicy:
    status: str
    records_read_before_unseal: int
    metrics_read_before_unseal: int
    selection_use_before_unseal: str
    explicit_unseal_authorization_required: bool
    prerequisites: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HoldoutPolicy:
        value = _object(value, "final_holdout_policy")
        _exact_keys(
            value,
            {
                "status",
                "records_read_before_unseal",
                "metrics_read_before_unseal",
                "selection_use_before_unseal",
                "explicit_unseal_authorization_required",
                "unseal_prerequisites",
            },
            "final_holdout_policy",
        )
        required = [
            "external_fair_model_frozen",
            "signal_set_frozen",
            "execution_model_frozen",
            "quote_eligibility_rules_frozen",
            "risk_policy_frozen",
        ]
        if value["unseal_prerequisites"] != required:
            raise ValueError("final holdout prerequisites are not the frozen sequence")
        return cls(
            status=_literal(value["status"], "sealed_unread", "holdout status"),
            records_read_before_unseal=_literal(
                value["records_read_before_unseal"], 0, "records_read_before_unseal"
            ),
            metrics_read_before_unseal=_literal(
                value["metrics_read_before_unseal"], 0, "metrics_read_before_unseal"
            ),
            selection_use_before_unseal=_literal(
                value["selection_use_before_unseal"],
                "forbidden",
                "selection_use_before_unseal",
            ),
            explicit_unseal_authorization_required=_literal(
                value["explicit_unseal_authorization_required"],
                True,
                "explicit_unseal_authorization_required",
            ),
            prerequisites=tuple(required),
        )


@dataclass(frozen=True)
class LegacyDataPolicy:
    interval: TimeInterval
    designation: str
    eligible_for_final_holdout: bool
    eligible_for_trading_claim: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LegacyDataPolicy:
        value = _object(value, "preexisting_data_policy")
        _exact_keys(
            value,
            {
                "interval",
                "designation",
                "eligible_for_final_holdout",
                "eligible_for_trading_claim",
            },
            "preexisting_data_policy",
        )
        interval = TimeInterval.from_mapping(value["interval"], "preexisting interval")
        _literal(
            interval.start,
            "2026-07-31T00:00:00Z",
            "preexisting interval.start",
        )
        _literal(
            interval.end,
            "2026-08-31T00:00:00Z",
            "preexisting interval.end",
        )
        return cls(
            interval=interval,
            designation=_literal(
                value["designation"],
                "research_and_plumbing_only",
                "preexisting designation",
            ),
            eligible_for_final_holdout=_literal(
                value["eligible_for_final_holdout"],
                False,
                "preexisting eligible_for_final_holdout",
            ),
            eligible_for_trading_claim=_literal(
                value["eligible_for_trading_claim"],
                False,
                "preexisting eligible_for_trading_claim",
            ),
        )


@dataclass(frozen=True)
class VersionedStatements:
    version: str
    items: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> VersionedStatements:
        value = _object(value, label)
        _exact_keys(value, {"version", "items"}, label)
        items = value["items"]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{label}.items must be a non-empty array")
        parsed: list[tuple[str, str]] = []
        for index, item in enumerate(items):
            item = _object(item, f"{label}.items[{index}]")
            _exact_keys(item, {"id", "statement"}, f"{label}.items[{index}]")
            parsed.append(
                (
                    _nonempty(item["id"], f"{label}.items[{index}].id"),
                    _nonempty(item["statement"], f"{label}.items[{index}].statement"),
                )
            )
        if len({item[0] for item in parsed}) != len(parsed):
            raise ValueError(f"{label} ids must be unique")
        return cls(
            version=_reviewed_version(value["version"], f"{label}.version"),
            items=tuple(parsed),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "items": [
                {"id": identifier, "statement": statement}
                for identifier, statement in self.items
            ],
        }


@dataclass(frozen=True)
class ResearchDesignPolicy:
    design_version: str
    frozen_at: str
    research_only: bool
    no_live: bool
    no_orders: bool
    provisional_label: ProvisionalExchangeLabel
    exact_pyth_label: ExactPythLabel
    observation_unit: WindowUnit
    splits: SplitPlan
    leakage_control: LeakageControl
    confidence_intervals: ConfidenceIntervalPolicy
    final_holdout_policy: HoldoutPolicy
    preexisting_data_policy: LegacyDataPolicy
    assumptions: VersionedStatements
    limitations: VersionedStatements
    schema_version: str = RESEARCH_DESIGN_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchDesignPolicy:
        value = _object(value, "research design")
        _exact_keys(
            value,
            {
                "schema_version",
                "design_version",
                "frozen_at",
                "research_only",
                "no_live",
                "no_orders",
                "label_contract",
                "observation_unit",
                "splits",
                "leakage_control",
                "confidence_intervals",
                "final_holdout_policy",
                "preexisting_data_policy",
                "assumptions",
                "limitations",
            },
            "research design",
        )
        _literal(
            value["schema_version"],
            RESEARCH_DESIGN_SCHEMA_VERSION,
            "schema_version",
        )
        for flag in ("research_only", "no_live", "no_orders"):
            _literal(value[flag], True, flag)
        frozen_at = _parse_utc(value["frozen_at"], "frozen_at")
        if frozen_at > datetime.now(UTC):
            raise ValueError("frozen_at cannot be in the future")
        labels = _object(value["label_contract"], "label_contract")
        _exact_keys(
            labels,
            {"provisional_exchange_reported", "independently_reconstructed_exact_pyth"},
            "label_contract",
        )
        policy = cls(
            design_version=_reviewed_version(value["design_version"], "design_version"),
            frozen_at=value["frozen_at"],
            research_only=True,
            no_live=True,
            no_orders=True,
            provisional_label=ProvisionalExchangeLabel.from_mapping(
                labels["provisional_exchange_reported"]
            ),
            exact_pyth_label=ExactPythLabel.from_mapping(
                labels["independently_reconstructed_exact_pyth"]
            ),
            observation_unit=WindowUnit.from_mapping(value["observation_unit"]),
            splits=SplitPlan.from_mapping(value["splits"]),
            leakage_control=LeakageControl.from_mapping(value["leakage_control"]),
            confidence_intervals=ConfidenceIntervalPolicy.from_mapping(
                value["confidence_intervals"]
            ),
            final_holdout_policy=HoldoutPolicy.from_mapping(
                value["final_holdout_policy"]
            ),
            preexisting_data_policy=LegacyDataPolicy.from_mapping(
                value["preexisting_data_policy"]
            ),
            assumptions=VersionedStatements.from_mapping(
                value["assumptions"], "assumptions"
            ),
            limitations=VersionedStatements.from_mapping(
                value["limitations"], "limitations"
            ),
        )
        if (
            policy.preexisting_data_policy.interval.end_datetime
            > policy.splits.engineering_burn_in.start_datetime
        ):
            raise ValueError("preexisting plumbing data overlaps engineering burn-in")
        if policy.splits.interval("final_holdout").start_datetime <= frozen_at:
            raise ValueError(
                "final holdout must be future and unread when design freezes"
            )
        return policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_version": self.design_version,
            "frozen_at": self.frozen_at,
            "research_only": self.research_only,
            "no_live": self.no_live,
            "no_orders": self.no_orders,
            "label_contract": {
                "provisional_exchange_reported": self.provisional_label.to_dict(),
                "independently_reconstructed_exact_pyth": (
                    self.exact_pyth_label.to_dict()
                ),
            },
            "observation_unit": {
                "unit": self.observation_unit.unit,
                "group_key": list(self.observation_unit.group_key),
                "atomic_split_assignment": (
                    self.observation_unit.atomic_split_assignment
                ),
                "required_terminal_streams": list(
                    self.observation_unit.required_terminal_streams
                ),
                "completeness_rule": self.observation_unit.completeness_rule,
                "partial_window_action": self.observation_unit.partial_window_action,
            },
            "splits": self.splits.to_dict(),
            "leakage_control": {
                "forward_label_horizons_seconds": list(
                    self.leakage_control.forward_label_horizons_seconds
                ),
                "max_forward_label_horizon_seconds": (
                    self.leakage_control.max_forward_label_horizon_seconds
                ),
                "max_order_lifetime_seconds": (
                    self.leakage_control.max_order_lifetime_seconds
                ),
                "embargo_seconds": self.leakage_control.embargo_seconds,
                "purge_rule": self.leakage_control.purge_rule,
                "embargo_rule": self.leakage_control.embargo_rule,
            },
            "confidence_intervals": asdict(self.confidence_intervals),
            "final_holdout_policy": {
                "status": self.final_holdout_policy.status,
                "records_read_before_unseal": (
                    self.final_holdout_policy.records_read_before_unseal
                ),
                "metrics_read_before_unseal": (
                    self.final_holdout_policy.metrics_read_before_unseal
                ),
                "selection_use_before_unseal": (
                    self.final_holdout_policy.selection_use_before_unseal
                ),
                "explicit_unseal_authorization_required": (
                    self.final_holdout_policy.explicit_unseal_authorization_required
                ),
                "unseal_prerequisites": list(self.final_holdout_policy.prerequisites),
            },
            "preexisting_data_policy": {
                "interval": asdict(self.preexisting_data_policy.interval),
                "designation": self.preexisting_data_policy.designation,
                "eligible_for_final_holdout": (
                    self.preexisting_data_policy.eligible_for_final_holdout
                ),
                "eligible_for_trading_claim": (
                    self.preexisting_data_policy.eligible_for_trading_claim
                ),
            },
            "assumptions": self.assumptions.to_dict(),
            "limitations": self.limitations.to_dict(),
        }


@dataclass(frozen=True)
class ResearchDesignArtifact:
    design_id: str
    directory: Path
    design_path: Path
    manifest_path: Path
    policy: ResearchDesignPolicy

    def public_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "design_version": self.policy.design_version,
            "design_path": str(self.design_path),
            "manifest_path": str(self.manifest_path),
            "research_only": True,
            "no_live": True,
            "no_orders": True,
            "final_holdout_status": self.policy.final_holdout_policy.status,
        }


@dataclass(frozen=True)
class WindowAssignment:
    split: str | None
    eligible: bool
    exclusion_reason: str | None
    utc_trading_day: str


def assign_complete_window(
    policy: ResearchDesignPolicy,
    *,
    window_open: datetime,
    window_close: datetime,
    maximum_forward_label_end: datetime,
    maximum_order_lifetime_end: datetime,
    complete: bool,
) -> WindowAssignment:
    """Assign one atomic window without reading rows or splitting its observations.

    Final-holdout assignment is intentionally unavailable.  A future integration
    must require an independently verified model/quote/risk freeze and unseal
    artifact; a boolean escape hatch would not be an adequate control.
    """

    values = {
        "window_open": window_open,
        "window_close": window_close,
        "maximum_forward_label_end": maximum_forward_label_end,
        "maximum_order_lifetime_end": maximum_order_lifetime_end,
    }
    for name, value in values.items():
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError(f"{name} must be timezone-aware UTC")
    if window_close <= window_open:
        raise ValueError("window_close must be after window_open")
    if maximum_forward_label_end < window_open:
        raise ValueError("maximum_forward_label_end cannot precede window_open")
    if maximum_order_lifetime_end < window_open:
        raise ValueError("maximum_order_lifetime_end cannot precede window_open")
    trading_day = window_open.date().isoformat()
    if not complete:
        return WindowAssignment(None, False, "incomplete_window", trading_day)

    legacy = policy.preexisting_data_policy.interval
    if legacy.start_datetime <= window_open < legacy.end_datetime:
        return WindowAssignment(
            None, False, "preexisting_research_plumbing_only", trading_day
        )

    for name, interval in policy.splits.intervals:
        start = interval.start_datetime
        end = interval.end_datetime
        if not (start <= window_open < end):
            continue
        if name == "final_holdout":
            raise FinalHoldoutSealedError(
                "final holdout is sealed; membership and contents must remain unread"
            )
        if window_close > end:
            return WindowAssignment(
                name, False, "window_crosses_split_end", trading_day
            )
        if maximum_forward_label_end > end or maximum_order_lifetime_end > end:
            return WindowAssignment(name, False, "purged_at_split_end", trading_day)
        if name != "research" and window_open < start + timedelta(
            seconds=policy.leakage_control.embargo_seconds
        ):
            return WindowAssignment(
                name, False, "embargoed_at_split_start", trading_day
            )
        return WindowAssignment(name, True, None, trading_day)
    return WindowAssignment(None, False, "outside_frozen_panel", trading_day)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResearchDesignError(f"Cannot read {label}") from error
    if not isinstance(value, dict):
        raise ResearchDesignError(f"{label} must be a JSON object")
    return value


def _resolve_inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ResearchDesignError(f"{label} escapes the storage root") from error
    return resolved


def materialize_research_design(
    storage_root: Path, value: Mapping[str, Any]
) -> ResearchDesignArtifact:
    """Freeze one secret-free, content-addressed research-design artifact."""

    policy = ResearchDesignPolicy.from_mapping(value)
    canonical = policy.to_dict()
    design_id = _sha256_bytes(_canonical_bytes(canonical))
    root = Path(storage_root).resolve()
    destination = root / "policies" / "research_design" / f"design={design_id}"
    if destination.exists():
        return verify_research_design(root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".research-design-", dir=destination.parent)
    )
    try:
        design_path = temporary / "design.json"
        _write_json(design_path, {"design_id": design_id, **canonical})
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": RESEARCH_DESIGN_SCHEMA_VERSION,
                "design_id": design_id,
                "outputs": [
                    {
                        "path": "design.json",
                        "size_bytes": design_path.stat().st_size,
                        "sha256": _sha256_path(design_path),
                    }
                ],
            },
        )
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            shutil.rmtree(temporary)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_research_design(root, destination)


def verify_research_design(
    storage_root: Path, design_path: Path
) -> ResearchDesignArtifact:
    """Verify strict schema, content address, storage location, and manifest hashes."""

    root = Path(storage_root).resolve()
    requested = _resolve_inside(root, Path(design_path), "Research design")
    directory = (
        requested.parent
        if requested.name in {"design.json", "manifest.json"}
        else requested
    )
    relative = directory.relative_to(root)
    if (
        len(relative.parts) < 3
        or relative.parts[-2] != "research_design"
        or not directory.name.startswith("design=")
    ):
        raise ResearchDesignError("Research design location is invalid")
    design_file = directory / "design.json"
    manifest_file = directory / "manifest.json"
    design = _read_object(design_file, "research design")
    manifest = _read_object(manifest_file, "research design manifest")
    design_id = design.get("design_id")
    if not _is_sha256(design_id) or directory.name != f"design={design_id}":
        raise ResearchDesignError("Research design identity is invalid")
    policy_value = dict(design)
    policy_value.pop("design_id", None)
    try:
        policy = ResearchDesignPolicy.from_mapping(policy_value)
    except ValueError as error:
        raise ResearchDesignError("Research design schema is invalid") from error
    if _sha256_bytes(_canonical_bytes(policy.to_dict())) != design_id:
        raise ResearchDesignError("Research design content hash is invalid")
    outputs = manifest.get("outputs")
    if (
        set(manifest) != {"schema_version", "design_id", "outputs"}
        or manifest.get("schema_version") != RESEARCH_DESIGN_SCHEMA_VERSION
        or manifest.get("design_id") != design_id
        or not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0], Mapping)
        or set(outputs[0]) != {"path", "size_bytes", "sha256"}
        or outputs[0].get("path") != "design.json"
        or outputs[0].get("size_bytes") != design_file.stat().st_size
        or outputs[0].get("sha256") != _sha256_path(design_file)
    ):
        raise ResearchDesignError("Research design manifest is invalid")
    return ResearchDesignArtifact(
        design_id=design_id,
        directory=directory,
        design_path=design_file,
        manifest_path=manifest_file,
        policy=policy,
    )
