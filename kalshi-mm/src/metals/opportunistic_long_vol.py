"""Research-only screen for fair-filtered underdog purchases.

The economic hypothesis is narrower than an unconditional favorite fade.  A
current underdog is eligible only when an external terminal-distribution model
values that same outcome above its executable ask by enough to pay the taker
fee and a frozen uncertainty hurdle.  Kalshi prices select the outcome and the
entry price; they never enter the external fair.

Historical one-minute candles do not have receipt timestamps.  Results from
``build_historical_screen`` are therefore opportunity screens, not executable
backtests or permission to trade.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence


GAUSSIAN_FAIR = "fair_yes_frozen_tod_ewma_gaussian"
EMPIRICAL_FAIR = "fair_yes_frozen_empirical_cdf"
SCHEMA_VERSION = "metals-opportunistic-long-vol-screen/v1"


@dataclass(frozen=True)
class LongVolPolicy:
    """Frozen one-contract entry policy used by the exploratory screen."""

    min_underdog_ask: float = 0.07
    max_underdog_ask: float = 0.31
    min_remaining_seconds: int = 120
    max_remaining_seconds: int = 900
    taker_fee_coefficient: float = 0.07
    min_net_external_edge: float = 0.03
    uncertainty_charge: float = 0.00
    max_book_age_ms: int = 250
    max_external_fair_age_ms: int = 250
    persistence_ms: int = 1_000
    max_entries_per_market: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.min_underdog_ask < self.max_underdog_ask < 0.5:
            raise ValueError("underdog ask bounds must lie strictly below 0.5")
        if not 0 < self.min_remaining_seconds <= self.max_remaining_seconds <= 900:
            raise ValueError("invalid remaining-time bounds")
        if self.taker_fee_coefficient < 0:
            raise ValueError("taker fee coefficient cannot be negative")
        if self.min_net_external_edge < 0 or self.uncertainty_charge < 0:
            raise ValueError("edge and uncertainty charges cannot be negative")
        if min(
            self.max_book_age_ms,
            self.max_external_fair_age_ms,
            self.persistence_ms,
        ) < 0:
            raise ValueError("freshness and persistence values cannot be negative")
        if self.max_entries_per_market != 1:
            raise ValueError("this research policy freezes one entry per market")


@dataclass(frozen=True)
class Opportunity:
    market_ticker: str
    split: str
    utc_day: str
    decision_ts_ns: int
    remaining_seconds: int
    side: str
    underdog_bid: float
    underdog_ask: float
    external_fair: float
    taker_fee: float
    gross_external_edge: float
    net_external_edge: float
    implied_vol_ratio_to_external: float | None
    outcome_won: bool
    terminal_pnl: float


def one_contract_taker_fee(price: float, coefficient: float = 0.07) -> float:
    """Return the one-contract fee, rounded up to the next centicent.

    The coefficient is an input because a versioned effective fee policy, rather
    than this default, must be used before any live order is enabled.
    """

    if not 0 <= price <= 1:
        raise ValueError("price must be between zero and one")
    if coefficient < 0:
        raise ValueError("coefficient cannot be negative")
    raw = coefficient * price * (1 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def executable_underdog(
    *, yes_bid: float, yes_ask: float
) -> tuple[str, float, float]:
    """Return current underdog side and its executable bid/ask."""

    # Kalshi's historical candles can legitimately carry a zero bid or a
    # one-dollar ask near settlement.  The derived core underdog ask is filtered
    # separately by the policy bounds.
    if not 0 <= yes_bid < yes_ask <= 1:
        raise ValueError("invalid two-sided YES quote")
    midpoint = (yes_bid + yes_ask) / 2
    if midpoint < 0.5:
        return "yes", yes_bid, yes_ask
    if midpoint > 0.5:
        return "no", 1 - yes_ask, 1 - yes_bid
    raise ValueError("there is no unique underdog at a 50-cent midpoint")


def implied_vol_ratio(
    *, external_underdog_fair: float, market_underdog_price: float
) -> float | None:
    """Infer market/external volatility scale under a zero-mean Gaussian digital.

    If both prices describe the same distance to strike and horizon, the ratio is
    ``abs(Phi^-1(p_external)) / abs(Phi^-1(p_market))``.  It is descriptive only:
    drift, skew, jumps, rounding, or stale spot can also create a price gap.
    """

    if not 0 < external_underdog_fair < 0.5:
        return None
    if not 0 < market_underdog_price < 0.5:
        return None
    external_z = abs(NormalDist().inv_cdf(external_underdog_fair))
    market_z = abs(NormalDist().inv_cdf(market_underdog_price))
    if market_z == 0:
        return None
    return external_z / market_z


def evaluate_entry(
    *,
    market_ticker: str,
    split: str,
    utc_day: str,
    decision_ts_ns: int,
    remaining_seconds: int,
    yes_bid: float,
    yes_ask: float,
    fair_yes: float,
    outcome_yes: bool,
    policy: LongVolPolicy,
) -> Opportunity | None:
    """Evaluate a single quote without using Kalshi price in the external fair."""

    if not 0 < fair_yes < 1:
        raise ValueError("external YES fair must be between zero and one")
    if (yes_bid + yes_ask) / 2 == 0.5:
        return None
    side, bid, ask = executable_underdog(yes_bid=yes_bid, yes_ask=yes_ask)
    if not policy.min_underdog_ask <= ask <= policy.max_underdog_ask:
        return None
    if not policy.min_remaining_seconds <= remaining_seconds <= policy.max_remaining_seconds:
        return None
    fair = fair_yes if side == "yes" else 1 - fair_yes
    fee = one_contract_taker_fee(ask, policy.taker_fee_coefficient)
    gross_edge = fair - ask
    net_edge = gross_edge - fee - policy.uncertainty_charge
    if net_edge < policy.min_net_external_edge:
        return None
    won = outcome_yes if side == "yes" else not outcome_yes
    return Opportunity(
        market_ticker=market_ticker,
        split=split,
        utc_day=utc_day,
        decision_ts_ns=decision_ts_ns,
        remaining_seconds=remaining_seconds,
        side=side,
        underdog_bid=bid,
        underdog_ask=ask,
        external_fair=fair,
        taker_fee=fee,
        gross_external_edge=gross_edge,
        net_external_edge=net_edge,
        implied_vol_ratio_to_external=implied_vol_ratio(
            external_underdog_fair=fair,
            market_underdog_price=ask,
        ),
        outcome_won=won,
        terminal_pnl=float(won) - ask - fee,
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty score rows: {path}")
    return rows


def _fair_yes(
    row: Mapping[str, Any],
    fair_definition: str,
    *,
    yes_bid: float,
    yes_ask: float,
) -> float:
    if fair_definition == "robust_minimum":
        gaussian = float(row[GAUSSIAN_FAIR])
        empirical = float(row[EMPIRICAL_FAIR])
        # Conservatism is side-specific.  The lower YES fair is conservative
        # when YES is the underdog; the higher YES fair creates the lower NO
        # fair when NO is the underdog.
        return min(gaussian, empirical) if (yes_bid + yes_ask) / 2 < 0.5 else max(
            gaussian, empirical
        )
    return float(row[fair_definition])


def select_historical_opportunities(
    rows: Sequence[Mapping[str, Any]],
    *,
    fair_definition: str,
    policy: LongVolPolicy,
) -> list[Opportunity]:
    """Select the earliest eligible row in each whole Kalshi window."""

    candidates: list[Opportunity] = []
    for row in rows:
        if row.get("split") == "embargo" or not row.get("kalshi_midpoint_available"):
            continue
        yes_bid = float(row["kalshi_yes_bid_close"])
        yes_ask = float(row["kalshi_yes_ask_close"])
        opportunity = evaluate_entry(
            market_ticker=str(row["market_ticker"]),
            split=str(row["split"]),
            utc_day=str(row["utc_day"]),
            decision_ts_ns=int(row["decision_ts_ns"]),
            remaining_seconds=int(row["remaining_seconds"]),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            fair_yes=_fair_yes(
                row,
                fair_definition,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
            ),
            outcome_yes=bool(int(row["outcome"])),
            policy=policy,
        )
        if opportunity is not None:
            candidates.append(opportunity)

    selected: list[Opportunity] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(candidates, key=lambda value: value.decision_ts_ns):
        key = (row.split, row.market_ticker)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


def _bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = 10_000, seed: int = 71_923
) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(seed)
    means = [
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]
    return [float(_percentile(means, 0.025)), float(_percentile(means, 0.975))]


def _summary(rows: Sequence[Opportunity]) -> dict[str, Any]:
    pnl = [row.terminal_pnl for row in rows]
    ratios = [
        row.implied_vol_ratio_to_external
        for row in rows
        if row.implied_vol_ratio_to_external is not None
    ]
    return {
        "entries": len(rows),
        "windows": len({row.market_ticker for row in rows}),
        "effective_utc_days": len({row.utc_day for row in rows}),
        "yes_entries": sum(row.side == "yes" for row in rows),
        "no_entries": sum(row.side == "no" for row in rows),
        "wins": sum(row.outcome_won for row in rows),
        "win_rate": sum(row.outcome_won for row in rows) / len(rows) if rows else None,
        "mean_entry_ask": sum(row.underdog_ask for row in rows) / len(rows) if rows else None,
        "mean_predicted_net_edge": (
            sum(row.net_external_edge for row in rows) / len(rows) if rows else None
        ),
        "mean_terminal_pnl": sum(pnl) / len(pnl) if pnl else None,
        "total_terminal_pnl": sum(pnl),
        "market_bootstrap_95pct_mean_pnl": _bootstrap_mean_interval(pnl),
        "median_implied_vol_ratio_to_external": _percentile(ratios, 0.5),
    }


def _historical_markouts(
    all_rows: Sequence[Mapping[str, Any]], opportunities: Sequence[Opportunity]
) -> dict[str, Any]:
    """Mark one-minute candle quotes after entry, holding the purchased side fixed."""

    indexed = {
        (str(row["market_ticker"]), int(row["decision_ts_ns"])): row
        for row in all_rows
        if row.get("kalshi_midpoint_available")
    }
    result: dict[str, Any] = {}
    for horizon_seconds in (60, 180, 300):
        midpoint_markouts: list[float] = []
        liquidation_markouts: list[float] = []
        for opportunity in opportunities:
            future = indexed.get(
                (
                    opportunity.market_ticker,
                    opportunity.decision_ts_ns + horizon_seconds * 1_000_000_000,
                )
            )
            if future is None:
                continue
            if opportunity.side == "yes":
                future_midpoint = float(future["kalshi_yes_midpoint"])
                future_bid = float(future["kalshi_yes_bid_close"])
            else:
                future_midpoint = 1 - float(future["kalshi_yes_midpoint"])
                future_bid = 1 - float(future["kalshi_yes_ask_close"])
            midpoint_markouts.append(
                future_midpoint - opportunity.underdog_ask - opportunity.taker_fee
            )
            liquidation_markouts.append(
                future_bid
                - opportunity.underdog_ask
                - opportunity.taker_fee
                - one_contract_taker_fee(future_bid)
            )
        result[f"{horizon_seconds}s"] = {
            "observations": len(midpoint_markouts),
            "mean_midpoint_markout": (
                sum(midpoint_markouts) / len(midpoint_markouts)
                if midpoint_markouts
                else None
            ),
            "mean_executable_liquidation_pnl": (
                sum(liquidation_markouts) / len(liquidation_markouts)
                if liquidation_markouts
                else None
            ),
        }
    return result


def build_historical_screen(
    score_rows_path: str | Path,
    *,
    policy: LongVolPolicy = LongVolPolicy(),
) -> dict[str, Any]:
    """Build a controlled screen without opening the formal final holdout."""

    path = Path(score_rows_path).resolve()
    rows = _read_rows(path)
    if any(bool(row.get("final_holdout")) for row in rows):
        raise ValueError("formal final-holdout rows must not enter this screen")
    required_splits = {"development", "embargo", "test"}
    if not required_splits.issubset({str(row.get("split")) for row in rows}):
        raise ValueError("score rows do not contain the frozen development/embargo/test split")

    fair_definitions = (GAUSSIAN_FAIR, EMPIRICAL_FAIR, "robust_minimum")
    primary: dict[str, Any] = {}
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for fair_definition in fair_definitions:
        selected = select_historical_opportunities(
            rows, fair_definition=fair_definition, policy=policy
        )
        primary[fair_definition] = {
            split: {
                **_summary([row for row in selected if row.split == split]),
                "candle_boundary_markouts": _historical_markouts(
                    rows, [row for row in selected if row.split == split]
                ),
            }
            for split in ("development", "test")
        }
        selected_rows[fair_definition] = [asdict(row) for row in selected]

    sensitivity: list[dict[str, Any]] = []
    for threshold in (0.00, 0.01, 0.02, 0.03, 0.05, 0.08):
        threshold_policy = LongVolPolicy(
            **{
                **asdict(policy),
                "min_net_external_edge": threshold,
            }
        )
        for fair_definition in fair_definitions:
            selected = select_historical_opportunities(
                rows,
                fair_definition=fair_definition,
                policy=threshold_policy,
            )
            sensitivity.append(
                {
                    "fair_definition": fair_definition,
                    "min_net_external_edge": threshold,
                    "development": _summary(
                        [row for row in selected if row.split == "development"]
                    ),
                    "test": _summary([row for row in selected if row.split == "test"]),
                }
            )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "exploratory_preexisting_research_only",
        "trading_eligible": False,
        "formal_final_holdout_accessed": False,
        "hypothesis": (
            "Buy the current Kalshi underdog only when an external return-distribution "
            "fair exceeds the executable underdog ask by the taker fee plus a frozen hurdle."
        ),
        "economic_exposure": (
            "OTM digital crossing probability: positive variance sensitivity in the core "
            "region, plus contrarian delta, skew, jump, and model-risk exposure; not pure vega."
        ),
        "policy": asdict(policy),
        "input": {"score_rows_path": str(path), "sha256": digest, "rows": len(rows)},
        "primary_results": primary,
        "threshold_sensitivity_not_for_test_optimization": sensitivity,
        "selected_opportunities": selected_rows,
        "limitations": [
            "The test partition contains 27 outcomes from one UTC day; its day-block confidence interval is unidentified.",
            "Historical candle boundary timestamps do not certify when the quote was receivable or executable.",
            "The frozen external fair uses provisional settlement semantics and a short volatility history.",
            "The test partition is not the untouched formal final holdout.",
            "Threshold sensitivity is diagnostic and must not be used to optimize against the test outcomes.",
            "No queue position, partial fill, order acknowledgment, or live taker latency is modeled.",
        ],
        "decision": "PROMISING_HYPOTHESIS_FORWARD_SHADOW_ONLY",
    }


def write_screen(report: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_screen(build_historical_screen(arguments.score_rows), arguments.output)
