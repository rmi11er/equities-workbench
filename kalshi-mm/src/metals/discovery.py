"""Dynamic metals market discovery with config hints and no rule hard-coding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import MetalsConfig, ProductConfig
from .kalshi import KalshiReadClient, ReceivedResponse


@dataclass(frozen=True)
class DiscoveredMarket:
    metal: str
    raw: Dict[str, Any]
    series_ticker: str = ""

    @property
    def ticker(self) -> str:
        return str(self.raw.get("ticker", self.raw.get("market_ticker", "")))


def market_matches_product(raw: Dict[str, Any], product: ProductConfig) -> bool:
    ticker = str(raw.get("ticker", raw.get("market_ticker", ""))).upper()
    series = str(raw.get("series_ticker", "")).upper()
    if series and series in product.series_tickers:
        return True
    if any(ticker.startswith(prefix) for prefix in product.ticker_prefixes):
        return True
    searchable = " ".join(
        str(raw.get(name, "")).lower()
        for name in ("title", "subtitle", "rules_primary", "rules_secondary")
    )
    return bool(product.title_keywords) and any(
        keyword in searchable for keyword in product.title_keywords
    ) and ("15 minute" in searchable or "15-minute" in searchable or "15m" in ticker.lower())


async def discover_open_markets(
    client: KalshiReadClient, config: MetalsConfig
) -> Tuple[List[DiscoveredMarket], List[ReceivedResponse]]:
    """Fetch open markets by configured series, falling back to broad discovery."""

    found: Dict[str, DiscoveredMarket] = {}
    responses: List[ReceivedResponse] = []
    attempted_series = False

    for metal, product in config.enabled_products.items():
        for series_ticker in product.series_tickers:
            attempted_series = True
            for status in ("open", "unopened"):
                pages = await client.get_all_pages(
                    "/markets",
                    "markets",
                    {"status": status, "series_ticker": series_ticker},
                )
                responses.extend(pages)
                for page in pages:
                    for raw in page.payload.get("markets", []):
                        if market_matches_product(raw, product):
                            normalized_raw = dict(raw)
                            normalized_raw.setdefault("series_ticker", series_ticker)
                            discovered = DiscoveredMarket(
                                metal=metal,
                                raw=normalized_raw,
                                series_ticker=series_ticker,
                            )
                            if discovered.ticker:
                                found[discovered.ticker] = discovered

    if found or attempted_series:
        return sorted(found.values(), key=lambda value: value.ticker), responses

    for status in ("open", "unopened"):
        pages = await client.get_all_pages("/markets", "markets", {"status": status})
        responses.extend(pages)
        for page in pages:
            for raw in page.payload.get("markets", []):
                for metal, product in config.enabled_products.items():
                    if market_matches_product(raw, product):
                        discovered = DiscoveredMarket(
                            metal=metal,
                            raw=dict(raw),
                            series_ticker=str(raw.get("series_ticker", "")),
                        )
                        if discovered.ticker:
                            found[discovered.ticker] = discovered
                        break
    return sorted(found.values(), key=lambda value: value.ticker), responses
