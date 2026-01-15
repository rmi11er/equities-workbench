"""
RFQ Responder Module

Responds to Kalshi RFQ (Request for Quote) requests for combos/parlays.

This is a separate workflow from the main market maker, focused exclusively
on responding to RFQs rather than providing continuous quotes.

Key components:
- RFQResponder: Main orchestrator class
- PricingEngine: Computes combo fair values from leg prices
- RiskManager: Tracks exposure and enforces limits
- FilterConfig: Configurable RFQ filtering rules

Usage:
    from src.rfq import RFQConfig, RFQResponder, load_rfq_config

    config = load_rfq_config("config_rfq.toml")
    responder = RFQResponder(config)
    await responder.start()
"""

from .config import RFQConfig, load_rfq_config
from .types import RFQ, RFQLeg, QuoteResponse, ActiveQuote, QuoteStatus
from .pricing import PricingEngine, BBOPriceSource, PriceSource
from .risk import RiskManager
from .responder import RFQResponder, run_rfq_responder
from .decision_logger import RFQDecisionLogger, RFQDecisionRecord, QuoteOutcomeRecord

__all__ = [
    # Config
    "RFQConfig",
    "load_rfq_config",
    # Types
    "RFQ",
    "RFQLeg",
    "QuoteResponse",
    "ActiveQuote",
    "QuoteStatus",
    # Pricing
    "PricingEngine",
    "BBOPriceSource",
    "PriceSource",
    # Risk
    "RiskManager",
    # Responder
    "RFQResponder",
    "run_rfq_responder",
    # Logging
    "RFQDecisionLogger",
    "RFQDecisionRecord",
    "QuoteOutcomeRecord",
]
