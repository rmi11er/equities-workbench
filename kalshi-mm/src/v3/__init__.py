"""
Passive Market Maker v3

Simple, defensive liquidity provision:
- Only quote markets with sufficient spread
- Always join existing depth (never be alone at front)
- Gently fade inventory toward neutral
- Exception: can be alone on inventory-reducing side with edge
"""

from .config import ConfigV3, load_config_v3
from .quoter import PassiveQuoter

__all__ = ["ConfigV3", "load_config_v3", "PassiveQuoter"]
