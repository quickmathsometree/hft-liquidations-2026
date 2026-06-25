"""
Configuration, constants, and dataclasses for the liquidation filter strategy.

All task-specific parameters live here. core/ never imports from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TAUS: tuple[int, ...] = (30, 120, 300)
SYMBOLS: tuple[str, ...] = ("btcusdt", "ethusdt")
TURNOVER_FLOOR_PER_DAY: float = 500_000.0


@dataclass
class FeatureConfig:
    """Controls which features to compute and their hyperparameters."""
    liq_halflives_s: tuple[float, ...] = (1.0, 5.0, 30.0)
    flow_windows_s: tuple[float, ...] = (1.0, 5.0, 30.0)
    book_windows_s: tuple[float, ...] = (1.0, 10.0)
    vol_windows_s: tuple[float, ...] = (60.0, 300.0)
    vol_rank_window: int = 1000
    use_opp_side_liq: bool = True


@dataclass
class FittedPipeline:
    """
    Container for everything needed at inference time.
    Created by run_train(), consumed by make_filter().
    """
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    models: dict[tuple[str, int], Any] = field(default_factory=dict)
    use_ml: bool = False
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY
