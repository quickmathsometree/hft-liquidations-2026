"""
Base class for feature blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureBlock(ABC):

    @abstractmethod
    def compute(self, trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute feature columns for each trade.

        Parameters
        ----------
        trades   : DataFrame with at least [timestamp, side, price, amount]
        **kwargs : additional frames — bbo=, liq_binance=, liq_bybit=

        Returns
        -------
        DataFrame with same length/index as trades, containing only feature columns.
        """
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """List of column names this block produces."""
        ...
