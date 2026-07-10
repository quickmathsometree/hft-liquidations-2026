"""
Base class for transform blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Transform(ABC):

    @abstractmethod
    def apply(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the feature matrix.

        Parameters
        ----------
        features : feature DataFrame (columns = feature names)
        trades   : original trades frame — available for context (e.g. side for direction)

        Returns
        -------
        Transformed DataFrame, same index as features.
        """
        ...
