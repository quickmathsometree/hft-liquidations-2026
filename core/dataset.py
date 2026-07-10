"""
DatasetBuilder — Orchestrates feature computation, transforms, sampling, and labeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.features.base import FeatureBlock
from core.sampling.samplers import Sampler, EveryTrade
from core.transforms.base import Transform


@dataclass
class DatasetBuilder:
    """
    Declarative dataset construction pipeline.

    Usage:
        builder = DatasetBuilder(
            features=[
                LiqFeatures('binance', 'buy',  halflives_s=[1.0, 5.0, 30.0],
                             count_windows_s=[5.0, 30.0]),
                LiqFeatures('binance', 'sell', halflives_s=[1.0, 5.0, 30.0],
                             count_windows_s=[5.0, 30.0]),
                LiqFeatures('bybit',   'buy',  halflives_s=[5.0, 30.0]),
                LiqFeatures('bybit',   'sell', halflives_s=[5.0, 30.0]),
                BookFeatures(depth_delta_windows_s=[1.0, 10.0]),
                FlowFeatures(windows_s=[1.0, 5.0, 30.0]),
                VolFeatures(windows_s=[5.0, 60.0, 300.0]),
                TimeFeatures(),
            ],
            transforms=[
                DirectionRelativize(),
                Winsorize(0.01),
                FillNanInf(),
            ],
            sampler=EveryTrade(),
        )
        dataset = builder.build(trades, bbo, liq_binance, liq_bybit)
    """

    features:   list[FeatureBlock] = field(default_factory=list)
    transforms: list[Transform]    = field(default_factory=list)
    sampler:    Sampler            = field(default_factory=EveryTrade)

    def build(
        self,
        trades:      pd.DataFrame,
        bbo:         pd.DataFrame,
        liq_binance: pd.DataFrame | None = None,
        liq_bybit:   pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build the feature matrix.

        Steps:
          1. Compute features from each block → concatenate columns.
          2. Apply transforms in order.
          3. Apply sampler mask → subset rows.
          4. Return the feature DataFrame (same index as sampled trades).

        Target columns (markout, pnl) are NOT computed here — they are added
        by the strategy layer using core.targets.
        """
        kwargs = dict(bbo=bbo, liq_binance=liq_binance, liq_bybit=liq_bybit)

        feature_df = pd.concat(
            [f.compute(trades, **kwargs) for f in self.features],
            axis=1,
        )

        for t in self.transforms:
            feature_df = t.apply(feature_df, trades)

        mask = self.sampler.sample_mask(trades)
        return feature_df.loc[mask]
