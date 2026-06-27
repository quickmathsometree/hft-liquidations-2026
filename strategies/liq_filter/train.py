"""
Training and evaluation orchestration.

Wires together: data loading → targets → features → model/strategy → threshold → scoring.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from core.data import load_data_with_required_preprocess, compute_num_days
from core.dataset import DatasetBuilder
from core.features.book import BookFeatures
from core.features.flow import FlowFeatures
from core.features.liq import LiqFeatures
from core.features.time import TimeFeatures
from core.features.vol import VolFeatures
from core.model import train_model, predict
from core.sampling.samplers import EveryTrade
from core.targets.markout import add_mid, compute_markout
from core.targets.pnl import compute_pnl
from core.transforms.direction import DirectionRelativize
from core.transforms.normalize import FillNanInf

from strategies.liq_filter.config import (
    TAUS, SYMBOLS, TURNOVER_FLOOR_PER_DAY, FeatureConfig, FittedPipeline,
)
from strategies.liq_filter.scoring import ScoreReport, score_all, reports_to_frame
from strategies.liq_filter.strategy import strategy_raw_score
from strategies.liq_filter.threshold import fit_threshold, apply_filter


Symbol = Literal["btcusdt", "ethusdt"]


def _build_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    builder = DatasetBuilder(
        features=[
            LiqFeatures("binance", "buy", halflives_s=cfg.liq_halflives_s, count_windows_s=cfg.flow_windows_s),
            LiqFeatures("binance", "sell", halflives_s=cfg.liq_halflives_s, count_windows_s=cfg.flow_windows_s),
            LiqFeatures("bybit", "buy", halflives_s=cfg.liq_halflives_s, count_windows_s=cfg.flow_windows_s),
            LiqFeatures("bybit", "sell", halflives_s=cfg.liq_halflives_s, count_windows_s=cfg.flow_windows_s),
            BookFeatures(depth_delta_windows_s=cfg.book_windows_s),
            FlowFeatures(windows_s=cfg.flow_windows_s),
            TimeFeatures(),
            VolFeatures(windows_s=cfg.vol_windows_s, rank_window=cfg.vol_rank_window),
        ],
        transforms=[
            DirectionRelativize(),
            FillNanInf(),
        ],
        sampler=EveryTrade(),
    )

    X = builder.build(
        trades=trades,
        bbo=bbo,
        liq_binance=liq_binance,
        liq_bybit=liq_bybit,
    )

    if not cfg.use_opp_side_liq:
        drop_cols = [
            c for c in X.columns
            if str(c).startswith("liq_") and "_opp_" in str(c)
        ]
        X = X.drop(columns=drop_cols)

    return X


def run_train(
    data_dir: str,
    use_ml: bool = False,
    symbols: tuple[str, ...] = SYMBOLS,
    feature_config: FeatureConfig | None = None,
    model_params: dict | None = None,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
    verbose: bool = False,
) -> FittedPipeline:
    """
    Train on the train split for the given symbols. Produces a FittedPipeline.

    Threshold is NOT fitted here — calibrated at inference time via fit_threshold.
    """
    cfg = feature_config or FeatureConfig()

    fitted = FittedPipeline(
        feature_config=cfg,
        models={},
        use_ml=use_ml,
        target_turnover_per_day=float(target_turnover_per_day),
    )

    if not use_ml:
        for symbol in symbols:
            for tau in TAUS:
                fitted.models[(symbol, int(tau))] = None
        return fitted

    for symbol in symbols:
        if verbose:
            print(f"[train] loading {symbol}", flush=True)

        trades, bbo, liq_binance, liq_bybit = load_data_with_required_preprocess(
            data_dir=data_dir,
            symbol=symbol,  # type: ignore[arg-type]
            split="train",
        )

        bbo = add_mid(bbo)
        trades = compute_markout(trades, bbo, TAUS)
        trades = compute_pnl(trades, TAUS)

        X = _build_features(trades, bbo, liq_binance, liq_bybit, cfg)
        w = trades.loc[X.index, "w"]

        for tau in TAUS:
            y = trades.loc[X.index, f"pnl_{tau}"]
            ok = y.notna() & w.notna()

            if verbose:
                print(f"[train] {symbol} tau={tau} rows={int(ok.sum()):,}", flush=True)

            model = train_model(
                features=X.loc[ok],
                target=y.loc[ok],
                sample_weight=w.loc[ok],
                model_params=model_params,
            )

            fitted.models[(symbol, int(tau))] = model

    return fitted


def run_eval(
    data_dir: str,
    split: str,
    fitted: FittedPipeline,
    symbols: tuple[str, ...] = SYMBOLS,
    verbose: bool = False,
) -> dict[str, dict[int, ScoreReport]]:
    """
    Evaluate on the given split using a pre-trained FittedPipeline.

    Returns {symbol: {tau: ScoreReport}}.
    ONE-SHOT evaluation — do NOT iterate on the threshold to improve val Score.
    """
    result: dict[str, dict[int, ScoreReport]] = {}

    for symbol in symbols:
        if verbose:
            print(f"[eval] loading {symbol} split={split}", flush=True)

        trades, bbo, liq_binance, liq_bybit = load_data_with_required_preprocess(
            data_dir=data_dir,
            symbol=symbol,  # type: ignore[arg-type]
            split=split,
        )

        num_days = compute_num_days(trades)

        bbo = add_mid(bbo)
        trades = compute_markout(trades, bbo, TAUS)
        trades = compute_pnl(trades, TAUS)

        X = _build_features(trades, bbo, liq_binance, liq_bybit, fitted.feature_config)
        scored = trades.loc[X.index].copy()
        w = scored["w"].to_numpy(dtype="float64")

        f_by_tau: dict[int, np.ndarray] = {}

        for tau in TAUS:
            model = fitted.models.get((symbol, int(tau)))

            if fitted.use_ml:
                if model is None:
                    raise ValueError(f"No fitted model for {(symbol, tau)}")
                raw_score = predict(model, X)
            else:
                raw_score = strategy_raw_score(X, scored)

            threshold = fit_threshold(
                raw_score=raw_score,
                w=w,
                num_days=num_days,
                target_turnover_per_day=fitted.target_turnover_per_day,
            )

            edge_mask = scored[f"edge_{tau}"].to_numpy(dtype=bool)

            f_by_tau[int(tau)] = apply_filter(
                raw_score=raw_score,
                threshold=threshold,
                edge_mask=edge_mask,
            )

            if verbose:
                kept = int((f_by_tau[int(tau)] == 0).sum())
                filt = int((f_by_tau[int(tau)] == 1).sum())
                print(
                    f"[eval] {symbol} tau={tau} threshold={threshold:.6g} "
                    f"kept={kept:,} filtered={filt:,}",
                    flush=True,
                )

        result[symbol] = score_all(scored, f_by_tau, num_days)

    return result


def run_experiment(
    data_dir: str,
    name: str = "experiment",
    use_ml: bool = False,
    symbols: tuple[str, ...] = SYMBOLS,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
    model_params: dict | None = None,
    verbose: bool = False,
) -> tuple[FittedPipeline, dict, pd.DataFrame]:
    """
    Full experiment: train on train split, evaluate on both train and validation.
    Returns (fitted_pipeline, raw_reports_dict, summary_dataframe).
    """
    fitted = run_train(
        data_dir=data_dir,
        use_ml=use_ml,
        symbols=symbols,
        model_params=model_params,
        target_turnover_per_day=target_turnover_per_day,
        verbose=verbose,
    )

    reports = {
        "train": run_eval(data_dir, "train", fitted, symbols=symbols, verbose=verbose),
        "validation": run_eval(data_dir, "validation", fitted, symbols=symbols, verbose=verbose),
    }

    frames: list[pd.DataFrame] = []

    for split, by_symbol in reports.items():
        for symbol, rep in by_symbol.items():
            df = reports_to_frame(
                rep,
                symbol=symbol,
                experiment=f"{name}_{split}",
            )
            df.insert(0, "split", split)
            frames.append(df)

    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return fitted, reports, summary