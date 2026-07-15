# utilities.py - summary tables and plots for the walk-forward result notebooks.
#
# Consumed by 2/3.*_Results.ipynb and 6/7.*_Results_Synthetic.ipynb as:
#     import utilities as util
#
# Public API (signatures match the notebook call sites exactly):
#     build_wf_summary_table(results, max_filter_rate)          -> pd.DataFrame
#     plot_score_by_window_one_chart(summary_table, variant,
#                                    taus, title)               -> pd.DataFrame (pivot)
#     plot_daily_score_distribution(results, variant, tau,
#                                   max_filter_rate,
#                                   instrument, model_name)     -> pd.DataFrame (per day)
#     plot_per_trade_distribution(results, variant, tau,
#                                 max_filter_rate,
#                                 instrument, model_name)       -> pd.DataFrame (stats)
#     plot_lgbm_feature_importance(models, top_n, ...)          -> pd.DataFrame
#
# `results` is the notebooks' dict[(variant: str, tau: int) -> WalkForwardResult].
# All Score/PnL numbers are recomputed via the pipeline's own scoring functions -
# this module adds no metric logic of its own.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from experiment import WalkForwardResult, collect_oos_arrays
from scoring import (
    compute_daily_scores_from_arrays,
    compute_daily_pnl_kept_from_arrays,
)

US_PER_DAY = 86_400 * 1_000_000


# ---------------------------------------------------------------------------
# summary table.
# ---------------------------------------------------------------------------
def build_wf_summary_table(
    results: dict[tuple[str, int], WalkForwardResult],
    max_filter_rate: float,
) -> pd.DataFrame:
    """
    One row per (variant, tau, window) at the selected max_filter_rate.

    Columns: variant, tau, window, max_filter_rate, pnl_all, pnl_kept,
    pnl_filtered, score, "kept_turnover_per_day, mln$", turnover_ok, filter_rate.
    """
    rows = []

    for (variant, tau), result in results.items():
        for r in result.windows:
            if not np.isclose(r.max_filter_rate, max_filter_rate):
                continue

            rows.append({
                "variant": variant,
                "tau": int(tau),
                "window": r.window_id,
                "max_filter_rate": r.max_filter_rate,
                "pnl_all": r.test_pnl_all,
                "pnl_kept": r.test_pnl_kept,
                "pnl_filtered": r.test_pnl_filtered,
                "score": r.test_score,
                "kept_turnover_per_day, mln$": round(
                    r.test_kept_turnover_per_day / 1e6, 2
                ),
                "turnover_ok": bool(r.test_turnover_ok),
                "filter_rate": round(r.test_filter_rate, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# score by window, one chart.
# ---------------------------------------------------------------------------
def plot_score_by_window_one_chart(
    summary_table: pd.DataFrame,
    variant: str,
    taus: tuple[int, ...] = (30, 120, 300),
    title: str = "OOS score by window",
) -> pd.DataFrame:
    """
    Line chart: OOS test score per window, one line per tau. Returns the
    window x tau pivot of scores.
    """
    sub = summary_table[
        (summary_table["variant"] == variant)
        & (summary_table["tau"].isin(list(taus)))
    ]

    pivot = sub.pivot_table(index="window", columns="tau", values="score")
    pivot = pivot.reindex(columns=list(taus))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pivot.index))

    for tau in taus:
        if tau not in pivot.columns:
            continue
        ax.plot(x, pivot[tau].values, "o-", lw=1.8, ms=6, label=f"tau={tau}s")

    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("window")
    ax.set_ylabel("OOS score, bps")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return pivot


# ---------------------------------------------------------------------------
# internals shared by the distribution plots.
# ---------------------------------------------------------------------------
def _oos_arrays(results, variant, tau, max_filter_rate):
    result = results[(variant, int(tau))]
    return collect_oos_arrays(result, max_filter_rate=max_filter_rate)


def _scoring_valid_mask(y, w, ts, f):
    """Same validity rule the scoring functions apply internally - keeps the
    day axis here aligned with the daily arrays they return."""
    return (
        np.isfinite(y) & np.isfinite(w) & (w > 0)
        & np.isfinite(ts) & np.isfinite(f)
    )


def _oos_dates(y, w, ts, f):
    valid = _scoring_valid_mask(y, w, ts, f)
    days = np.unique(ts[valid] // US_PER_DAY)
    return pd.to_datetime(days, unit="D")


def _run_tag(instrument, model_name, variant, tau, max_filter_rate):
    return f"{instrument.upper()}_{model_name}_{variant}_tau{tau}s_fr{max_filter_rate}"


def _print_daily_stats(tag: str, label: str, x: np.ndarray) -> None:
    n = len(x)
    n_pos = int((x > 0).sum())
    n_neg = int((x < 0).sum())
    std = np.std(x, ddof=1) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    t_stat = np.mean(x) / sem if sem > 0 else float("nan")
    print(tag)
    print(f"{label}:")
    print(f"  mean   = {np.mean(x):+f}")
    print(f"  std    = {std:f}")
    print(f"  t-stat = {t_stat:+.2f}  (mean / sem, {n} days)")
    print(f"  median = {np.median(x):+f}")
    print(f"  min    = {np.min(x):+f}")
    print(f"  max    = {np.max(x):+f}")
    print(f"  >0 days: {n_pos}/{n} ({round(100.0 * n_pos / n)}%)")
    print(f"  <0 days: {n_neg}/{n} ({round(100.0 * n_neg / n)}%)")
    print()


# ---------------------------------------------------------------------------
# daily score distribution.
# ---------------------------------------------------------------------------
def plot_daily_score_distribution(
    results: dict[tuple[str, int], WalkForwardResult],
    variant: str,
    tau: int,
    max_filter_rate: float,
    instrument: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Daily OOS Score and daily PnL_kept across all windows (concatenated test
    periods). Prints summary stats and draws a 2x2 panel:
    time series + histogram for each of the two daily series.
    Returns a per-day DataFrame.
    """
    y, w, ts, f = _oos_arrays(results, variant, tau, max_filter_rate)

    daily_score = compute_daily_scores_from_arrays(y=y, w=w, ts=ts, f=f)
    daily_kept = compute_daily_pnl_kept_from_arrays(y=y, w=w, ts=ts, f=f)
    # Same valid-row rule as the scoring functions, so dates align 1:1 with
    # the daily arrays.
    dates = _oos_dates(y, w, ts, f)

    tag = _run_tag(instrument, model_name, variant, tau, max_filter_rate)
    _print_daily_stats(tag, "Daily Score", daily_score)
    _print_daily_stats(tag, "Daily PnL kept", daily_kept)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(dates, daily_score, "o-", ms=3, lw=1, color="tab:blue")
    axes[0, 0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0, 0].set_title("Daily Score over time")
    axes[0, 0].set_ylabel("Score, bps")

    axes[0, 1].hist(daily_score, bins=40, color="tab:blue", alpha=0.75)
    axes[0, 1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[0, 1].axvline(np.mean(daily_score), color="red", lw=1.2, ls="--",
                       label=f"mean {np.mean(daily_score):+.2f}")
    axes[0, 1].set_title("Daily Score distribution")
    axes[0, 1].legend()

    axes[1, 0].plot(dates, daily_kept, "o-", ms=3, lw=1, color="tab:green")
    axes[1, 0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1, 0].set_title("Daily PnL kept over time")
    axes[1, 0].set_ylabel("PnL kept, bps")

    axes[1, 1].hist(daily_kept, bins=40, color="tab:green", alpha=0.75)
    axes[1, 1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1, 1].axvline(np.mean(daily_kept), color="red", lw=1.2, ls="--",
                       label=f"mean {np.mean(daily_kept):+.2f}")
    axes[1, 1].set_title("Daily PnL kept distribution")
    axes[1, 1].legend()

    # Cumulative score and drawdown.
    cum_score = np.cumsum(daily_score)
    running_max = np.maximum.accumulate(cum_score)
    drawdown = cum_score - running_max

    axes[2, 0].plot(dates, cum_score, lw=1.5, color="tab:blue")
    axes[2, 0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[2, 0].set_title("Cumulative daily Score")
    axes[2, 0].set_ylabel("bps (cum)")

    axes[2, 1].fill_between(dates, drawdown, 0, color="tab:red", alpha=0.5)
    axes[2, 1].set_title(
        f"Drawdown of cumulative Score (max {drawdown.min():.2f} bps)"
    )
    axes[2, 1].set_ylabel("bps")

    for ax in (axes[0, 0], axes[1, 0], axes[2, 0], axes[2, 1]):
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.suptitle(tag)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "day": dates,
        "daily_score": daily_score,
        "daily_pnl_kept": daily_kept,
    })


# ---------------------------------------------------------------------------
# per-trade distribution.
# ---------------------------------------------------------------------------
def plot_per_trade_distribution(
    results: dict[tuple[str, int], WalkForwardResult],
    variant: str,
    tau: int,
    max_filter_rate: float,
    instrument: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Per-trade PnL distributions for all / kept / filtered OOS trades. Prints
    weighted aggregates and unweighted per-trade stats, draws a 2x2 panel.
    Returns a small stats DataFrame (one row per group).
    """
    y, w, ts, f = _oos_arrays(results, variant, tau, max_filter_rate)

    kept_mask = f == 0
    filt_mask = f == 1

    n_all = len(y)
    n_kept = int(kept_mask.sum())
    n_filt = int(filt_mask.sum())

    def _wmean(yy, ww):
        s = ww.sum()
        return float((ww * yy).sum() / s) if s > 0 else float("nan")

    pnl_all = _wmean(y, w)
    pnl_kept = _wmean(y[kept_mask], w[kept_mask])
    pnl_filt = _wmean(y[filt_mask], w[filt_mask])
    score = pnl_kept - pnl_all

    print("=" * 80)
    print(f"{instrument.upper()} | {model_name} | {variant} | tau={tau}s | "
          f"max_filter_rate={max_filter_rate}")
    print("-" * 80)
    print(f"n_all      = {n_all:,}")
    print(f"n_kept     = {n_kept:,} ({100.0 * n_kept / n_all:.2f}%)")
    print(f"n_filtered = {n_filt:,} ({100.0 * n_filt / n_all:.2f}%)")
    print()
    print("Weighted PnL:")
    print(f"  pnl_all      = {pnl_all:+f} bps")
    print(f"  pnl_kept     = {pnl_kept:+f} bps")
    print(f"  pnl_filtered = {pnl_filt:+f} bps")
    print(f"  score        = {score:+f} bps")
    print()
    print("Unweighted per-trade PnL:")

    rows = []
    for label, mask in [("all", np.ones(n_all, dtype=bool)),
                        ("kept", kept_mask),
                        ("filtered", filt_mask)]:
        yy = y[mask]
        neg_pct = 100.0 * float((yy < 0).mean()) if len(yy) else float("nan")
        print(f"  {label:<8s} mean={np.mean(yy):+f}, "
              f"median={np.median(yy):+f}, <0={neg_pct:.1f}%")
        rows.append({
            "group": label,
            "n": int(mask.sum()),
            "weighted_pnl": _wmean(yy, w[mask]),
            "mean": float(np.mean(yy)),
            "median": float(np.median(yy)),
            "pct_negative": neg_pct,
        })
    print()

    # 2x2 panel: kept vs all overlay, kept alone, filtered, weighted-contribution.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    clip = np.nanpercentile(np.abs(y), 99.5)
    bins = np.linspace(-clip, clip, 80)

    axes[0, 0].hist(y, bins=bins, color="gray", alpha=0.6, label="all", density=True)
    axes[0, 0].hist(y[kept_mask], bins=bins, color="tab:green", alpha=0.6,
                    label="kept", density=True)
    axes[0, 0].set_title("Per-trade PnL: all vs kept (density)")
    axes[0, 0].legend()

    axes[0, 1].hist(y[kept_mask], bins=60, color="tab:green", alpha=0.8)
    axes[0, 1].axvline(np.mean(y[kept_mask]) if n_kept else 0, color="red",
                       lw=1.2, ls="--")
    axes[0, 1].set_title(f"Kept trades (n={n_kept:,})")

    axes[1, 0].hist(y[filt_mask], bins=bins, color="tab:red", alpha=0.7)
    axes[1, 0].set_title(f"Filtered trades (n={n_filt:,})")
    axes[1, 0].set_yscale("log")

    contrib = w * y
    axes[1, 1].hist(contrib[kept_mask], bins=60, color="tab:green", alpha=0.6,
                    label="kept")
    axes[1, 1].hist(contrib[filt_mask], bins=60, color="tab:red", alpha=0.4,
                    label="filtered")
    axes[1, 1].set_title("Weighted contribution w*pnl")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{instrument.upper()} {model_name} {variant} tau={tau}s "
        f"fr={max_filter_rate}: per-trade PnL"
    )
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LGBM feature importance across windows.
# ---------------------------------------------------------------------------
def _extract_boosters(model_obj):
    """Return a list of fitted LGBM models inside a train_fn output."""
    if hasattr(model_obj, "models"):                    # LGBMEnsembleRegressor
        out = []
        for m in model_obj.models:
            out.extend(_extract_boosters(m))
        return out
    if isinstance(model_obj, dict) and "model" in model_obj:
        return _extract_boosters(model_obj["model"])
    if hasattr(model_obj, "booster_"):                  # LGBMRegressor / Classifier
        return [model_obj]
    return []


def plot_lgbm_feature_importance(
    models: list[dict],
    top_n: int = 30,
    importance_type: str = "gain",
    interval: str = "std",
    normalize_per_model: bool = True,
    figsize: tuple[float, float] = (10, 8),
    title: str = "LGBM feature importance",
) -> pd.DataFrame:
    """
    Aggregate LGBM feature importances across per-window models
    (`models` = WalkForwardResult.models, a list of model_info dicts).

    Returns a DataFrame sorted by mean_importance with columns:
    feature, mean_importance, std_importance, n_models, sem_importance, interval.
    `interval` holds the chosen half-width: std_importance ("std") or
    sem_importance ("sem").
    """
    per_model = []
    feature_names = None

    for info in models:
        names = info.get("features")
        boosters = _extract_boosters(info.get("model"))
        for b in boosters:
            imp = np.asarray(
                b.booster_.feature_importance(importance_type=importance_type),
                dtype=np.float64,
            )
            if normalize_per_model and imp.sum() > 0:
                imp = imp / imp.sum()
            per_model.append(imp)
            if feature_names is None:
                feature_names = list(names)

    if not per_model:
        raise ValueError(
            "No LGBM models found in `models` - feature importance is only "
            "available for LGBM-based model families."
        )

    M = np.vstack(per_model)                     # (n_models, n_features)
    n_models = M.shape[0]
    mean_imp = M.mean(axis=0)
    std_imp = M.std(axis=0, ddof=1) if n_models > 1 else np.zeros_like(mean_imp)
    sem_imp = std_imp / np.sqrt(n_models)

    fi = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": mean_imp,
        "std_importance": std_imp,
        "n_models": n_models,
        "sem_importance": sem_imp,
    })
    fi["interval"] = fi["sem_importance"] if interval == "sem" else fi["std_importance"]
    fi = fi.sort_values("mean_importance", ascending=False)

    top = fi.head(top_n).iloc[::-1]              # reversed for barh (best on top).

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        top["feature"], top["mean_importance"],
        xerr=top["interval"], color="#2c7fb8", alpha=0.85,
        error_kw={"ecolor": "#444", "capsize": 2, "lw": 1},
    )
    ax.set_xlabel(
        f"{importance_type} importance"
        + (" (normalized per model)" if normalize_per_model else "")
    )
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()

    return fi


# ---------------------------------------------------------------------------
# paired variant comparison (ablations: e.g. base vs base + diffusion).
# ---------------------------------------------------------------------------
def compare_variants_daily(
    results: dict[tuple[str, int], WalkForwardResult],
    variant_a: str,
    variant_b: str,
    tau: int,
    max_filter_rate: float,
    instrument: str = "",
    model_name: str = "",
) -> pd.DataFrame:
    """
    Paired daily-score comparison of two variants on the same OOS days.

    Prints a paired t-test on (B - A), plots both cumulative curves and the
    cumulative difference. Returns a per-day DataFrame with both scores and
    the difference.
    """
    frames = {}
    for variant in (variant_a, variant_b):
        y, w, ts, f = _oos_arrays(results, variant, tau, max_filter_rate)
        daily = compute_daily_scores_from_arrays(y=y, w=w, ts=ts, f=f)
        frames[variant] = pd.Series(daily, index=_oos_dates(y, w, ts, f))

    joined = pd.DataFrame({
        variant_a: frames[variant_a],
        variant_b: frames[variant_b],
    }).dropna()
    joined["diff"] = joined[variant_b] - joined[variant_a]

    d = joined["diff"].to_numpy()
    n = len(d)
    sem = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t_stat = d.mean() / sem if sem > 0 else float("nan")

    tag = f"{instrument.upper()} {model_name} tau={tau}s fr={max_filter_rate}"
    print(tag)
    print(f"Paired daily score, {n} common days:")
    print(f"  {variant_a:<24s} mean = {joined[variant_a].mean():+f}")
    print(f"  {variant_b:<24s} mean = {joined[variant_b].mean():+f}")
    print(f"  diff (B - A)             mean = {d.mean():+f}")
    print(f"  paired t-stat = {t_stat:+.2f}, "
          f"B wins {int((d > 0).sum())}/{n} days "
          f"({100.0 * (d > 0).mean():.0f}%)")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(joined.index, joined[variant_a].cumsum(), lw=1.5,
                 label=variant_a, color="tab:gray")
    axes[0].plot(joined.index, joined[variant_b].cumsum(), lw=1.5,
                 label=variant_b, color="tab:blue")
    axes[0].set_title("Cumulative daily Score")
    axes[0].set_ylabel("bps (cum)")
    axes[0].legend()

    axes[1].plot(joined.index, joined["diff"].cumsum(), lw=1.5,
                 color="tab:purple")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_title(
        f"Cumulative difference ({variant_b} - {variant_a}), "
        f"t = {t_stat:+.2f}"
    )
    axes[1].set_ylabel("bps (cum)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(tag)
    plt.tight_layout()
    plt.show()

    return joined.reset_index(names="day")


# ---------------------------------------------------------------------------
# offline pilot scripts (btc_pilot*.py) -> notebook plots.
# ---------------------------------------------------------------------------
def load_pilot_results(path: str | Path) -> pd.DataFrame:
    """
    Load a pilot*_results.parquet written by btc_pilot*.py.

    Expected columns: variant, tau, window, max_fr, test_score, test_fr
    (pilot4/5 omit tau; pilot6 includes it).
    """
    df = pd.read_parquet(path)
    if "max_fr" in df.columns and "max_filter_rate" not in df.columns:
        df = df.rename(columns={"max_fr": "max_filter_rate"})
    return df


def pilot_aggregate_summary(
    pilot_df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("variant", "tau", "max_filter_rate"),
) -> pd.DataFrame:
    """Mean / min / std of test_score and mean filter rate per group."""
    g = pilot_df.groupby(list(group_cols), as_index=False)
    return g.agg(
        mean_score=("test_score", "mean"),
        worst_window_score=("test_score", "min"),
        std_score=("test_score", "std"),
        mean_filter_rate=("test_fr", "mean"),
        n_windows=("window", "count"),
    )


def pilot_consistency_table(
    pilot_df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("variant", "tau", "max_filter_rate"),
) -> pd.DataFrame:
    """
    Robustness view per config: mean / std / worst per-window OOS score,
    number of positive windows, mean/std ratio, mean achieved filter rate.

    mean_over_std is a crude per-window Sharpe of the config across W1-W5:
    configs with high mean but low mean_over_std win on a couple of lucky
    windows rather than consistently.
    """
    g = pilot_df.groupby(list(group_cols), as_index=False)
    out = g.agg(
        mean_score=("test_score", "mean"),
        std_score=("test_score", "std"),
        worst_window=("test_score", "min"),
        pos_windows=("test_score", lambda s: int((s > 0).sum())),
        n_windows=("test_score", "size"),
        mean_filter_rate=("test_fr", "mean"),
    )
    out["mean_over_std"] = out["mean_score"] / out["std_score"]
    return out.sort_values(["max_filter_rate", "tau", "variant"]).reset_index(drop=True)


def pilot_paired_delta_table(
    pilot_df: pd.DataFrame,
    variant_a: str = "advanced",
    variant_b: str = "extended",
) -> pd.DataFrame:
    """
    Paired per-window comparison (variant_b - variant_a) per (tau, max_filter_rate).

    Pairing on the window removes the shared window-to-window variance, so
    b_wins ("how many windows variant_b is ahead") is a more honest signal
    than the difference of the means.
    """
    p = pilot_df.pivot_table(
        index=["tau", "max_filter_rate", "window"],
        columns="variant", values="test_score",
    ).reset_index()
    if variant_a not in p.columns or variant_b not in p.columns:
        raise KeyError(f"variants {variant_a!r} / {variant_b!r} not found in pilot_df")
    p["delta"] = p[variant_b] - p[variant_a]
    g = p.groupby(["tau", "max_filter_rate"], as_index=False)
    out = g.agg(
        mean_delta=("delta", "mean"),
        worst_delta=("delta", "min"),
        b_wins=("delta", lambda s: int((s > 0).sum())),
        n_windows=("delta", "size"),
    )
    return out.sort_values(["max_filter_rate", "tau"]).reset_index(drop=True)


def plot_pilot_score_by_window(
    pilot_df: pd.DataFrame,
    variant: str,
    max_filter_rate: float,
    taus: tuple[int, ...] = (30, 120, 300),
    title: str = "Pilot OOS score by window",
) -> pd.DataFrame:
    """Line chart: one line per tau for a single variant. Returns window x tau pivot."""
    sub = pilot_df[
        (pilot_df["variant"] == variant)
        & np.isclose(pilot_df["max_filter_rate"], max_filter_rate)
        & pilot_df["tau"].isin(list(taus))
    ]
    pivot = sub.pivot_table(index="window", columns="tau", values="test_score")
    pivot = pivot.reindex(columns=list(taus))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pivot.index))
    for tau in taus:
        if tau not in pivot.columns:
            continue
        ax.plot(x, pivot[tau].values, "o-", lw=1.8, ms=6, label=f"tau={tau}s")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("window")
    ax.set_ylabel("OOS score, bps")
    ax.set_title(f"{title} | {variant} | fr={max_filter_rate}")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return pivot


def plot_pilot_variant_comparison(
    pilot_df: pd.DataFrame,
    max_filter_rate: float,
    taus: tuple[int, ...] = (30, 120, 300),
    variant_a: str = "advanced",
    variant_b: str = "extended",
) -> pd.DataFrame:
    """
    Grouped bar chart of mean OOS score (variant_a vs variant_b) per tau,
    plus a delta panel. Returns the aggregated summary table.
    """
    agg = pilot_aggregate_summary(pilot_df)
    sub = agg[
        agg["variant"].isin([variant_a, variant_b])
        & np.isclose(agg["max_filter_rate"], max_filter_rate)
        & agg["tau"].isin(list(taus))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(taus))
    width = 0.35

    for i, variant in enumerate((variant_a, variant_b)):
        vals = [
            sub.loc[(sub["variant"] == variant) & (sub["tau"] == tau), "mean_score"].iloc[0]
            if len(sub.loc[(sub["variant"] == variant) & (sub["tau"] == tau)]) else np.nan
            for tau in taus
        ]
        axes[0].bar(x + (i - 0.5) * width, vals, width, label=variant)

    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"tau={t}s" for t in taus])
    axes[0].set_ylabel("mean OOS score, bps")
    axes[0].set_title(f"Mean score across windows (fr={max_filter_rate})")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    pivot = sub.pivot_table(index="tau", columns="variant", values="mean_score")
    if variant_a in pivot.columns and variant_b in pivot.columns:
        delta = pivot[variant_b] - pivot[variant_a]
        colors = ["tab:green" if v >= 0 else "tab:red" for v in delta.values]
        axes[1].bar([f"tau={t}s" for t in delta.index], delta.values, color=colors)
        axes[1].axhline(0, color="gray", lw=0.8, ls="--")
        axes[1].set_ylabel(f"delta ({variant_b} - {variant_a}), bps")
        axes[1].set_title("Extended minus advanced")
        axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()
    return sub.sort_values(["tau", "variant"])


def load_pilot_oos(
    npz_path: str | Path,
    meta_path: str | Path,
    variant: str,
    tau: int,
    max_filter_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load concatenated OOS (y, w, ts, f) saved by btc_pilot6."""
    meta = pd.read_parquet(meta_path)
    row = meta[
        (meta["variant"] == variant)
        & (meta["tau"] == int(tau))
        & np.isclose(meta["max_fr"], max_filter_rate)
    ]
    if row.empty:
        raise KeyError(
            f"No OOS arrays for variant={variant!r}, tau={tau}, "
            f"max_filter_rate={max_filter_rate}"
        )
    tag = row.iloc[0]["tag"]
    data = np.load(npz_path)
    return data[f"{tag}/y"], data[f"{tag}/w"], data[f"{tag}/ts"], data[f"{tag}/f"]


def pilot_paired_daily_table(
    npz_path: str | Path,
    meta_path: str | Path,
    pairs: list[tuple[str, str, int]],
    max_filter_rates: tuple[float, ...] = (0.10, 0.30),
) -> pd.DataFrame:
    """
    Paired daily-score t-test challenger vs control on common OOS days.

    pairs: list of (challenger, control, tau). Both configs must be present
    in the pilot oos npz. The paired test removes the shared day-to-day
    variance, so it is the honest way to compare two configs at the same tau.
    """
    def _daily(variant, tau, fr):
        y, w, ts, f = load_pilot_oos(npz_path, meta_path, variant, tau, fr)
        return compute_daily_scores_from_arrays(y=y, w=w, ts=ts, f=f)

    rows = []
    for challenger, control, tau in pairs:
        for fr in max_filter_rates:
            da = _daily(challenger, tau, fr)
            db = _daily(control, tau, fr)
            n = min(len(da), len(db))
            d = da[:n] - db[:n]
            rows.append({
                "challenger": challenger,
                "control": control,
                "tau": tau,
                "max_filter_rate": fr,
                "mean_daily_diff": d.mean(),
                "paired_t": d.mean() / (d.std(ddof=1) / np.sqrt(n)),
                "share_days_ahead": float((d > 0).mean()),
                "n_days": n,
            })
    return pd.DataFrame(rows)


def plot_pilot_daily_score(
    npz_path: str | Path,
    meta_path: str | Path,
    variant: str,
    tau: int,
    max_filter_rate: float,
    instrument: str = "btc",
    model_name: str = "pilot6_ens_rank",
) -> pd.DataFrame:
    """
    Daily OOS score / PnL_kept for an offline pilot run (btc_pilot6 oos npz).
    Same 2x2 + cumulative panels as plot_daily_score_distribution.
    """
    y, w, ts, f = load_pilot_oos(npz_path, meta_path, variant, tau, max_filter_rate)
    daily_score = compute_daily_scores_from_arrays(y=y, w=w, ts=ts, f=f)
    daily_kept = compute_daily_pnl_kept_from_arrays(y=y, w=w, ts=ts, f=f)
    dates = _oos_dates(y, w, ts, f)

    tag = _run_tag(instrument, model_name, variant, tau, max_filter_rate)
    _print_daily_stats(tag, "Daily Score", daily_score)
    _print_daily_stats(tag, "Daily PnL kept", daily_kept)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes[0, 0].plot(dates, daily_score, "o-", ms=3, lw=1, color="tab:blue")
    axes[0, 0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0, 0].set_title("Daily Score over time")
    axes[0, 0].set_ylabel("Score, bps")

    axes[0, 1].hist(daily_score, bins=40, color="tab:blue", alpha=0.75)
    axes[0, 1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[0, 1].axvline(np.mean(daily_score), color="red", lw=1.2, ls="--",
                       label=f"mean {np.mean(daily_score):+.2f}")
    axes[0, 1].set_title("Daily Score distribution")
    axes[0, 1].legend()

    axes[1, 0].plot(dates, daily_kept, "o-", ms=3, lw=1, color="tab:green")
    axes[1, 0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1, 0].set_title("Daily PnL kept over time")
    axes[1, 0].set_ylabel("PnL kept, bps")

    axes[1, 1].hist(daily_kept, bins=40, color="tab:green", alpha=0.75)
    axes[1, 1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1, 1].axvline(np.mean(daily_kept), color="red", lw=1.2, ls="--",
                       label=f"mean {np.mean(daily_kept):+.2f}")
    axes[1, 1].set_title("Daily PnL kept distribution")
    axes[1, 1].legend()

    cum_score = np.cumsum(daily_score)
    running_max = np.maximum.accumulate(cum_score)
    drawdown = cum_score - running_max
    axes[2, 0].plot(dates, cum_score, lw=1.5, color="tab:blue")
    axes[2, 0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[2, 0].set_title("Cumulative daily Score")
    axes[2, 0].set_ylabel("bps (cum)")

    axes[2, 1].fill_between(dates, drawdown, 0, color="tab:red", alpha=0.5)
    axes[2, 1].set_title(f"Drawdown of cumulative Score (max {drawdown.min():.2f} bps)")
    axes[2, 1].set_ylabel("bps")

    for ax in (axes[0, 0], axes[1, 0], axes[2, 0], axes[2, 1]):
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(tag)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({"day": dates, "daily_score": daily_score, "daily_pnl_kept": daily_kept})
