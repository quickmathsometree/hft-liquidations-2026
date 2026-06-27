def score_one(
    pnl: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    num_days: float,
    tau: int,
) -> ScoreReport:
    """
    Compute ScoreReport for one tau.

    Formulas (over valid trades only, where pnl is not NaN):
        PnL_all      = Σ w·pnl         / Σ w
        PnL_kept     = Σ (1-f)·w·pnl   / Σ (1-f)·w
        PnL_filtered = Σ f·w·pnl       / Σ f·w
        Score        = PnL_kept - PnL_all

    Division-by-zero: returns NaN for the affected metric.
    """
    pnl = np.asarray(pnl, dtype="float64")
    w = np.asarray(w, dtype="float64")
    f = np.asarray(f, dtype="int8")

    valid = np.isfinite(pnl) & np.isfinite(w) & (w > 0)
    kept = f == 0
    filtered = f == 1

    def wavg(mask: np.ndarray) -> float:
        den = float(w[mask].sum())
        if den <= 0:
            return float("nan")
        return float(np.sum(w[mask] * pnl[mask]) / den)

    pnl_all = wavg(valid)
    pnl_kept = wavg(valid & kept)
    pnl_filtered = wavg(valid & filtered)

    if np.isfinite(pnl_kept) and np.isfinite(pnl_all):
        score = float(pnl_kept - pnl_all)
    else:
        score = float("nan")

    days = max(float(num_days), 1e-12)

    kept_turnover = float(w[kept & np.isfinite(w)].sum() / days)
    filtered_turnover = float(w[filtered & np.isfinite(w)].sum() / days)

    return ScoreReport(
        tau=int(tau),
        score=score,
        pnl_all=pnl_all,
        pnl_kept=pnl_kept,
        pnl_filtered=pnl_filtered,
        kept_turnover_per_day=kept_turnover,
        filtered_turnover_per_day=filtered_turnover,
        constraint_ok=kept_turnover >= 500_000.0,
        n_trades=int(len(pnl)),
        n_valid=int(valid.sum()),
        n_kept=int((valid & kept).sum()),
        n_filtered=int((valid & filtered).sum()),
        n_edge=int((~valid).sum()),
    )


def score_all(
    trades_with_pnl: pd.DataFrame,
    f_by_tau: dict[int, np.ndarray],
    num_days: float,
) -> dict[int, ScoreReport]:
    """Run score_one for all taus. Unified output format."""
    if "w" not in trades_with_pnl.columns:
        raise ValueError("trades_with_pnl must contain column 'w'")

    w = trades_with_pnl["w"].to_numpy(dtype="float64")

    reports: dict[int, ScoreReport] = {}

    for tau in TAUS:
        if tau not in f_by_tau:
            continue

        pnl_col = f"pnl_{tau}"
        if pnl_col not in trades_with_pnl.columns:
            raise ValueError(f"trades_with_pnl must contain column {pnl_col!r}")

        reports[int(tau)] = score_one(
            pnl=trades_with_pnl[pnl_col].to_numpy(dtype="float64"),
            w=w,
            f=f_by_tau[int(tau)],
            num_days=num_days,
            tau=int(tau),
        )

    return reports


def reports_to_frame(
    reports: dict[int, ScoreReport],
    symbol: str = "",
    experiment: str = "",
) -> pd.DataFrame:
    """Convert ScoreReport dict to a summary DataFrame for display."""
    rows = []

    for tau, r in reports.items():
        rows.append(
            {
                "experiment": experiment,
                "symbol": symbol,
                "tau": tau,
                "score": r.score,
                "pnl_all": r.pnl_all,
                "pnl_kept": r.pnl_kept,
                "pnl_filtered": r.pnl_filtered,
                "kept_turnover_per_day": r.kept_turnover_per_day,
                "filtered_turnover_per_day": r.filtered_turnover_per_day,
                "constraint_ok": r.constraint_ok,
                "n_trades": r.n_trades,
                "n_valid": r.n_valid,
                "n_kept": r.n_kept,
                "n_filtered": r.n_filtered,
                "n_edge": r.n_edge,
            }
        )

    return pd.DataFrame(rows)