def strategy_raw_score(features: pd.DataFrame, trades: pd.DataFrame) -> np.ndarray:
    """
    Heuristic raw score: higher = better trade (higher expected pnl, keep it).

    Simplest version: score = -(same-side liq pressure at preferred halflife).
    Uses direction-relative features (same_side / opp_side), not absolute buy/sell.

    Returns float64 array of length len(features).
    """
    score = np.zeros(len(features), dtype="float64")

    cols = [
        c for c in features.columns
        if str(c).startswith("liq_ewma_") and "_same_" in str(c)
    ]

    if not cols:
        raise ValueError(
            "No same-side liquidation EWMA features found. "
            "Expected columns like 'liq_ewma_binance_same_5s'."
        )

    for c in cols:
        x = features[c].to_numpy(dtype="float64")
        score -= np.log1p(np.maximum(x, 0.0))

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)