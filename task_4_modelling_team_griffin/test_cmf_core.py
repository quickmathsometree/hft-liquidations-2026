"""Self-validation of cmf_core against the official spec + causality guarantees."""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import cmf_core as cc

US = cc.US_PER_SECOND
ok = lambda name, cond: print(("PASS" if cond else "FAIL"), name) or (cond or sys.exit(1))

# 1) markout/pnl matches the official docstring worked example -------------------
# price=100000, side='buy' (s=+1). mid_30=100050 -> pnl=-4.5 ; mid_30=99950 -> +5.5
bbo = pd.DataFrame({
    "timestamp": [0, 30*US, 30*US],
    "bid_price": [99999.5, 100049.5, 99949.5],
    "ask_price": [100000.5, 100050.5, 99950.5],
    "bid_amount": [1.0, 1.0, 1.0],
    "ask_amount": [1.0, 1.0, 1.0],
})
bbo = cc.add_mid(bbo.sort_values("timestamp").reset_index(drop=True))
tr = pd.DataFrame({"timestamp": [0], "price": [100000.0], "amount": [1.0], "side": ["buy"]})
tr = cc.compute_markout(tr, bbo, taus=(30,))
tr = cc.compute_pnl(tr, taus=(30,))
# mid at t=30s forward-fill -> last row with ts<=30s is the 99950 row (index 2)
ok("markout asof picks last <= t+tau", abs(tr["mid_30"].iloc[0] - 99950.0) < 1e-9)
ok("pnl good fill = +5.5 bps", abs(tr["pnl_30"].iloc[0] - 5.5) < 1e-9)

# flip: make the only post-tau quote the +50 one
bbo2 = pd.DataFrame({
    "timestamp": [0, 30*US],
    "bid_price": [99999.5, 100049.5], "ask_price": [100000.5, 100050.5],
    "bid_amount": [1.0, 1.0], "ask_amount": [1.0, 1.0]})
bbo2 = cc.add_mid(bbo2)
tr2 = cc.compute_markout(tr[["timestamp","price","amount","side"]].copy(), bbo2, taus=(30,))
tr2 = cc.compute_pnl(tr2, taus=(30,))
ok("pnl bad fill = -4.5 bps", abs(tr2["pnl_30"].iloc[0] - (-4.5)) < 1e-9)

# 2) microprice formula ----------------------------------------------------------
b = cc.add_mid(pd.DataFrame({"timestamp":[0],"bid_price":[100.0],"ask_price":[101.0],
                             "bid_amount":[3.0],"ask_amount":[1.0]}))
# micro = (100*1 + 101*3)/(3+1) = (100+303)/4 = 100.75  (skew toward ask = big bid pressure)
ok("microprice = 100.75", abs(b["microprice"].iloc[0]-100.75) < 1e-9)
ok("mid = 100.5", abs(b["mid"].iloc[0]-100.5) < 1e-9)

# 3) edge flag: trade whose t+tau exceeds max bbo ts -----------------------------
tr3 = pd.DataFrame({"timestamp":[100*US],"price":[100.0],"amount":[1.0],"side":["sell"]})
tr3 = cc.compute_markout(tr3, bbo, taus=(30,))
ok("edge flag set beyond bbo range", bool(tr3["edge_30"].iloc[0]) and np.isnan(tr3["mid_30"].iloc[0]))

# 4) official score on a tiny hand-checked set -----------------------------------
# 4 trades, equal w=1. pnl=[+2,-2,+4,-4]. filter out the two losers (f for negatives).
trades = pd.DataFrame({"timestamp":[0,1,2,3],"w":[1.,1.,1.,1.],
                       "pnl_30":[2.,-2.,4.,-4.]})
f = np.array([0,1,0,1])  # remove the losers
rep = cc.score_one(trades["pnl_30"].values, trades["w"].values, f, num_days=1.0, tau=30)
ok("pnl_all = 0", abs(rep.pnl_all-0.0) < 1e-9)
ok("pnl_kept = +3", abs(rep.pnl_kept-3.0) < 1e-9)
ok("pnl_filtered = -3", abs(rep.pnl_filtered-(-3.0)) < 1e-9)
ok("score = +3", abs(rep.score-3.0) < 1e-9)
ok("kept turnover/day = 2", abs(rep.kept_turnover_per_day-2.0) < 1e-9)

# 5) fit_threshold respects the turnover floor (label-free) -----------------------
rng = np.random.default_rng(0)
n = 1000
raw = rng.normal(size=n)
w = np.full(n, 1000.0)              # total 1e6 over num_days=1 -> 1e6/day
t = cc.fit_threshold(raw, w, num_days=1.0, target_turnover_per_day=500_000.0)
f5 = cc.apply_filter(raw, t)
kept_to = w[f5 == 0].sum() / 1.0
ok("fit_threshold keeps >= floor turnover", kept_to >= 500_000.0 - 1e-6)
ok("fit_threshold keeps ~ floor (not everything)", kept_to <= 0.6e6 + 2000)

# 6) causal_window_sum is strictly causal (excludes simultaneous + future) --------
ev_ts = np.array([10, 20, 30, 40], dtype=np.int64)
ev_v = np.array([1., 1., 1., 1.])
q = np.array([25, 30, 100], dtype=np.int64)
s = cc.causal_window_sum(q, ev_ts, ev_v, window_us=1000)
ok("window_sum strict-before @25 -> {10,20}=2", abs(s[0]-2.0) < 1e-9)
ok("window_sum strict-before @30 EXCLUDES event@30 -> {10,20}=2", abs(s[1]-2.0) < 1e-9)
ok("window_sum @100 all 4 = 4", abs(s[2]-4.0) < 1e-9)
win = cc.causal_window_sum(np.array([35],dtype=np.int64), ev_ts, ev_v, window_us=12)
ok("window respects [q-w, q): @35 w=12 -> {30}=1", abs(win[0]-1.0)<1e-9)

# 7) causal_decaying_sum matches brute-force reference, strictly causal -----------
rng = np.random.default_rng(1)
ev_ts = np.sort(rng.integers(0, 10_000_000, size=400)).astype(np.int64)
ev_v = rng.random(400)
q = np.sort(rng.integers(0, 10_000_000, size=120)).astype(np.int64)
hl = 1_000_000.0
fast = cc.causal_decaying_sum(q, ev_ts, ev_v, hl)
tau = hl/np.log(2)
brute = np.array([np.sum(ev_v[ev_ts < qq]*np.exp(-(qq-ev_ts[ev_ts<qq])/tau)) for qq in q])
ok("decaying_sum matches brute force", np.allclose(fast, brute, atol=1e-9, rtol=1e-7))
# strict: an event exactly at query must be excluded
q0 = np.array([ev_ts[50]], dtype=np.int64)
fast0 = cc.causal_decaying_sum(q0, ev_ts, ev_v, hl)
brute0 = np.sum(ev_v[ev_ts < q0[0]]*np.exp(-(q0[0]-ev_ts[ev_ts<q0[0]])/tau))
ok("decaying_sum excludes simultaneous event", abs(fast0[0]-brute0) < 1e-9)

# 8) time_since + count ----------------------------------------------------------
ts2 = cc.causal_time_since(np.array([35],dtype=np.int64), np.array([10,20,30],dtype=np.int64))
ok("time_since last (strict) = 5", abs(ts2[0]-5) < 1e-9)
cnt = cc.causal_window_count(np.array([30],dtype=np.int64), np.array([10,20,30],dtype=np.int64), 100)
ok("count strict-before @30 = 2", abs(cnt[0]-2) < 1e-9)

# 9) purged/CPCV splits have no test/train leakage at the boundary ---------------
days = np.repeat(np.arange(30), 5)
for tr_d, te_d in cc.purged_kfold_splits(days, n_splits=5, embargo=1):
    inter = np.intersect1d(tr_d, te_d)
    ok("purged kfold: no train/test overlap", len(inter)==0)
    # embargo: no train day within 1 of test block
    gap = np.min(np.abs(tr_d[:,None]-te_d[None,:])) if len(tr_d) and len(te_d) else 99
    ok("purged kfold: embargo gap >= 2", gap >= 2)
    break
combos = list(cc.combinatorial_purged_splits(days, n_groups=6, k_test=2, embargo=1))
ok("CPCV yields C(6,2)=15 splits", len(combos)==15)

print("\nALL CORE TESTS PASSED")
