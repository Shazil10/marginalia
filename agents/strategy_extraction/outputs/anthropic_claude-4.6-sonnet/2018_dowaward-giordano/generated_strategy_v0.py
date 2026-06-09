import yfinance as yf
import numpy as np
import pandas as pd


def run_backtest():
    """Ranked Asset Allocation Model (RAAM) — Giordano 2018.

    Sector ETF rotation using 4 signals: momentum, EWMA volatility,
    average pairwise correlation, and ATR trend breakout.
    Regime-adaptive allocation (SPY vs 200-SMA + market breadth).
    Cash filter replaces negative-momentum sectors with SHY.
    """

    SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
    BENCHMARK = "SPY"
    CASH_ETF = "SHY"
    ALL_TICKERS = list(dict.fromkeys(SECTOR_ETFS + [BENCHMARK, CASH_ETF]))

    # ── Download data ──
    raw = yf.download(ALL_TICKERS, start="2004-07-01", end="2026-03-15",
                      progress=False, group_by="ticker")
    closes = pd.DataFrame()
    for t in ALL_TICKERS:
        try:
            c = raw[t]["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            closes[t] = c
        except Exception:
            pass
    closes = closes.sort_index().dropna(how="all")

    ohlc_raw = yf.download(SECTOR_ETFS, start="2004-01-01", end="2026-03-15",
                           progress=False, group_by="ticker")
    ohlc_dict = {}
    for t in SECTOR_ETFS:
        try:
            df = ohlc_raw[t][["Open", "High", "Low", "Close"]].dropna()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            ohlc_dict[t] = df
        except Exception:
            pass

    sector_closes = closes[SECTOR_ETFS].dropna(how="all")
    sector_rets = sector_closes.pct_change()

    # ══════════════════════════════════════════════════════════════
    # Precompute all signals (vectorised, outside the day loop)
    # ══════════════════════════════════════════════════════════════

    MOM_LB = 84  # ~4 months

    mom_all = sector_closes.pct_change(periods=MOM_LB)

    # EWMA volatility (RiskMetrics λ=0.94, 84-day window, restarted each window)
    def _ewma_vol(closes_df, lam=0.94, window=84):
        rets = closes_df.pct_change()
        rv = rets.values
        n, k = rv.shape
        out = np.full((n, k), np.nan)
        for i in range(window, n):
            r = rv[i - window + 1: i + 1].copy()
            any_valid = ~np.all(np.isnan(r), axis=0)
            r = np.nan_to_num(r, nan=0.0)
            var = r[0] ** 2
            for j in range(1, len(r)):
                var = lam * var + (1 - lam) * r[j] ** 2
            v = np.sqrt(var * 252)
            v[~any_valid] = np.nan
            out[i] = v
        return pd.DataFrame(out, index=closes_df.index, columns=closes_df.columns)

    vol_all = _ewma_vol(sector_closes)

    # Average pairwise correlation (84-day window)
    def _avg_corr(closes_df, lookback=84):
        rv = closes_df.pct_change().values
        n, k = rv.shape
        out = np.full((n, k), np.nan)
        for i in range(lookback, n):
            r = rv[i - lookback: i + 1]
            valid_c = ~np.all(np.isnan(r), axis=0)
            if valid_c.sum() < 3:
                continue
            rs = r[:, valid_c]
            row_ok = ~np.any(np.isnan(rs), axis=1)
            rc = rs[row_ok]
            if len(rc) < 10:
                continue
            cm = np.corrcoef(rc.T)
            if np.any(np.isnan(cm)):
                continue
            nc = cm.shape[0]
            avg = (cm.sum(axis=1) - 1) / (nc - 1)
            vi = np.where(valid_c)[0]
            for idx_k, idx_j in enumerate(vi):
                out[i, idx_j] = avg[idx_k]
        return pd.DataFrame(out, index=closes_df.index, columns=closes_df.columns)

    corr_all = _avg_corr(sector_closes)

    # ATR trend signal (+1 uptrend, -1 downtrend, 0 neutral) — precomputed per ticker
    def _trend_all(ohlc_dict, tickers, dates, atr_p=42, hi_p=63, lo_p=105):
        out = pd.DataFrame(0.0, index=dates, columns=tickers)
        for t in tickers:
            if t not in ohlc_dict:
                continue
            df = ohlc_dict[t]
            h, l, c = df["High"], df["Low"], df["Close"]
            tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(atr_p).mean()
            upper = c.rolling(hi_p).max() + atr
            lower = l.rolling(lo_p).min() - atr
            sig = pd.Series(0.0, index=df.index)
            sig[c >= upper] = 1.0
            sig[c <= lower] = -1.0
            out[t] = sig.reindex(dates).fillna(0)
        return out

    trend_all = _trend_all(ohlc_dict, SECTOR_ETFS, sector_closes.index)

    # ══════════════════════════════════════════════════════════════
    # Build rank matrices from precomputed signals
    # ══════════════════════════════════════════════════════════════

    tickers = list(sector_closes.columns)
    n_tickers = len(tickers)
    n_days = len(sector_closes)
    MIN_IDX = 110  # warmup

    M_ranks = np.full((n_days, n_tickers), np.nan)
    V_ranks = np.full((n_days, n_tickers), np.nan)
    C_ranks = np.full((n_days, n_tickers), np.nan)
    T_vals  = np.full((n_days, n_tickers), np.nan)
    mom_raw = np.full((n_days, n_tickers), np.nan)

    mv = mom_all.values
    vv = vol_all.values
    cv = corr_all.values
    tv = trend_all.values

    for idx in range(MIN_IDX, n_days):
        m, v, c = mv[idx], vv[idx], cv[idx]
        valid = ~np.isnan(m) & ~np.isnan(v) & ~np.isnan(c)
        if valid.sum() < 3:
            continue
        vi = np.where(valid)[0]
        nv = len(vi)

        # Momentum: highest value → rank 1 (ascending=False)
        m_sub = m[vi]
        m_r = np.argsort(np.argsort(-m_sub)) + 1.0

        # Volatility: lowest value → rank 1 (ascending=True)
        v_sub = v[vi]
        v_r = np.argsort(np.argsort(v_sub)) + 1.0

        # Correlation: lowest value → rank 1 (ascending=True)
        c_sub = c[vi]
        c_r = np.argsort(np.argsort(c_sub)) + 1.0

        # ATR trend: negate so uptrend (+1) → -1 (lower composite = better)
        t_sub = -tv[idx, vi]

        for k, j in enumerate(vi):
            M_ranks[idx, j] = m_r[k]
            V_ranks[idx, j] = v_r[k]
            C_ranks[idx, j] = c_r[k]
            T_vals[idx, j]  = t_sub[k]
            mom_raw[idx, j] = m_sub[k]

    # Forward returns (next-day)
    fwd_rets = sector_closes.pct_change().shift(-1).fillna(0).values
    cash_r   = closes[CASH_ETF].pct_change().shift(-1).fillna(0).reindex(sector_closes.index).fillna(0).values
    bench_r  = closes[BENCHMARK].pct_change().shift(-1).fillna(0).reindex(sector_closes.index).fillna(0).values
    valid_mask = ~np.isnan(M_ranks)

    mat = dict(tickers=tickers, dates=sector_closes.index,
               M=M_ranks, V=V_ranks, C=C_ranks, T=T_vals,
               mom=mom_raw, fwd_rets=fwd_rets,
               cash_rets=cash_r, bench_rets=bench_r, valid=valid_mask)

    # ══════════════════════════════════════════════════════════════
    # Regime indicators (precompute once)
    # ══════════════════════════════════════════════════════════════

    mat_dates = mat["dates"]
    spy_p = closes[BENCHMARK].reindex(mat_dates).ffill().values
    spy_sma200 = pd.Series(spy_p, index=mat_dates).rolling(200).mean().values
    sec_p = closes[SECTOR_ETFS].reindex(mat_dates).ffill()
    sec_sma50 = sec_p.rolling(50).mean()
    breadth_arr = (sec_p > sec_sma50).sum(axis=1).values / np.maximum(sec_p.notna().sum(axis=1).values, 1)

    # Inverse-vol weights (precompute)
    sec_rets_full = closes[SECTOR_ETFS].reindex(mat_dates).ffill().pct_change()
    inv_vol_arr = (sec_rets_full.rolling(63, min_periods=20).std().values * np.sqrt(252))
    tick_to_sec = {t: SECTOR_ETFS.index(t) for t in tickers if t in SECTOR_ETFS}

    # ══════════════════════════════════════════════════════════════
    # Backtest engine
    # ══════════════════════════════════════════════════════════════

    def fast_backtest(weights, top_n=5, rebal_freq="M", cash_filter=True,
                      sizing="equal", breadth_thr=0.6):
        wM, wV, wC, wT = weights
        M = mat["M"]; V = mat["V"]; C = mat["C"]; T = mat["T"]
        mm = mat["mom"]; fwd = mat["fwd_rets"]
        cr = mat["cash_rets"]; br = mat["bench_rets"]
        vld = mat["valid"]
        n_d, n_t = M.shape

        big = 9999.0
        composite = (wM * np.where(vld, M, big) +
                     wV * np.where(vld, V, big) +
                     wC * np.where(vld, C, big) +
                     wT * np.where(vld, T, big))

        vda = np.where(vld.any(axis=1))[0]
        if len(vda) == 0:
            return None
        first, last = vda[0], vda[-1]
        if last - first < 40:
            return None

        active = mat_dates[first:last + 1]
        rmask = np.zeros(n_d, dtype=bool)
        if rebal_freq == "W":
            rd = pd.Series(active).groupby(active.to_period("W")).last().values
        elif rebal_freq == "2W":
            wk = pd.Series(active).groupby(active.to_period("W")).last()
            rd = wk.iloc[::2].values
        else:
            rd = pd.Series(active).groupby(active.to_period("M")).last().values
        for d in rd:
            loc = mat_dates.get_indexer([d], method="ffill")[0]
            if first <= loc <= last:
                rmask[loc] = True
        rmask[first] = True

        pw = np.zeros((n_d, n_t))
        cw = np.zeros(n_d)
        cur_w = np.zeros(n_t)
        cur_c = 1.0

        for i in range(first, last + 1):
            if rmask[i]:
                sp, ss, brd = spy_p[i], spy_sma200[i], breadth_arr[i]
                if np.isnan(ss):
                    regime = "N"
                elif sp > ss and brd >= breadth_thr:
                    regime = "B"
                elif sp > ss:
                    regime = "N"
                else:
                    regime = "D"

                if regime == "B":
                    tn = min(top_n + 3, n_t)
                    use_cf = False
                elif regime == "N":
                    tn = top_n
                    use_cf = cash_filter
                else:
                    tn = max(top_n - 2, 2)
                    use_cf = True

                row = composite[i]
                vc = int(vld[i].sum())
                if vc < tn:
                    cur_w = np.zeros(n_t)
                    cur_c = 1.0
                else:
                    vi = np.where(vld[i])[0]
                    sv = vi[np.argsort(row[vi])]
                    ti = sv[:tn]
                    ns = len(ti)

                    if sizing == "inv_vol":
                        rw = np.zeros(ns)
                        for k, j in enumerate(ti):
                            t = tickers[j]
                            if t in tick_to_sec:
                                vl = inv_vol_arr[i, tick_to_sec[t]]
                                rw[k] = 1.0 / vl if (not np.isnan(vl) and vl > 0) else 1.0
                            else:
                                rw[k] = 1.0
                    else:
                        rw = np.ones(ns)

                    tw = rw.sum()
                    rw = rw / tw if tw > 0 else np.full(ns, 1.0 / ns)

                    nw = np.zeros(n_t)
                    nc = 0.0
                    for k, j in enumerate(ti):
                        mval = mm[i, j]
                        if use_cf and (np.isnan(mval) or mval < 0):
                            nc += rw[k]
                        else:
                            nw[j] = rw[k]
                    cur_w = nw
                    cur_c = nc

            pw[i] = cur_w
            cw[i] = cur_c

        pd_daily = (pw[first:last+1] * fwd[first:last+1]).sum(axis=1) + \
                    cw[first:last+1] * cr[first:last+1]

        eq = np.cumprod(1 + pd_daily)
        dti = mat_dates[first:last + 1]
        equity = pd.Series(eq, index=dti[:len(eq)])
        daily_s = pd.Series(pd_daily, index=dti[:len(pd_daily)])

        td = (equity.index[-1] - equity.index[0]).days
        ty = td / 365.25
        if ty <= 0:
            return None

        tr = float(equity.iloc[-1] / equity.iloc[0] - 1)
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / ty) - 1)
        dr = daily_s
        sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
        rm = equity.cummax()
        dd = (equity - rm) / rm
        mdd = float(dd.min())
        calmar = float(cagr / abs(mdd)) if abs(mdd) > 0 else 0.0
        wr = float((dr > 0).sum() / len(dr)) if len(dr) > 0 else 0.0
        neg = dr[dr < 0]
        ds = float(neg.std() * np.sqrt(252)) if len(neg) > 0 else 0.001
        sortino = float((dr.mean() * 252) / ds) if ds > 0 else 0.0

        return dict(cagr=cagr, sharpe=sharpe, max_drawdown=mdd, calmar=calmar,
                    win_rate=wr, total_return=tr, sortino=sortino,
                    _daily=daily_s,
                    start_date=str(dti[0].date()), end_date=str(dti[-1].date()))

    # ══════════════════════════════════════════════════════════════
    # Parameter optimisation (grid search)
    # ══════════════════════════════════════════════════════════════

    weight_combos = [
        (0.25, 0.25, 0.25, 0.25),
        (0.40, 0.20, 0.20, 0.20),
        (0.30, 0.30, 0.20, 0.20),
        (0.20, 0.30, 0.20, 0.30),
        (0.35, 0.25, 0.15, 0.25),
        (0.30, 0.20, 0.20, 0.30),
    ]

    best_sharpe = -np.inf
    best_result = None
    best_params = None

    for w in weight_combos:
        for tn in [4, 5, 6]:
            for sz in ["equal", "inv_vol"]:
                try:
                    r = fast_backtest(w, top_n=tn, rebal_freq="M",
                                     cash_filter=True, sizing=sz)
                    if r is not None and r["sharpe"] > best_sharpe:
                        best_sharpe = r["sharpe"]
                        best_result = r
                        best_params = {
                            "signal_weights_MVCT": list(w),
                            "top_n_base": tn,
                            "rebal_freq": "M",
                            "cash_filter": True,
                            "sizing": sz,
                            "breadth_thr": 0.6,
                            "regime_adaptive_top_n": True,
                        }
                except Exception:
                    continue

    if best_result is None:
        return {"error": "All backtest configurations failed"}

    ds = best_result.pop("_daily")
    daily_returns_list = [[str(d.date()), float(v)] for d, v in ds.items()]

    return {
        "cagr": best_result["cagr"],
        "sharpe": best_result["sharpe"],
        "max_drawdown": best_result["max_drawdown"],
        "calmar": best_result["calmar"],
        "win_rate": best_result["win_rate"],
        "total_return": best_result["total_return"],
        "sortino": best_result["sortino"],
        "optimized_parameters": best_params,
        "daily_returns": daily_returns_list,
        "start_date": best_result["start_date"],
        "end_date": best_result["end_date"],
    }

if __name__ == '__main__':
    import json
    import traceback
    try:
        res = run_backtest()
        # Extract and persist daily returns without bloating stdout
        daily_returns_data = res.pop('daily_returns', None)
        if daily_returns_data:
            import pandas as pd
            returns_s = pd.Series(
                [float(r) for _, r in daily_returns_data],
                index=pd.to_datetime([d for d, _ in daily_returns_data]),
                name='daily_return',
            )
            returns_s.to_csv(r'agents/strategy_extraction/outputs/anthropic_claude-4.6-sonnet/2018_dowaward-giordano/daily_returns.csv', header=True)
        print('BACKTEST_RESULT:' + json.dumps(res))
    except Exception as e:
        print('BACKTEST_ERROR:' + traceback.format_exc())
