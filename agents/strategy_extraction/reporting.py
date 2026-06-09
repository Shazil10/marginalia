import os
import json
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY CLASSIFICATION & ABSTRACT
# ═══════════════════════════════════════════════════════════════════════════════


def classify_risk_profile(metrics):
    sharpe = metrics.get('sharpe', 0)
    max_dd = abs(metrics.get('max_drawdown', 1))
    if sharpe > 1.0 and max_dd < 0.20:
        return {'label': 'Conservative', 'color': '#27ae60',
                'desc': 'Best for risk-averse investors prioritizing capital preservation with limited drawdowns.'}
    elif sharpe > 0.5 and max_dd < 0.35:
        return {'label': 'Moderate', 'color': '#f39c12',
                'desc': 'Best for balanced investors accepting moderate drawdowns for above-average returns.'}
    else:
        return {'label': 'Aggressive', 'color': '#e74c3c',
                'desc': 'Best for growth-oriented investors comfortable with significant volatility and deep drawdowns.'}


def generate_abstract(sj):
    name = sj.get('strategy_name', 'Unknown')
    stype = sj.get('strategy_type', 'unknown').replace('_', ' ')
    source = sj.get('source_paper', 'Unknown')
    rebal = sj.get('rebalance_frequency', 'periodic')
    universe = sj.get('asset_universe', 'broad market')
    if isinstance(universe, list):
        universe = ', '.join(str(u) for u in universe[:3]) + ('...' if len(universe) > 3 else '')
    entry = sj.get('entry_signal', '')
    entry = (entry[:200].rstrip('.') + '...') if len(entry) > 200 else entry.rstrip('.')
    exit_s = sj.get('exit_signal', '')
    exit_s = (exit_s[:160].rstrip('.') + '...') if len(exit_s) > 160 else exit_s.rstrip('.')
    return (f"{name} is a {stype} strategy from {source}. "
            f"Universe: {universe}; rebalances {rebal}. "
            f"Entry: {entry}. Exit: {exit_s}.")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def _load_returns(output_dir):
    path = os.path.join(output_dir, 'daily_returns.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df.iloc[:, 0].dropna()
    s = s[~s.index.duplicated(keep='first')].sort_index()
    s.index = pd.to_datetime(s.index)
    return s if len(s) > 10 else None


def _download_benchmark(start, end, ticker='SPY'):
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            close = data[('Close', ticker)]
        else:
            close = data['Close']
        return close.pct_change().dropna()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE STATISTICS (no QuantStats dependency)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_stats(returns, benchmark=None):
    s = {}
    n = len(returns)
    yrs = n / 252.0
    cumul = (1 + returns).cumprod()
    total = cumul.iloc[-1] - 1

    s['start'] = returns.index[0].strftime('%Y-%m-%d')
    s['end'] = returns.index[-1].strftime('%Y-%m-%d')
    s['years'] = round(yrs, 2)
    s['total_return'] = total
    s['cagr'] = (1 + total) ** (1 / yrs) - 1 if yrs > 0 else 0

    vol = returns.std()
    s['ann_vol'] = vol * np.sqrt(252)
    s['sharpe'] = (returns.mean() / vol) * np.sqrt(252) if vol > 0 else 0

    down = returns[returns < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 0 else 1
    s['sortino'] = (returns.mean() * 252) / down_vol if down_vol > 0 else 0

    rm = cumul.cummax()
    dd = (cumul - rm) / rm
    s['max_dd'] = dd.min()

    # Max drawdown duration (trading days)
    is_dd = dd < 0
    groups = (~is_dd).cumsum()
    dur = is_dd.groupby(groups).sum()
    s['max_dd_days'] = int(dur.max()) if len(dur) > 0 else 0

    s['calmar'] = s['cagr'] / abs(s['max_dd']) if s['max_dd'] != 0 else 0
    s['skew'] = returns.skew()
    s['kurtosis'] = returns.kurtosis()

    s['win_rate'] = (returns > 0).sum() / n
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    s['win_rate_m'] = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0

    s['best_day'] = returns.max()
    s['worst_day'] = returns.min()
    s['best_month'] = monthly.max() if len(monthly) > 0 else 0
    s['worst_month'] = monthly.min() if len(monthly) > 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    s['avg_win'] = wins.mean() if len(wins) > 0 else 0
    s['avg_loss'] = losses.mean() if len(losses) > 0 else 0

    gp = wins.sum() if len(wins) > 0 else 0
    gl = abs(losses.sum()) if len(losses) > 0 else 1
    s['profit_factor'] = gp / gl if gl > 0 else 0
    s['tail_ratio'] = abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
    s['var_daily'] = np.percentile(returns, 5)
    s['expected_daily'] = returns.mean()
    s['expected_monthly'] = monthly.mean() if len(monthly) > 0 else 0

    # Benchmark stats
    if benchmark is not None and len(benchmark) > 30:
        ab = pd.DataFrame({'s': returns, 'b': benchmark}).dropna()
        if len(ab) > 30:
            beta, alpha = np.polyfit(ab['b'], ab['s'], 1)
            s['beta'] = beta
            s['alpha'] = alpha * 252
            s['correlation'] = ab['s'].corr(ab['b'])
            te = (ab['s'] - ab['b']).std() * np.sqrt(252)
            s['tracking_error'] = te
            s['info_ratio'] = ((ab['s'].mean() - ab['b'].mean()) * 252) / te if te > 0 else 0

            bc = (1 + ab['b']).cumprod()
            bt = bc.iloc[-1] - 1
            by = len(ab) / 252.0
            s['bm_total'] = bt
            s['bm_cagr'] = (1 + bt) ** (1 / by) - 1 if by > 0 else 0
            s['bm_sharpe'] = (ab['b'].mean() / ab['b'].std()) * np.sqrt(252) if ab['b'].std() > 0 else 0
            s['bm_vol'] = ab['b'].std() * np.sqrt(252)
            bdd = (bc - bc.cummax()) / bc.cummax()
            s['bm_max_dd'] = bdd.min()
            bd = ab['b'][ab['b'] < 0]
            bdv = bd.std() * np.sqrt(252) if len(bd) > 0 else 1
            s['bm_sortino'] = (ab['b'].mean() * 252) / bdv if bdv > 0 else 0
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# CHART DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════


def _chart_data(returns, benchmark=None):
    d = {}
    cumul = (1 + returns).cumprod()
    dates = [x.strftime('%Y-%m-%d') for x in cumul.index]
    d['cum_dates'] = dates
    d['cum_strat'] = [round(float(x), 4) for x in cumul]

    rm = cumul.cummax()
    dd = (cumul - rm) / rm
    d['dd_vals'] = [round(float(x) * 100, 2) for x in dd]

    if benchmark is not None and len(benchmark) > 0:
        ab = pd.DataFrame({'s': returns, 'b': benchmark}).dropna()
        bc = (1 + ab['b']).cumprod()
        d['cum_bench'] = [round(float(x), 4) for x in bc]
        bdd = (bc - bc.cummax()) / bc.cummax()
        d['dd_bench'] = [round(float(x) * 100, 2) for x in bdd]
        d['cum_dates'] = [x.strftime('%Y-%m-%d') for x in ab.index]
        sc = (1 + ab['s']).cumprod()
        d['cum_strat'] = [round(float(x), 4) for x in sc]
        d['dd_vals'] = [round(float((sc.iloc[i] - sc.iloc[:i + 1].max()) / sc.iloc[:i + 1].max()) * 100, 2) for i in range(len(sc))]

    # EOY returns
    yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    d['eoy_years'] = [str(y.year) for y in yearly.index]
    d['eoy_strat'] = [round(float(x) * 100, 2) for x in yearly]
    if benchmark is not None and len(benchmark) > 0:
        by = benchmark.reindex(returns.index).dropna().resample('YE').apply(lambda x: (1 + x).prod() - 1)
        d['eoy_bench'] = [round(float(x) * 100, 2) for x in by.reindex(yearly.index).fillna(0)]

    # Monthly distribution
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    bins = np.linspace(float(monthly.min()) - 0.005, float(monthly.max()) + 0.005, 25)
    counts, edges = np.histogram(monthly.dropna(), bins=bins)
    d['dist_labels'] = [f"{(edges[i] + edges[i + 1]) / 2 * 100:.1f}" for i in range(len(counts))]
    d['dist_counts'] = [int(c) for c in counts]

    # Rolling Sharpe (126-day = 6 months)
    r_mean = returns.rolling(126).mean() * 252
    r_std = returns.rolling(126).std() * np.sqrt(252)
    r_sharpe = (r_mean / r_std).dropna()
    weekly_sharpe = r_sharpe.resample('W').last().dropna()
    d['rsharpe_dates'] = [x.strftime('%Y-%m-%d') for x in weekly_sharpe.index]
    d['rsharpe_vals'] = [round(float(x), 3) for x in weekly_sharpe]

    # Rolling Volatility (126-day)
    r_vol = (returns.rolling(126).std() * np.sqrt(252)).dropna()
    wv = r_vol.resample('W').last().dropna()
    d['rvol_dates'] = [x.strftime('%Y-%m-%d') for x in wv.index]
    d['rvol_strat'] = [round(float(x) * 100, 2) for x in wv]
    if benchmark is not None and len(benchmark) > 0:
        bv = (benchmark.reindex(returns.index).dropna().rolling(126).std() * np.sqrt(252)).dropna()
        bvw = bv.resample('W').last().dropna()
        d['rvol_bench'] = [round(float(x) * 100, 2) for x in bvw.reindex(wv.index, method='nearest').fillna(0)]

    # Rolling Sortino (126-day)
    def rolling_sortino(ret, w=126):
        out = pd.Series(index=ret.index, dtype=float)
        for i in range(w, len(ret)):
            window = ret.iloc[i - w:i]
            dw = window[window < 0]
            dv = dw.std() * np.sqrt(252) if len(dw) > 5 else np.nan
            out.iloc[i] = (window.mean() * 252) / dv if dv and dv > 0 else np.nan
        return out.dropna()

    rs = rolling_sortino(returns)
    ws = rs.resample('W').last().dropna()
    d['rsortino_dates'] = [x.strftime('%Y-%m-%d') for x in ws.index]
    d['rsortino_vals'] = [round(float(x), 3) for x in ws]

    # Rolling Beta (126-day)
    if benchmark is not None and len(benchmark) > 0:
        ab = pd.DataFrame({'s': returns, 'b': benchmark}).dropna()
        cov = ab['s'].rolling(126).cov(ab['b'])
        var = ab['b'].rolling(126).var()
        rbeta = (cov / var).dropna()
        wb = rbeta.resample('W').last().dropna()
        d['rbeta_dates'] = [x.strftime('%Y-%m-%d') for x in wb.index]
        d['rbeta_vals'] = [round(float(x), 3) for x in wb]

    # Daily returns (subsample to weekly for scatter-like view)
    d['daily_dates'] = [x.strftime('%Y-%m-%d') for x in returns.index]
    d['daily_vals'] = [round(float(x) * 100, 3) for x in returns]

    return d


def _monthly_heatmap(returns):
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    ytd_map = {y.year: float(v) for y, v in zip(yearly.index, yearly)}

    rows = []
    for year in sorted(set(monthly.index.year)):
        row = {'year': int(year)}
        for mi in range(1, 13):
            vals = monthly[(monthly.index.year == year) & (monthly.index.month == mi)]
            row[months[mi - 1]] = round(float(vals.iloc[0]) * 100, 2) if len(vals) > 0 else None
        row['YTD'] = round(ytd_map.get(year, 0) * 100, 2) if year in ytd_map else None
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# HTML RENDERING (QuantStats-style tear sheet)
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt_pct(v):
    return f"{v * 100:.2f}%"


def _fmt_f(v, d=2):
    return f"{v:.{d}f}"


def _metrics_row(label, strat_val, bench_val=None, fmt='pct'):
    f = _fmt_pct if fmt == 'pct' else (lambda x: _fmt_f(x))
    sv = f(strat_val)
    bv = f(bench_val) if bench_val is not None else '-'
    return f'<tr><td class="ml">{label}</td><td class="mv">{sv}</td><td class="mv bv">{bv}</td></tr>'


def _render_html(strategy_json, stats, chart_data, heatmap, risk, abstract, params):
    name = strategy_json.get('strategy_name', 'Strategy')
    source = strategy_json.get('source_paper', 'Unknown')
    stype = strategy_json.get('strategy_type', 'unknown').replace('_', ' ').title()
    has_bm = 'bm_cagr' in stats
    period = f"{stats['start']} — {stats['end']} ({stats['years']} years)"
    rc = risk['color']

    # Metrics table rows
    mr = []
    mr.append(_metrics_row('Cumulative Return', stats['total_return'], stats.get('bm_total')))
    mr.append(_metrics_row('CAGR', stats['cagr'], stats.get('bm_cagr')))
    mr.append(_metrics_row('Sharpe', stats['sharpe'], stats.get('bm_sharpe'), 'f'))
    mr.append(_metrics_row('Sortino', stats['sortino'], stats.get('bm_sortino'), 'f'))
    mr.append(_metrics_row('Max Drawdown', stats['max_dd'], stats.get('bm_max_dd')))
    mr.append(f'<tr><td class="ml">Longest DD (days)</td><td class="mv">{stats["max_dd_days"]}</td><td class="mv bv">-</td></tr>')
    mr.append(_metrics_row('Volatility (ann.)', stats['ann_vol'], stats.get('bm_vol')))
    mr.append(_metrics_row('Calmar', stats['calmar'], fmt='f'))
    mr.append(_metrics_row('Skew', stats['skew'], fmt='f'))
    mr.append(_metrics_row('Kurtosis', stats['kurtosis'], fmt='f'))

    mr.append('<tr><td colspan="3" class="sep"></td></tr>')
    mr.append(_metrics_row('Expected Daily', stats['expected_daily']))
    mr.append(_metrics_row('Expected Monthly', stats['expected_monthly']))
    mr.append(_metrics_row('Expected Yearly', stats['cagr']))
    mr.append(_metrics_row('Daily VaR (95%)', stats['var_daily']))

    mr.append('<tr><td colspan="3" class="sep"></td></tr>')
    mr.append(_metrics_row('Win Rate (daily)', stats['win_rate']))
    mr.append(_metrics_row('Win Rate (monthly)', stats['win_rate_m']))
    mr.append(_metrics_row('Best Day', stats['best_day']))
    mr.append(_metrics_row('Worst Day', stats['worst_day']))
    mr.append(_metrics_row('Best Month', stats['best_month']))
    mr.append(_metrics_row('Worst Month', stats['worst_month']))
    mr.append(_metrics_row('Avg Win (daily)', stats['avg_win']))
    mr.append(_metrics_row('Avg Loss (daily)', stats['avg_loss']))
    mr.append(_metrics_row('Profit Factor', stats['profit_factor'], fmt='f'))
    mr.append(_metrics_row('Tail Ratio', stats['tail_ratio'], fmt='f'))

    if has_bm:
        mr.append('<tr><td colspan="3" class="sep"></td></tr>')
        mr.append(_metrics_row('Alpha (ann.)', stats.get('alpha', 0)))
        mr.append(_metrics_row('Beta', stats.get('beta', 0), fmt='f'))
        mr.append(_metrics_row('Correlation', stats.get('correlation', 0), fmt='f'))
        mr.append(_metrics_row('Information Ratio', stats.get('info_ratio', 0), fmt='f'))
        mr.append(_metrics_row('Tracking Error', stats.get('tracking_error', 0)))

    metrics_html = '\n'.join(mr)

    # Param rows
    pr = ''.join(f'<tr><td class="ml">{k}</td><td class="mv" colspan="2">{v}</td></tr>' for k, v in params.items())

    # Heatmap
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    hm_header = '<th>Year</th>' + ''.join(f'<th>{m}</th>' for m in months) + '<th>YTD</th>'
    hm_rows = ''
    for row in heatmap:
        hm_rows += f'<tr><td class="hy">{row["year"]}</td>'
        for m in months:
            v = row.get(m)
            if v is None:
                hm_rows += '<td class="hc"></td>'
            else:
                intensity = min(abs(v) / 8.0, 1.0)
                if v >= 0:
                    bg = f'rgba(39,174,96,{intensity * 0.7 + 0.05})'
                    tc = '#fff' if intensity > 0.4 else '#333'
                else:
                    bg = f'rgba(231,76,60,{intensity * 0.7 + 0.05})'
                    tc = '#fff' if intensity > 0.4 else '#333'
                hm_rows += f'<td class="hc" style="background:{bg};color:{tc}">{v:.1f}%</td>'
        ytd = row.get('YTD')
        if ytd is not None:
            intensity = min(abs(ytd) / 30.0, 1.0)
            bg = f'rgba(39,174,96,{intensity * 0.7 + 0.05})' if ytd >= 0 else f'rgba(231,76,60,{intensity * 0.7 + 0.05})'
            tc = '#fff' if intensity > 0.4 else '#333'
            hm_rows += f'<td class="hc hytd" style="background:{bg};color:{tc}">{ytd:.1f}%</td>'
        else:
            hm_rows += '<td class="hc"></td>'
        hm_rows += '</tr>'

    cd = json.dumps(chart_data)

    # Benchmark dataset snippets for charts
    bm_cum = f',{{label:"Benchmark (SPY)",data:D.cum_bench,borderColor:"#f28e2c",backgroundColor:"rgba(242,142,44,0.08)",fill:true,borderWidth:1.5,pointRadius:0,tension:0.3}}' if 'cum_bench' in chart_data else ''
    bm_dd = f',{{label:"Benchmark",data:D.dd_bench,borderColor:"#f28e2c",backgroundColor:"rgba(242,142,44,0.10)",fill:true,borderWidth:1,pointRadius:0,tension:0.3}}' if 'dd_bench' in chart_data else ''
    bm_eoy = f',{{label:"Benchmark (SPY)",data:D.eoy_bench,backgroundColor:"rgba(242,142,44,0.7)",borderColor:"#f28e2c",borderWidth:1,borderRadius:3}}' if 'eoy_bench' in chart_data else ''
    bm_rvol = f',{{label:"Benchmark",data:D.rvol_bench,borderColor:"#f28e2c",borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}}' if 'rvol_bench' in chart_data else ''
    bm_rbeta_chart = ''
    if 'rbeta_dates' in chart_data:
        bm_rbeta_chart = f"""
      new Chart($('rbetaC'),{{type:'line',data:{{labels:D.rbeta_dates,datasets:[{{label:"Rolling Beta",data:D.rbeta_vals,borderColor:"#8e44ad",borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}}]}},options:LO("Rolling Beta (6 months)")}});"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{name} — Tear Sheet</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#fafbfc;color:#333;line-height:1.5}}
.wrap{{max-width:1200px;margin:0 auto;padding:20px}}
.hdr{{background:linear-gradient(135deg,#1a1d2e 0%,#0d1117 100%);color:#fff;border-radius:12px;padding:28px 32px;margin-bottom:24px}}
.hdr h1{{font-size:20px;font-weight:700;margin-bottom:2px}}
.hdr .sub{{font-size:13px;color:#8b95a5;margin-top:4px}}
.hdr .badges{{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}}
.hdr .badge{{font-size:11px;padding:4px 12px;border-radius:20px;font-weight:600}}
.badge-period{{background:rgba(255,255,255,0.1);color:#aab}}
.badge-risk{{background:{rc};color:#fff}}
.badge-type{{background:rgba(91,126,201,0.25);color:#a0bfff}}
.abstract{{font-size:13px;color:#99a;margin-top:14px;line-height:1.55}}
.row{{display:flex;gap:20px;margin-bottom:20px;flex-wrap:wrap}}
.chart-card{{background:#fff;border:1px solid #e8ecf1;border-radius:10px;padding:16px;flex:1;min-width:300px}}
.chart-card h3{{font-size:13px;font-weight:600;color:#555;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px}}
.chart-card canvas{{width:100%!important;max-height:220px}}
.full{{min-width:100%}}
.metrics-card{{background:#fff;border:1px solid #e8ecf1;border-radius:10px;padding:16px;min-width:320px;max-width:380px}}
.metrics-card h3{{font-size:13px;font-weight:600;color:#555;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px}}
.metrics-card table{{width:100%;border-collapse:collapse;font-size:12px}}
.ml{{color:#555;padding:3px 8px;white-space:nowrap}}
.mv{{font-weight:600;color:#1a1a2e;padding:3px 8px;text-align:right}}
.bv{{color:#888;font-weight:400}}
.sep{{border-bottom:1px solid #eee;padding:0;height:1px}}
.metrics-card thead th{{font-size:10px;text-transform:uppercase;color:#999;padding:3px 8px;text-align:right;border-bottom:1px solid #eee}}
.metrics-card thead th:first-child{{text-align:left}}
.heatmap-wrap{{background:#fff;border:1px solid #e8ecf1;border-radius:10px;padding:16px 12px;margin-bottom:20px}}
.heatmap-wrap h3{{font-size:13px;font-weight:600;color:#555;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px}}
.hm{{width:100%;border-collapse:collapse;font-size:12px;text-align:center}}
.hm th{{padding:5px 6px;font-size:10px;color:#888;text-transform:uppercase;font-weight:600;border-bottom:1px solid #e8ecf1}}
.hc{{padding:5px 4px;font-size:11px;font-weight:500;border:1px solid rgba(255,255,255,0.15);min-width:52px}}
.hy{{font-weight:700;color:#555;padding:5px 8px;text-align:left}}
.hytd{{font-weight:700}}
.details{{background:#fff;border:1px solid #e8ecf1;border-radius:10px;padding:20px;margin-bottom:20px;font-size:13px;color:#555}}
.details h3{{font-size:13px;font-weight:600;color:#555;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px}}
.risk-badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;color:#fff;background:{rc};margin-right:8px}}
</style>
</head>
<body>
<div class="wrap">

<div class="hdr">
  <h1>{name}</h1>
  <div class="sub">Based on: {source} &nbsp;|&nbsp; {period}</div>
  <div class="badges">
    <span class="badge badge-type">{stype}</span>
    <span class="badge badge-period">{period}</span>
    <span class="badge badge-risk">{risk['label']} Risk</span>
  </div>
  <p class="abstract">{abstract}</p>
</div>

<!-- ROW 1: Cumulative Returns + Metrics -->
<div class="row">
  <div class="chart-card" style="flex:2">
    <h3>Cumulative Returns</h3>
    <canvas id="cumC" height="220"></canvas>
  </div>
  <div class="metrics-card">
    <h3>Performance Metrics</h3>
    <table>
      <thead><tr><th></th><th>Strategy</th><th>{'Benchmark' if has_bm else ''}</th></tr></thead>
      <tbody>{metrics_html}</tbody>
    </table>
  </div>
</div>

<!-- ROW 2: Drawdown -->
<div class="row">
  <div class="chart-card full">
    <h3>Underwater Plot (Drawdown)</h3>
    <canvas id="ddC" height="160"></canvas>
  </div>
</div>

<!-- ROW 3: EOY Returns + Monthly Distribution -->
<div class="row">
  <div class="chart-card">
    <h3>End-of-Year Returns vs Benchmark</h3>
    <canvas id="eoyC" height="220"></canvas>
  </div>
  <div class="chart-card">
    <h3>Distribution of Monthly Returns</h3>
    <canvas id="distC" height="220"></canvas>
  </div>
</div>

<!-- ROW 4: Rolling Sharpe + Rolling Volatility -->
<div class="row">
  <div class="chart-card">
    <h3>Rolling Sharpe (6 months)</h3>
    <canvas id="rsharpeC" height="200"></canvas>
  </div>
  <div class="chart-card">
    <h3>Rolling Volatility (6 months)</h3>
    <canvas id="rvolC" height="200"></canvas>
  </div>
</div>

<!-- ROW 5: Rolling Sortino + Rolling Beta -->
<div class="row">
  <div class="chart-card">
    <h3>Rolling Sortino (6 months)</h3>
    <canvas id="rsortinoC" height="200"></canvas>
  </div>
  <div class="chart-card">
    <h3>Rolling Beta (6 months)</h3>
    <canvas id="rbetaC" height="200"></canvas>
  </div>
</div>

<!-- ROW 6: Daily Returns -->
<div class="row">
  <div class="chart-card full">
    <h3>Daily Returns</h3>
    <canvas id="dailyC" height="140"></canvas>
  </div>
</div>

<!-- Heatmap -->
<div class="heatmap-wrap">
  <h3>Monthly Returns (%)</h3>
  <table class="hm">
    <thead><tr>{hm_header}</tr></thead>
    <tbody>{hm_rows}</tbody>
  </table>
</div>

<!-- Strategy Details -->
<div class="details">
  <h3>Strategy Details</h3>
  <p style="margin-bottom:10px"><span class="risk-badge">{risk['label']}</span> {risk['desc']}</p>
  <table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:14px">
    <thead><tr><th style="text-align:left;color:#999;font-size:10px;padding:3px 8px;border-bottom:1px solid #eee">OPTIMIZED PARAMETER</th><th style="text-align:right;color:#999;font-size:10px;padding:3px 8px;border-bottom:1px solid #eee">VALUE</th></tr></thead>
    <tbody>{pr}</tbody>
  </table>
</div>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
(function(){{
  var D = {cd};
  var $ = function(id){{ return document.getElementById(id); }};
  Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.color = '#888';

  function LO(title,yPct){{
    return {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},title:{{display:false}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{
        x:{{display:true,ticks:{{maxTicksLimit:8,font:{{size:10}}}},grid:{{display:false}}}},
        y:{{display:true,ticks:{{font:{{size:10}},callback:function(v){{return yPct?v.toFixed(1)+'%':v.toFixed(2)}}}},grid:{{color:'#f0f0f0'}}}}
      }},
      elements:{{point:{{radius:0}},line:{{borderWidth:1.5}}}}
    }};
  }}

  // Cumulative Returns
  new Chart($('cumC'),{{type:'line',data:{{labels:D.cum_dates,datasets:[
    {{label:"Strategy",data:D.cum_strat,borderColor:"#2962ff",backgroundColor:"rgba(41,98,255,0.08)",fill:true,borderWidth:2,pointRadius:0,tension:0.3}}{bm_cum}
  ]}},options:Object.assign({{}},LO("",false),{{plugins:{{legend:{{display:true,position:'top',labels:{{boxWidth:12,font:{{size:10}}}}}}}}}})}}
  );

  // Drawdown
  new Chart($('ddC'),{{type:'line',data:{{labels:D.cum_dates,datasets:[
    {{label:"Strategy",data:D.dd_vals,borderColor:"#e74c3c",backgroundColor:"rgba(231,76,60,0.15)",fill:true,borderWidth:1.5,pointRadius:0,tension:0.3}}{bm_dd}
  ]}},options:Object.assign({{}},LO("",true),{{plugins:{{legend:{{display:{'true' if has_bm else 'false'},position:'top',labels:{{boxWidth:12,font:{{size:10}}}}}}}}}})}}
  );

  // EOY Returns
  new Chart($('eoyC'),{{type:'bar',data:{{labels:D.eoy_years,datasets:[
    {{label:"Strategy",data:D.eoy_strat,backgroundColor:"rgba(41,98,255,0.7)",borderColor:"#2962ff",borderWidth:1,borderRadius:3}}{bm_eoy}
  ]}},options:Object.assign({{}},LO("",true),{{plugins:{{legend:{{display:true,position:'top',labels:{{boxWidth:12,font:{{size:10}}}}}}}}}})}}
  );

  // Distribution
  new Chart($('distC'),{{type:'bar',data:{{labels:D.dist_labels,datasets:[
    {{data:D.dist_counts,backgroundColor:"rgba(41,98,255,0.6)",borderColor:"#2962ff",borderWidth:1,borderRadius:2}}
  ]}},options:LO("",false)}});

  // Rolling Sharpe
  new Chart($('rsharpeC'),{{type:'line',data:{{labels:D.rsharpe_dates,datasets:[
    {{label:"Rolling Sharpe",data:D.rsharpe_vals,borderColor:"#2962ff",borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}}
  ]}},options:LO("",false)}});

  // Rolling Volatility
  new Chart($('rvolC'),{{type:'line',data:{{labels:D.rvol_dates,datasets:[
    {{label:"Strategy",data:D.rvol_strat,borderColor:"#2962ff",borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}}{bm_rvol}
  ]}},options:Object.assign({{}},LO("",true),{{plugins:{{legend:{{display:{'true' if has_bm else 'false'},position:'top',labels:{{boxWidth:12,font:{{size:10}}}}}}}}}})}}
  );

  // Rolling Sortino
  new Chart($('rsortinoC'),{{type:'line',data:{{labels:D.rsortino_dates,datasets:[
    {{label:"Rolling Sortino",data:D.rsortino_vals,borderColor:"#27ae60",borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}}
  ]}},options:LO("",false)}});

  // Rolling Beta
  {bm_rbeta_chart}

  // Daily Returns
  new Chart($('dailyC'),{{type:'bar',data:{{labels:D.daily_dates,datasets:[
    {{data:D.daily_vals,backgroundColor:D.daily_vals.map(function(v){{return v>=0?'rgba(41,98,255,0.5)':'rgba(231,76,60,0.5)'}}),borderWidth:0,barPercentage:1.0,categoryPercentage:1.0}}
  ]}},options:Object.assign({{}},LO("",true),{{scales:{{x:{{display:true,ticks:{{maxTicksLimit:8,font:{{size:10}}}},grid:{{display:false}}}}}}}})}}
  );

}})();
</script>
</body>
</html>"""
    return html


def _render_minimal_html(strategy_json, metrics, risk, abstract, params):
    """Fallback report when no daily_returns.csv exists."""
    name = strategy_json.get('strategy_name', 'Strategy')
    source = strategy_json.get('source_paper', 'Unknown')
    stype = strategy_json.get('strategy_type', 'unknown').replace('_', ' ').title()
    rc = risk['color']

    mr = []
    for k, label, fmt in [
        ('cagr', 'CAGR', 'pct'), ('sharpe', 'Sharpe', 'f'), ('max_drawdown', 'Max Drawdown', 'pct'),
        ('calmar', 'Calmar', 'f'), ('win_rate', 'Win Rate', 'pct'), ('total_return', 'Total Return', 'pct'),
    ]:
        v = metrics.get(k, 0)
        vs = _fmt_pct(v) if fmt == 'pct' else _fmt_f(v)
        mr.append(f'<tr><td class="ml">{label}</td><td class="mv">{vs}</td></tr>')
    metrics_html = '\n'.join(mr)
    pr = ''.join(f'<tr><td class="ml">{k}</td><td class="mv">{v}</td></tr>'
                 for k, v in params.items())

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{name}</title>
<style>
body{{font-family:-apple-system,sans-serif;background:#fafbfc;color:#333;max-width:800px;margin:0 auto;padding:24px}}
.hdr{{background:#1a1d2e;color:#fff;border-radius:12px;padding:28px;margin-bottom:20px}}
.hdr h1{{font-size:20px}} .hdr .sub{{font-size:13px;color:#8b95a5;margin-top:4px}}
.card{{background:#fff;border:1px solid #e8ecf1;border-radius:10px;padding:20px;margin-bottom:16px}}
h3{{font-size:13px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
.ml{{color:#555;padding:4px 8px}} .mv{{font-weight:600;padding:4px 8px;text-align:right}}
.note{{background:#f0f4f8;border-radius:8px;padding:16px;font-size:13px;color:#666;margin-top:14px}}
</style></head><body>
<div class="hdr"><h1>{name}</h1><div class="sub">Based on: {source}</div></div>
<div class="card"><h3>Performance Metrics</h3><table>{metrics_html}</table>
<div class="note"><strong>Charts unavailable.</strong> Run the full pipeline to generate daily returns and the complete tear sheet:
<br><code>python agents/strategy_extraction/backfill_daily_returns.py</code>
<br><code>python agents/strategy_extraction/generate_reports.py</code></div></div>
<div class="card"><h3>Optimized Parameters</h3><table>{pr}</table></div>
<div class="card"><h3>Strategy Details</h3><p style="font-size:13px">{abstract}</p>
<p style="margin-top:10px;font-size:13px"><span style="display:inline-block;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600;color:#fff;background:{rc}">{risk['label']}</span> {risk['desc']}</p></div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def generate_tearsheet(output_dir, strategy_json, metrics):
    risk = classify_risk_profile(metrics)
    abstract = generate_abstract(strategy_json)
    params = metrics.get('optimized_parameters', {})

    # Load daily returns
    returns = _load_returns(output_dir)

    if returns is not None:
        print("  Computing comprehensive statistics...")
        start, end = returns.index[0], returns.index[-1]
        pad_start = (start - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        pad_end = (end + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        benchmark = _download_benchmark(pad_start, pad_end)

        stats = _compute_stats(returns, benchmark)
        chart_data = _chart_data(returns, benchmark)
        heatmap = _monthly_heatmap(returns)

        html = _render_html(strategy_json, stats, chart_data, heatmap, risk, abstract, params)
    else:
        html = _render_minimal_html(strategy_json, metrics, risk, abstract, params)

    # Save profile JSON
    profile = {
        'strategy_name': strategy_json.get('strategy_name', 'Unknown'),
        'source_paper': strategy_json.get('source_paper', 'Unknown'),
        'strategy_type': strategy_json.get('strategy_type', 'unknown'),
        'abstract': abstract,
        'risk_profile': {'label': risk['label'], 'description': risk['desc']},
        'optimized_parameters': params,
    }
    ppath = os.path.join(output_dir, 'strategy_profile.json')
    with open(ppath, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"  Strategy profile  -> {ppath}")

    html_path = os.path.join(output_dir, 'report.html')
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  Strategy report   -> {html_path}")
    return profile
