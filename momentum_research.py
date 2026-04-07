"""
================================================================================
 MOMENTUM RESEARCH — One-Time Validation Script
================================================================================
 Purpose : Prove (or disprove) that the 12-1 momentum anomaly exists in the
           S&P 500 + Nasdaq-100 universe before deploying any momentum-based
           strategy.  Seven rigorous statistical tests.

 Usage   : python momentum_research.py
 Output  : momentum_research_report.txt
================================================================================
"""

# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
import warnings, datetime, os, sys, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from nightly_scanner import (
    get_sp500_with_sectors,
    get_nasdaq100_tickers,
    bulk_download,
    clean_data,
)
from config import SCANNER_BATCH_SIZE, MISSING_THRESHOLD

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
LOOKBACK_YEARS       = 6        # 5+ years of history (extra buffer for 12-mo signal)
MOMENTUM_WINDOW      = 252      # ~12 months of trading days
SKIP_RECENT          = 21       # skip most recent month (21 trading days)
REPORT_FILE          = os.path.join(_SCRIPT_DIR, "momentum_research_report.txt")
SUBSAMPLE_SPLIT_DATE = "2022-01-01"
TRAILING_STOP_PCT    = 0.10     # 10% trailing stop for Test 7


# ==========================================================================
#  HELPER — Month-End Rebalance Dates
# ==========================================================================

def get_month_end_dates(index):
    """Return a list of month-end trading dates from a DatetimeIndex."""
    s = pd.Series(index, index=index)
    return s.groupby([s.dt.year, s.dt.month]).last().values


# ==========================================================================
#  STEP 1 — Download Data
# ==========================================================================

def download_universe():
    """Fetch S&P 500 + Nasdaq-100 tickers and download daily closes."""
    print("=" * 70)
    print("  MOMENTUM RESEARCH SCRIPT")
    print("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print()

    # Get ticker lists (reuse nightly_scanner helpers)
    sp500_df = get_sp500_with_sectors()
    ndx_df   = get_nasdaq100_tickers()

    combined = pd.concat([sp500_df, ndx_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="Symbol", keep="first").reset_index(drop=True)
    tickers  = combined["Symbol"].tolist()
    print(f"[DATA] Combined universe: {len(tickers)} unique tickers.\n")

    # Bulk download
    prices = bulk_download(tickers, LOOKBACK_YEARS)
    prices = clean_data(prices, missing_thresh=MISSING_THRESHOLD)

    print(f"[DATA] Final price matrix: {prices.shape[0]} days x "
          f"{prices.shape[1]} tickers.")
    print(f"[DATA] Date range: {prices.index[0].date()} to "
          f"{prices.index[-1].date()}\n")

    return prices


# ==========================================================================
#  STEP 2 — Compute 12-1 Momentum Signal (no lookahead)
# ==========================================================================

def compute_momentum_signal(prices):
    """
    At each date t, momentum = price[t - SKIP_RECENT] / price[t - MOMENTUM_WINDOW] - 1.
    This is the 12-month return skipping the most recent month.
    No lookahead: signal at month-end t uses prices up to t - SKIP_RECENT.
    """
    print("[SIGNAL] Computing 12-1 momentum signal ...")

    # Shift prices by SKIP_RECENT to avoid using the most recent month
    lagged = prices.shift(SKIP_RECENT)
    # Price MOMENTUM_WINDOW days ago (from the lagged perspective)
    base   = prices.shift(MOMENTUM_WINDOW)

    momentum = lagged / base - 1.0

    # Drop rows where signal is not yet available
    momentum = momentum.iloc[MOMENTUM_WINDOW:]
    momentum = momentum.dropna(how="all")

    print(f"[SIGNAL] Momentum signal computed: {momentum.shape[0]} dates x "
          f"{momentum.shape[1]} tickers.\n")
    return momentum


# ==========================================================================
#  STEP 3 — Build Monthly Decile Portfolios
# ==========================================================================

def build_monthly_deciles(prices, momentum):
    """
    At each month-end, rank stocks by momentum into deciles (D1=top winners).
    Compute next-month equal-weight return for each decile.
    Returns a DataFrame of shape (n_months, 10) with decile returns.
    """
    print("[DECILES] Building monthly decile portfolios ...")

    # Monthly returns for forward return calculation
    monthly_prices = prices.resample("ME").last()
    monthly_ret    = monthly_prices.pct_change().shift(-1)  # next-month return

    # Align momentum to month-end
    monthly_mom = momentum.resample("ME").last()

    # Common dates and tickers
    common_dates   = monthly_mom.index.intersection(monthly_ret.index)
    common_tickers = monthly_mom.columns.intersection(monthly_ret.columns)

    monthly_mom = monthly_mom.loc[common_dates, common_tickers]
    monthly_ret = monthly_ret.loc[common_dates, common_tickers]

    # Drop last row (no forward return available)
    monthly_mom = monthly_mom.iloc[:-1]
    monthly_ret = monthly_ret.iloc[:-1]

    decile_returns = []
    dates_used     = []

    for date in monthly_mom.index:
        mom_row = monthly_mom.loc[date].dropna()
        ret_row = monthly_ret.loc[date].dropna()

        # Intersect tickers with valid momentum AND valid forward return
        valid = mom_row.index.intersection(ret_row.index)
        if len(valid) < 50:
            continue  # need enough stocks for meaningful deciles

        mom_vals = mom_row[valid]
        ret_vals = ret_row[valid]

        # Rank into deciles (1=highest momentum, 10=lowest)
        ranks = mom_vals.rank(ascending=False, method="first")
        n = len(ranks)
        decile_labels = pd.cut(ranks, bins=10, labels=False) + 1

        row_returns = {}
        for d in range(1, 11):
            mask = decile_labels == d
            if mask.sum() > 0:
                row_returns[f"D{d}"] = ret_vals[mask].mean()
            else:
                row_returns[f"D{d}"] = np.nan

        decile_returns.append(row_returns)
        dates_used.append(date)

    df = pd.DataFrame(decile_returns, index=dates_used)
    print(f"[DECILES] Built {len(df)} monthly observations with decile returns.\n")
    return df


# ==========================================================================
#  TEST 1 — Decile Sort
# ==========================================================================

def test_decile_sort(decile_df):
    """Average monthly return per decile."""
    print("[TEST 1] Decile Sort Analysis ...")
    means = decile_df.mean() * 100  # convert to percent
    stds  = decile_df.std() * 100

    lines = []
    lines.append("TEST 1 -- DECILE SORT (Average Monthly Return %)")
    lines.append("-" * 55)
    lines.append(f"{'Decile':<10} {'Avg Return %':<15} {'Std Dev %':<15}")
    lines.append("-" * 55)
    for d in range(1, 11):
        col = f"D{d}"
        label = "Top Winners" if d == 1 else ("Bottom Losers" if d == 10 else "")
        lines.append(f"  D{d:<8} {means[col]:>+10.3f}      {stds[col]:>10.3f}      {label}")
    lines.append("-" * 55)
    spread = means["D1"] - means["D10"]
    lines.append(f"  D1 - D10 spread: {spread:+.3f}% per month")
    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines


# ==========================================================================
#  TEST 2 — Long-Short Spread
# ==========================================================================

def test_long_short_spread(decile_df):
    """D1 minus D10 portfolio return each month."""
    print("[TEST 2] Long-Short Spread Analysis ...")

    spread = decile_df["D1"] - decile_df["D10"]
    spread = spread.dropna()

    mean_ret = spread.mean()
    std_ret  = spread.std()
    t_stat   = mean_ret / (std_ret / np.sqrt(len(spread)))
    p_value  = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(spread) - 1))

    go = t_stat > 2.0

    lines = []
    lines.append("TEST 2 -- LONG-SHORT SPREAD (D1 minus D10)")
    lines.append("-" * 55)
    lines.append(f"  Months:           {len(spread)}")
    lines.append(f"  Mean monthly ret: {mean_ret*100:+.3f}%")
    lines.append(f"  Std dev:          {std_ret*100:.3f}%")
    lines.append(f"  t-statistic:      {t_stat:.3f}")
    lines.append(f"  p-value:          {p_value:.6f}")
    lines.append(f"  Annualized ret:   {mean_ret*12*100:+.2f}%")
    lines.append(f"  Annualized vol:   {std_ret*np.sqrt(12)*100:.2f}%")
    lines.append(f"  Sharpe (ann):     {(mean_ret*12)/(std_ret*np.sqrt(12)):.3f}")
    lines.append("-" * 55)
    lines.append(f"  GO CONDITION (t-stat > 2.0): {'PASS' if go else 'FAIL'}  "
                 f"(t = {t_stat:.3f})")
    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines, spread


# ==========================================================================
#  TEST 3 — Fama-MacBeth Cross-Sectional Regression
# ==========================================================================

def test_fama_macbeth(prices, momentum):
    """
    Each month: regress next-month return on current momentum signal.
    Average the slope coefficients across months (Fama-MacBeth 1973).
    """
    print("[TEST 3] Fama-MacBeth Cross-Sectional Regression ...")

    monthly_prices = prices.resample("ME").last()
    monthly_ret    = monthly_prices.pct_change().shift(-1)
    monthly_mom    = momentum.resample("ME").last()

    common_dates   = monthly_mom.index.intersection(monthly_ret.index)
    common_tickers = monthly_mom.columns.intersection(monthly_ret.columns)

    monthly_mom = monthly_mom.loc[common_dates, common_tickers]
    monthly_ret = monthly_ret.loc[common_dates, common_tickers]

    # Drop last row
    monthly_mom = monthly_mom.iloc[:-1]
    monthly_ret = monthly_ret.iloc[:-1]

    betas = []
    for date in monthly_mom.index:
        mom = monthly_mom.loc[date].dropna()
        ret = monthly_ret.loc[date].dropna()
        valid = mom.index.intersection(ret.index)
        if len(valid) < 50:
            continue

        x = mom[valid].values
        y = ret[valid].values

        # Simple OLS: y = alpha + beta * x
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            betas.append(coeffs[1])
        except Exception:
            continue

    betas = np.array(betas)
    avg_beta = betas.mean()
    se_beta  = betas.std() / np.sqrt(len(betas))
    t_stat   = avg_beta / se_beta
    p_value  = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(betas) - 1))

    lines = []
    lines.append("TEST 3 -- FAMA-MACBETH CROSS-SECTIONAL REGRESSION")
    lines.append("-" * 55)
    lines.append(f"  Monthly regressions:  {len(betas)}")
    lines.append(f"  Avg slope (beta):     {avg_beta:.6f}")
    lines.append(f"  Std error:            {se_beta:.6f}")
    lines.append(f"  t-statistic:          {t_stat:.3f}")
    lines.append(f"  p-value:              {p_value:.6f}")
    lines.append("-" * 55)
    lines.append(f"  Interpretation: A 1pp increase in 12-1 momentum predicts")
    lines.append(f"  a {avg_beta*100:.4f}% change in next-month return.")
    lines.append(f"  Significant (t > 2): {'YES' if abs(t_stat) > 2.0 else 'NO'}")
    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines


# ==========================================================================
#  TEST 4 — Crash Analysis (Long-Short Drawdowns)
# ==========================================================================

def test_crash_analysis(ls_spread):
    """Compute cumulative P&L of the long-short portfolio and find worst drawdowns."""
    print("[TEST 4] Crash / Drawdown Analysis ...")

    cum_ret    = (1 + ls_spread).cumprod()
    peak       = cum_ret.cummax()
    drawdown   = (cum_ret - peak) / peak

    # Find the 5 worst drawdowns
    dd_periods = []
    dd_series  = drawdown.copy()

    for _ in range(5):
        if dd_series.min() >= 0:
            break

        trough_idx  = dd_series.idxmin()
        trough_val  = dd_series[trough_idx]

        # Walk backward to find start (last peak before trough)
        pre_trough  = cum_ret.loc[:trough_idx]
        start_idx   = pre_trough.idxmax()

        # Walk forward to find recovery (next time cum_ret >= peak)
        peak_val    = cum_ret[start_idx]
        post_trough = cum_ret.loc[trough_idx:]
        recovered   = post_trough[post_trough >= peak_val]
        if len(recovered) > 0:
            end_idx  = recovered.index[0]
            duration = (end_idx - start_idx).days
        else:
            end_idx  = cum_ret.index[-1]
            duration = (end_idx - start_idx).days

        dd_periods.append({
            "start":    start_idx,
            "trough":   trough_idx,
            "end":      end_idx,
            "depth":    trough_val,
            "duration": duration,
            "recovered": len(recovered) > 0,
        })

        # Zero out this drawdown period so we find the next one
        dd_series.loc[start_idx:end_idx] = 0.0

    max_dd = drawdown.min()
    avg_duration = np.mean([d["duration"] for d in dd_periods]) if dd_periods else 0

    lines = []
    lines.append("TEST 4 -- CRASH / DRAWDOWN ANALYSIS (Long-Short Portfolio)")
    lines.append("-" * 70)
    lines.append(f"  Max drawdown: {max_dd*100:.2f}%")
    lines.append(f"  Avg drawdown duration: {avg_duration:.0f} days")
    lines.append("")
    lines.append(f"  {'#':<4} {'Start':<12} {'Trough':<12} {'End':<12} "
                 f"{'Depth %':<10} {'Days':<8} {'Recovered'}")
    lines.append("  " + "-" * 68)

    for i, d in enumerate(dd_periods):
        lines.append(
            f"  {i+1:<4} "
            f"{str(d['start'].date()):<12} "
            f"{str(d['trough'].date()):<12} "
            f"{str(d['end'].date()):<12} "
            f"{d['depth']*100:>+8.2f}%  "
            f"{d['duration']:<8} "
            f"{'Yes' if d['recovered'] else 'No'}"
        )
    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines


# ==========================================================================
#  TEST 5 — Volume Confirmation (Lee & Swaminathan 2000)
# ==========================================================================

def test_volume_confirmation(prices, momentum):
    """
    Split top decile into volume-confirmed vs unconfirmed.
    Volume-confirmed: 20d avg volume > 50d avg volume.
    """
    print("[TEST 5] Volume Confirmation Analysis ...")
    print("  Downloading volume data (this may take a few minutes) ...")

    # We need volume data -- download separately
    tickers = list(prices.columns)
    end   = prices.index[-1]
    start = prices.index[0]

    # Download volume in batches
    vol_frames = []
    batch_size = SCANNER_BATCH_SIZE
    n_batches  = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch   = tickers[i : i + batch_size]
        batch_n = i // batch_size + 1
        print(f"    Volume batch {batch_n}/{n_batches} ...", end=" ", flush=True)
        try:
            raw = yf.download(
                batch,
                start=str(start.date()), end=str(end.date()),
                auto_adjust=True, progress=False, threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                vol = raw["Volume"]
            else:
                vol = raw[["Volume"]].rename(columns={"Volume": batch[0]})
            vol_frames.append(vol)
            print(f"OK ({vol.shape[1]} tickers)")
        except Exception as exc:
            print(f"FAILED ({exc})")
        time.sleep(0.5)

    if not vol_frames:
        lines = []
        lines.append("TEST 5 -- VOLUME CONFIRMATION")
        lines.append("-" * 55)
        lines.append("  SKIPPED: Could not download volume data.")
        lines.append("")
        for line in lines:
            print("  " + line)
        return lines

    volume = pd.concat(vol_frames, axis=1)
    volume = volume.loc[:, ~volume.columns.duplicated()]

    # Compute volume indicators
    vol_20d = volume.rolling(20).mean()
    vol_50d = volume.rolling(50).mean()
    vol_confirmed = vol_20d > vol_50d  # True = volume expanding

    # Monthly analysis
    monthly_prices = prices.resample("ME").last()
    monthly_ret    = monthly_prices.pct_change().shift(-1)
    monthly_mom    = momentum.resample("ME").last()
    monthly_volc   = vol_confirmed.resample("ME").last()

    common_dates   = monthly_mom.index.intersection(monthly_ret.index)
    common_dates   = common_dates.intersection(monthly_volc.index)
    common_tickers = monthly_mom.columns.intersection(monthly_ret.columns)
    common_tickers = common_tickers.intersection(monthly_volc.columns)

    monthly_mom  = monthly_mom.loc[common_dates, common_tickers].iloc[:-1]
    monthly_ret  = monthly_ret.loc[common_dates, common_tickers].iloc[:-1]
    monthly_volc = monthly_volc.loc[common_dates, common_tickers].iloc[:-1]

    confirmed_rets   = []
    unconfirmed_rets = []

    for date in monthly_mom.index:
        mom = monthly_mom.loc[date].dropna()
        ret = monthly_ret.loc[date].dropna()
        vc  = monthly_volc.loc[date].dropna()

        valid = mom.index.intersection(ret.index).intersection(vc.index)
        if len(valid) < 50:
            continue

        mom_vals = mom[valid]
        ret_vals = ret[valid]
        vc_vals  = vc[valid]

        # Top decile
        ranks = mom_vals.rank(ascending=False, method="first")
        n = len(ranks)
        cutoff = n * 0.1
        top_mask = ranks <= cutoff

        top_ret = ret_vals[top_mask]
        top_vc  = vc_vals[top_mask]

        conf_mask   = top_vc == True
        unconf_mask = top_vc == False

        if conf_mask.sum() > 0:
            confirmed_rets.append(top_ret[conf_mask].mean())
        if unconf_mask.sum() > 0:
            unconfirmed_rets.append(top_ret[unconf_mask].mean())

    confirmed_rets   = np.array(confirmed_rets)
    unconfirmed_rets = np.array(unconfirmed_rets)

    lines = []
    lines.append("TEST 5 -- VOLUME CONFIRMATION (Lee & Swaminathan 2000)")
    lines.append("-" * 55)

    if len(confirmed_rets) > 0 and len(unconfirmed_rets) > 0:
        conf_mean   = confirmed_rets.mean() * 100
        unconf_mean = unconfirmed_rets.mean() * 100
        diff        = conf_mean - unconf_mean

        # Two-sample t-test
        t_val, p_val = stats.ttest_ind(confirmed_rets, unconfirmed_rets)

        lines.append(f"  Volume-Confirmed (20d > 50d avg vol):")
        lines.append(f"    Months: {len(confirmed_rets)},  Avg monthly ret: {conf_mean:+.3f}%")
        lines.append(f"  Volume-Unconfirmed:")
        lines.append(f"    Months: {len(unconfirmed_rets)},  Avg monthly ret: {unconf_mean:+.3f}%")
        lines.append(f"  Difference: {diff:+.3f}%  (t = {t_val:.3f}, p = {p_val:.4f})")
        lines.append(f"  Volume adds edge: {'YES' if diff > 0 and p_val < 0.10 else 'INCONCLUSIVE'}")
    else:
        lines.append("  Insufficient data for volume split.")

    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines


# ==========================================================================
#  TEST 6 — Subsample Stability
# ==========================================================================

def test_subsample_stability(ls_spread):
    """Split at SUBSAMPLE_SPLIT_DATE and check both halves."""
    print("[TEST 6] Subsample Stability Analysis ...")

    split = pd.Timestamp(SUBSAMPLE_SPLIT_DATE)
    pre   = ls_spread[ls_spread.index < split].dropna()
    post  = ls_spread[ls_spread.index >= split].dropna()

    results = {}
    for label, data in [("Pre-2022", pre), ("Post-2022", post)]:
        if len(data) < 6:
            results[label] = {"mean": np.nan, "t_stat": np.nan, "p_val": np.nan,
                              "n": len(data)}
            continue
        m = data.mean()
        s = data.std()
        t = m / (s / np.sqrt(len(data)))
        p = 2 * (1 - stats.t.cdf(abs(t), df=len(data) - 1))
        results[label] = {"mean": m, "t_stat": t, "p_val": p, "n": len(data)}

    lines = []
    lines.append("TEST 6 -- SUBSAMPLE STABILITY")
    lines.append("-" * 55)
    lines.append(f"  Split date: {SUBSAMPLE_SPLIT_DATE}")
    lines.append("")
    lines.append(f"  {'Period':<15} {'Months':<10} {'Spread %/mo':<15} "
                 f"{'t-stat':<10} {'p-value':<10}")
    lines.append("  " + "-" * 55)

    both_positive = True
    for label, r in results.items():
        if np.isnan(r["mean"]):
            lines.append(f"  {label:<15} {r['n']:<10} {'N/A':<15} {'N/A':<10} {'N/A':<10}")
            both_positive = False
        else:
            lines.append(f"  {label:<15} {r['n']:<10} {r['mean']*100:>+10.3f}     "
                         f"{r['t_stat']:>8.3f}  {r['p_val']:>8.4f}")
            if r["mean"] <= 0:
                both_positive = False

    lines.append("")
    lines.append(f"  Edge positive in both subsamples: "
                 f"{'YES' if both_positive else 'NO'}")
    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines


# ==========================================================================
#  TEST 7 — Trailing Stop Backtest
# ==========================================================================

def test_trailing_stop(prices, momentum):
    """
    For top decile stocks, simulate:
      - Buy at month-end
      - Exit at 10% trailing stop OR end of month (whichever first)
    Using daily prices for intra-month tracking.
    """
    print("[TEST 7] Trailing Stop Backtest ...")

    monthly_mom = momentum.resample("ME").last()
    month_ends  = monthly_mom.index[:-1]  # skip last (no forward data)

    all_trades = []

    for i, entry_date in enumerate(month_ends):
        if (i + 1) % 12 == 0:
            print(f"  Processing month {i+1}/{len(month_ends)} ...", flush=True)

        mom_row = monthly_mom.loc[entry_date].dropna()
        if len(mom_row) < 50:
            continue

        # Top decile
        ranks  = mom_row.rank(ascending=False, method="first")
        n      = len(ranks)
        cutoff = n * 0.1
        top    = ranks[ranks <= cutoff].index.tolist()

        # Determine next month-end for exit deadline
        if i + 1 < len(month_ends):
            exit_deadline = month_ends[i + 1]
        else:
            exit_deadline = prices.index[-1]

        # Get daily prices for the holding period
        mask         = (prices.index > entry_date) & (prices.index <= exit_deadline)
        daily_window = prices.loc[mask]
        if len(daily_window) == 0:
            continue

        for ticker in top:
            if ticker not in daily_window.columns:
                continue

            ticker_prices = daily_window[ticker].dropna()
            if len(ticker_prices) == 0:
                continue

            entry_price = ticker_prices.iloc[0]
            peak_price  = entry_price
            exit_price  = ticker_prices.iloc[-1]  # default: hold to month-end
            stopped_out = False

            for price in ticker_prices.values:
                if price > peak_price:
                    peak_price = price
                drawdown = (price - peak_price) / peak_price
                if drawdown <= -TRAILING_STOP_PCT:
                    exit_price  = price
                    stopped_out = True
                    break

            trade_ret = (exit_price - entry_price) / entry_price
            all_trades.append({
                "entry_date": entry_date,
                "ticker":     ticker,
                "return":     trade_ret,
                "stopped":    stopped_out,
            })

    trades_df = pd.DataFrame(all_trades)

    if len(trades_df) == 0:
        lines = []
        lines.append("TEST 7 -- TRAILING STOP BACKTEST")
        lines.append("-" * 55)
        lines.append("  No trades generated. Insufficient data.")
        lines.append("")
        for line in lines:
            print("  " + line)
        return lines

    returns   = trades_df["return"]
    total_ret = (1 + returns).prod() - 1

    # Group by entry month for Sharpe calculation
    monthly_groups = trades_df.groupby("entry_date")["return"].mean()
    sharpe_ann     = (monthly_groups.mean() / monthly_groups.std()) * np.sqrt(12) \
                     if monthly_groups.std() > 0 else 0.0

    # Max drawdown of the equal-weight monthly portfolio
    cum         = (1 + monthly_groups).cumprod()
    peak        = cum.cummax()
    max_dd      = ((cum - peak) / peak).min()

    win_rate    = (returns > 0).mean() * 100
    stop_rate   = trades_df["stopped"].mean() * 100

    lines = []
    lines.append("TEST 7 -- TRAILING STOP BACKTEST (10% Trailing Stop)")
    lines.append("-" * 55)
    lines.append(f"  Total trades:       {len(trades_df)}")
    lines.append(f"  Avg return/trade:   {returns.mean()*100:+.3f}%")
    lines.append(f"  Win rate:           {win_rate:.1f}%")
    lines.append(f"  Stop-out rate:      {stop_rate:.1f}%")
    lines.append(f"  Portfolio Sharpe:   {sharpe_ann:.3f} (annualized)")
    lines.append(f"  Max drawdown:       {max_dd*100:.2f}%")
    lines.append(f"  Cumulative return:  {total_ret*100:+.2f}%")
    lines.append(f"  Period:             {trades_df['entry_date'].min().date()} to "
                 f"{trades_df['entry_date'].max().date()}")
    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines


# ==========================================================================
#  GO / NO-GO Recommendation
# ==========================================================================

def make_recommendation(decile_df, ls_spread):
    """Evaluate all evidence and produce a final recommendation."""
    spread_mean = ls_spread.mean()
    spread_std  = ls_spread.std()
    t_stat      = spread_mean / (spread_std / np.sqrt(len(ls_spread)))

    # Criteria
    c1 = t_stat > 2.0                           # statistical significance
    c2 = spread_mean > 0                         # positive spread
    c3 = decile_df["D1"].mean() > decile_df["D10"].mean()  # monotonic decile sort

    # Subsample check
    split = pd.Timestamp(SUBSAMPLE_SPLIT_DATE)
    pre   = ls_spread[ls_spread.index < split]
    post  = ls_spread[ls_spread.index >= split]
    c4    = pre.mean() > 0 and post.mean() > 0  # positive in both halves

    go = c1 and c2 and c3

    lines = []
    lines.append("=" * 55)
    lines.append("  FINAL RECOMMENDATION")
    lines.append("=" * 55)
    lines.append("")
    lines.append("  Checklist:")
    lines.append(f"    [{'x' if c1 else ' '}] Long-short t-stat > 2.0  (t = {t_stat:.3f})")
    lines.append(f"    [{'x' if c2 else ' '}] Positive long-short spread  "
                 f"({spread_mean*100:+.3f}%/mo)")
    lines.append(f"    [{'x' if c3 else ' '}] D1 > D10 in decile sort")
    lines.append(f"    [{'x' if c4 else ' '}] Positive in both subsamples (bonus)")
    lines.append("")

    if go:
        verdict = "GO -- Momentum edge is statistically significant."
        if not c4:
            verdict += " (Note: subsample stability is weak.)"
    else:
        failures = []
        if not c1:
            failures.append("t-stat below 2.0")
        if not c2:
            failures.append("negative spread")
        if not c3:
            failures.append("non-monotonic decile sort")
        verdict = f"NO-GO -- Insufficient evidence. Failures: {', '.join(failures)}."

    lines.append(f"  >>> {verdict}")
    lines.append("")

    for line in lines:
        print("  " + line)
    print()
    return lines


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    start_time = time.time()

    # Step 1 -- Download
    prices = download_universe()

    # Step 2 -- Momentum signal
    momentum = compute_momentum_signal(prices)

    # Step 3 -- Monthly decile portfolios
    decile_df = build_monthly_deciles(prices, momentum)

    if len(decile_df) < 12:
        print("ERROR: Not enough monthly observations to run tests. "
              "Need at least 12 months.")
        return

    # Collect all report sections
    report = []
    report.append("=" * 70)
    report.append("  MOMENTUM RESEARCH REPORT")
    report.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Universe: S&P 500 + Nasdaq-100")
    report.append(f"  Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    report.append(f"  Signal: 12-1 momentum (12-month return, skip recent month)")
    report.append(f"  Tickers: {prices.shape[1]}")
    report.append(f"  Monthly observations: {len(decile_df)}")
    report.append("=" * 70)
    report.append("")

    # Test 1
    report.extend(test_decile_sort(decile_df))

    # Test 2
    t2_lines, ls_spread = test_long_short_spread(decile_df)
    report.extend(t2_lines)

    # Test 3
    report.extend(test_fama_macbeth(prices, momentum))

    # Test 4
    report.extend(test_crash_analysis(ls_spread))

    # Test 5
    report.extend(test_volume_confirmation(prices, momentum))

    # Test 6
    report.extend(test_subsample_stability(ls_spread))

    # Test 7
    report.extend(test_trailing_stop(prices, momentum))

    # Final recommendation
    report.extend(make_recommendation(decile_df, ls_spread))

    elapsed = time.time() - start_time
    report.append(f"  Script runtime: {elapsed/60:.1f} minutes")
    report.append("")

    # Write report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("-" * 70)
    print(f"  Report written to: {REPORT_FILE}")
    print(f"  Total runtime: {elapsed/60:.1f} minutes")
    print("-" * 70)


if __name__ == "__main__":
    main()
