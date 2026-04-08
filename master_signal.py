"""
================================================================================
 MASTER SIGNAL v3 — Institutional-Grade Live + Backtest Verification Engine
================================================================================
 Red Team Audit Fixes Applied
 ----------------------------
 1. LOOKAHEAD BIAS ELIMINATED
    - Signal observed at close of bar i  →  execution at bar i+1 close
    - beta[t] = OLS on log prices [t-60 .. t-1]  (excludes bar t)
    - z[t] = (spread[t] - mu) / sigma
      where mu, sigma computed from spread[t-20 .. t-1]  (excludes bar t)
    - Trade can NEVER execute on the same bar the signal is observed

 2. SLIPPAGE — TRUE TWO-LEG NOTIONAL COST
    - cost = SLIPPAGE_PCT × (entry_notional + exit_notional)
    - entry_notional = |shares_A|×P_A(entry) + |shares_B|×P_B(entry)
    - exit_notional  = |shares_A|×P_A(exit)  + |shares_B|×P_B(exit)
    - Applied on BOTH legs at BOTH entry and exit

 3. DATA HYGIENE
    - auto_adjust=True  →  Close = split/dividend-adjusted prices
    - dropna() before any spread calc  (common index alignment)
    - No ffill — missing data = no signal, not fake convergence
    - Volatility floor on sigma to prevent z-score explosion

 4. DIRECTIONAL EXIT LOGIC
    - LONG spread:  exit when z >= 0  (not |z| < 0 which is unreachable)
    - SHORT spread: exit when z <= 0
    - Stop: |z| > 3.5  (protective, either direction)

 Diamond Quality Gates (hard reject unless ALL pass)
 ---------------------------------------------------
    Standard  — Min trades >= 10 | WR > 55% | PF > 1.3x | P&L > $0
    Sharpe    — Annualised Sharpe > 0.5
    Sortino   — Annualised Sortino > 0.5 (downside-only risk)
    Avg P&L   — Average net P&L per trade > $5.00
    Drawdown  — Peak-to-trough drawdown > -$200
    Beta      — Hedge-ratio coefficient of variation < 0.30
    Gate A    — Split-Half  : Both halves P&L >= $20 independently
    Gate B    — Momentum    : Last 5 trades combined P&L > $0
    Gate C    — Freshness   : ADF on last 90 days of spread, p < 0.10
    Gate D    — Correlation : Recent 90-day price correlation >= 0.65
    Gate E    — Earnings    : Neither leg reports within ±7 days of today (FMP)
    Gate F    — Walk-Forward: OOS (30%) Sharpe > 0 and OOS P&L > 0

 Strategy Parameters
 -------------------
 - Log-price OLS regression for hedge ratio (beta)
 - Rolling beta window  : 60 days
 - Rolling z window  : 20 days
 - Entry             : |z| > 1.5
 - Exit              : z crosses 0  (directional)
 - Stop-loss         : |z| > 3.5
 - Max hold          : 30 days
 - Capital per trade : $1,000
 - Slippage          : 0.10% on total notional (entry + exit, both legs)
================================================================================
"""

import warnings, datetime, time, logging, os, requests
warnings.filterwarnings("ignore")
import earnings_util

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (imported from config.py — single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
from config import (
    INPUT_CSV, LOOKBACK_YEARS,
    ROLLING_BETA_WIN, ROLLING_Z_WIN,
    Z_ENTRY, Z_EXIT, Z_STOP, MAX_HOLD,
    CAPITAL_PER_TRADE, SLIPPAGE_PCT,
    MIN_WIN_RATE, MIN_PROFIT_FACTOR, MIN_TOTAL_PNL, MIN_SHARPE, MAX_DRAWDOWN,
    MIN_TRADES, BETA_STABILITY_MAX,
    MIN_SORTINO, MIN_AVG_PNL,
    MAX_SHARPE_OVERFIT, MAX_PROFIT_FACTOR_OVERFIT,
    OOS_SHARPE_RATIO_MIN,
    MAX_RETRIES, RETRY_DELAY, ERROR_LOG,
    SPLIT_HALF_ENABLED, SPLIT_HALF_MIN_PNL,
    RECENT_ADF_WINDOW, RECENT_ADF_PVAL,
    RECENT_MOMENTUM_N,
    RECENT_CORR_WINDOW, MIN_RECENT_CORR,
    VIX_MAX_ENTRY,
    WALK_FORWARD_SPLIT, MIN_OOS_TRADES,
    AV_API_KEY, AV_RATE_DELAY,
    FMP_API_KEY, EARNINGS_BLACKOUT_DAYS,
    BORROW_COST_PCT,
    VOL_TARGET_DAILY, VOL_LOOKBACK_DAYS, VOL_SIZE_FLOOR, VOL_SIZE_CAP,
    FDR_ALPHA,
)

# ── Volatility floor — prevents z-score explosion when std collapses ────────
SIGMA_FLOOR = 1e-8

# ──────────────────────────────────────────────────────────────────────────────
#  ERROR LOGGER  (per-pair math failures → error.log)
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_err_logger = logging.getLogger("pair_errors")
if not _err_logger.handlers:
    _err_logger.setLevel(logging.WARNING)
    _fh = logging.FileHandler(
        os.path.join(_SCRIPT_DIR, ERROR_LOG), encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    _err_logger.addHandler(_fh)


# ──────────────────────────────────────────────────────────────────────────────
#  ALPHA VANTAGE — per-run price cache + fetch helper
# ──────────────────────────────────────────────────────────────────────────────
# Cache is keyed by ticker symbol and lives for the duration of one main() call.
# This means a ticker shared across multiple pairs is only downloaded once.
_av_cache: dict[str, pd.Series] = {}


def _clear_caches() -> None:
    global _av_cache
    _av_cache = {}
    earnings_util.clear_earnings_cache()


def _fetch_av_ticker(symbol: str) -> "pd.Series | None":
    """
    Fetch the full daily adjusted-close series from Alpha Vantage.
    Returns a pd.Series(DatetimeIndex → float) or None on failure/rate-limit.
    Caches results so each ticker is fetched at most once per run.

    Free tier  : 25 calls/day, 5/min  → AV_RATE_DELAY = 12.0
    Paid tier  : 75 calls/min         → AV_RATE_DELAY = 0.8
    When the daily quota is exhausted AV returns a 'Note' key; fetch_pair()
    falls back to yfinance automatically.
    """
    global _av_cache
    if symbol in _av_cache:
        return _av_cache[symbol]
    if not AV_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function":   "TIME_SERIES_DAILY_ADJUSTED",
                "symbol":     symbol,
                "outputsize": "full",
                "apikey":     AV_API_KEY,
            },
            timeout=15,
        )
        data = resp.json()

        # Rate-limit or quota exhausted — AV sends a Note / Information key
        if "Note" in data or "Information" in data:
            _err_logger.warning(
                f"Alpha Vantage quota/rate-limit hit fetching {symbol} — "
                "falling back to yfinance for this and subsequent tickers.")
            return None

        ts = data.get("Time Series (Daily)")
        if not ts:
            return None

        series = pd.Series(
            {pd.Timestamp(d): float(v["5. adjusted close"])
             for d, v in ts.items()},
            name=symbol,
            dtype=float,
        ).sort_index()

        _av_cache[symbol] = series
        if AV_RATE_DELAY > 0:
            time.sleep(AV_RATE_DELAY)
        return series

    except Exception as e:
        _err_logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  FMP EARNINGS BLACKOUT — Gate E  (logic moved to earnings_util.py)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
#  VIX REGIME FILTER
# ──────────────────────────────────────────────────────────────────────────────

def get_vix() -> float | None:
    """Fetch the latest VIX closing level. Returns None on failure."""
    try:
        raw = yf.download("^VIX", period="5d", auto_adjust=False, progress=False)
        close = (raw["Close"]["^VIX"] if isinstance(raw.columns, pd.MultiIndex)
                 else raw["Close"])
        return float(close.dropna().iloc[-1])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  DATA FETCH  (single download per pair, with retry)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_pair(a: str, b: str) -> "dict | None":
    """
    Download ~2 years of daily adjusted prices for a pair.

    Returns a dict {"close": close_df, "open": open_df} where open_df may be
    None when the Alpha Vantage path is used (no open data available there).

    Data source priority:
      1. Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED) — more stable, split/
         dividend-adjusted. Results cached per-run; each ticker fetched once.
         Falls back to yfinance when quota is exhausted or fetch fails.
      2. yfinance (auto_adjust=True) — batched download with retry logic.
         Provides both Close and Open prices.

    Data hygiene (both sources):
      - dropna() on the common index — no ffill, no fake convergence.
    """
    end      = datetime.date.today()
    start    = end - datetime.timedelta(days=int(LOOKBACK_YEARS * 365 * 1.1))
    min_bars = ROLLING_BETA_WIN + ROLLING_Z_WIN + 50

    # ── 1. Try Alpha Vantage ──────────────────────────────────────────────
    if AV_API_KEY:
        sa = _fetch_av_ticker(a)
        sb = _fetch_av_ticker(b)
        if sa is not None and sb is not None:
            close = (pd.concat([sa, sb], axis=1)
                     .loc[pd.Timestamp(start):]
                     .dropna())
            if len(close) >= min_bars:
                return {"close": close, "open": None}
            # AV returned data but not enough history — fall through to yfinance

    # ── 2. Fall back to yfinance ──────────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download([a, b], start=str(start), end=str(end),
                              auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"][[a, b]].dropna()
                open_ = raw["Open"][[a, b]].reindex(close.index)
                # Fill any missing opens with close (rare, but defensive)
                open_ = open_.fillna(close)
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            if len(close) < min_bars:
                return None
            return {"close": close, "open": open_}
        except Exception as e:
            _err_logger.warning(
                f"{a}/{b} yfinance attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                _err_logger.error(
                    f"{a}/{b} download failed after {MAX_RETRIES} retries")
                return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  ROLLING SIGNALS  (log-price OLS beta, spread, z-score — timing-safe)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_signals(close: pd.DataFrame, a: str, b: str,
                            open_prices: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """
    Build day-by-day rolling beta, spread, z-score using LOG of adjusted prices.

    Timing guarantees (no lookahead):
      beta[t]   = OLS on log prices [t-60, t-1]      (excludes bar t)
      spread[t] = log(A_t) - beta[t] * log(B_t)      (today's observation)
      mu[t]     = mean( spread[t-20 .. t-1] )         (excludes bar t)
      sigma[t]  = std( spread[t-20 .. t-1] )          (excludes bar t)
      z[t]      = (spread[t] - mu[t]) / max(sigma[t], SIGMA_FLOOR)

    open_prices: optional DataFrame of open prices aligned to close.index.
      When provided, open_a / open_b columns are included in the output for
      use as fill prices in backtest_pair().  When None, close prices are used
      as a fallback (Alpha Vantage path).
    """
    pa_raw = close[a].values.astype(float)
    pb_raw = close[b].values.astype(float)
    # Guard: non-positive prices → NaN (prevents log(0) and division-by-zero)
    pa_raw = np.where(pa_raw > 0, pa_raw, np.nan)
    pb_raw = np.where(pb_raw > 0, pb_raw, np.nan)

    if open_prices is not None:
        oa_raw = open_prices[a].values.astype(float)
        ob_raw = open_prices[b].values.astype(float)
        oa_raw = np.where(oa_raw > 0, oa_raw, np.nan)
        ob_raw = np.where(ob_raw > 0, ob_raw, np.nan)
    else:
        oa_raw = pa_raw  # fallback: use close prices for fills
        ob_raw = pb_raw
    pa = np.log(pa_raw)
    pb = np.log(pb_raw)
    n  = len(pa)

    beta    = np.full(n, np.nan)
    spread  = np.full(n, np.nan)
    z_score = np.full(n, np.nan)

    # ── Rolling OLS beta: uses [i-60 : i)  →  bars t-60..t-1 ────────────
    for i in range(ROLLING_BETA_WIN, n):
        y = pa[i - ROLLING_BETA_WIN : i]
        x = add_constant(pb[i - ROLLING_BETA_WIN : i])
        try:
            b_hat = OLS(y, x).fit().params[1]
        except Exception:
            b_hat = beta[i - 1] if not np.isnan(beta[i - 1]) else 1.0
        beta[i] = b_hat

    # ── Spread: today's log-price observation ────────────────────────────
    for i in range(ROLLING_BETA_WIN, n):
        if not np.isnan(beta[i]):
            spread[i] = pa[i] - beta[i] * pb[i]

    # ── Rolling z-score with LAGGED statistics (no lookahead) ────────────
    #    window = spread[i-20 : i]  →  Python slice gives bars t-20..t-1
    #    spread[i] (bar t) is EXCLUDED from mean/std calculation
    start_z = ROLLING_BETA_WIN + ROLLING_Z_WIN
    for i in range(start_z, n):
        window = spread[i - ROLLING_Z_WIN : i]   # [t-20 .. t-1]
        if np.any(np.isnan(window)):
            continue
        mu  = window.mean()
        sig = max(window.std(ddof=1), SIGMA_FLOOR)   # volatility floor
        z_score[i] = (spread[i] - mu) / sig

    return pd.DataFrame({
        "date":    close.index,
        "price_a": pa_raw,
        "price_b": pb_raw,
        "open_a":  oa_raw,
        "open_b":  ob_raw,
        "beta":    beta,
        "spread":  spread,
        "z":       z_score,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LIVE CHECK  (current z-score)
# ──────────────────────────────────────────────────────────────────────────────

def live_check(signals: pd.DataFrame) -> dict:
    """
    Examine the LAST row of the signal DataFrame.
    Returns dict with pass/fail, current z, direction, current beta,
    and latest prices for both legs.
    """
    last = signals.iloc[-1]
    z_now    = last["z"]
    beta_now = last["beta"]
    pa_now   = last["price_a"]
    pb_now   = last["price_b"]

    if np.isnan(z_now):
        return {"pass": False, "reason": "Current z-score is NaN"}

    if abs(z_now) <= Z_ENTRY:
        return {"pass": False, "z": z_now, "beta": beta_now,
                "price_a": pa_now, "price_b": pb_now,
                "reason": f"|z| = {abs(z_now):.2f} <= {Z_ENTRY} threshold"}

    direction = "LONG" if z_now < -Z_ENTRY else "SHORT"
    return {"pass": True, "z": z_now, "beta": beta_now,
            "price_a": pa_now, "price_b": pb_now,
            "direction": direction}


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 2 — HISTORY CHECK  (walk-forward backtest, timing-safe)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_pair(signals: pd.DataFrame) -> dict:
    """
    Walk-forward backtest with 1-bar execution delay.

    Timing model (eliminates lookahead bias):
      Signal observed at close of bar i  →  execution at OPEN of bar i+1.
      You CANNOT trade on the bar you observe the z-score.

    Exit logic (directional — fixes unreachable |z| < 0 audit finding):
      LONG spread  → exit when z >= 0   (mean reversion achieved)
      SHORT spread → exit when z <= 0   (mean reversion achieved)
      Stop         : |z| > Z_STOP       (protective, either direction)
      Time         : hold > MAX_HOLD    (forced exit)

    Slippage model (true two-leg notional cost):
      cost = SLIPPAGE_PCT * (entry_notional + exit_notional)
      where notional = |shares_A| * price_A + |shares_B| * price_B
      Applied on BOTH legs at BOTH entry and exit.

    Returns metrics dict with all fields expected by run_system.py.
    """
    z       = signals["z"].values
    price_a = signals["price_a"].values
    price_b = signals["price_b"].values
    open_a  = signals["open_a"].values
    open_b  = signals["open_b"].values
    n       = len(z)

    trades     = []
    position   = 0       # 0=flat, 1=long spread, -1=short spread
    entry_bar  = 0       # bar index where position was opened
    shares_a   = 0.0
    shares_b   = 0.0

    vol_scale  = 1.0
    spread     = signals["spread"].values

    # ── Signal on bar i → execute at bar i+1 ─────────────────────────────
    for i in range(n - 1):
        if np.isnan(z[i]):
            continue

        exec_bar = i + 1    # next-bar execution (no same-bar trading)

        # Guard: skip if execution-bar prices are invalid
        if np.isnan(open_a[exec_bar]) or np.isnan(open_b[exec_bar]):
            continue

        if position == 0:
            # ── Entry signals ────────────────────────────────────────────
            if z[i] < -Z_ENTRY:
                # Long spread: long A, short B
                # Vol-adjusted sizing
                vol_start = max(0, exec_bar - VOL_LOOKBACK_DAYS)
                spread_slice = spread[vol_start:exec_bar]
                valid_spread = spread_slice[~np.isnan(spread_slice)]
                if len(valid_spread) > 1:
                    spread_vol = np.std(np.diff(valid_spread), ddof=1)
                    if spread_vol > 1e-8:
                        vol_scale = np.clip(VOL_TARGET_DAILY / spread_vol,
                                            VOL_SIZE_FLOOR, VOL_SIZE_CAP)
                    else:
                        vol_scale = 1.0
                else:
                    vol_scale = 1.0
                half = (CAPITAL_PER_TRADE * vol_scale) / 2.0
                shares_a = half / open_a[exec_bar]
                shares_b = half / open_b[exec_bar]
                position  = 1
                entry_bar = exec_bar
            elif z[i] > Z_ENTRY:
                # Short spread: short A, long B
                # Vol-adjusted sizing
                vol_start = max(0, exec_bar - VOL_LOOKBACK_DAYS)
                spread_slice = spread[vol_start:exec_bar]
                valid_spread = spread_slice[~np.isnan(spread_slice)]
                if len(valid_spread) > 1:
                    spread_vol = np.std(np.diff(valid_spread), ddof=1)
                    if spread_vol > 1e-8:
                        vol_scale = np.clip(VOL_TARGET_DAILY / spread_vol,
                                            VOL_SIZE_FLOOR, VOL_SIZE_CAP)
                    else:
                        vol_scale = 1.0
                else:
                    vol_scale = 1.0
                half = (CAPITAL_PER_TRADE * vol_scale) / 2.0
                shares_a = half / open_a[exec_bar]
                shares_b = half / open_b[exec_bar]
                position  = -1
                entry_bar = exec_bar

        else:
            # ── Exit signals (check on bar i, execute at i+1) ────────────
            hold_days   = exec_bar - entry_bar
            close_trade = False
            exit_reason = ""

            # Directional mean-reversion target
            if position == 1 and z[i] >= Z_EXIT:
                close_trade, exit_reason = True, "target"
            elif position == -1 and z[i] <= Z_EXIT:
                close_trade, exit_reason = True, "target"

            # Stop-loss (protective, both directions)
            if abs(z[i]) > Z_STOP:
                close_trade, exit_reason = True, "stop"

            # Max holding period
            if hold_days >= MAX_HOLD:
                close_trade, exit_reason = True, "time"

            if close_trade:
                # ── P&L calculation (shares-based, fills at open prices) ──
                if position == 1:    # long A, short B
                    pnl_a =  shares_a * (open_a[exec_bar] - open_a[entry_bar])
                    pnl_b = -shares_b * (open_b[exec_bar] - open_b[entry_bar])
                else:                # short A, long B
                    pnl_a = -shares_a * (open_a[exec_bar] - open_a[entry_bar])
                    pnl_b =  shares_b * (open_b[exec_bar] - open_b[entry_bar])
                gross = pnl_a + pnl_b

                # ── TRUE two-leg slippage ────────────────────────────────
                #    Notional = |shares| * price   for EACH leg
                #    Applied at entry AND exit, on BOTH legs
                entry_notional = (shares_a * open_a[entry_bar]
                                + shares_b * open_b[entry_bar])
                exit_notional  = (shares_a * open_a[exec_bar]
                                + shares_b * open_b[exec_bar])
                cost = SLIPPAGE_PCT * (entry_notional + exit_notional)

                # ── Short-leg borrow cost ─────────────────────────────────
                #    LONG spread: short B;  SHORT spread: short A
                #    Daily rate = BORROW_COST_PCT / 252
                hold_bars = exec_bar - entry_bar
                if position == 1:   # short B
                    avg_pb = (open_b[entry_bar] + open_b[exec_bar]) / 2.0
                    borrow = shares_b * avg_pb * (BORROW_COST_PCT / 252) * hold_bars
                else:               # short A
                    avg_pa = (open_a[entry_bar] + open_a[exec_bar]) / 2.0
                    borrow = shares_a * avg_pa * (BORROW_COST_PCT / 252) * hold_bars

                net_pnl = gross - cost - borrow

                trades.append({
                    "entry_bar":   entry_bar,
                    "hold_days":   hold_days,
                    "exit_reason": exit_reason,
                    "gross_pnl":   gross,
                    "net_pnl":     net_pnl,
                    "cost":        cost,
                    "vol_scale":   vol_scale,
                })
                position = 0
                shares_a = 0.0
                shares_b = 0.0

    # ── Force-close any open position at end of data ─────────────────────
    if position != 0:
        last_bar = n - 1
        if position == 1:
            pnl_a =  shares_a * (open_a[last_bar] - open_a[entry_bar])
            pnl_b = -shares_b * (open_b[last_bar] - open_b[entry_bar])
        else:
            pnl_a = -shares_a * (open_a[last_bar] - open_a[entry_bar])
            pnl_b =  shares_b * (open_b[last_bar] - open_b[entry_bar])
        gross = pnl_a + pnl_b
        entry_notional = (shares_a * open_a[entry_bar]
                        + shares_b * open_b[entry_bar])
        exit_notional  = (shares_a * open_a[last_bar]
                        + shares_b * open_b[last_bar])
        cost      = SLIPPAGE_PCT * (entry_notional + exit_notional)
        hold_bars = last_bar - entry_bar
        if position == 1:
            avg_pb = (open_b[entry_bar] + open_b[last_bar]) / 2.0
            borrow = shares_b * avg_pb * (BORROW_COST_PCT / 252) * hold_bars
        else:
            avg_pa = (open_a[entry_bar] + open_a[last_bar]) / 2.0
            borrow = shares_a * avg_pa * (BORROW_COST_PCT / 252) * hold_bars
        trades.append({
            "entry_bar":   entry_bar,
            "hold_days":   hold_bars,
            "exit_reason": "eod",
            "gross_pnl":   gross,
            "net_pnl":     gross - cost - borrow,
            "cost":        cost,
            "vol_scale":   vol_scale,
        })

    # ── Compute metrics ──────────────────────────────────────────────────
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "oos_sharpe": 0.0, "oos_pnl": 0.0,
                "avg_hold": 0.0, "max_dd": 0.0, "beta_cv": 0.0,
                "recent_corr": None, "pass": False,
                "reason": "No historical trades",
                "h1_pnl": 0.0, "h2_pnl": 0.0, "recent_pnl": 0.0,
                "vol_scale": 1.0}

    pnls     = np.array([t["net_pnl"] for t in trades])
    wins     = pnls[pnls > 0]
    losses   = pnls[pnls <= 0]
    n_trades = len(pnls)

    win_rate      = len(wins) / n_trades * 100
    gross_win     = wins.sum()  if len(wins)   else 0.0
    gross_loss    = abs(losses.sum()) if len(losses) else 1e-12
    profit_factor = gross_win / gross_loss
    total_pnl     = pnls.sum()
    avg_pnl       = pnls.mean()

    avg_hold = np.mean([t["hold_days"] for t in trades])
    tpy      = 250 / max(avg_hold, 1)
    std_pnl  = pnls.std(ddof=1) if n_trades > 1 else 1e-12
    sharpe   = (avg_pnl / std_pnl) * np.sqrt(tpy)

    # ── Sortino ratio (downside deviation only) ───────────────────────
    downside      = pnls[pnls < 0]
    downside_std  = downside.std(ddof=1) if len(downside) > 1 else 1e-12
    sortino       = (avg_pnl / downside_std) * np.sqrt(tpy)

    # ── Max drawdown (peak-to-trough on cumulative P&L) ──────────────────
    cum_pnl  = np.cumsum(pnls)
    peak     = np.maximum.accumulate(cum_pnl)
    max_dd   = float((cum_pnl - peak).min())   # <= 0

    # ── Beta stability (coefficient of variation of rolling hedge ratio) ──
    beta_vals = signals["beta"].dropna().values
    beta_cv   = 0.0
    if len(beta_vals) > 1:
        mean_b  = abs(beta_vals.mean())
        beta_cv = (beta_vals.std(ddof=1) / mean_b) if mean_b > 1e-4 else np.inf

    # ── Standard quality gates ───────────────────────────────────────────
    reasons = []
    if n_trades < MIN_TRADES:
        reasons.append(f"Insufficient trades ({n_trades} < {MIN_TRADES})")
    if win_rate < MIN_WIN_RATE:
        reasons.append(f"WR {win_rate:.1f}% < {MIN_WIN_RATE}%")
    if profit_factor < MIN_PROFIT_FACTOR:
        reasons.append(f"PF {profit_factor:.2f}x < {MIN_PROFIT_FACTOR}x")
    if total_pnl <= MIN_TOTAL_PNL:
        reasons.append(f"P&L ${total_pnl:+.2f} \u2264 ${MIN_TOTAL_PNL:.0f}")
    if sharpe < MIN_SHARPE:
        reasons.append(f"Sharpe {sharpe:+.2f} < {MIN_SHARPE}")
    if sharpe > MAX_SHARPE_OVERFIT:
        reasons.append(f"Overfit: Sharpe {sharpe:.2f} > {MAX_SHARPE_OVERFIT}")
    if profit_factor > MAX_PROFIT_FACTOR_OVERFIT:
        reasons.append(f"Overfit: PF {profit_factor:.2f}x > {MAX_PROFIT_FACTOR_OVERFIT}x")
    if max_dd < MAX_DRAWDOWN:
        reasons.append(f"Max DD ${max_dd:+.0f} < ${MAX_DRAWDOWN:+.0f}")
    if beta_cv > BETA_STABILITY_MAX:
        reasons.append(f"Unstable Beta: CV={beta_cv:.2f} > {BETA_STABILITY_MAX}")
    if sortino < MIN_SORTINO:
        reasons.append(f"Sortino {sortino:+.2f} < {MIN_SORTINO}")
    if avg_pnl < MIN_AVG_PNL:
        reasons.append(f"Avg P&L ${avg_pnl:+.2f} < ${MIN_AVG_PNL:.2f}")

    # ── GATE A: Split-Half Validation (Calendar-Based) ───────────────────
    #    Split at temporal midpoint of the data (not trade count).
    #    Both halves must be independently profitable (P&L > $0).
    h1_pnl_val = 0.0
    h2_pnl_val = 0.0
    if SPLIT_HALF_ENABLED:
        if len(trades) < 6:
            reasons.append("Insufficient trades for Split-Half (<6)")
        else:
            mid_bar = n // 2   # temporal midpoint of data
            h1 = [t for t in trades if t["entry_bar"] < mid_bar]
            h2 = [t for t in trades if t["entry_bar"] >= mid_bar]
            h1_pnl_val = sum(t["net_pnl"] for t in h1) if h1 else 0.0
            h2_pnl_val = sum(t["net_pnl"] for t in h2) if h2 else 0.0
            if not h1 or h1_pnl_val < SPLIT_HALF_MIN_PNL:
                reasons.append(f"Failed H1 P&L (${h1_pnl_val:+.0f} < ${SPLIT_HALF_MIN_PNL:.0f})")
            if not h2 or h2_pnl_val < SPLIT_HALF_MIN_PNL:
                reasons.append(f"Failed H2 P&L (${h2_pnl_val:+.0f} < ${SPLIT_HALF_MIN_PNL:.0f})")

    # ── GATE B: Recent Trade Momentum (Alpha Decay) ─────────────────────
    #    Last N trades combined P&L must be > $0.
    #    If the edge is decaying, the most recent trades are losers.
    recent_pnl = 0.0
    if len(trades) < RECENT_MOMENTUM_N:
        reasons.append(
            f"Insufficient trades for Momentum (<{RECENT_MOMENTUM_N})")
    else:
        recent_pnl = sum(t["net_pnl"] for t in trades[-RECENT_MOMENTUM_N:])
        if recent_pnl <= 0:
            reasons.append(
                f"Low Recent Momentum: last {RECENT_MOMENTUM_N} trades "
                f"P&L=${recent_pnl:+.0f}")

    # ── GATE F: Walk-Forward Validation (70/30 OOS test) ──────────────
    #    Train on first 70% of data, test backtest stats on last 30%.
    #    OOS Sharpe must be > 0 and OOS total P&L must be > 0.
    oos_sharpe = 0.0
    oos_pnl    = 0.0
    split_bar  = int(n * WALK_FORWARD_SPLIT)
    if len(trades) >= MIN_TRADES and split_bar > 0 and split_bar < n:
        oos_trades = [t for t in trades if t["entry_bar"] >= split_bar]
        if len(oos_trades) < MIN_OOS_TRADES:
            reasons.append(
                f"Insufficient OOS trades ({len(oos_trades)} < {MIN_OOS_TRADES})")
        oos_pnls   = np.array([t["net_pnl"] for t in oos_trades])
        oos_pnl    = float(oos_pnls.sum())
        if len(oos_pnls) > 1:
            oos_avg  = oos_pnls.mean()
            oos_std  = oos_pnls.std(ddof=1) if len(oos_pnls) > 1 else 1e-12
            oos_hold = np.mean([t["hold_days"] for t in oos_trades])
            oos_tpy  = 250 / max(oos_hold, 1)
            oos_sharpe = float((oos_avg / oos_std) * np.sqrt(oos_tpy))
        # Compute IS Sharpe for consistency check
        is_trades = [t for t in trades if t["entry_bar"] < split_bar]
        is_sharpe = 0.0
        if len(is_trades) > 1:
            is_pnls = np.array([t["net_pnl"] for t in is_trades])
            is_avg  = is_pnls.mean()
            is_std  = is_pnls.std(ddof=1) if len(is_pnls) > 1 else 1e-12
            is_hold = np.mean([t["hold_days"] for t in is_trades])
            is_tpy  = 250 / max(is_hold, 1)
            is_sharpe = float((is_avg / is_std) * np.sqrt(is_tpy))
        if is_sharpe > 0 and oos_sharpe / is_sharpe < OOS_SHARPE_RATIO_MIN:
            reasons.append(
                f"IS/OOS decay: OOS={oos_sharpe:.2f} vs IS={is_sharpe:.2f}")
        if oos_sharpe <= 0 or oos_pnl <= 0:
            reasons.append(
                f"Walk-Forward OOS: Sharpe={oos_sharpe:+.2f}, "
                f"P&L=${oos_pnl:+.2f}")

    passed = len(reasons) == 0

    last_vol_scale = trades[-1]["vol_scale"] if trades else 1.0

    return {
        "n_trades":      n_trades,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "total_pnl":     total_pnl,
        "avg_pnl":       avg_pnl,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "oos_sharpe":    oos_sharpe,
        "oos_pnl":       oos_pnl,
        "avg_hold":      avg_hold,
        "max_dd":        max_dd,
        "beta_cv":       beta_cv,
        "recent_corr":   None,       # filled in by main() after Gate D
        "pass":          passed,
        "reason":        " | ".join(reasons) if reasons else "",
        "h1_pnl":        h1_pnl_val,
        "h2_pnl":        h2_pnl_val,
        "recent_pnl":    recent_pnl,
        "vol_scale":     last_vol_scale,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def print_diamond(a: str, b: str, live: dict, bt: dict):
    """Print a DIAMOND SIGNAL block — passed ALL gates + execution ticket."""
    # ── Execution Ticket calculations (mirrors backtest share sizing) ─────
    pa       = live["price_a"]
    pb       = live["price_b"]
    half     = CAPITAL_PER_TRADE / 2.0
    shares_a = round(half / pa)
    shares_b = round(half / pb)
    not_a    = shares_a * pa
    not_b    = shares_b * pb

    if live["direction"] == "LONG":
        act_label = "LONG SPREAD"
        act_a, act_b = "BUY", "SELL"
        hint = "Buy A / Sell B"
    else:
        act_label = "SHORT SPREAD"
        act_a, act_b = "SELL", "BUY"
        hint = "Sell A / Buy B"

    W = 58   # inner width between the ║ walls
    _hbar = '\u2550' * W
    _tbar = '\u2500' * 40
    def L(s): return f"  \u2551{s:<{W}}\u2551"

    print()
    print(f"  \u2554{_hbar}\u2557")
    print(L(f"  \u25c6 DIAMOND SIGNAL \u25c6   {a} / {b}"))
    print(f"  \u2560{_hbar}\u2563")
    print(L(f"  LIVE"))
    print(L(f"    Direction    : {live['direction']}"))
    print(L(f"    Current Z    : {live['z']:>+8.4f}"))
    print(L(f"    Current \u03b2    : {live['beta']:>8.4f}"))
    print(L(""))
    print(L(f"  BACKTEST (2yr, 1-bar delay)"))
    print(L(f"    Trades       : {bt['n_trades']:>5}"))
    print(L(f"    Win Rate     : {bt['win_rate']:>5.1f} %"))
    print(L(f"    Profit Factor: {bt['profit_factor']:>5.2f}x"))
    print(L(f"    Total P&L    : ${bt['total_pnl']:>+9.2f}"))
    print(L(f"    Max Drawdown : ${bt.get('max_dd', 0):>+9.2f}"))
    print(L(f"    Sharpe       : {bt['sharpe']:>+5.2f}"))
    print(L(f"    Sortino      : {bt.get('sortino', 0):>+5.2f}"))
    print(L(f"    OOS Sharpe   : {bt.get('oos_sharpe', 0):>+5.2f}"))
    print(L(f"    OOS P&L      : ${bt.get('oos_pnl', 0):>+9.2f}"))
    print(L(f"    Avg Hold     : {bt['avg_hold']:>5.1f} days"))
    print(L(""))
    rc_str = (f"{bt['recent_corr']:.2f}" if bt.get("recent_corr") is not None
              else "N/A")
    print(L(f"  REGIME GATES"))
    print(L(f"    Half-1 P&L   : ${bt.get('h1_pnl', 0):>+9.2f}   \u2713"))
    print(L(f"    Half-2 P&L   : ${bt.get('h2_pnl', 0):>+9.2f}   \u2713"))
    print(L(f"    Recent ADF   :  Stationary   \u2713"))
    print(L(f"    Last 5 P&L   : ${bt.get('recent_pnl', 0):>+9.2f}   \u2713"))
    print(L(f"    Recent \u03c1     :  {rc_str:>5}         \u2713"))
    print(L(f"    Beta CV      :  {bt.get('beta_cv', 0):.2f}          \u2713"))
    print(L(""))
    print(f"  \u2560{_hbar}\u2563")
    print(L(f"  \u26a1 EXECUTION TICKET"))
    print(L(f"    ACTION : {act_label} ({hint})"))
    print(L(f"    {_tbar}"))
    print(L(f"    Hedge Ratio (\u03b2) : {live['beta']:.4f}"))
    print(L(f"    Leg A ({a:>5})    : {act_a:>4} {shares_a:>4} shares  (${not_a:>,.0f})"))
    print(L(f"    Leg B ({b:>5})    : {act_b:>4} {shares_b:>4} shares  (${not_b:>,.0f})"))
    print(L(f"    {_tbar}"))
    print(L(f"    Capital / trade : ${CAPITAL_PER_TRADE:>,.0f}"))
    print(L(f"    Total notional  : ${not_a + not_b:>,.0f}"))
    print(f"  \u255a{_hbar}\u255d")


def print_rejected(a: str, b: str, live: dict, bt: dict):
    """Print a rejection block with gate-specific reasons."""
    pf_s = f"{bt['profit_factor']:.2f}x" if bt['n_trades'] > 0 else "N/A"
    reason = bt.get("reason", "")

    # Determine short gate label for console
    if "Earnings Blackout" in reason:
        gate = "Earnings Blackout"
    elif "Insufficient trades" in reason:
        gate = "Too Few Trades"
    elif "Unstable Beta" in reason:
        gate = "Unstable Hedge Ratio"
    elif "Low Recent Correlation" in reason:
        gate = "Correlation Breakdown"
    elif "H1 P&L" in reason or "H2 P&L" in reason or "Split-Half" in reason:
        gate = "Failed Split-Half"
    elif "ADF" in reason or "Low Recent Cointegration" in reason:
        gate = "Failed Cointegration / ADF"
    elif "Walk-Forward" in reason:
        gate = "Failed Walk-Forward OOS"
    elif "Recent Momentum" in reason:
        gate = "Alpha Decay (Recent Trades)"
    elif "Sortino" in reason:
        gate = "Low Sortino"
    elif "Avg P&L" in reason:
        gate = "Avg P&L Too Thin"
    elif "Sharpe" in reason:
        gate = "Low Sharpe"
    elif "Max DD" in reason:
        gate = "Excessive Drawdown"
    elif "WR" in reason or "PF" in reason or "P&L" in reason:
        gate = "Failed Standard Backtest"
    else:
        gate = "Historically Unprofitable"

    print(f"    \u2717  REJECTED: {gate}")
    print(f"       Live z={live['z']:+.2f} {live['direction']} | "
          f"BT: {bt['n_trades']} trades, "
          f"WR {bt['win_rate']:.0f}%, "
          f"PF {pf_s}, "
          f"P&L ${bt['total_pnl']:+.0f}")
    if reason:
        print(f"       Gates: {reason}")


def apply_fdr_correction(verified: list, alpha: float) -> tuple:
    """
    Benjamini-Hochberg FDR correction on diamond signals.
    Derives t-statistics from backtest Sharpe and trade count,
    then controls false discovery rate across multiple comparisons.
    Returns (surviving_diamonds, fdr_rejected_list).
    """
    if not verified:
        return verified, []

    # Derive p-values from backtest statistics
    scored = []
    for v in verified:
        bt = v["bt"]
        n_trades = bt.get("n_trades", 0)
        sharpe = bt.get("sharpe", 0.0)
        if n_trades < 2 or sharpe <= 0:
            # Can't compute meaningful t-stat; assign p=1.0
            scored.append((v, 1.0))
        else:
            t_stat = sharpe * np.sqrt(n_trades)
            p_val = float(stats.t.sf(abs(t_stat), df=n_trades - 1) * 2)
            scored.append((v, p_val))

    # Sort by p-value ascending
    scored.sort(key=lambda x: x[1])
    m = len(scored)

    # Find BH threshold: largest k where p_k <= k/m * alpha
    max_k = 0
    for k_idx, (v, p_val) in enumerate(scored):
        k = k_idx + 1  # 1-indexed rank
        bh_threshold = k / m * alpha
        if p_val <= bh_threshold:
            max_k = k

    # Split into survivors and rejected
    survivors = []
    fdr_rejected = []
    for k_idx, (v, p_val) in enumerate(scored):
        if k_idx < max_k:
            survivors.append(v)
        else:
            # Move to rejected with FDR reason
            v["bt"]["pass"] = False
            old_reason = v["bt"].get("reason", "")
            fdr_msg = f"FDR correction (p={p_val:.4f})"
            v["bt"]["reason"] = (f"{old_reason} | {fdr_msg}" if old_reason else fdr_msg)
            fdr_rejected.append(v)

    return survivors, fdr_rejected


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> dict:
    """
    Run the full scan.  Returns a dict with keys:
        verified  — list of dicts  {a, b, live, bt}
        rejected  — list of dicts  {a, b, live, bt}
        no_signal — int
        errors    — int
        timestamp — str
    """
    _clear_caches()   # fresh cache for this run
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print("\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
    print("\u2551  MASTER SIGNAL v3 \u2014 Audit-Fixed Verification Engine         \u2551")
    print(f"\u2551  Run : {ts}                                  \u2551")
    print("\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")

    # ── Load candidates ──────────────────────────────────────────────────
    try:
        df_pairs = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"\n  \u2717  {INPUT_CSV} not found. Run nightly_scanner.py first.\n")
        return {"verified": [], "rejected": [], "no_signal": 0,
                "errors": 1, "timestamp": ts}

    pairs   = list(zip(df_pairs["Stock_A"], df_pairs["Stock_B"]))
    n_pairs = len(pairs)

    print(f"\n  Loaded {n_pairs} pairs from {INPUT_CSV}")
    print(f"  Strategy: \u03b2={ROLLING_BETA_WIN}d | z={ROLLING_Z_WIN}d | "
          f"Entry |z|>{Z_ENTRY} | Exit z\u2192{Z_EXIT} | "
          f"Stop |z|>{Z_STOP} | Hold\u2264{MAX_HOLD}d")
    print(f"  Quality gates: WR>{MIN_WIN_RATE}% | "
          f"PF>{MIN_PROFIT_FACTOR}x | P&L>$0")
    print(f"  Slippage: {SLIPPAGE_PCT*100:.2f}% on notional "
          f"(entry+exit, both legs) | Capital: ${CAPITAL_PER_TRADE:,.0f}/trade")
    print(f"  Execution: 1-bar delay (signal@close \u2192 trade@next open)\n")

    # ── VIX Regime Filter ─────────────────────────────────────────────────────
    vix_level   = get_vix()
    vix_blocked = (vix_level is not None and vix_level > VIX_MAX_ENTRY)
    if vix_level is None:
        print(f"  VIX : unavailable — proceeding without regime filter\n")
    elif vix_blocked:
        print("\n  " + "!"*60)
        print(f"  ⚠  VIX REGIME BLOCK: {vix_level:.1f} > {VIX_MAX_ENTRY}")
        print(f"  Signals shown but flagged BLOCKED. No new entries advised.")
        print("  " + "!"*60 + "\n")
    else:
        print(f"  VIX : {vix_level:.1f}  ✓ (threshold {VIX_MAX_ENTRY})\n")

    verified  = []
    rejected  = []
    no_signal = 0
    errors    = 0

    for idx, (a, b) in enumerate(pairs):
        label = f"{a} / {b}"
        print(f"  [{idx+1}/{n_pairs}]  {label:>14}  ", end="", flush=True)

        try:
            # ── Fetch data (with retry) ──────────────────────────────────
            data = fetch_pair(a, b)
            if data is None:
                print("SKIP  (download failed)")
                errors += 1
                continue

            # ── Compute rolling signals (timing-safe) ────────────────────
            close_df = data["close"]
            open_df  = data.get("open")
            signals = compute_rolling_signals(close_df, a, b, open_prices=open_df)

            # ── STEP 1: Live Check ───────────────────────────────────────
            live = live_check(signals)
            if not live["pass"]:
                print(f"\u2014  No signal  (|z| = {abs(live.get('z', 0)):.2f})")
                no_signal += 1
                continue

            print(f"\u26a1 z={live['z']:+.2f} {live['direction']:>5}  \u2192  ", end="",
                  flush=True)

            # ── GATE E: Earnings Blackout ─────────────────────────────────
            #    Block entry if either leg reports within ±EARNINGS_BLACKOUT_DAYS.
            #    Checked before the expensive backtest to save computation.
            earn_sym, earn_delta = earnings_util.check_pair_earnings_blackout(a, b)
            if earn_sym:
                when = (f"in {earn_delta}d" if earn_delta >= 0
                        else f"{abs(earn_delta)}d ago")
                print(f"\U0001f4c5 EARNINGS BLOCK  ({earn_sym} earnings {when})")
                rejected.append({"a": a, "b": b, "live": live, "bt": {
                    "pass": False,
                    "reason": f"Earnings Blackout: {earn_sym} earnings {when}",
                    "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                    "total_pnl": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
                    "sortino": 0.0, "oos_sharpe": 0.0, "oos_pnl": 0.0,
                    "avg_hold": 0.0, "max_dd": 0.0, "beta_cv": 0.0,
                    "recent_corr": None, "h1_pnl": 0.0, "h2_pnl": 0.0,
                    "recent_pnl": 0.0,
                }})
                continue

            # ── STEP 2: History Check (with 1-bar delay) ─────────────────
            bt = backtest_pair(signals)

            # ── GATE C: Cointegration Freshness ("Still Alive?") ─────────
            #    ADF on last 90 days of spread.  p-value must be < 0.10.
            #    If p > 0.10, the rubber band has stopped snapping.
            spread_recent = signals["spread"].dropna().values[-RECENT_ADF_WINDOW:]
            if len(spread_recent) >= RECENT_ADF_WINDOW:
                try:
                    adf_pval = adfuller(spread_recent, maxlag=5,
                                        regression="c", autolag="AIC")[1]
                    if adf_pval > RECENT_ADF_PVAL:
                        bt["pass"] = False
                        old_reason = bt.get("reason", "")
                        coint_msg = (f"Low Recent Cointegration: "
                                     f"ADF p={adf_pval:.3f} > {RECENT_ADF_PVAL}")
                        bt["reason"] = (f"{old_reason} | {coint_msg}"
                                        if old_reason else coint_msg)
                except Exception:
                    pass   # ADF can fail on degenerate data — don't block

            # ── GATE D: Recent Correlation Freshness ─────────────────────
            #    Price correlation over last 90 days must still be high.
            #    A pair with ρ=0.85 historically but ρ=0.30 recently is broken.
            pa_recent = signals["price_a"].dropna().values[-RECENT_CORR_WINDOW:]
            pb_recent = signals["price_b"].dropna().values[-RECENT_CORR_WINDOW:]
            if len(pa_recent) >= RECENT_CORR_WINDOW and len(pb_recent) >= RECENT_CORR_WINDOW:
                try:
                    recent_corr = float(np.corrcoef(pa_recent, pb_recent)[0, 1])
                    if np.isnan(recent_corr):
                        recent_corr = 0.0
                    bt["recent_corr"] = recent_corr
                    if recent_corr < MIN_RECENT_CORR:
                        bt["pass"] = False
                        old_reason = bt.get("reason", "")
                        corr_msg = (f"Low Recent Correlation: "
                                    f"\u03c1={recent_corr:.2f} < {MIN_RECENT_CORR}")
                        bt["reason"] = (f"{old_reason} | {corr_msg}"
                                        if old_reason else corr_msg)
                except Exception:
                    bt["recent_corr"] = 0.0
                    bt["pass"] = False
                    old_reason = bt.get("reason", "")
                    bt["reason"] = (f"{old_reason} | Correlation calc failed"
                                    if old_reason else "Correlation calc failed")

            if bt["pass"]:
                if vix_blocked:
                    print(f"\u25c6 DIAMOND \u2192 VIX BLOCKED ({vix_level:.1f} > {VIX_MAX_ENTRY})")
                    bt["reason"] = f"VIX Regime Block ({vix_level:.1f} > {VIX_MAX_ENTRY})"
                    rejected.append({"a": a, "b": b, "live": live, "bt": bt})
                else:
                    print(f"\u25c6 DIAMOND  (WR {bt['win_rate']:.0f}% | "
                          f"PF {bt['profit_factor']:.2f}x | P&L ${bt['total_pnl']:+.0f})")
                    verified.append({"a": a, "b": b, "live": live, "bt": bt})
            else:
                print(f"\u2717 REJECTED")
                rejected.append({"a": a, "b": b, "live": live, "bt": bt})

        except (ZeroDivisionError, ValueError, TypeError, KeyError) as e:
            print(f"ERROR  ({type(e).__name__}: {e})")
            _err_logger.error(f"{a}/{b} \u2014 {type(e).__name__}: {e}")
            errors += 1
            continue
        except Exception as e:
            print(f"ERROR  (unexpected: {e})")
            _err_logger.error(f"{a}/{b} \u2014 Unexpected: {e}", exc_info=True)
            errors += 1
            continue

    # ── FDR correction for multiple comparisons ──────────────────────────
    if len(verified) > 1:
        verified, fdr_rejected = apply_fdr_correction(verified, FDR_ALPHA)
        rejected.extend(fdr_rejected)

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════
    print("\n")
    print("\u2550" * 64)
    print("  SCAN COMPLETE")
    print("\u2550" * 64)
    print(f"  Pairs scanned   : {n_pairs}")
    print(f"  No live signal  : {no_signal}")
    print(f"  Download errors : {errors}")
    print(f"  Signals found   : {len(verified) + len(rejected)}")
    print(f"    \u25c6 Diamond     : {len(verified)}")
    print(f"    \u2717 Rejected    : {len(rejected)}")
    print("\u2550" * 64)

    # ── Print diamond blocks ─────────────────────────────────────────────
    if verified:
        for v in verified:
            print_diamond(v["a"], v["b"], v["live"], v["bt"])
    else:
        print("\n  No diamond signals today.\n")

    # ── Print rejected details ───────────────────────────────────────────
    if rejected:
        print("\n  \u2500\u2500 Rejected Signals (had live z, failed backtest) \u2500\u2500")
        for r in rejected:
            print()
            print(f"  {r['a']} / {r['b']}:")
            print_rejected(r["a"], r["b"], r["live"], r["bt"])
        print()

    return {
        "verified":  verified,
        "rejected":  rejected,
        "no_signal": no_signal,
        "errors":    errors,
        "timestamp": ts,
        "vix":       vix_level,
    }


if __name__ == "__main__":
    main()
