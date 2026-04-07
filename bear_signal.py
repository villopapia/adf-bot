"""
================================================================================
 bear_signal.py -- Bear Market Signal Engine
================================================================================
 Two modules for elevated-volatility / bear-market environments:

   Module 1 - Mean-Reversion Bounce
   ---------------------------------
   Uses Connors RSI(2) + Internal Bar Strength (IBS) to identify
   short-duration oversold bounces in SPY.  Entry when RSI(2) < 10 AND
   IBS < 0.20.  Exit when price recovers above 5-day SMA, IBS > 0.80,
   or max hold (5 days).  No individual stop loss (Connors research shows
   stops hurt RSI(2) systems on SPY).

   Module 2 - Inverse ETF Trend Short
   ------------------------------------
   Uses SH (ProShares Short S&P500) to capture sustained downtrends.
   Entry when SPY is below both 50-day and 200-day SMA AND 20-day return
   is below -5%.  Exit when SPY recovers above its 20-day SMA or max
   hold (15 days).  Half-sized position to account for inverse ETF decay.

 No-Lookahead Rule
 -----------------
 Signal observed at close of bar i-1  ->  execution at bar i close.
 A trade can NEVER execute on the same bar the signal is observed.

 Quality Gates (both modules, all must pass)
 --------------------------------------------
 - min_trades       : >= BEAR_BT_MIN_TRADES
 - win_rate         : >= BEAR_BT_MIN_WIN_RATE %
 - profit_factor    : >= BEAR_BT_MIN_PROFIT_FACTOR
 - sharpe           : >= BEAR_BT_MIN_SHARPE
 - total_pnl        : >= BEAR_BT_MIN_TOTAL_PNL
 - oos_pnl          : > 0  (walk-forward OOS 30%)

 Activation
 ----------
 Bear module only activates when VIX > BEAR_VIX_ACTIVATE (default 25).
================================================================================
"""

import warnings
import datetime
import time
import os
import sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
#  PATH SETUP
# ------------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ------------------------------------------------------------------------------
#  CONFIGURATION  (single source of truth in config.py)
# ------------------------------------------------------------------------------
from config import (
    BEAR_MODULE_ENABLED, BEAR_VIX_ACTIVATE,
    BOUNCE_INSTRUMENTS,
    BOUNCE_INSTRUMENT, BOUNCE_RSI_PERIOD, BOUNCE_RSI_ENTRY,
    BOUNCE_IBS_ENTRY, BOUNCE_EXIT_SMA, BOUNCE_IBS_EXIT,
    BOUNCE_MAX_HOLD, BOUNCE_CAPITAL_BASE, BOUNCE_USE_STOP,
    BOUNCE_SLIPPAGE_PCT, BOUNCE_VIX_TIERS,
    SHORT_ENABLED, SHORT_INSTRUMENT,
    SHORT_SPY_BELOW_50SMA, SHORT_SPY_BELOW_200SMA,
    SHORT_MOM_LOOKBACK, SHORT_MOM_THRESHOLD,
    SHORT_EXIT_SMA, SHORT_MAX_HOLD,
    SHORT_CAPITAL_SCALE, SHORT_SLIPPAGE_PCT,
    CAPIT_VIX_SPIKE, CAPIT_VIX_DECLINE_PCT, CAPIT_BOOST_SCALE,
    BEAR_BT_MIN_TRADES, BEAR_BT_MIN_WIN_RATE,
    BEAR_BT_MIN_PROFIT_FACTOR, BEAR_BT_MIN_SHARPE, BEAR_BT_MIN_TOTAL_PNL,
    MAX_RETRIES, RETRY_DELAY, ERROR_LOG,
)


# ==============================================================================
#  HELPER -- LOG TO error.log
# ==============================================================================

def _log_error(timestamp: str, context: str, ticker: str, exc: Exception) -> None:
    """Append a single error line to error.log."""
    try:
        path = os.path.join(_SCRIPT_DIR, ERROR_LOG)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                f"{timestamp} | bear_signal | {context} | {ticker} | {exc}\n"
            )
    except Exception:
        pass


# ==============================================================================
#  1. FETCH INSTRUMENT
# ==============================================================================

def fetch_instrument(ticker: str, years: int = 2) -> "dict | None":
    """
    Download OHLCV data for a single instrument (SPY, SH, etc.) via yfinance.

    Returns a dict with keys: high, low, close, volume (all pd.Series with
    DatetimeIndex).  Returns None on failure after MAX_RETRIES attempts.
    """
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=int(years * 365.25) + 30)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                ticker,
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError("empty dataframe returned")

            # yfinance sometimes returns a MultiIndex with the ticker as level 0
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            # Normalise column names to lowercase
            raw.columns = [c.lower() for c in raw.columns]

            required = {"high", "low", "close", "volume"}
            if not required.issubset(set(raw.columns)):
                raise KeyError(f"Missing columns: {required - set(raw.columns)}")

            raw = raw.dropna(subset=["close"])
            if len(raw) < 100:
                raise ValueError(
                    f"Only {len(raw)} rows after dropna -- insufficient history"
                )

            return {
                "high":   raw["high"],
                "low":    raw["low"],
                "close":  raw["close"],
                "volume": raw["volume"],
            }

        except Exception as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _log_error(ts, "fetch_instrument", ticker, exc)
                return None

    return None


# ==============================================================================
#  2. RSI  (Wilder smoothing, Connors RSI-2 by default)
# ==============================================================================

def compute_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """
    Standard RSI with Wilder smoothing.
    Default period=2 matches Connors RSI(2) mean-reversion research.
    Returns a Series with the same index as close.
    """
    delta  = close.diff()
    gain   = delta.clip(lower=0.0)
    loss   = (-delta).clip(lower=0.0)

    alpha  = 1.0 / period
    avg_g  = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi    = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ==============================================================================
#  3. INTERNAL BAR STRENGTH
# ==============================================================================

def compute_ibs(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Internal Bar Strength = (Close - Low) / (High - Low).
    Range: 0 to 1.  Values near 0 indicate a weak close (bearish day);
    values near 1 indicate a strong close (bullish day).
    Div-by-zero (High == Low) is set to 0.5.
    """
    bar_range = high - low
    ibs = (close - low) / bar_range.replace(0, np.nan)
    ibs = ibs.fillna(0.5)
    return ibs


# ==============================================================================
#  4. VIX-BASED POSITION SCALE  (bounce module)
# ==============================================================================

def get_bounce_vix_scale(vix_level: "float | None") -> float:
    """
    Look up the position scale factor for the bounce module based on the
    current VIX level.  Higher VIX -> smaller position (already stressed market).
    Uses BOUNCE_VIX_TIERS from config.py.
    """
    if vix_level is None:
        return 1.0
    scale = 1.0
    for thresh in sorted(BOUNCE_VIX_TIERS.keys()):
        if vix_level > thresh:
            scale = BOUNCE_VIX_TIERS[thresh]
    return scale


# ==============================================================================
#  5. BEAR REGIME DETECTION
# ==============================================================================

def detect_bear_regime(spy_data: dict, vix_level: "float | None") -> dict:
    """
    Compute regime indicators from SPY price data and the current VIX level.

    Returns a dict with regime classification, SMA levels, momentum metrics,
    VIX scale, and capitulation flag.

    Regime hierarchy:
      stressed  -- VIX > 30
      elevated  -- VIX > 25
      bear      -- SPY below 200-day SMA
      bull      -- all other conditions
    """
    close   = spy_data["close"]
    sma_50  = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    sma_20  = close.rolling(20).mean()

    latest_price   = float(close.iloc[-1])
    latest_sma_50  = float(sma_50.iloc[-1])
    latest_sma_200 = float(sma_200.iloc[-1])
    latest_sma_20  = float(sma_20.iloc[-1])

    spy_below_200sma = latest_price < latest_sma_200
    spy_below_50sma  = latest_price < latest_sma_50

    # 20-day return (avoid index error on short series)
    spy_20d_return = (
        float(close.iloc[-1] / close.iloc[-21] - 1)
        if len(close) > 21 else 0.0
    )

    # Regime classification (VIX-aware, not just SMA cross)
    if vix_level is not None and vix_level > 30:
        regime = "stressed"
    elif vix_level is not None and vix_level > 25:
        regime = "elevated"
    elif spy_below_200sma:
        regime = "bear"
    else:
        regime = "bull"

    # VIX position scaling
    vix_scale = get_bounce_vix_scale(vix_level)

    # Capitulation detection: VIX peaked above CAPIT_VIX_SPIKE and then
    # declined by at least CAPIT_VIX_DECLINE_PCT from that peak.
    capitulation = False
    try:
        vix_raw = yf.download(
            "^VIX", period="60d", auto_adjust=True, progress=False
        )
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_close = vix_raw["Close"].iloc[:, 0]
        else:
            vix_close = vix_raw["Close"]
        vix_close = vix_close.dropna()

        if len(vix_close) > 5:
            vix_peak    = float(vix_close.max())
            vix_current = float(vix_close.iloc[-1])
            if vix_peak >= CAPIT_VIX_SPIKE:
                decline_pct = (vix_peak - vix_current) / vix_peak
                if decline_pct >= CAPIT_VIX_DECLINE_PCT:
                    capitulation = True
    except Exception:
        pass

    # Console output
    vix_display = f"{vix_level:.1f}" if vix_level is not None else "N/A"
    print(f"  [BEAR] Regime: {regime.upper()}")
    print(
        f"  [BEAR] SPY: ${latest_price:.2f}  "
        f"50SMA: ${latest_sma_50:.2f}  "
        f"200SMA: ${latest_sma_200:.2f}"
    )
    print(f"  [BEAR] SPY 20d return: {spy_20d_return*100:+.1f}%")
    print(f"  [BEAR] VIX: {vix_display}  Scale: {vix_scale:.2f}")
    if capitulation:
        print(
            f"  [BEAR] *** CAPITULATION DETECTED -- "
            f"VIX peaked >{CAPIT_VIX_SPIKE} and declined ***"
        )

    return {
        "regime":          regime,
        "spy_below_200sma": spy_below_200sma,
        "spy_below_50sma":  spy_below_50sma,
        "spy_below_20sma":  latest_price < latest_sma_20,
        "spy_20d_return":   spy_20d_return,
        "spy_price":        latest_price,
        "spy_sma_50":       latest_sma_50,
        "spy_sma_200":      latest_sma_200,
        "spy_sma_20":       latest_sma_20,
        "vix":              vix_level,
        "vix_scale":        vix_scale,
        "capitulation":     capitulation,
    }


# ==============================================================================
#  6. BOUNCE LIVE CHECK
# ==============================================================================

def bounce_live_check(spy_data: dict, regime: dict) -> dict:
    """
    Check whether SPY has a live bounce entry signal for TODAY.

    Signal bar = index -2 (yesterday's close, no lookahead).
    Execution would be at today's close (bar -1).

    Conditions:
      1. VIX above BEAR_VIX_ACTIVATE  (regime already stressed)
      2. RSI(2) < BOUNCE_RSI_ENTRY    (deeply oversold)
      3. IBS < BOUNCE_IBS_ENTRY       (closed near session low)
    """
    close = spy_data["close"]
    high  = spy_data["high"]
    low   = spy_data["low"]

    rsi_2 = compute_rsi(close, BOUNCE_RSI_PERIOD)
    ibs   = compute_ibs(high, low, close)

    # Signal bar: second-to-last (-2); execution bar: last (-1)
    sig = -2

    conditions = {
        "vix_elevated": (
            regime["vix"] is not None and regime["vix"] > BEAR_VIX_ACTIVATE
        ),
        "rsi_oversold": float(rsi_2.iloc[sig]) < BOUNCE_RSI_ENTRY,
        "ibs_low":      float(ibs.iloc[sig]) < BOUNCE_IBS_ENTRY,
    }

    all_pass = all(conditions.values())
    failed   = [k for k, v in conditions.items() if not v]

    return {
        "pass":      all_pass,
        "ticker":    BOUNCE_INSTRUMENT,
        "price":     float(close.iloc[-1]),
        "rsi_2":     float(rsi_2.iloc[sig]),
        "ibs":       float(ibs.iloc[sig]),
        "direction": "LONG",
        "reason":    ", ".join(failed) if not all_pass else "",
    }


# ==============================================================================
#  SHARED METRICS HELPER
# ==============================================================================

def _empty_metrics() -> dict:
    """Return a zeroed-out metrics dict for cases with no trades."""
    return {
        "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "total_pnl": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
        "max_dd": 0.0, "avg_hold": 0.0,
    }


def _compute_metrics(trade_list: list) -> dict:
    """
    Compute performance statistics for a list of trade dicts.
    Each trade must have keys: net_pnl, hold_days.

    Returns:
      n_trades, win_rate, profit_factor, total_pnl, avg_pnl,
      sharpe (annualised), max_dd, avg_hold
    """
    if not trade_list:
        return _empty_metrics()

    pnls   = [t["net_pnl"] for t in trade_list]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n_trades  = len(trade_list)
    win_rate  = len(wins) / n_trades * 100.0
    total_pnl = sum(pnls)
    avg_pnl   = float(np.mean(pnls))

    if losses and sum(losses) != 0:
        profit_factor = sum(wins) / abs(sum(losses))
    else:
        profit_factor = 999.0

    # Annualised Sharpe (per-trade approach)
    avg_hold      = float(np.mean([t["hold_days"] for t in trade_list]))
    trades_per_yr = 252.0 / max(avg_hold, 1.0)

    if len(pnls) > 1:
        pnl_std = float(np.std(pnls))
        sharpe  = (
            (float(np.mean(pnls)) / pnl_std) * np.sqrt(trades_per_yr)
            if pnl_std > 0 else 0.0
        )
    else:
        sharpe = 0.0

    # Max drawdown (cumulative P&L peak-to-trough)
    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    dd      = cum_pnl - peak
    max_dd  = float(dd.min())

    return {
        "n_trades":      n_trades,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "total_pnl":     total_pnl,
        "avg_pnl":       avg_pnl,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "avg_hold":      avg_hold,
    }


# ==============================================================================
#  7. BOUNCE BACKTEST
# ==============================================================================

def backtest_bounce(spy_data: dict) -> dict:
    """
    Walk-forward bar-by-bar backtest of the RSI(2) + IBS bounce strategy on SPY.

    Signal bar  : i - 1  (no lookahead)
    Execution   : bar i close

    Entry conditions (evaluated at signal bar i-1):
      - RSI(2) < BOUNCE_RSI_ENTRY
      - IBS < BOUNCE_IBS_ENTRY

    Exit conditions (evaluated at execution bar i):
      1. Close above 5-day SMA  (sma signal from prior bar, avoiding lookahead)
      2. IBS > BOUNCE_IBS_EXIT  (closed near session high = strength)
      3. Hold >= BOUNCE_MAX_HOLD  (time stop)

    No stop loss: Connors research demonstrates stops are detrimental to
    RSI(2) mean-reversion systems on SPY.  Portfolio-level kill switch
    in bear_tracker.py handles system-wide risk.

    Returns a dict with full/IS/OOS metrics and pass/fail verdict.
    """
    close = spy_data["close"]
    high  = spy_data["high"]
    low   = spy_data["low"]

    rsi_2  = compute_rsi(close, BOUNCE_RSI_PERIOD)
    ibs    = compute_ibs(high, low, close)
    sma_5  = close.rolling(BOUNCE_EXIT_SMA).mean()

    n      = len(close)
    warmup = max(BOUNCE_RSI_PERIOD + 5, BOUNCE_EXIT_SMA + 5, 20)

    if n < warmup + 10:
        return {
            **_empty_metrics(),
            "oos_pnl": 0.0, "oos_trades": 0, "oos_sharpe": 0.0,
            "is_pnl": 0.0, "is_trades": 0,
            "pass": False, "reason": "insufficient_history",
        }

    # Walk-forward split (70% IS, 30% OOS)
    split_idx  = int(n * 0.70)
    split_date = close.index[split_idx]

    position   = 0   # 0 = flat, 1 = long
    entry_bar  = 0
    entry_price = 0.0
    shares     = 0.0
    capital    = 0.0
    trades     = []

    for i in range(warmup, n):
        sig = i - 1  # signal bar (no lookahead)

        if position == 0:
            # FLAT -- check entry conditions on prior-bar indicators
            if (rsi_2.iloc[sig] < BOUNCE_RSI_ENTRY and
                    ibs.iloc[sig] < BOUNCE_IBS_ENTRY):
                entry_price = float(close.iloc[i])
                capital     = BOUNCE_CAPITAL_BASE   # historical bt: no live VIX scale
                shares      = capital / entry_price
                entry_bar   = i
                position    = 1

        else:
            # IN POSITION -- check exit conditions
            hold_days     = i - entry_bar
            current_close = float(close.iloc[i])
            exit_reason   = ""

            # Exit 1: price recovered above 5-day SMA (use prior-bar SMA to
            #         avoid lookahead -- the SMA value at sig uses data up to
            #         bar sig, which is prior to execution bar i)
            if current_close > float(sma_5.iloc[sig]):
                exit_reason = "sma_recovery"
            # Exit 2: IBS shows strength (current bar close = execution bar)
            elif float(ibs.iloc[i]) > BOUNCE_IBS_EXIT:
                exit_reason = "ibs_strength"
            # Exit 3: time stop
            elif hold_days >= BOUNCE_MAX_HOLD:
                exit_reason = "time_stop"

            if exit_reason:
                gross    = shares * (current_close - entry_price)
                slippage = BOUNCE_SLIPPAGE_PCT * shares * (entry_price + current_close)
                net      = gross - slippage

                trades.append({
                    "entry_bar":   entry_bar,
                    "exit_bar":    i,
                    "entry_price": entry_price,
                    "exit_price":  current_close,
                    "shares":      shares,
                    "capital":     capital,
                    "hold_days":   hold_days,
                    "gross_pnl":   gross,
                    "net_pnl":     net,
                    "exit_reason": exit_reason,
                    "entry_date":  close.index[entry_bar],
                    "exit_date":   close.index[i],
                })
                position = 0

    # Split into IS and OOS buckets
    is_trades  = [t for t in trades if t["entry_date"] <  split_date]
    oos_trades = [t for t in trades if t["entry_date"] >= split_date]

    all_metrics = _compute_metrics(trades)
    oos_metrics = _compute_metrics(oos_trades)
    is_metrics  = _compute_metrics(is_trades)

    # Quality gates
    gates = []
    gates.append(("min_trades",    all_metrics["n_trades"]      >= BEAR_BT_MIN_TRADES))
    gates.append(("win_rate",      all_metrics["win_rate"]       >= BEAR_BT_MIN_WIN_RATE))
    gates.append(("profit_factor", all_metrics["profit_factor"]  >= BEAR_BT_MIN_PROFIT_FACTOR))
    gates.append(("sharpe",        all_metrics["sharpe"]         >= BEAR_BT_MIN_SHARPE))
    gates.append(("total_pnl",     all_metrics["total_pnl"]      >= BEAR_BT_MIN_TOTAL_PNL))
    gates.append(("oos_pnl",       oos_metrics["total_pnl"]      >  0))

    failed_gates = [name for name, passed in gates if not passed]
    all_pass     = len(failed_gates) == 0

    return {
        # Full-sample metrics
        **all_metrics,
        # OOS metrics
        "oos_pnl":    oos_metrics["total_pnl"],
        "oos_trades": oos_metrics["n_trades"],
        "oos_sharpe": oos_metrics["sharpe"],
        # IS metrics
        "is_pnl":     is_metrics["total_pnl"],
        "is_trades":  is_metrics["n_trades"],
        # Verdict
        "pass":   all_pass,
        "reason": ", ".join(failed_gates) if not all_pass else "",
    }


# ==============================================================================
#  8. SHORT LIVE CHECK
# ==============================================================================

def short_live_check(spy_data: dict, regime: dict) -> dict:
    """
    Check whether there is a live inverse-ETF entry signal for TODAY.

    Conditions (all must pass):
      1. SPY below 200-day SMA
      2. SPY below 50-day SMA
      3. SPY 20-day return < SHORT_MOM_THRESHOLD  (sustained negative momentum)
      4. VIX > BEAR_VIX_ACTIVATE

    If signal fires, also fetches current SH price for the entry dict.
    """
    conditions = {
        "spy_below_200sma":  regime["spy_below_200sma"],
        "spy_below_50sma":   regime["spy_below_50sma"],
        "negative_momentum": regime["spy_20d_return"] < SHORT_MOM_THRESHOLD,
        "vix_elevated": (
            regime["vix"] is not None and regime["vix"] > BEAR_VIX_ACTIVATE
        ),
    }

    all_pass = all(conditions.values())
    failed   = [k for k, v in conditions.items() if not v]

    # Fetch current SH price if signal passes
    sh_price = None
    if all_pass:
        try:
            sh_raw = yf.download(
                SHORT_INSTRUMENT, period="5d",
                auto_adjust=True, progress=False
            )
            if isinstance(sh_raw.columns, pd.MultiIndex):
                sh_close = sh_raw["Close"].iloc[:, 0]
            else:
                sh_close = sh_raw["Close"]
            sh_close = sh_close.dropna()
            if not sh_close.empty:
                sh_price = float(sh_close.iloc[-1])
        except Exception:
            pass

    return {
        "pass":      all_pass,
        "ticker":    SHORT_INSTRUMENT,
        "price":     sh_price,
        "spy_price": regime["spy_price"],
        "direction": "LONG",   # we BUY SH to express a short view on the market
        "reason":    ", ".join(failed) if not all_pass else "",
    }


# ==============================================================================
#  9. SHORT BACKTEST
# ==============================================================================

def backtest_short(spy_data: dict, sh_data: dict) -> dict:
    """
    Walk-forward bar-by-bar backtest of the inverse ETF trend-following strategy.

    We BUY SH when SPY is in a confirmed downtrend and SELL SH when SPY
    recovers above its 20-day SMA.

    Signal bar  : i - 1  (no lookahead)
    Execution   : bar i close on SH

    Entry conditions (evaluated at signal bar i-1):
      - SPY < SPY 200-day SMA
      - SPY < SPY 50-day SMA
      - SPY 20-day return < SHORT_MOM_THRESHOLD

    Exit conditions:
      1. SPY recovers above 20-day SMA  (trend reversal signal)
      2. Hold >= SHORT_MAX_HOLD  (time stop -- inverse ETF decay protection)

    Returns a dict with full/IS/OOS metrics and pass/fail verdict.
    """
    spy_close = spy_data["close"]
    sh_close  = sh_data["close"]

    # Align SPY and SH on common trading dates
    common    = spy_close.index.intersection(sh_close.index)
    spy_close = spy_close.loc[common]
    sh_close  = sh_close.loc[common]

    spy_sma_50  = spy_close.rolling(50).mean()
    spy_sma_200 = spy_close.rolling(200).mean()
    spy_sma_20  = spy_close.rolling(SHORT_EXIT_SMA).mean()

    n      = len(spy_close)
    warmup = max(200 + SHORT_MOM_LOOKBACK + 5, SHORT_EXIT_SMA + 5, 50)

    if n < warmup + 10:
        return {
            **_empty_metrics(),
            "oos_pnl": 0.0, "oos_trades": 0, "oos_sharpe": 0.0,
            "is_pnl": 0.0, "is_trades": 0,
            "pass": False, "reason": "insufficient_history",
        }

    # Walk-forward split (70% IS, 30% OOS)
    split_idx  = int(n * 0.70)
    split_date = spy_close.index[split_idx]

    position   = 0
    entry_bar  = 0
    entry_price = 0.0
    shares     = 0.0
    capital    = 0.0
    trades     = []

    for i in range(warmup, n):
        sig = i - 1  # signal bar (no lookahead)

        if position == 0:
            # FLAT -- check entry conditions on prior-bar data
            lookback_bar  = max(0, sig - SHORT_MOM_LOOKBACK)
            spy_20d_ret   = float(
                spy_close.iloc[sig] / spy_close.iloc[lookback_bar] - 1
            )

            if (float(spy_close.iloc[sig]) < float(spy_sma_200.iloc[sig]) and
                    float(spy_close.iloc[sig]) < float(spy_sma_50.iloc[sig]) and
                    spy_20d_ret < SHORT_MOM_THRESHOLD):
                # Buy SH at bar i close
                entry_price = float(sh_close.iloc[i])
                capital     = BOUNCE_CAPITAL_BASE * SHORT_CAPITAL_SCALE
                shares      = capital / entry_price
                entry_bar   = i
                position    = 1

        else:
            # IN POSITION -- check exit conditions
            hold_days   = i - entry_bar
            current_sh  = float(sh_close.iloc[i])
            exit_reason = ""

            # Exit 1: SPY closes above 20-day SMA (trend reversal)
            # Use prior-bar SMA value to maintain no-lookahead discipline
            if float(spy_close.iloc[i]) > float(spy_sma_20.iloc[sig]):
                exit_reason = "spy_above_20sma"
            # Exit 2: time stop (inverse ETF decay protection)
            elif hold_days >= SHORT_MAX_HOLD:
                exit_reason = "time_stop"

            if exit_reason:
                gross    = shares * (current_sh - entry_price)
                slippage = SHORT_SLIPPAGE_PCT * shares * (entry_price + current_sh)
                net      = gross - slippage

                trades.append({
                    "entry_bar":   entry_bar,
                    "exit_bar":    i,
                    "entry_price": entry_price,
                    "exit_price":  current_sh,
                    "shares":      shares,
                    "capital":     capital,
                    "hold_days":   hold_days,
                    "gross_pnl":   gross,
                    "net_pnl":     net,
                    "exit_reason": exit_reason,
                    "entry_date":  spy_close.index[entry_bar],
                    "exit_date":   spy_close.index[i],
                })
                position = 0

    # Split into IS and OOS buckets
    is_trades  = [t for t in trades if t["entry_date"] <  split_date]
    oos_trades = [t for t in trades if t["entry_date"] >= split_date]

    all_metrics = _compute_metrics(trades)
    oos_metrics = _compute_metrics(oos_trades)
    is_metrics  = _compute_metrics(is_trades)

    # Quality gates (same as bounce)
    gates = []
    gates.append(("min_trades",    all_metrics["n_trades"]      >= BEAR_BT_MIN_TRADES))
    gates.append(("win_rate",      all_metrics["win_rate"]       >= BEAR_BT_MIN_WIN_RATE))
    gates.append(("profit_factor", all_metrics["profit_factor"]  >= BEAR_BT_MIN_PROFIT_FACTOR))
    gates.append(("sharpe",        all_metrics["sharpe"]         >= BEAR_BT_MIN_SHARPE))
    gates.append(("total_pnl",     all_metrics["total_pnl"]      >= BEAR_BT_MIN_TOTAL_PNL))
    gates.append(("oos_pnl",       oos_metrics["total_pnl"]      >  0))

    failed_gates = [name for name, passed in gates if not passed]
    all_pass     = len(failed_gates) == 0

    return {
        **all_metrics,
        "oos_pnl":    oos_metrics["total_pnl"],
        "oos_trades": oos_metrics["n_trades"],
        "oos_sharpe": oos_metrics["sharpe"],
        "is_pnl":     is_metrics["total_pnl"],
        "is_trades":  is_metrics["n_trades"],
        "pass":   all_pass,
        "reason": ", ".join(failed_gates) if not all_pass else "",
    }


# ==============================================================================
#  11. MAIN PIPELINE
# ==============================================================================

def main(vix_level: "float | None" = None) -> "dict | None":
    """
    Full bear-market signal pipeline.

    1. Fetch SPY data.
    2. Detect regime (VIX-aware SMA/momentum analysis).
    3. Gate: exit early if VIX <= BEAR_VIX_ACTIVATE.
    4. Module 1 -- Bounce: live check + walk-forward backtest validation.
    5. Module 2 -- Short:  live check + walk-forward backtest validation.
    6. Return result dict with verified signals and regime context.

    Arguments
    ---------
    vix_level : float | None
        Pass a pre-fetched VIX value to avoid a redundant download.
        If None, capitulation detection will still fetch VIX history
        internally inside detect_bear_regime().

    Returns
    -------
    dict with keys: bounce_verified, bounce_rejected, short_verified,
                    short_rejected, regime, capitulation, timestamp, vix
    Returns None if SPY data download fails or bear module is inactive.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n  [BEAR] {'=' * 55}")
    print(f"  [BEAR] Bear Market Signal Engine")
    print(f"  [BEAR] {timestamp}")
    print(f"  [BEAR] {'=' * 55}\n")

    # 1. Fetch SPY data
    spy_data = fetch_instrument(BOUNCE_INSTRUMENT, years=2)
    if spy_data is None:
        print(f"  [BEAR] Failed to download {BOUNCE_INSTRUMENT} data.")
        return None

    # 2. Detect regime
    regime = detect_bear_regime(spy_data, vix_level)

    # 3. Gate: bear module only active when VIX is elevated
    vix_display = f"{regime['vix']:.1f}" if regime["vix"] is not None else "N/A"
    if regime["vix"] is None or regime["vix"] <= BEAR_VIX_ACTIVATE:
        print(
            f"  [BEAR] VIX {vix_display} <= {BEAR_VIX_ACTIVATE} "
            f"-- bear module inactive."
        )
        return None

    result = {
        "bounce_verified": [],
        "bounce_rejected": [],
        "short_verified":  [],
        "short_rejected":  [],
        "regime":          regime,
        "capitulation":    regime["capitulation"],
        "timestamp":       timestamp,
        "vix":             regime["vix"],
    }

    # 4. Module 1 -- Mean-Reversion Bounce (multi-instrument)
    bounce_tickers = BOUNCE_INSTRUMENTS if BOUNCE_INSTRUMENTS else [BOUNCE_INSTRUMENT]
    for bounce_ticker in bounce_tickers:
        print(f"\n  [BOUNCE] {'-' * 50}")
        print(f"  [BOUNCE] Checking RSI({BOUNCE_RSI_PERIOD}) + IBS on {bounce_ticker} ...")

        # Fetch instrument data (reuse spy_data if ticker is SPY)
        if bounce_ticker == BOUNCE_INSTRUMENT:
            instr_data = spy_data
        else:
            instr_data = fetch_instrument(bounce_ticker, years=2)
            if instr_data is None:
                print(f"  [BOUNCE] Failed to download {bounce_ticker} data.")
                continue

        bounce_live = bounce_live_check(instr_data, regime)
        # Override ticker in result (may be QQQ/IWM, not SPY)
        bounce_live["ticker"] = bounce_ticker

        if bounce_live["pass"]:
            print(
                f"  [BOUNCE] SIGNAL: RSI({BOUNCE_RSI_PERIOD})={bounce_live['rsi_2']:.1f}  "
                f"IBS={bounce_live['ibs']:.3f}  "
                f"price=${bounce_live['price']:.2f}"
            )

            # Backtest validation
            print(f"  [BOUNCE] Running walk-forward backtest on {bounce_ticker} ...")
            bt = backtest_bounce(instr_data)

            entry = {"ticker": bounce_ticker, "live": bounce_live, "bt": bt}

            if bt["pass"]:
                result["bounce_verified"].append(entry)
                print(
                    f"  [BOUNCE] DIAMOND: WR={bt['win_rate']:.0f}%  "
                    f"PF={bt['profit_factor']:.2f}x  "
                    f"P&L=${bt['total_pnl']:+.0f}  "
                    f"Sharpe={bt['sharpe']:+.2f}"
                )
                print(
                    f"  [BOUNCE]          OOS P&L=${bt['oos_pnl']:+.0f}  "
                    f"OOS trades={bt['oos_trades']}"
                )
            else:
                result["bounce_rejected"].append(entry)
                print(f"  [BOUNCE] REJECTED: {bt['reason']}")
        else:
            print(f"  [BOUNCE] No signal: {bounce_live['reason']}")

    # 5. Module 2 -- Inverse ETF Trend Short
    if SHORT_ENABLED:
        print(f"\n  [SHORT] {'-' * 50}")
        print(f"  [SHORT] Checking inverse ETF trend signal ({SHORT_INSTRUMENT}) ...")
        short_live = short_live_check(spy_data, regime)

        if short_live["pass"]:
            spy_20d_pct = regime["spy_20d_return"] * 100
            sh_price_display = (
                f"${short_live['price']:.2f}"
                if short_live["price"] is not None else "N/A"
            )
            print(
                f"  [SHORT] SIGNAL: SPY below 50/200 SMA, "
                f"20d return={spy_20d_pct:+.1f}%  "
                f"{SHORT_INSTRUMENT}={sh_price_display}"
            )

            # Fetch SH data for backtest
            sh_data = fetch_instrument(SHORT_INSTRUMENT, years=2)
            if sh_data is not None:
                print(f"  [SHORT] Running walk-forward backtest on {SHORT_INSTRUMENT} ...")
                bt = backtest_short(spy_data, sh_data)

                entry = {
                    "ticker": SHORT_INSTRUMENT,
                    "live":   short_live,
                    "bt":     bt,
                }

                if bt["pass"]:
                    result["short_verified"].append(entry)
                    print(
                        f"  [SHORT] DIAMOND: WR={bt['win_rate']:.0f}%  "
                        f"PF={bt['profit_factor']:.2f}x  "
                        f"P&L=${bt['total_pnl']:+.0f}  "
                        f"Sharpe={bt['sharpe']:+.2f}"
                    )
                    print(
                        f"  [SHORT]          OOS P&L=${bt['oos_pnl']:+.0f}  "
                        f"OOS trades={bt['oos_trades']}"
                    )
                else:
                    result["short_rejected"].append(entry)
                    print(f"  [SHORT] REJECTED: {bt['reason']}")
            else:
                print(f"  [SHORT] Failed to download {SHORT_INSTRUMENT} data.")
        else:
            print(f"  [SHORT] No signal: {short_live['reason']}")

    # 6. Summary
    n_bounce = len(result["bounce_verified"])
    n_short  = len(result["short_verified"])
    print(f"\n  [BEAR] {'=' * 55}")
    print(
        f"  [BEAR] Results: {n_bounce} bounce diamond(s), "
        f"{n_short} short diamond(s)"
    )
    if regime["capitulation"]:
        boost_pct = int((CAPIT_BOOST_SCALE - 1.0) * 100)
        print(
            f"  [BEAR] Capitulation boost active -- "
            f"position size +{boost_pct}%"
        )
    print(f"  [BEAR] {'=' * 55}\n")

    return result


# ==============================================================================
#  STANDALONE ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    result = main()
    if result:
        regime = result["regime"]
        print(f"\n  Regime : {regime['regime'].upper()}")
        print(f"  VIX    : {regime['vix']:.1f}")
        print(f"  SPY    : ${regime['spy_price']:.2f}")
        print(f"  Capit  : {result['capitulation']}")
        print()

        if result["bounce_verified"]:
            print(f"  BOUNCE DIAMONDS ({len(result['bounce_verified'])}):")
            for v in result["bounce_verified"]:
                print(
                    f"    {v['ticker']}  "
                    f"${v['live']['price']:.2f}  "
                    f"RSI={v['live']['rsi_2']:.1f}  "
                    f"IBS={v['live']['ibs']:.3f}"
                )
        else:
            print("  No bounce diamonds.")

        if result["short_verified"]:
            print(f"  SHORT DIAMONDS ({len(result['short_verified'])}):")
            for v in result["short_verified"]:
                price_str = (
                    f"${v['live']['price']:.2f}"
                    if v["live"]["price"] is not None else "N/A"
                )
                print(f"    {v['ticker']}  {price_str}")
        else:
            print("  No short diamonds.")
