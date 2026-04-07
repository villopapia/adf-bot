"""
================================================================================
 momentum_signal.py -- Momentum Backtest Engine & Live Signal Checker
================================================================================
 Processes momentum candidates (from momentum_candidates.csv or a passed
 DataFrame), computes technical indicators, runs a walk-forward backtest to
 validate signal quality historically, checks for live entry signals today,
 applies quality gates, and returns verified "Momentum Diamonds".

 No-Lookahead Rule
 -----------------
 Signal observed at close of bar i-1  ->  execution at bar i close.
 A trade can NEVER execute on the same bar the signal is observed.

 Exit Rules
 ----------
 1. ATR trailing stop  : peak_price - MOM_ATR_TRAIL_MULT * ATR(14)
 2. ATR take profit    : entry_price + MOM_ATR_TP_MULT * ATR(14)
 3. Time stop          : MOM_TIME_STOP_DAYS bars held

 Quality Gates (all must pass)
 --------------------------------
 - min_trades        : >= MOM_MIN_TRADES
 - win_rate          : >= MOM_MIN_WIN_RATE %
 - profit_factor     : >= MOM_MIN_PROFIT_FACTOR
 - total_pnl         : >= MOM_MIN_TOTAL_PNL
 - sharpe            : >= MOM_MIN_SHARPE
 - max_dd            : >= MOM_MAX_MOM_DRAWDOWN
 - avg_gain_loss     : >= MOM_MIN_AVG_GAIN_LOSS
 - oos_pnl           : > 0  (walk-forward OOS)
 - oos_sharpe        : > 0  (walk-forward OOS)
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
    LOOKBACK_YEARS,
    MOM_FORMATION_DAYS, MOM_FORMATION_SHORT, MOM_FORMATION_LONG,
    MOM_FORMATION_BLEND_VIX, MOM_SKIP_DAYS,
    MOM_MIN_PRICE, MOM_MIN_AVG_VOLUME, MOM_TOP_PCT,
    MOM_52W_HIGH_PCT, MOM_VOLUME_RATIO_MIN,
    MOM_RSI_MIN, MOM_RSI_MAX, MOM_ADX_MIN,
    MOM_ABSOLUTE_FILTER, MOM_REQUIRE_ABOVE_200SMA,
    MOM_ATR_PERIOD, MOM_ATR_TRAIL_MULT, MOM_ATR_TP_MULT,
    MOM_TRAILING_STOP_PCT, MOM_TIME_STOP_DAYS, MOM_TAKE_PROFIT_PCT,
    MOM_CAPITAL_PER_TRADE, MOM_SLIPPAGE_PCT,
    MOM_VOL_TARGET_ANN, MOM_VOL_LOOKBACK_DAYS, MOM_VOL_FLOOR,
    MOM_MIN_TRADES, MOM_MIN_WIN_RATE, MOM_MIN_PROFIT_FACTOR,
    MOM_MIN_SHARPE, MOM_MIN_TOTAL_PNL, MOM_MAX_MOM_DRAWDOWN,
    MOM_MIN_AVG_GAIN_LOSS, MOM_WALK_FORWARD_SPLIT,
    MOM_VIX_TIERS,
    MOM_MARKET_REGIME_EMA,
    MOMENTUM_CANDIDATES_CSV,
    MAX_RETRIES, RETRY_DELAY,
    ERROR_LOG,
)


# ==============================================================================
#  HELPER — LOG TO error.log
# ==============================================================================

def _log_error(timestamp: str, context: str, ticker: str, exc: Exception) -> None:
    """Append a single error line to error.log."""
    try:
        path = os.path.join(_SCRIPT_DIR, ERROR_LOG)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} | momentum_signal | {context} | {ticker} | {exc}\n")
    except Exception:
        pass


# ==============================================================================
#  1. FETCH STOCK DATA
# ==============================================================================

def fetch_stock(ticker: str, years: int = LOOKBACK_YEARS) -> "dict | None":
    """
    Download OHLCV data for a single stock using yfinance.

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
            if len(raw) < 200:
                raise ValueError(f"Only {len(raw)} rows after dropna — insufficient history")

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
                _log_error(ts, "fetch_stock", ticker, exc)
                return None

    return None


# ==============================================================================
#  2. RSI
# ==============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI.  Returns a Series with the same index as close."""
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
#  3. ADX
# ==============================================================================

def compute_adx(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Proper ADX using Wilder smoothing.

    TR  = max(high - low, |high - prev_close|, |low - prev_close|)
    +DM = high - prev_high  if positive and > -(low - prev_low), else 0
    -DM = prev_low - low    if positive and >  (high - prev_high), else 0
    ADX = Wilder smooth of DX = 100 * |+DI - -DI| / (+DI + -DI)
    """
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    # True Range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    alpha = 1.0 / period

    atr_w     = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di_w = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_di_w= minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di  = 100.0 * plus_di_w  / atr_w.replace(0, np.nan)
    minus_di = 100.0 * minus_di_w / atr_w.replace(0, np.nan)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx    = 100.0 * (plus_di - minus_di).abs() / denom

    adx   = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx


# ==============================================================================
#  4. ATR
# ==============================================================================

def compute_atr(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """True Range with Wilder smoothing."""
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr   = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return atr


# ==============================================================================
#  5. COMPUTE MOMENTUM SIGNALS
# ==============================================================================

def compute_momentum_signals(data: dict) -> "pd.DataFrame | None":
    """
    Build a DataFrame of all momentum signals from raw OHLCV data.

    Columns:  price, high, low, volume, mom_long, mom_short, mom_blended,
              rsi, adx, atr, sma_200, high_52w, high_52w_pct,
              vol_20d, vol_50d, vol_ratio, vol_6mo, mom_pctile
    """
    try:
        close  = data["close"].astype(float)
        high   = data["high"].astype(float)
        low    = data["low"].astype(float)
        volume = data["volume"].astype(float)

        skip = MOM_SKIP_DAYS
        formation_short = MOM_FORMATION_SHORT
        formation_long  = MOM_FORMATION_LONG

        df = pd.DataFrame(index=close.index)
        df["price"]  = close
        df["high"]   = high
        df["low"]    = low
        df["volume"] = volume

        # Momentum returns (skip most recent month to avoid short-term reversal)
        df["mom_long"]  = close.shift(skip) / close.shift(skip + formation_long)  - 1.0
        df["mom_short"] = close.shift(skip) / close.shift(skip + formation_short) - 1.0

        # Default blended = long (may be overridden in main() based on VIX)
        df["mom_blended"] = df["mom_long"]

        # Technical indicators
        df["rsi"] = compute_rsi(close, period=14)
        df["adx"] = compute_adx(high, low, close, period=14)
        df["atr"] = compute_atr(high, low, close, period=MOM_ATR_PERIOD)

        # Trend / proximity
        df["sma_200"]     = close.rolling(200).mean()
        df["high_52w"]    = close.rolling(252).max()
        df["high_52w_pct"]= df["price"] / df["high_52w"]

        # Volume regime
        df["vol_20d"]  = volume.rolling(20).mean()
        df["vol_50d"]  = volume.rolling(50).mean()
        df["vol_ratio"]= df["vol_20d"] / df["vol_50d"].replace(0, np.nan)

        # Realized volatility (annualised)
        df["vol_6mo"] = (
            close.pct_change()
                 .rolling(MOM_VOL_LOOKBACK_DAYS)
                 .std()
            * np.sqrt(252)
        )

        # Percentile rank of blended momentum over rolling 252-bar window
        # raw=False so we can use pd.Series.rank inside the lambda
        df["mom_pctile"] = (
            df["mom_blended"]
              .rolling(252, min_periods=60)
              .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
        )

        # Drop warmup rows that are all NaN
        df = df.dropna(subset=["mom_long", "sma_200", "rsi", "adx", "atr", "mom_pctile"])

        if df.empty:
            return None

        return df

    except Exception:
        return None


# ==============================================================================
#  6. VOLATILITY-TARGETED POSITION SIZING
# ==============================================================================

def compute_vol_target_size(vol_6mo: float, vix_scale: float) -> float:
    """
    Scale capital allocation using realised vol targeting
    (Barroso & Santa-Clara 2015) and a VIX-based regime multiplier.
    """
    realized   = max(vol_6mo, MOM_VOL_FLOOR)
    raw_scale  = MOM_VOL_TARGET_ANN / realized
    adjusted   = raw_scale * vix_scale
    capital    = MOM_CAPITAL_PER_TRADE * min(max(adjusted, 0.25), 2.0)
    return capital


# ==============================================================================
#  7. VIX SCALING
# ==============================================================================

def get_vix_scale(vix_level: "float | None") -> float:
    """
    Graduated VIX scaling from MOM_VIX_TIERS dict.
    Higher VIX -> lower scale (reduce position size).
    """
    if vix_level is None:
        return 1.0
    scale = 1.0
    for thresh in sorted(MOM_VIX_TIERS.keys()):
        if vix_level > thresh:
            scale = MOM_VIX_TIERS[thresh]
    return scale


# ==============================================================================
#  8. BACKTEST ENGINE
# ==============================================================================

def _empty_metrics() -> dict:
    return {
        "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "total_pnl": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "max_dd": 0.0, "avg_hold": 0.0,
        "avg_gain_loss": 0.0,
    }


def _compute_metrics(trade_list: list) -> dict:
    """Compute performance statistics for a list of trade dicts."""
    if not trade_list:
        return _empty_metrics()

    pnls   = [t["net_pnl"] for t in trade_list]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n_trades     = len(trade_list)
    win_rate     = len(wins) / n_trades * 100.0
    total_pnl    = sum(pnls)
    avg_pnl      = float(np.mean(pnls))

    if losses and sum(losses) != 0:
        profit_factor = sum(wins) / abs(sum(losses))
    else:
        profit_factor = 999.0

    # Annualised Sharpe (per-trade approach)
    if len(pnls) > 1:
        avg_hold      = float(np.mean([t["hold_days"] for t in trade_list]))
        trades_per_yr = 252.0 / max(avg_hold, 1.0)
        pnl_std       = float(np.std(pnls))
        sharpe        = (float(np.mean(pnls)) / pnl_std) * np.sqrt(trades_per_yr) if pnl_std > 0 else 0.0
    else:
        avg_hold      = float(trade_list[0]["hold_days"]) if trade_list else 0.0
        trades_per_yr = 252.0 / max(avg_hold, 1.0)
        sharpe        = 0.0

    # Sortino (downside std)
    downside      = [p for p in pnls if p < 0]
    downside_std  = float(np.std(downside)) if len(downside) > 1 else 1.0
    sortino       = (float(np.mean(pnls)) / downside_std) * np.sqrt(trades_per_yr) if downside_std > 0 else 0.0

    # Max drawdown (cumulative P&L peak-to-trough)
    cum_pnl  = np.cumsum(pnls)
    peak     = np.maximum.accumulate(cum_pnl)
    dd       = cum_pnl - peak
    max_dd   = float(dd.min())

    # Avg gain / loss ratio
    avg_win       = float(np.mean(wins))  if wins   else 0.0
    avg_loss_abs  = abs(float(np.mean(losses))) if losses else 1.0
    avg_gain_loss = avg_win / avg_loss_abs if avg_loss_abs > 0 else 999.0

    return {
        "n_trades": n_trades, "win_rate": win_rate,
        "profit_factor": profit_factor, "total_pnl": total_pnl,
        "avg_pnl": avg_pnl, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "avg_hold": avg_hold,
        "avg_gain_loss": avg_gain_loss,
    }


def backtest_momentum(
    signals: pd.DataFrame,
    ticker: str,
    vix_scale: float = 1.0,
) -> dict:
    """
    Walk-forward bar-by-bar backtest.

    Signal bar  : i - 1  (no lookahead)
    Execution   : bar i close

    Returns a dict with full/IS/OOS metrics and pass/fail result.
    """
    # Warmup: enough bars for all lookbacks
    warmup = max(MOM_FORMATION_LONG + MOM_SKIP_DAYS, 252, 200) + 1
    n      = len(signals)

    if n < warmup + 2:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "total_pnl": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "max_dd": 0.0, "avg_hold": 0.0,
            "avg_gain_loss": 0.0,
            "oos_pnl": 0.0, "oos_sharpe": 0.0, "oos_trades": 0,
            "pass": False, "reason": "insufficient_history",
        }

    # ----- main loop -----
    position    = 0          # 0 = flat, 1 = long
    trades      = []
    entry_price = 0.0
    entry_atr   = 0.0
    capital     = MOM_CAPITAL_PER_TRADE
    shares      = 0.0
    peak_price  = 0.0
    trail_stop  = 0.0
    take_profit = 0.0
    entry_bar   = 0

    for i in range(warmup, n):
        sig = i - 1          # signal bar (no lookahead)

        if position == 0:    # FLAT — check for entry
            entry_conditions = (
                signals["mom_pctile"].iloc[sig] >= (100.0 - MOM_TOP_PCT)
                and (not MOM_ABSOLUTE_FILTER or signals["mom_blended"].iloc[sig] > 0.0)
                and MOM_RSI_MIN <= signals["rsi"].iloc[sig] <= MOM_RSI_MAX
                and signals["adx"].iloc[sig] >= MOM_ADX_MIN
                and (
                    not MOM_REQUIRE_ABOVE_200SMA
                    or signals["price"].iloc[sig] > signals["sma_200"].iloc[sig]
                )
                and signals["high_52w_pct"].iloc[sig] >= MOM_52W_HIGH_PCT
                and signals["vol_ratio"].iloc[sig] >= MOM_VOLUME_RATIO_MIN
                and signals["price"].iloc[sig] >= MOM_MIN_PRICE
                and signals["vol_20d"].iloc[sig] >= MOM_MIN_AVG_VOLUME
            )

            if entry_conditions:
                entry_price = float(signals["price"].iloc[i])   # execute at bar i
                entry_atr   = float(signals["atr"].iloc[sig])
                vol_6mo_val = float(signals["vol_6mo"].iloc[sig])
                capital     = compute_vol_target_size(vol_6mo_val, vix_scale)
                shares      = capital / entry_price if entry_price > 0 else 0.0
                peak_price  = entry_price
                trail_stop  = entry_price - MOM_ATR_TRAIL_MULT * entry_atr
                take_profit = entry_price + MOM_ATR_TP_MULT  * entry_atr
                entry_bar   = i
                position    = 1

        else:                # IN POSITION — check for exit
            hold_days     = i - entry_bar
            current_price = float(signals["price"].iloc[i])
            current_atr   = float(signals["atr"].iloc[sig])

            # Update peak and trailing stop
            if current_price > peak_price:
                peak_price = current_price
            trail_stop = peak_price - MOM_ATR_TRAIL_MULT * current_atr

            exit_reason = ""
            if current_price <= trail_stop:
                exit_reason = "trail_stop"
            elif current_price >= take_profit:
                exit_reason = "take_profit"
            elif hold_days >= MOM_TIME_STOP_DAYS:
                exit_reason = "time_stop"

            if exit_reason:
                gross    = shares * (current_price - entry_price)
                slippage = MOM_SLIPPAGE_PCT * (
                    shares * entry_price + shares * current_price
                )
                net = gross - slippage

                trades.append({
                    "entry_bar":   entry_bar,
                    "exit_bar":    i,
                    "entry_price": entry_price,
                    "exit_price":  current_price,
                    "shares":      shares,
                    "capital":     capital,
                    "hold_days":   hold_days,
                    "gross_pnl":   gross,
                    "net_pnl":     net,
                    "exit_reason": exit_reason,
                    "entry_date":  signals.index[entry_bar],
                    "exit_date":   signals.index[i],
                })
                position = 0

    # ----- walk-forward split -----
    split_idx  = int(len(signals) * MOM_WALK_FORWARD_SPLIT)
    split_date = signals.index[split_idx]

    is_trades  = [t for t in trades if t["entry_date"] <  split_date]
    oos_trades = [t for t in trades if t["entry_date"] >= split_date]

    is_metrics  = _compute_metrics(is_trades)
    oos_metrics = _compute_metrics(oos_trades)
    all_metrics = _compute_metrics(trades)

    # ----- quality gates -----
    gates = [
        ("min_trades",   all_metrics["n_trades"]     >= MOM_MIN_TRADES),
        ("win_rate",     all_metrics["win_rate"]      >= MOM_MIN_WIN_RATE),
        ("profit_factor",all_metrics["profit_factor"] >= MOM_MIN_PROFIT_FACTOR),
        ("total_pnl",    all_metrics["total_pnl"]     >= MOM_MIN_TOTAL_PNL),
        ("sharpe",       all_metrics["sharpe"]        >= MOM_MIN_SHARPE),
        ("max_dd",       all_metrics["max_dd"]        >= MOM_MAX_MOM_DRAWDOWN),
        ("avg_gain_loss",all_metrics["avg_gain_loss"] >= MOM_MIN_AVG_GAIN_LOSS),
        ("oos_pnl",      oos_metrics["total_pnl"]     >  0.0),
        ("oos_sharpe",   oos_metrics["sharpe"]        >  0.0),
    ]

    passed = all(g[1] for g in gates)
    reason = ", ".join(name for name, ok in gates if not ok) if not passed else ""

    return {
        "n_trades":     all_metrics["n_trades"],
        "win_rate":     all_metrics["win_rate"],
        "profit_factor":all_metrics["profit_factor"],
        "total_pnl":    all_metrics["total_pnl"],
        "avg_pnl":      all_metrics["avg_pnl"],
        "sharpe":       all_metrics["sharpe"],
        "sortino":      all_metrics["sortino"],
        "max_dd":       all_metrics["max_dd"],
        "avg_hold":     all_metrics["avg_hold"],
        "avg_gain_loss":all_metrics["avg_gain_loss"],
        "oos_pnl":      oos_metrics["total_pnl"],
        "oos_sharpe":   oos_metrics["sharpe"],
        "oos_trades":   oos_metrics["n_trades"],
        "pass":         passed,
        "reason":       reason,
    }


# ==============================================================================
#  9. LIVE SIGNAL CHECK
# ==============================================================================

def live_check(
    signals: pd.DataFrame,
    ticker: str,
    market_regime: "dict | None" = None,
) -> dict:
    """
    Evaluate whether the stock has a live entry signal as of today.

    sig    = second-to-last bar  (signal bar, no lookahead)
    latest = last bar            (execution price)
    """
    if len(signals) < 3:
        return {
            "pass": False, "ticker": ticker,
            "price": 0.0, "atr_14": 0.0, "vol_6mo": 0.0,
            "mom_score": 0.0, "direction": "LONG",
            "reason": "insufficient_data",
        }

    sig    = -2   # signal bar
    latest = -1   # execution bar

    conditions = {
        "mom_pctile":  signals["mom_pctile"].iloc[sig]  >= (100.0 - MOM_TOP_PCT),
        "absolute_mom":(not MOM_ABSOLUTE_FILTER) or signals["mom_blended"].iloc[sig] > 0.0,
        "rsi":         MOM_RSI_MIN <= signals["rsi"].iloc[sig] <= MOM_RSI_MAX,
        "adx":         signals["adx"].iloc[sig] >= MOM_ADX_MIN,
        "above_200sma":(not MOM_REQUIRE_ABOVE_200SMA) or
                       signals["price"].iloc[sig] > signals["sma_200"].iloc[sig],
        "52w_high":    signals["high_52w_pct"].iloc[sig] >= MOM_52W_HIGH_PCT,
        "vol_ratio":   signals["vol_ratio"].iloc[sig]   >= MOM_VOLUME_RATIO_MIN,
        "min_price":   signals["price"].iloc[sig]       >= MOM_MIN_PRICE,
        "min_volume":  signals["vol_20d"].iloc[sig]     >= MOM_MIN_AVG_VOLUME,
    }

    # Market regime check (soft -- warn but allow by failing the gate)
    if market_regime and MOM_MARKET_REGIME_EMA:
        if not market_regime.get("sp500_above_200ema", True):
            conditions["market_regime"] = False

    all_pass = all(conditions.values())
    failed   = [k for k, v in conditions.items() if not v]

    return {
        "pass":      all_pass,
        "ticker":    ticker,
        "price":     float(signals["price"].iloc[latest]),
        "atr_14":    float(signals["atr"].iloc[sig]),
        "vol_6mo":   float(signals["vol_6mo"].iloc[sig]),
        "mom_score": float(signals["mom_blended"].iloc[sig]),
        "direction": "LONG",
        "reason":    ", ".join(failed) if not all_pass else "",
    }


# ==============================================================================
#  10. MAIN PIPELINE
# ==============================================================================

def main(
    candidates_df: "pd.DataFrame | None" = None,
    vix_level: "float | None" = None,
    market_regime: "dict | None" = None,
) -> "dict | None":
    """
    Full momentum signal pipeline.

    Loads candidates, downloads data, computes signals, runs live check and
    backtest, applies quality gates, and returns verified Momentum Diamonds.

    Parameters
    ----------
    candidates_df  : Pre-loaded candidates DataFrame (ticker column required).
                     Loaded from MOMENTUM_CANDIDATES_CSV if None.
    vix_level      : Current VIX level for regime scaling.  None = ignore.
    market_regime  : Optional dict with keys like 'sp500_above_200ema'.

    Returns
    -------
    dict with keys: verified, rejected, no_signal, errors, timestamp, vix
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vix_scale = get_vix_scale(vix_level)

    # -- Load candidates --
    if candidates_df is None:
        path = os.path.join(_SCRIPT_DIR, MOMENTUM_CANDIDATES_CSV)
        if not os.path.exists(path):
            print("  [MOM-SIG] No candidates file found.")
            return None
        try:
            candidates_df = pd.read_csv(path, encoding="utf-8")
        except Exception as exc:
            print(f"  [MOM-SIG] Failed to read candidates CSV: {exc}")
            return None

    if "ticker" not in candidates_df.columns:
        print("  [MOM-SIG] Candidates CSV missing 'ticker' column.")
        return None

    tickers = candidates_df["ticker"].dropna().unique().tolist()
    print(f"  [MOM-SIG] Processing {len(tickers)} momentum candidates ...")
    print("  " + "-" * 60)

    verified  = []
    rejected  = []
    no_signal = 0
    errors    = 0

    for ticker in tickers:
        try:
            # 1. Download OHLCV
            data = fetch_stock(ticker)
            if data is None:
                errors += 1
                continue

            # 2. Compute base signals
            signals = compute_momentum_signals(data)
            if signals is None or len(signals) < 252:
                errors += 1
                continue

            # 3. Set blended momentum based on VIX regime
            if vix_level is not None and vix_level > MOM_FORMATION_BLEND_VIX:
                signals["mom_blended"] = (
                    0.5 * signals["mom_long"] + 0.5 * signals["mom_short"]
                )
            else:
                signals["mom_blended"] = signals["mom_long"]

            # Recompute percentile rank using the (potentially updated) blended
            signals["mom_pctile"] = (
                signals["mom_blended"]
                  .rolling(252, min_periods=60)
                  .apply(
                      lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
                      raw=False,
                  )
            )

            # 4. Live entry check
            live = live_check(signals, ticker, market_regime)
            if not live["pass"]:
                no_signal += 1
                continue

            # 5. Walk-forward backtest
            bt = backtest_momentum(signals, ticker, vix_scale)

            entry = {"ticker": ticker, "live": live, "bt": bt}

            if bt["pass"]:
                verified.append(entry)
                print(
                    f"    [DIAMOND] {ticker:>6}  "
                    f"mom={live['mom_score']:+.3f}  "
                    f"WR={bt['win_rate']:.0f}%  "
                    f"PF={bt['profit_factor']:.2f}x  "
                    f"P&L=${bt['total_pnl']:+.0f}  "
                    f"Sharpe={bt['sharpe']:+.2f}"
                )
            else:
                rejected.append(entry)
                print(f"    [REJECT]  {ticker:>6}  {bt['reason']}")

        except Exception as exc:
            errors += 1
            _log_error(timestamp, "main_loop", ticker, exc)

    print("  " + "-" * 60)
    print(
        f"  [MOM-SIG] Results: {len(verified)} diamonds, "
        f"{len(rejected)} rejected, {no_signal} no signal, {errors} errors"
    )

    return {
        "verified":  verified,
        "rejected":  rejected,
        "no_signal": no_signal,
        "errors":    errors,
        "timestamp": timestamp,
        "vix":       vix_level,
    }


# ==============================================================================
#  STANDALONE MODE
# ==============================================================================

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n  Momentum Diamonds: {len(result['verified'])}")
        for v in result["verified"]:
            print(f"    {v['ticker']}  P&L=${v['bt']['total_pnl']:+.2f}")
