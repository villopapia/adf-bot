"""
================================================================================
 shock_signal.py -- Policy Shock Bounce Signal Engine
================================================================================
 Detects rapid market drawdowns caused by policy announcements (tariffs,
 trade war escalation, geopolitical threats) and generates long entry signals
 betting on the reversal/walkback pattern.

 Detection Logic
 ---------------
 A "policy shock" is identified when:
   1. SPY/QQQ drops >= SHOCK_DROP_THRESHOLD over SHOCK_LOOKBACK_DAYS
   2. VIX spikes >= SHOCK_VIX_SPIKE_MIN points over the same window
   3. RSI(14) is below SHOCK_RSI_MAX (confirming oversold, not just drift)

 The combination of rapid price drop + VIX spike filters out slow grinds
 and isolates genuine shock events.

 Entry
 -----
 After a shock is detected, wait SHOCK_WAIT_DAYS (let panic exhaust),
 then enter LONG at the close.  No 200d SMA requirement (shocks often
 break below it temporarily).

 Exit
 ----
   1. Take profit at SHOCK_TAKE_PROFIT_PCT (+4%)
   2. Stop loss at SHOCK_STOP_LOSS_PCT (-3%)
   3. VIX mean-reversion: VIX declines SHOCK_EXIT_VIX_DECLINE from peak
   4. Time stop: SHOCK_MAX_HOLD days

 No-Lookahead Rule
 -----------------
 Signal observed at close of bar i-1  ->  execution at bar i close.

 Quality Gates (all must pass)
 -----------------------------
 - min_trades       : >= SHOCK_BT_MIN_TRADES
 - win_rate         : >= SHOCK_BT_MIN_WIN_RATE %
 - profit_factor    : >= SHOCK_BT_MIN_PROFIT_FACTOR
 - sharpe           : >= SHOCK_BT_MIN_SHARPE
 - total_pnl        : >= SHOCK_BT_MIN_TOTAL_PNL
 - oos_pnl          : > 0  (walk-forward OOS 30%)
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
#  CONFIGURATION
# ------------------------------------------------------------------------------
from config import (
    SHOCK_MODULE_ENABLED, SHOCK_INSTRUMENTS,
    SHOCK_LOOKBACK_DAYS, SHOCK_DROP_THRESHOLD, SHOCK_VIX_SPIKE_MIN,
    SHOCK_COOLDOWN_DAYS, SHOCK_WAIT_DAYS,
    SHOCK_RSI_MAX, SHOCK_ABOVE_200SMA,
    SHOCK_TAKE_PROFIT_PCT, SHOCK_STOP_LOSS_PCT,
    SHOCK_MAX_HOLD, SHOCK_EXIT_VIX_DECLINE,
    SHOCK_CAPITAL_PER_TRADE, SHOCK_SLIPPAGE_PCT, SHOCK_VIX_TIERS,
    SHOCK_BT_MIN_TRADES, SHOCK_BT_MIN_WIN_RATE,
    SHOCK_BT_MIN_PROFIT_FACTOR, SHOCK_BT_MIN_SHARPE, SHOCK_BT_MIN_TOTAL_PNL,
    MAX_RETRIES, RETRY_DELAY, ERROR_LOG,
)


# ==============================================================================
#  HELPER -- LOG TO error.log
# ==============================================================================

def _log_error(timestamp: str, context: str, ticker: str, exc: Exception) -> None:
    try:
        path = os.path.join(_SCRIPT_DIR, ERROR_LOG)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                f"{timestamp} | shock_signal | {context} | {ticker} | {exc}\n"
            )
    except Exception:
        pass


# ==============================================================================
#  1. FETCH INSTRUMENT
# ==============================================================================

def fetch_instrument(ticker: str, years: int = 5) -> "dict | None":
    """Download OHLCV data via yfinance. Returns dict or None on failure."""
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

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.columns = [c.lower() for c in raw.columns]

            required = {"high", "low", "close", "volume"}
            if not required.issubset(set(raw.columns)):
                raise KeyError(f"Missing columns: {required - set(raw.columns)}")

            raw = raw.dropna(subset=["close"])
            if len(raw) < 252:
                raise ValueError(
                    f"Only {len(raw)} rows after dropna -- need at least 1 year"
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
#  2. FETCH VIX HISTORY
# ==============================================================================

def fetch_vix(years: int = 5) -> "pd.Series | None":
    """Download VIX close history. Returns Series or None."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=int(years * 365.25) + 30)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                "^VIX",
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError("empty VIX data")

            if isinstance(raw.columns, pd.MultiIndex):
                vix_close = raw[("Close", "^VIX")]
            else:
                vix_close = raw["Close"]

            vix_close = vix_close.dropna()
            if len(vix_close) < 100:
                raise ValueError(f"Only {len(vix_close)} VIX rows")

            return vix_close

        except Exception as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _log_error(ts, "fetch_vix", "^VIX", exc)
                return None

    return None


# ==============================================================================
#  3. RSI (standard 14-period for shock module)
# ==============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI with Wilder smoothing."""
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
#  4. SHOCK DETECTION
# ==============================================================================

def detect_shock(close: pd.Series, vix: pd.Series, bar: int,
                 lookback: int = SHOCK_LOOKBACK_DAYS) -> "dict | None":
    """
    Check if a policy shock occurred as of bar index `bar`.

    A shock is detected when:
      1. Price dropped >= SHOCK_DROP_THRESHOLD over `lookback` days
      2. VIX spiked >= SHOCK_VIX_SPIKE_MIN over the same window

    Returns dict with shock details or None if no shock.
    Uses bar-1 data (no lookahead) for signal detection.
    """
    if bar < lookback + 1:
        return None

    # Price return over lookback window (signal bar = bar, not execution bar)
    price_now  = float(close.iloc[bar])
    price_prev = float(close.iloc[bar - lookback])
    if price_prev == 0:
        return None
    price_return = (price_now / price_prev) - 1.0

    # Align VIX to same dates
    close_dates = close.index
    bar_date    = close_dates[bar]
    prev_date   = close_dates[bar - lookback]

    # Find nearest VIX dates
    vix_at_bar  = vix.asof(bar_date)
    vix_at_prev = vix.asof(prev_date)

    if pd.isna(vix_at_bar) or pd.isna(vix_at_prev):
        return None

    vix_change = float(vix_at_bar) - float(vix_at_prev)

    # Check shock conditions
    if price_return <= SHOCK_DROP_THRESHOLD and vix_change >= SHOCK_VIX_SPIKE_MIN:
        return {
            "price_return": price_return,
            "vix_change":   vix_change,
            "vix_level":    float(vix_at_bar),
            "price":        price_now,
        }

    return None


# ==============================================================================
#  5. VIX-BASED POSITION SCALE
# ==============================================================================

def get_shock_vix_scale(vix_level: "float | None") -> float:
    """Higher VIX -> smaller position (already stressed)."""
    if vix_level is None:
        return 1.0
    scale = 1.0
    for thresh in sorted(SHOCK_VIX_TIERS.keys()):
        if vix_level > thresh:
            scale = SHOCK_VIX_TIERS[thresh]
    return scale


# ==============================================================================
#  6. METRICS HELPER
# ==============================================================================

def _empty_metrics() -> dict:
    return {
        "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "total_pnl": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
        "max_dd": 0.0, "avg_hold": 0.0,
    }


def _compute_metrics(trade_list: list) -> dict:
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
#  7. SHOCK BOUNCE BACKTEST
# ==============================================================================

def backtest_shock(instr_data: dict, vix: pd.Series, ticker: str = "SPY") -> dict:
    """
    Walk-forward bar-by-bar backtest of the shock bounce strategy.

    Signal detection at bar i-1 (no lookahead).
    Execution at bar i close.

    Entry:
      - Shock detected (price drop + VIX spike) at signal bar
      - Wait SHOCK_WAIT_DAYS after shock
      - RSI(14) < SHOCK_RSI_MAX at signal bar

    Exit:
      1. Take profit at +SHOCK_TAKE_PROFIT_PCT from entry
      2. Stop loss at -SHOCK_STOP_LOSS_PCT from entry
      3. VIX declines SHOCK_EXIT_VIX_DECLINE from shock-peak VIX
      4. Time stop at SHOCK_MAX_HOLD days

    Returns dict with full/IS/OOS metrics and pass/fail verdict.
    """
    close = instr_data["close"]
    rsi   = compute_rsi(close, 14)

    n      = len(close)
    warmup = max(200, SHOCK_LOOKBACK_DAYS + 20)

    if n < warmup + 50:
        return {
            **_empty_metrics(),
            "oos_pnl": 0.0, "oos_trades": 0, "oos_sharpe": 0.0,
            "is_pnl": 0.0, "is_trades": 0,
            "pass": False, "reason": "insufficient_history",
        }

    # Walk-forward split (70% IS, 30% OOS)
    split_idx  = int(n * 0.70)
    split_date = close.index[split_idx]

    position    = 0     # 0 = flat, 1 = long
    entry_bar   = 0
    entry_price = 0.0
    shares      = 0.0
    shock_vix   = 0.0   # VIX at shock detection (for exit condition)
    wait_counter = 0    # countdown after shock detection
    shock_pending = False
    last_entry_bar = -999  # for cooldown
    trades      = []

    for i in range(warmup, n):
        sig = i - 1  # signal bar (no lookahead)

        if position == 0:
            # Check for new shock at signal bar
            if not shock_pending:
                shock = detect_shock(close, vix, sig)
                if shock is not None:
                    # Cooldown check
                    bars_since_last = i - last_entry_bar
                    cooldown_bars = SHOCK_COOLDOWN_DAYS
                    if bars_since_last >= cooldown_bars:
                        shock_pending = True
                        wait_counter  = SHOCK_WAIT_DAYS
                        shock_vix     = shock["vix_level"]

            # If shock detected and wait period elapsed, check entry
            if shock_pending:
                if wait_counter > 0:
                    wait_counter -= 1
                else:
                    # Entry filters
                    rsi_ok = float(rsi.iloc[sig]) < SHOCK_RSI_MAX

                    sma_ok = True
                    if SHOCK_ABOVE_200SMA:
                        sma_200 = close.rolling(200).mean()
                        sma_ok = float(close.iloc[sig]) > float(sma_200.iloc[sig])

                    if rsi_ok and sma_ok:
                        entry_price = float(close.iloc[i])
                        shares      = SHOCK_CAPITAL_PER_TRADE / entry_price
                        entry_bar   = i
                        position    = 1
                        last_entry_bar = i
                        shock_pending  = False
                    else:
                        # Filters failed, cancel this shock
                        shock_pending = False

        else:
            # IN POSITION -- check exits
            hold_days     = i - entry_bar
            current_close = float(close.iloc[i])
            exit_reason   = ""

            # Exit 1: take profit
            pnl_pct = (current_close - entry_price) / entry_price
            if pnl_pct >= SHOCK_TAKE_PROFIT_PCT:
                exit_reason = "take_profit"

            # Exit 2: stop loss
            if not exit_reason and pnl_pct <= -SHOCK_STOP_LOSS_PCT:
                exit_reason = "stop_loss"

            # Exit 3: VIX mean-reversion (shock subsiding)
            if not exit_reason:
                bar_date    = close.index[i]
                vix_current = vix.asof(bar_date)
                if not pd.isna(vix_current) and shock_vix > 0:
                    vix_decline = (shock_vix - float(vix_current)) / shock_vix
                    if vix_decline >= SHOCK_EXIT_VIX_DECLINE:
                        exit_reason = "vix_recovery"

            # Exit 4: time stop
            if not exit_reason and hold_days >= SHOCK_MAX_HOLD:
                exit_reason = "time_stop"

            if exit_reason:
                gross    = shares * (current_close - entry_price)
                slippage = SHOCK_SLIPPAGE_PCT * shares * (entry_price + current_close)
                net      = gross - slippage

                trades.append({
                    "entry_bar":   entry_bar,
                    "exit_bar":    i,
                    "entry_price": entry_price,
                    "exit_price":  current_close,
                    "shares":      shares,
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
    gates.append(("min_trades",    all_metrics["n_trades"]      >= SHOCK_BT_MIN_TRADES))
    gates.append(("win_rate",      all_metrics["win_rate"]       >= SHOCK_BT_MIN_WIN_RATE))
    gates.append(("profit_factor", all_metrics["profit_factor"]  >= SHOCK_BT_MIN_PROFIT_FACTOR))
    gates.append(("sharpe",        all_metrics["sharpe"]         >= SHOCK_BT_MIN_SHARPE))
    gates.append(("total_pnl",     all_metrics["total_pnl"]      >= SHOCK_BT_MIN_TOTAL_PNL))
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
#  8. LIVE SHOCK CHECK
# ==============================================================================

def shock_live_check(instr_data: dict, vix: pd.Series,
                     ticker: str = "SPY") -> dict:
    """
    Check whether a policy shock bounce entry is live for TODAY.

    Signal bar = index -2 (yesterday's close, no lookahead).
    We check if a shock occurred SHOCK_WAIT_DAYS ago and entry
    conditions are met now.

    Returns dict with pass/fail, price, shock details.
    """
    close = instr_data["close"]
    rsi   = compute_rsi(close, 14)

    # We need to check for a shock that occurred SHOCK_WAIT_DAYS + 1 bars ago
    # (signal bar is -2, so shock was at -(SHOCK_WAIT_DAYS + 2))
    sig = -2  # signal bar

    # Check for shock in the recent window (SHOCK_WAIT_DAYS ago from signal bar)
    shock_bar = len(close) + sig - SHOCK_WAIT_DAYS
    if shock_bar < SHOCK_LOOKBACK_DAYS + 1:
        return {
            "pass": False, "ticker": ticker, "price": None,
            "direction": "LONG", "reason": "insufficient_bars",
            "shock_return": 0.0, "vix_spike": 0.0, "rsi": 0.0,
        }

    shock = detect_shock(close, vix, shock_bar)

    if shock is None:
        return {
            "pass": False, "ticker": ticker,
            "price": float(close.iloc[-1]),
            "direction": "LONG", "reason": "no_shock_detected",
            "shock_return": 0.0, "vix_spike": 0.0,
            "rsi": float(rsi.iloc[sig]),
        }

    # Entry filters at signal bar
    conditions = {
        "shock_detected": True,
        "rsi_oversold":   float(rsi.iloc[sig]) < SHOCK_RSI_MAX,
    }

    if SHOCK_ABOVE_200SMA:
        sma_200 = close.rolling(200).mean()
        conditions["above_200sma"] = (
            float(close.iloc[sig]) > float(sma_200.iloc[sig])
        )

    all_pass = all(conditions.values())
    failed   = [k for k, v in conditions.items() if not v]

    return {
        "pass":         all_pass,
        "ticker":       ticker,
        "price":        float(close.iloc[-1]),
        "direction":    "LONG",
        "reason":       ", ".join(failed) if not all_pass else "",
        "shock_return": shock["price_return"],
        "vix_spike":    shock["vix_change"],
        "vix_level":    shock["vix_level"],
        "rsi":          float(rsi.iloc[sig]),
    }


# ==============================================================================
#  9. MAIN PIPELINE
# ==============================================================================

def main(vix_level: "float | None" = None,
         open_tickers: "set | None" = None) -> "dict | None":
    """
    Full policy shock bounce signal pipeline.

    1. Fetch VIX history.
    2. For each instrument (SPY, QQQ):
       a. Fetch price data.
       b. Live shock check.
       c. Backtest validation.
    3. Return verified signals.

    Arguments
    ---------
    vix_level : float | None
        Current VIX (for display only; detection uses full VIX history).
    open_tickers : set | None
        Currently open shock tickers to avoid double-entry.

    Returns
    -------
    dict with keys: verified, rejected, timestamp, vix
    """
    if open_tickers is None:
        open_tickers = set()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n  [SHOCK] {'=' * 55}")
    print(f"  [SHOCK] Policy Shock Bounce Signal Engine")
    print(f"  [SHOCK] {timestamp}")
    print(f"  [SHOCK] {'=' * 55}\n")

    # 1. Fetch VIX history
    print(f"  [SHOCK] Fetching VIX history ...")
    vix = fetch_vix(years=5)
    if vix is None:
        print(f"  [SHOCK] Failed to download VIX history.")
        return None

    current_vix = float(vix.iloc[-1])
    vix_display = f"{current_vix:.1f}"
    print(f"  [SHOCK] VIX: {vix_display}")

    result = {
        "verified":  [],
        "rejected":  [],
        "timestamp": timestamp,
        "vix":       current_vix,
    }

    # 2. Check each instrument
    for ticker in SHOCK_INSTRUMENTS:
        print(f"\n  [SHOCK] {'-' * 50}")
        print(f"  [SHOCK] Checking {ticker} for policy shock bounce ...")

        if ticker in open_tickers:
            print(f"  [SHOCK] Skipped {ticker} -- already has open position")
            continue

        instr_data = fetch_instrument(ticker, years=5)
        if instr_data is None:
            print(f"  [SHOCK] Failed to download {ticker} data.")
            continue

        # Live check
        live = shock_live_check(instr_data, vix, ticker)

        if live["pass"]:
            print(
                f"  [SHOCK] SIGNAL: {ticker}  "
                f"drop={live['shock_return']*100:+.1f}%  "
                f"VIX spike={live['vix_spike']:+.1f}  "
                f"RSI={live['rsi']:.1f}  "
                f"price=${live['price']:.2f}"
            )

            # Backtest validation
            print(f"  [SHOCK] Running walk-forward backtest on {ticker} ...")
            bt = backtest_shock(instr_data, vix, ticker)

            entry = {"ticker": ticker, "live": live, "bt": bt}

            if bt["pass"]:
                result["verified"].append(entry)
                print(
                    f"  [SHOCK] DIAMOND: WR={bt['win_rate']:.0f}%  "
                    f"PF={bt['profit_factor']:.2f}x  "
                    f"P&L=${bt['total_pnl']:+.0f}  "
                    f"Sharpe={bt['sharpe']:+.2f}"
                )
                print(
                    f"  [SHOCK]          OOS P&L=${bt['oos_pnl']:+.0f}  "
                    f"OOS trades={bt['oos_trades']}  "
                    f"trades={bt['n_trades']}"
                )
            else:
                result["rejected"].append(entry)
                print(f"  [SHOCK] REJECTED: {bt['reason']}")
        else:
            print(f"  [SHOCK] No signal: {live['reason']}")

    # 3. Summary
    n_verified = len(result["verified"])
    n_rejected = len(result["rejected"])
    print(f"\n  [SHOCK] {'=' * 55}")
    print(
        f"  [SHOCK] Results: {n_verified} diamond(s), "
        f"{n_rejected} rejected"
    )
    print(f"  [SHOCK] {'=' * 55}\n")

    return result


# ==============================================================================
#  STANDALONE ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n  VIX: {result['vix']:.1f}")
        if result["verified"]:
            print(f"\n  SHOCK DIAMONDS ({len(result['verified'])}):")
            for v in result["verified"]:
                print(
                    f"    {v['ticker']}  "
                    f"${v['live']['price']:.2f}  "
                    f"drop={v['live']['shock_return']*100:+.1f}%  "
                    f"VIX+{v['live']['vix_spike']:.1f}"
                )
        else:
            print("  No shock diamonds.")
        if result["rejected"]:
            print(f"\n  REJECTED ({len(result['rejected'])}):")
            for r in result["rejected"]:
                print(f"    {r['ticker']}  {r['bt']['reason']}")
