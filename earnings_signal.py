"""
================================================================================
 EARNINGS SIGNAL  -  Entry Filter & Signal Generator
================================================================================
 Takes the raw candidates from earnings_scanner.py, applies entry quality
 gates, and returns verified / rejected signal dicts.

 Called from run_system.py Phase 5 via importlib.import_module("earnings_signal").

 Quality gates (in order):
   1. Already in open trade        5. EPS trend not declining
   2. Beat rate >= 75%             6. Revenue surprise >= 0%
   3. EPS surprise > 0             7. Price above 200-day SMA
   4. Minimum 3 quarters data      8. 10-day drift < 8% (sell-the-news)
   9. Walk-forward backtest pass (win rate, PF, Sharpe, total P&L, OOS P&L)
================================================================================
"""

import os
import sys
import datetime
import logging

import numpy as np
import yfinance as yf
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    EARN_MIN_BEAT_RATE,
    EARN_MIN_QUARTERS,
    EARN_ALLOW_DECLINING,
    EARN_REQUIRE_ABOVE_200SMA,
    EARN_ENTRY_DAYS_BEFORE,
    EARN_MIN_DAYS_BEFORE,
    EARN_MAX_RUNUP_PCT,
    EARN_MIN_REV_SURPRISE_PCT,
    EARN_SHORT_SQUEEZE_SI,
    EARN_CAPITAL_PER_TRADE,
    EARN_SLIPPAGE_PCT,
    EARN_STOP_PCT,
    EARN_MAX_HOLD,
    EARN_BT_MIN_TRADES,
    EARN_BT_MIN_WIN_RATE,
    EARN_BT_MIN_PROFIT_FACTOR,
    EARN_BT_MIN_SHARPE,
    EARN_BT_MIN_TOTAL_PNL,
)
from earnings_scanner import scan_earnings_candidates

_logger = logging.getLogger("earnings_signal")


# ------------------------------------------------------------------------------
#  Price / SMA fetch
# ------------------------------------------------------------------------------

def _get_price_data(ticker: str) -> dict:
    """
    Fetch latest close price, 200-day SMA, and 10-day drift for ticker.
    Returns dict with keys: price, sma200, drift_10d.
    All values are float or None on failure/insufficient data.
    Handles yfinance MultiIndex columns (single-ticker downloads).
    """
    _none = {"price": None, "sma200": None, "drift_10d": None}
    try:
        raw = yf.download(ticker, period="220d", auto_adjust=True, progress=False)
        if raw is None or len(raw) < 5:
            return _none

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw[("Close", ticker)]
        else:
            close = raw["Close"]

        price    = float(close.iloc[-1])
        sma200   = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        drift_10d = (float(price / close.iloc[-11] - 1) * 100) if len(close) >= 11 else None
        return {"price": price, "sma200": sma200, "drift_10d": drift_10d}
    except Exception as exc:
        _logger.warning(f"[earnings_signal] Price fetch failed for {ticker}: {exc}")
        return _none


def _get_short_interest(ticker: str) -> float:
    """Return short % of float via yfinance, or 0.0 on failure."""
    try:
        info = yf.Ticker(ticker).info
        return float(info.get("shortPercentOfFloat", 0) or 0)
    except Exception:
        return 0.0


# ------------------------------------------------------------------------------
#  Backtest helpers
# ------------------------------------------------------------------------------

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


def _fetch_earnings_history(ticker: str) -> list:
    """
    Fetch historical earnings dates from yfinance.
    Returns list of datetime.date for completed earnings (has Reported EPS),
    sorted oldest-first.
    """
    try:
        t  = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or len(ed) == 0:
            return []
        completed = ed[ed["Reported EPS"].notna()]
        # Convert tz-aware timestamps to naive dates
        dates = [ts.date() for ts in completed.index]
        return sorted(dates)  # oldest first
    except Exception:
        return []


def backtest_earnings(ticker: str, close: pd.Series, earnings_dates: list) -> dict:
    """
    Walk-forward bar-by-bar backtest of the pre-earnings anticipation strategy.

    For each earnings date:
      - Entry: EARN_ENTRY_DAYS_BEFORE trading days before earnings at close price
      - Exit 1: 1 trading day after earnings (capture the gap)
      - Exit 2: Stop loss at EARN_STOP_PCT from entry (bar-by-bar check)
      - Exit 3: Time stop at EARN_MAX_HOLD trading days

    Walk-forward split: 70% in-sample / 30% out-of-sample on earnings_dates list.
    """
    _fail_base = {
        **_empty_metrics(),
        "oos_pnl": 0.0, "oos_trades": 0, "oos_sharpe": 0.0,
        "is_pnl": 0.0, "is_trades": 0,
        "pass": False,
    }

    close_dates = close.index.date  # array of datetime.date, timezone-naive

    trades = []

    for earn_date in earnings_dates:
        # Find the bar index for the trading day at or just before the earnings date
        earn_idxs = [i for i, d in enumerate(close_dates) if d <= earn_date]
        if not earn_idxs:
            continue
        earn_bar = earn_idxs[-1]  # last trading day on or before earnings date

        # Entry bar: EARN_ENTRY_DAYS_BEFORE trading days before earnings bar
        # Clamp to at least EARN_MIN_DAYS_BEFORE days before (match live gate)
        entry_offset = min(EARN_ENTRY_DAYS_BEFORE, earn_bar)
        if entry_offset < EARN_MIN_DAYS_BEFORE:
            continue
        entry_bar = earn_bar - entry_offset
        if entry_bar < 0:
            continue

        entry_price = float(close.iloc[entry_bar])
        if entry_price <= 0:
            continue

        # Gate: check 10d drift at entry bar (retroactive runup filter)
        if entry_bar >= 10:
            prev_price = float(close.iloc[entry_bar - 10])
            drift_10d  = (entry_price / prev_price - 1) * 100.0
            if drift_10d > EARN_MAX_RUNUP_PCT:
                continue  # skip — sell-the-news risk

        # Exit bar target: 1 trading day after earnings bar
        exit_target = earn_bar + 1

        # Simulate hold bar-by-bar: check stop loss and time stop
        position    = True
        exit_bar    = None
        exit_reason = ""

        for bar in range(entry_bar + 1, min(entry_bar + EARN_MAX_HOLD + 1, len(close))):
            current_price = float(close.iloc[bar])
            pnl_pct       = (current_price - entry_price) / entry_price * 100.0
            hold_days     = bar - entry_bar

            # Stop loss
            if pnl_pct <= -EARN_STOP_PCT:
                exit_bar    = bar
                exit_reason = "stop_loss"
                break

            # Planned exit: 1 day after earnings
            if bar >= exit_target:
                exit_bar    = bar
                exit_reason = "post_earnings"
                break

            # Time stop
            if hold_days >= EARN_MAX_HOLD:
                exit_bar    = bar
                exit_reason = "time_stop"
                break

        if exit_bar is None:
            # Trade still open at end of data — skip it
            continue

        exit_price = float(close.iloc[exit_bar])
        shares     = EARN_CAPITAL_PER_TRADE / entry_price
        gross      = shares * (exit_price - entry_price)
        slippage   = EARN_SLIPPAGE_PCT * shares * (entry_price + exit_price)
        net_pnl    = gross - slippage
        hold_days  = exit_bar - entry_bar

        trades.append({
            "entry_bar":   entry_bar,
            "exit_bar":    exit_bar,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "shares":      shares,
            "hold_days":   hold_days,
            "gross_pnl":   gross,
            "net_pnl":     net_pnl,
            "exit_reason": exit_reason,
            "earn_date":   earn_date,
        })

    if not trades:
        return {**_fail_base, "reason": "no_trades_simulated"}

    # Walk-forward split: split earnings_dates list at 70%
    split_idx   = int(len(earnings_dates) * 0.70)
    split_date  = earnings_dates[split_idx] if split_idx < len(earnings_dates) else earnings_dates[-1]

    is_trades  = [t for t in trades if t["earn_date"] <  split_date]
    oos_trades = [t for t in trades if t["earn_date"] >= split_date]

    all_metrics = _compute_metrics(trades)
    oos_metrics = _compute_metrics(oos_trades)
    is_metrics  = _compute_metrics(is_trades)

    # Quality gates
    gates = []
    gates.append(("min_trades",    all_metrics["n_trades"]      >= EARN_BT_MIN_TRADES))
    gates.append(("win_rate",      all_metrics["win_rate"]       >= EARN_BT_MIN_WIN_RATE))
    gates.append(("profit_factor", all_metrics["profit_factor"]  >= EARN_BT_MIN_PROFIT_FACTOR))
    gates.append(("sharpe",        all_metrics["sharpe"]         >= EARN_BT_MIN_SHARPE))
    gates.append(("total_pnl",     all_metrics["total_pnl"]      >= EARN_BT_MIN_TOTAL_PNL))
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


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main(open_tickers: set = None) -> dict:
    """
    Entry point called by run_system.py.

    Parameters
    ----------
    open_tickers : set, optional
        Set of ticker strings already in open earnings trades (to avoid doubling up).

    Returns
    -------
    dict with keys:
        "verified"  : list of signal dicts that passed all gates
        "rejected"  : list of dicts with "ticker" and "bt.reason" explaining rejection
    """
    if open_tickers is None:
        open_tickers = set()

    print(f"\n  [Earnings Signal] Scanning for earnings within "
          f"{EARN_ENTRY_DAYS_BEFORE} days ...\n")

    candidates = scan_earnings_candidates()
    verified   = []
    rejected   = []

    for cand in candidates:
        ticker               = cand["symbol"]
        earnings_date        = cand["earnings_date"]
        days_until           = cand["days_until_earnings"]
        beat_rate            = cand["beat_rate"]
        avg_eps_surprise_pct = cand["avg_eps_surprise_pct"]
        avg_rev_surprise_pct = cand["avg_rev_surprise_pct"]
        eps_trend            = cand["eps_trend"]
        quarters_available   = cand["quarters_available"]

        # Gate 1: Already in an open earnings trade
        if ticker in open_tickers:
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": "already in open trade",
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} SKIP  already in open trade")
            continue

        # Gate 1b: Minimum days before earnings (no same-day or next-day gambles)
        if days_until < EARN_MIN_DAYS_BEFORE:
            reason = f"earnings too soon ({days_until}d < {EARN_MIN_DAYS_BEFORE}d minimum)"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        # Gate 2: Minimum historical beat rate
        if beat_rate < EARN_MIN_BEAT_RATE:
            reason = f"beat_rate {beat_rate:.0%} < {EARN_MIN_BEAT_RATE:.0%}"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        # Gate 3: Positive average EPS surprise
        if avg_eps_surprise_pct <= 0:
            reason = f"avg_eps_surprise {avg_eps_surprise_pct:.1f}% <= 0"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        # Gate 4: Minimum quarters of data
        if quarters_available < EARN_MIN_QUARTERS:
            reason = f"quarters {quarters_available} < {EARN_MIN_QUARTERS}"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        # Gate 5: EPS trend (no declining unless configured)
        if eps_trend == "declining" and not EARN_ALLOW_DECLINING:
            reason = "eps_trend declining"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        # Gate 6: Revenue surprise consistency
        if avg_rev_surprise_pct < EARN_MIN_REV_SURPRISE_PCT:
            reason = (f"avg_rev_surprise {avg_rev_surprise_pct:.1f}% "
                      f"< {EARN_MIN_REV_SURPRISE_PCT:.1f}%")
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        # Fetch current price + 200-day SMA + 10-day drift
        pdata     = _get_price_data(ticker)
        price     = pdata["price"]
        sma200    = pdata["sma200"]
        drift_10d = pdata["drift_10d"]
        if price is None:
            reason = "price fetch failed"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} SKIP  {reason}")
            continue

        # Gate 7: Price above 200-day SMA
        if EARN_REQUIRE_ABOVE_200SMA:
            if sma200 is None:
                reason = "SMA200 unavailable (< 200d history)"
                rejected.append({
                    "ticker": ticker,
                    "bt": {"pass": False, "reason": reason,
                           "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
                })
                print(f"    x {ticker:<8} FAIL  {reason}")
                continue
            if price < sma200:
                reason = f"price ${price:.2f} < SMA200 ${sma200:.2f}"
                rejected.append({
                    "ticker": ticker,
                    "bt": {"pass": False, "reason": reason,
                           "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
                })
                print(f"    x {ticker:<8} FAIL  {reason}")
                continue

        # Gate 8: Pre-earnings drift (sell-the-news risk)
        if drift_10d is not None and drift_10d > EARN_MAX_RUNUP_PCT:
            reason = (f"10d runup {drift_10d:+.1f}% "
                      f"> {EARN_MAX_RUNUP_PCT:.0f}% (sell-the-news risk)")
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        # All gates passed — fetch short interest as enrichment flag
        si_pct       = _get_short_interest(ticker)
        squeeze_flag = si_pct >= EARN_SHORT_SQUEEZE_SI

        # Backtest validation using yfinance earnings history + 5y price data
        earn_hist = _fetch_earnings_history(ticker)
        if len(earn_hist) < EARN_BT_MIN_TRADES:
            reason = f"insufficient earnings history ({len(earn_hist)} events)"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} FAIL  {reason}")
            continue

        bt_raw = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
        if bt_raw is None or len(bt_raw) < 50:
            reason = "5y price data unavailable for backtest"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} SKIP  {reason}")
            continue

        if isinstance(bt_raw.columns, pd.MultiIndex):
            bt_close = bt_raw[("Close", ticker)]
        else:
            bt_close = bt_raw["Close"]

        bt = backtest_earnings(ticker, bt_close, earn_hist)

        if not bt["pass"]:
            reason = bt["reason"]
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": bt["win_rate"], "profit_factor": bt["profit_factor"],
                       "total_pnl": bt["total_pnl"]},
            })
            print(f"    x {ticker:<8} FAIL  backtest: {reason}")
            continue

        sma_display   = f"${sma200:.2f}" if sma200 is not None else "N/A"
        drift_display = f"{drift_10d:+.1f}%" if drift_10d is not None else "N/A"
        si_display    = f"  SI={si_pct*100:.1f}% [SQUEEZE]" if squeeze_flag else ""
        print(f"    + {ticker:<8} DIAMOND  "
              f"BT: WR={bt['win_rate']:.0f}%  PF={bt['profit_factor']:.2f}x  "
              f"P&L=${bt['total_pnl']:+.0f}  Sharpe={bt['sharpe']:.2f}  "
              f"OOS=${bt['oos_pnl']:+.0f}  trades={bt['n_trades']}")

        signal = {
            "ticker": ticker,
            "live": {
                "price":                price,
                "direction":            "LONG",
                "earnings_date":        earnings_date,
                "days_until_earnings":  days_until,
                "beat_rate":            beat_rate,
                "avg_eps_surprise_pct": avg_eps_surprise_pct,
                "avg_rev_surprise_pct": avg_rev_surprise_pct,
                "eps_trend":            eps_trend,
                "drift_10d":            drift_10d,
                "short_interest":       si_pct,
                "squeeze_flag":         squeeze_flag,
            },
            "bt": {
                "pass":          True,
                "reason":        "",
                "win_rate":      bt["win_rate"],
                "profit_factor": bt["profit_factor"],
                "total_pnl":     bt["total_pnl"],
                "sharpe":        bt["sharpe"],
                "n_trades":      bt["n_trades"],
                "oos_pnl":       bt["oos_pnl"],
                "max_dd":        bt["max_dd"],
            },
        }
        verified.append(signal)

    # Summary
    print()
    print(f"  [Earnings Signal] Verified: {len(verified)}  |  "
          f"Rejected: {len(rejected)}")
    print()

    return {"verified": verified, "rejected": rejected}
