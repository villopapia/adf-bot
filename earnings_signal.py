"""
================================================================================
 EARNINGS SIGNAL  -  Entry Filter & Signal Generator
================================================================================
 Takes the raw candidates from earnings_scanner.py, applies entry quality
 gates, and returns verified / rejected signal dicts.

 Called from run_system.py Phase 5 via importlib.import_module("earnings_signal").

 Signal dict format (verified):
   {
     "ticker": str,
     "live": {
         "price": float,
         "direction": "LONG",
         "earnings_date": str,          # "YYYY-MM-DD"
         "days_until_earnings": int,
         "beat_rate": float,
         "avg_eps_surprise_pct": float,
         "avg_rev_surprise_pct": float,
         "eps_trend": str,
     },
     "bt": {
         "pass": True,
         "reason": "",
         "win_rate": float,             # == beat_rate * 100
         "profit_factor": 0.0,
         "total_pnl": 0.0,
     }
   }
================================================================================
"""

import os
import sys
import datetime
import logging

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
)
from earnings_scanner import scan_earnings_candidates

_logger = logging.getLogger("earnings_signal")


# ------------------------------------------------------------------------------
#  Price / SMA fetch
# ------------------------------------------------------------------------------

def _get_price_and_sma(ticker: str) -> tuple:
    """
    Fetch latest close price and 200-day SMA for ticker.
    Returns (price, sma200) or (None, None) on failure.
    Handles yfinance MultiIndex columns (single-ticker downloads).
    """
    try:
        raw = yf.download(ticker, period="220d", auto_adjust=True, progress=False)
        if raw is None or len(raw) < 5:
            return None, None

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw[("Close", ticker)]
        else:
            close = raw["Close"]

        price  = float(close.iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        return price, sma200
    except Exception as exc:
        _logger.warning(f"[earnings_signal] Price fetch failed for {ticker}: {exc}")
        return None, None


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

        # Fetch current price + 200-day SMA
        price, sma200 = _get_price_and_sma(ticker)
        if price is None:
            reason = "price fetch failed"
            rejected.append({
                "ticker": ticker,
                "bt": {"pass": False, "reason": reason,
                       "win_rate": beat_rate * 100, "profit_factor": 0.0, "total_pnl": 0.0},
            })
            print(f"    x {ticker:<8} SKIP  {reason}")
            continue

        # Gate 6: Price above 200-day SMA
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

        # All gates passed
        sma_display = f"${sma200:.2f}" if sma200 is not None else "N/A"
        print(f"    + {ticker:<8} PASS  "
              f"earnings={earnings_date}  days={days_until}  "
              f"beat={beat_rate:.0%}  eps_surp={avg_eps_surprise_pct:+.1f}%  "
              f"trend={eps_trend}  price=${price:.2f}  SMA200={sma_display}")

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
            },
            "bt": {
                "pass":          True,
                "reason":        "",
                "win_rate":      round(beat_rate * 100, 1),
                "profit_factor": 0.0,
                "total_pnl":     0.0,
            },
        }
        verified.append(signal)

    # Summary
    print()
    print(f"  [Earnings Signal] Verified: {len(verified)}  |  "
          f"Rejected: {len(rejected)}")
    print()

    return {"verified": verified, "rejected": rejected}
