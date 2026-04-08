"""
================================================================================
 EARNINGS SCANNER  -  Upcoming Earnings Calendar Scanner
================================================================================
 Scans the FMP earnings calendar for upcoming S&P 500 + Nasdaq-100 companies
 reporting within the configured lookahead window, then assesses earnings
 quality (beat rate, EPS surprise, trend) for each match.

 Main entry point:
   scan_earnings_candidates(lookahead_days=None)
       Returns list of candidate dicts ready for earnings_signal.py to filter.

 Universe:
   get_earnings_universe()
       Combines S&P 500 + Nasdaq-100, cached to earnings_universe_cache.json
       for 24 hours so repeated intraday calls don't hammer Wikipedia.

 Calendar:
   fetch_earnings_calendar(from_date, to_date)
       Single FMP /stable/earning_calendar call for the date range.
================================================================================
"""

import os
import sys
import json
import time
import datetime
import requests
import logging

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import FMP_API_KEY, EARN_ENTRY_DAYS_BEFORE
from earnings_util import assess_earnings_quality

_logger = logging.getLogger("earnings_scanner")

_UNIVERSE_CACHE_PATH = os.path.join(_SCRIPT_DIR, "earnings_universe_cache.json")
_UNIVERSE_CACHE_TTL  = 86400  # 24 hours in seconds


# ------------------------------------------------------------------------------
#  Universe
# ------------------------------------------------------------------------------

def get_earnings_universe() -> set:
    """
    Return the combined S&P 500 + Nasdaq-100 ticker universe as a set.
    Results are cached to earnings_universe_cache.json for 24 hours.
    """
    # Check cache freshness
    if os.path.exists(_UNIVERSE_CACHE_PATH):
        try:
            with open(_UNIVERSE_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            ts = cached.get("timestamp", 0)
            if time.time() - ts < _UNIVERSE_CACHE_TTL:
                tickers = set(cached.get("tickers", []))
                if tickers:
                    _logger.info(f"[earnings_scanner] Universe loaded from cache: {len(tickers)} tickers")
                    return tickers
        except Exception as exc:
            _logger.warning(f"[earnings_scanner] Cache read failed: {exc}")

    # Import here to avoid circular imports; same pattern as momentum_scanner.py
    from nightly_scanner import get_sp500_with_sectors, get_nasdaq100_tickers

    tickers = set()
    try:
        sp500_df = get_sp500_with_sectors()
        tickers.update(sp500_df["Symbol"].tolist())
    except Exception as exc:
        _logger.warning(f"[earnings_scanner] S&P 500 fetch failed: {exc}")

    try:
        ndx100_df = get_nasdaq100_tickers()
        tickers.update(ndx100_df["Symbol"].tolist())
    except Exception as exc:
        _logger.warning(f"[earnings_scanner] Nasdaq-100 fetch failed: {exc}")

    # Persist cache
    try:
        payload = {"timestamp": time.time(), "tickers": sorted(tickers)}
        with open(_UNIVERSE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        _logger.warning(f"[earnings_scanner] Cache write failed: {exc}")

    _logger.info(f"[earnings_scanner] Universe refreshed: {len(tickers)} tickers")
    return tickers


# ------------------------------------------------------------------------------
#  Calendar Fetch
# ------------------------------------------------------------------------------

def fetch_earnings_calendar(from_date: str, to_date: str) -> list:
    """
    Fetch FMP earnings calendar for [from_date, to_date].
    Dates should be "YYYY-MM-DD" strings.
    Returns list of dicts (may be empty on failure or missing API key).
    """
    if not FMP_API_KEY:
        _logger.warning("[earnings_scanner] FMP_API_KEY not set; cannot fetch calendar.")
        return []

    url = (
        f"https://financialmodelingprep.com/stable/earnings-calendar"
        f"?from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        _logger.warning(f"[earnings_scanner] Unexpected calendar response type: {type(data)}")
        return []
    except Exception as exc:
        _logger.warning(f"[earnings_scanner] Calendar fetch failed: {exc}")
        return []


# ------------------------------------------------------------------------------
#  Main Scanner
# ------------------------------------------------------------------------------

def scan_earnings_candidates(lookahead_days: int = None) -> list:
    """
    Scan for upcoming earnings in our S&P 500 + Nasdaq-100 universe.

    For each company reporting within lookahead_days trading days, calls
    assess_earnings_quality() to gather historical beat rate and EPS data.

    Parameters
    ----------
    lookahead_days : int, optional
        How many calendar days forward to scan. Defaults to EARN_ENTRY_DAYS_BEFORE.

    Returns
    -------
    list of dicts, each containing:
        symbol, earnings_date, days_until_earnings, beat_rate,
        avg_eps_surprise_pct, avg_rev_surprise_pct, eps_trend,
        quarters_available, eps_estimated, revenue_estimated
    """
    if lookahead_days is None:
        lookahead_days = EARN_ENTRY_DAYS_BEFORE

    today      = datetime.date.today()
    from_date  = today.strftime("%Y-%m-%d")
    to_date    = (today + datetime.timedelta(days=lookahead_days)).strftime("%Y-%m-%d")

    print(f"  [Earnings Scanner] Fetching calendar {from_date} -> {to_date} ...")

    universe = get_earnings_universe()
    if not universe:
        print("  [Earnings Scanner] Universe is empty -- aborting.")
        return []

    calendar = fetch_earnings_calendar(from_date, to_date)
    if not calendar:
        print("  [Earnings Scanner] No earnings calendar data returned.")
        return []

    # Filter calendar to our universe
    in_universe = []
    seen = set()
    for entry in calendar:
        sym = str(entry.get("symbol", "")).upper().strip()
        if not sym or sym in seen:
            continue
        if sym in universe:
            seen.add(sym)
            in_universe.append(entry)

    print(f"  [Earnings Scanner] {len(calendar)} calendar entries -> "
          f"{len(in_universe)} in universe")

    candidates = []
    for entry in in_universe:
        sym = str(entry.get("symbol", "")).upper().strip()

        # Parse earnings date
        date_str = entry.get("date", "")
        try:
            earnings_date = datetime.date.fromisoformat(date_str)
        except (ValueError, TypeError):
            continue

        days_until = (earnings_date - today).days

        # Assess earnings quality (FMP per-ticker call, cached per run)
        try:
            quality = assess_earnings_quality(sym)
        except Exception as exc:
            _logger.warning(f"[earnings_scanner] Quality assessment failed for {sym}: {exc}")
            time.sleep(0.3)
            continue

        time.sleep(0.3)  # respect FMP rate limit

        candidates.append({
            "symbol":                sym,
            "earnings_date":         earnings_date.strftime("%Y-%m-%d"),
            "days_until_earnings":   days_until,
            "beat_rate":             quality["beat_rate"],
            "avg_eps_surprise_pct":  quality["avg_eps_surprise_pct"],
            "avg_rev_surprise_pct":  quality["avg_rev_surprise_pct"],
            "eps_trend":             quality["eps_trend"],
            "quarters_available":    quality["quarters_available"],
            "eps_estimated":         entry.get("epsEstimated"),
            "revenue_estimated":     entry.get("revenueEstimated"),
        })

    print(f"  [Earnings Scanner] {len(candidates)} candidates with quality data.")
    return candidates
