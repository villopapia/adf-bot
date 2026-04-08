"""
================================================================================
 earnings_util.py -- Shared Earnings Blackout Utility
================================================================================
 Provides FMP-based earnings date fetching and blackout checks used by both
 the pairs strategy (master_signal.py) and the momentum strategy
 (momentum_signal.py).

 Results are cached per run (module-level dict) so each ticker is fetched
 at most once regardless of how many callers ask for it.
================================================================================
"""

import datetime
import logging
import requests

from config import (FMP_API_KEY, EARNINGS_BLACKOUT_DAYS,
                    MOM_EARNINGS_BEAT_RATE_MIN, MOM_EARNINGS_MIN_QUARTERS,
                    MOM_EARNINGS_ALLOW_DECLINING)

# ------------------------------------------------------------------------------
#  MODULE-LEVEL CACHE  (lives for the duration of one process invocation)
# ------------------------------------------------------------------------------
_fmp_earnings_cache: dict[str, list] = {}
_fmp_earnings_raw_cache: dict[str, list[dict]] = {}

_logger = logging.getLogger("earnings_util")


# ------------------------------------------------------------------------------
#  PUBLIC API
# ------------------------------------------------------------------------------

def clear_earnings_cache() -> None:
    """Reset the per-run earnings cache.  Call at the start of each main() run."""
    global _fmp_earnings_cache, _fmp_earnings_raw_cache
    _fmp_earnings_cache = {}
    _fmp_earnings_raw_cache = {}


def _fetch_earnings_raw(symbol: str) -> list[dict]:
    """Fetch full earnings data from FMP. Cached per-run."""
    global _fmp_earnings_raw_cache
    if symbol in _fmp_earnings_raw_cache:
        return _fmp_earnings_raw_cache[symbol]
    if not FMP_API_KEY:
        _fmp_earnings_raw_cache[symbol] = []
        return []
    try:
        resp = requests.get(
            "https://financialmodelingprep.com/stable/earnings",
            params={"symbol": symbol, "limit": 5, "apikey": FMP_API_KEY},
            timeout=10,
        )
        data = resp.json()
        if not isinstance(data, list):
            _fmp_earnings_raw_cache[symbol] = []
            return []
        _fmp_earnings_raw_cache[symbol] = data
        return data
    except Exception as exc:
        _logger.warning(f"FMP earnings raw fetch failed for {symbol}: {exc}")
        _fmp_earnings_raw_cache[symbol] = []
        return []


def fetch_earnings_dates(symbol: str) -> list:
    """
    Fetch recent + upcoming earnings dates for a ticker from FMP.
    Returns a list of datetime.date objects (empty on failure or if no API key).
    Results are cached so each ticker is fetched at most once per run.
    """
    global _fmp_earnings_cache
    if symbol in _fmp_earnings_cache:
        return _fmp_earnings_cache[symbol]
    raw = _fetch_earnings_raw(symbol)
    dates = []
    for row in raw:
        d = row.get("date")
        if d:
            try:
                dates.append(datetime.date.fromisoformat(d))
            except ValueError:
                pass
    _fmp_earnings_cache[symbol] = dates
    return dates


def check_earnings_blackout(
    symbol: str,
    pre_days: int,
    post_days: int,
) -> "tuple[str | None, int | None]":
    """
    Check whether *symbol* has earnings within the blackout window.

    The window spans [-post_days, +pre_days] relative to today:
      - delta > 0  =>  earnings is N days in the future  (upcoming)
      - delta < 0  =>  earnings was N days ago            (recently reported)

    Parameters
    ----------
    symbol    : Ticker to check.
    pre_days  : Block entry if earnings are upcoming within this many days.
    post_days : Block entry if earnings were reported within this many days.

    Returns
    -------
    (symbol, delta_days) if in blackout window, else (None, None).
    """
    today = datetime.date.today()
    for edate in fetch_earnings_dates(symbol):
        delta = (edate - today).days
        if -post_days <= delta <= pre_days:
            return symbol, delta
    return None, None


def check_pair_earnings_blackout(
    a: str,
    b: str,
) -> "tuple[str | None, int | None]":
    """
    Wrapper for the pairs strategy — checks both legs of a pair.

    Uses EARNINGS_BLACKOUT_DAYS from config for both pre and post windows,
    preserving backward compatibility with master_signal.py Gate E.

    Returns (ticker, delta_days) for the first leg that is in blackout,
    or (None, None) if both legs are clear.
    """
    for symbol in (a, b):
        sym, delta = check_earnings_blackout(
            symbol,
            pre_days=EARNINGS_BLACKOUT_DAYS,
            post_days=EARNINGS_BLACKOUT_DAYS,
        )
        if sym is not None:
            return sym, delta
    return None, None


def assess_earnings_quality(symbol: str) -> dict:
    """
    Analyze historical earnings to decide if pre-earnings entry is allowed.

    Returns dict with:
        beat_rate, avg_eps_surprise_pct, avg_rev_surprise_pct,
        eps_trend ("growing"/"flat"/"declining"), quarters_available,
        next_earnings_date, allow_pre_earnings (bool)
    """
    raw = _fetch_earnings_raw(symbol)

    completed = []
    next_earnings_date = None

    for row in raw:
        date_str = row.get("date")
        eps_actual = row.get("epsActual")
        eps_estimated = row.get("epsEstimated")
        rev_actual = row.get("revenueActual")
        rev_estimated = row.get("revenueEstimated")

        if eps_actual is not None and eps_estimated is not None:
            completed.append({
                "date": date_str,
                "eps_actual": eps_actual,
                "eps_estimated": eps_estimated,
                "rev_actual": rev_actual,
                "rev_estimated": rev_estimated,
            })
        elif date_str and eps_actual is None:
            try:
                candidate = datetime.date.fromisoformat(date_str)
                if candidate >= datetime.date.today():
                    next_earnings_date = candidate
            except ValueError:
                pass

    quarters_available = len(completed)

    if quarters_available == 0:
        return {
            "beat_rate": 0.0,
            "avg_eps_surprise_pct": 0.0,
            "avg_rev_surprise_pct": 0.0,
            "eps_trend": "flat",
            "quarters_available": 0,
            "next_earnings_date": next_earnings_date,
            "allow_pre_earnings": False,
        }

    # Beat rate
    beats = sum(1 for q in completed if q["eps_actual"] > q["eps_estimated"])
    beat_rate = beats / quarters_available

    # Average EPS surprise %
    eps_surprises = []
    for q in completed:
        est = q["eps_estimated"]
        if est != 0:
            eps_surprises.append((q["eps_actual"] - est) / abs(est) * 100)
    avg_eps_surprise = sum(eps_surprises) / len(eps_surprises) if eps_surprises else 0.0

    # Average revenue surprise %
    rev_surprises = []
    for q in completed:
        ra, re = q.get("rev_actual"), q.get("rev_estimated")
        if ra is not None and re is not None and re != 0:
            rev_surprises.append((ra - re) / abs(re) * 100)
    avg_rev_surprise = sum(rev_surprises) / len(rev_surprises) if rev_surprises else 0.0

    # EPS trend (split-half comparison)
    sorted_q = sorted(completed, key=lambda q: q["date"])
    if len(sorted_q) >= 2:
        mid = len(sorted_q) // 2
        avg_older = sum(q["eps_actual"] for q in sorted_q[:mid]) / mid
        avg_newer = sum(q["eps_actual"] for q in sorted_q[mid:]) / len(sorted_q[mid:])
        pct_change = (avg_newer - avg_older) / abs(avg_older) * 100 if avg_older != 0 else 0
        if pct_change > 5:
            eps_trend = "growing"
        elif pct_change < -5:
            eps_trend = "declining"
        else:
            eps_trend = "flat"
    else:
        eps_trend = "flat"

    # Decision
    allow = (
        beat_rate >= MOM_EARNINGS_BEAT_RATE_MIN
        and avg_eps_surprise > 0
        and quarters_available >= MOM_EARNINGS_MIN_QUARTERS
        and (eps_trend != "declining" or MOM_EARNINGS_ALLOW_DECLINING)
    )

    return {
        "beat_rate": beat_rate,
        "avg_eps_surprise_pct": round(avg_eps_surprise, 2),
        "avg_rev_surprise_pct": round(avg_rev_surprise, 2),
        "eps_trend": eps_trend,
        "quarters_available": quarters_available,
        "next_earnings_date": next_earnings_date,
        "allow_pre_earnings": allow,
    }
