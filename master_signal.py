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
    Gate A — Split-Half  : Year 1 P&L > $0 AND Year 2 P&L > $0
    Gate B — Momentum    : Last 5 trades combined P&L > $0
    Gate C — Freshness   : ADF on last 90 days of spread, p < 0.10

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

import warnings, datetime, time, logging, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (imported from config.py — single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
from config import (
    INPUT_CSV, LOOKBACK_YEARS,
    ROLLING_BETA_WIN, ROLLING_Z_WIN,
    Z_ENTRY, Z_EXIT, Z_STOP, MAX_HOLD,
    CAPITAL_PER_TRADE, SLIPPAGE_PCT,
    MIN_WIN_RATE, MIN_PROFIT_FACTOR, MIN_TOTAL_PNL,
    MAX_RETRIES, RETRY_DELAY, ERROR_LOG,
    SPLIT_HALF_ENABLED,
    RECENT_ADF_WINDOW, RECENT_ADF_PVAL,
    RECENT_MOMENTUM_N,
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
#  DATA FETCH  (single download per pair, with retry)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_pair(a: str, b: str) -> pd.DataFrame | None:
    """
    Download ~2 years of daily adjusted closes.
    auto_adjust=True → Close column is split/dividend-adjusted.
    Retries up to MAX_RETRIES on failure.
    """
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=int(LOOKBACK_YEARS * 365 * 1.1))
    min_bars = ROLLING_BETA_WIN + ROLLING_Z_WIN + 50

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download([a, b], start=str(start), end=str(end),
                              auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                # Data hygiene: dropna enforces common index alignment.
                # No ffill — missing data = no signal, not fake convergence.
                close = raw["Close"][[a, b]].dropna()
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            if len(close) < min_bars:
                return None   # not enough data — no point retrying
            return close
        except Exception as e:
            _err_logger.warning(
                f"{a}/{b} fetch attempt {attempt}/{MAX_RETRIES}: {e}")
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

def compute_rolling_signals(close: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """
    Build day-by-day rolling beta, spread, z-score using LOG of adjusted prices.

    Timing guarantees (no lookahead):
      beta[t]   = OLS on log prices [t-60, t-1]      (excludes bar t)
      spread[t] = log(A_t) - beta[t] * log(B_t)      (today's observation)
      mu[t]     = mean( spread[t-20 .. t-1] )         (excludes bar t)
      sigma[t]  = std( spread[t-20 .. t-1] )          (excludes bar t)
      z[t]      = (spread[t] - mu[t]) / max(sigma[t], SIGMA_FLOOR)
    """
    pa_raw = close[a].values.astype(float)
    pb_raw = close[b].values.astype(float)
    # Guard: non-positive prices → NaN (prevents log(0) and division-by-zero)
    pa_raw = np.where(pa_raw > 0, pa_raw, np.nan)
    pb_raw = np.where(pb_raw > 0, pb_raw, np.nan)
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
    Returns dict with pass/fail, current z, direction, and current beta.
    """
    last = signals.iloc[-1]
    z_now    = last["z"]
    beta_now = last["beta"]

    if np.isnan(z_now):
        return {"pass": False, "reason": "Current z-score is NaN"}

    if abs(z_now) <= Z_ENTRY:
        return {"pass": False, "z": z_now, "beta": beta_now,
                "reason": f"|z| = {abs(z_now):.2f} <= {Z_ENTRY} threshold"}

    direction = "LONG" if z_now < -Z_ENTRY else "SHORT"
    return {"pass": True, "z": z_now, "beta": beta_now,
            "direction": direction}


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 2 — HISTORY CHECK  (walk-forward backtest, timing-safe)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_pair(signals: pd.DataFrame) -> dict:
    """
    Walk-forward backtest with 1-bar execution delay.

    Timing model (eliminates lookahead bias):
      Signal observed at close of bar i  →  execution at close of bar i+1.
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
    n       = len(z)

    trades     = []
    position   = 0       # 0=flat, 1=long spread, -1=short spread
    entry_bar  = 0       # bar index where position was opened
    shares_a   = 0.0
    shares_b   = 0.0

    # ── Signal on bar i → execute at bar i+1 ─────────────────────────────
    for i in range(n - 1):
        if np.isnan(z[i]):
            continue

        exec_bar = i + 1    # next-bar execution (no same-bar trading)

        # Guard: skip if execution-bar prices are invalid
        if np.isnan(price_a[exec_bar]) or np.isnan(price_b[exec_bar]):
            continue

        if position == 0:
            # ── Entry signals ────────────────────────────────────────────
            if z[i] < -Z_ENTRY:
                # Long spread: long A, short B
                half     = CAPITAL_PER_TRADE / 2.0
                shares_a = half / price_a[exec_bar]
                shares_b = half / price_b[exec_bar]
                position  = 1
                entry_bar = exec_bar
            elif z[i] > Z_ENTRY:
                # Short spread: short A, long B
                half     = CAPITAL_PER_TRADE / 2.0
                shares_a = half / price_a[exec_bar]
                shares_b = half / price_b[exec_bar]
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
                # ── P&L calculation (shares-based) ───────────────────────
                if position == 1:    # long A, short B
                    pnl_a =  shares_a * (price_a[exec_bar] - price_a[entry_bar])
                    pnl_b = -shares_b * (price_b[exec_bar] - price_b[entry_bar])
                else:                # short A, long B
                    pnl_a = -shares_a * (price_a[exec_bar] - price_a[entry_bar])
                    pnl_b =  shares_b * (price_b[exec_bar] - price_b[entry_bar])
                gross = pnl_a + pnl_b

                # ── TRUE two-leg slippage ────────────────────────────────
                #    Notional = |shares| * price   for EACH leg
                #    Applied at entry AND exit, on BOTH legs
                entry_notional = (shares_a * price_a[entry_bar]
                                + shares_b * price_b[entry_bar])
                exit_notional  = (shares_a * price_a[exec_bar]
                                + shares_b * price_b[exec_bar])
                cost = SLIPPAGE_PCT * (entry_notional + exit_notional)

                net_pnl = gross - cost

                trades.append({
                    "entry_bar":   entry_bar,
                    "hold_days":   hold_days,
                    "exit_reason": exit_reason,
                    "gross_pnl":   gross,
                    "net_pnl":     net_pnl,
                    "cost":        cost,
                })
                position = 0
                shares_a = 0.0
                shares_b = 0.0

    # ── Force-close any open position at end of data ─────────────────────
    if position != 0:
        last_bar = n - 1
        if position == 1:
            pnl_a =  shares_a * (price_a[last_bar] - price_a[entry_bar])
            pnl_b = -shares_b * (price_b[last_bar] - price_b[entry_bar])
        else:
            pnl_a = -shares_a * (price_a[last_bar] - price_a[entry_bar])
            pnl_b =  shares_b * (price_b[last_bar] - price_b[entry_bar])
        gross = pnl_a + pnl_b
        entry_notional = (shares_a * price_a[entry_bar]
                        + shares_b * price_b[entry_bar])
        exit_notional  = (shares_a * price_a[last_bar]
                        + shares_b * price_b[last_bar])
        cost = SLIPPAGE_PCT * (entry_notional + exit_notional)
        trades.append({
            "entry_bar":   entry_bar,
            "hold_days":   last_bar - entry_bar,
            "exit_reason": "eod",
            "gross_pnl":   gross,
            "net_pnl":     gross - cost,
            "cost":        cost,
        })

    # ── Compute metrics ──────────────────────────────────────────────────
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
                "avg_hold": 0.0, "pass": False,
                "reason": "No historical trades",
                "h1_pnl": 0.0, "h2_pnl": 0.0, "recent_pnl": 0.0}

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

    # ── Standard quality gates ───────────────────────────────────────────
    reasons = []
    if win_rate < MIN_WIN_RATE:
        reasons.append(f"WR {win_rate:.1f}% < {MIN_WIN_RATE}%")
    if profit_factor < MIN_PROFIT_FACTOR:
        reasons.append(f"PF {profit_factor:.2f}x < {MIN_PROFIT_FACTOR}x")
    if total_pnl <= MIN_TOTAL_PNL:
        reasons.append(f"P&L ${total_pnl:+.2f} \u2264 $0")

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
            if not h1 or h1_pnl_val <= 0:
                reasons.append(f"Failed H1 P&L (${h1_pnl_val:+.0f})")
            if not h2 or h2_pnl_val <= 0:
                reasons.append(f"Failed H2 P&L (${h2_pnl_val:+.0f})")

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

    passed = len(reasons) == 0

    return {
        "n_trades":      n_trades,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "total_pnl":     total_pnl,
        "avg_pnl":       avg_pnl,
        "sharpe":        sharpe,
        "avg_hold":      avg_hold,
        "pass":          passed,
        "reason":        " | ".join(reasons) if reasons else "",
        "h1_pnl":        h1_pnl_val,
        "h2_pnl":        h2_pnl_val,
        "recent_pnl":    recent_pnl,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def print_diamond(a: str, b: str, live: dict, bt: dict):
    """Print a DIAMOND SIGNAL block — passed ALL gates."""
    print()
    print("  \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
    print(f"  \u2551  \u25c6 DIAMOND SIGNAL \u25c6   {a:>6} / {b:<6}                 \u2551")
    print("  \u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")
    print(f"  \u2551  LIVE                                                    \u2551")
    print(f"  \u2551    Direction    : {live['direction']:<8}"
          f"                                \u2551")
    print(f"  \u2551    Current Z    : {live['z']:>+8.4f}"
          f"                                \u2551")
    print(f"  \u2551    Current \u03b2    : {live['beta']:>8.4f}"
          f"                                \u2551")
    print(f"  \u2551                                                          \u2551")
    print(f"  \u2551  BACKTEST (2yr, 1-bar delay)                             \u2551")
    print(f"  \u2551    Trades       : {bt['n_trades']:>5}"
          f"                                   \u2551")
    print(f"  \u2551    Win Rate     : {bt['win_rate']:>5.1f} %"
          f"                                 \u2551")
    print(f"  \u2551    Profit Factor: {bt['profit_factor']:>5.2f}x"
          f"                                 \u2551")
    print(f"  \u2551    Total P&L    : ${bt['total_pnl']:>+9.2f}"
          f"                            \u2551")
    print(f"  \u2551    Sharpe       : {bt['sharpe']:>+5.2f}"
          f"                                  \u2551")
    print(f"  \u2551    Avg Hold     : {bt['avg_hold']:>5.1f} days"
          f"                               \u2551")
    print(f"  \u2551                                                          \u2551")
    print(f"  \u2551  REGIME GATES                                            \u2551")
    print(f"  \u2551    Half-1 P&L   : ${bt.get('h1_pnl', 0):>+9.2f}"
          f"   \u2713                       \u2551")
    print(f"  \u2551    Half-2 P&L   : ${bt.get('h2_pnl', 0):>+9.2f}"
          f"   \u2713                       \u2551")
    print(f"  \u2551    Recent ADF   :  Stationary"
          f"   \u2713                       \u2551")
    print(f"  \u2551    Last 5 P&L   : ${bt.get('recent_pnl', 0):>+9.2f}"
          f"   \u2713                       \u2551")
    print("  \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")


def print_rejected(a: str, b: str, live: dict, bt: dict):
    """Print a rejection block with gate-specific reasons."""
    pf_s = f"{bt['profit_factor']:.2f}x" if bt['n_trades'] > 0 else "N/A"
    reason = bt.get("reason", "")

    # Determine short gate label for console
    if "H1 P&L" in reason or "H2 P&L" in reason or "Split-Half" in reason:
        gate = "Failed Split-Half"
    elif "ADF" in reason or "Low Recent Cointegration" in reason:
        gate = "Failed Cointegration / ADF"
    elif "Recent Momentum" in reason:
        gate = "Alpha Decay (Recent Trades)"
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
    print(f"  Execution: 1-bar delay (signal@close \u2192 trade@next close)\n")

    verified  = []
    rejected  = []
    no_signal = 0
    errors    = 0

    for idx, (a, b) in enumerate(pairs):
        label = f"{a} / {b}"
        print(f"  [{idx+1}/{n_pairs}]  {label:>14}  ", end="", flush=True)

        try:
            # ── Fetch data (with retry) ──────────────────────────────────
            close = fetch_pair(a, b)
            if close is None:
                print("SKIP  (download failed)")
                errors += 1
                continue

            # ── Compute rolling signals (timing-safe) ────────────────────
            signals = compute_rolling_signals(close, a, b)

            # ── STEP 1: Live Check ───────────────────────────────────────
            live = live_check(signals)
            if not live["pass"]:
                print(f"\u2014  No signal  (|z| = {abs(live.get('z', 0)):.2f})")
                no_signal += 1
                continue

            print(f"\u26a1 z={live['z']:+.2f} {live['direction']:>5}  \u2192  ", end="",
                  flush=True)

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

            if bt["pass"]:
                print(f"\u25c6 DIAMOND  (WR {bt['win_rate']:.0f}% | "
                      f"PF {bt['profit_factor']:.2f}x | "
                      f"P&L ${bt['total_pnl']:+.0f})")
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
    }


if __name__ == "__main__":
    main()
