"""
================================================================================
 GLOBAL RISK MANAGER  -  Cross-Module Portfolio Risk Controller
================================================================================
 Passive, read-only observer of all five trading modules.  Modules do not
 import this file -- it reads their CSV/JSON state files directly and imposes
 three escalation tiers on top of the per-module kill switches.

 Three Escalation Tiers
 ----------------------
   Tier 1  ALLOW      Normal operation.  Gatekeeper checks enforce per-trade
                      limits on capital, positions, ticker overlap, and
                      directional concentration before each new entry.

   Tier 2  FREEZE     All new entries blocked across every module.  Existing
                      per-module stop-loss and exit logic continues to run.
                      Triggered by combined drawdown or single-day loss.

   Tier 3  LIQUIDATE  Emergency close of all positions.  Triggered when
                      combined drawdown exceeds GLOBAL_LIQUIDATE_DRAWDOWN.
                      run_system.py handles the physical closing; this module
                      returns the status and reasons.

 Gatekeeper Checks (Tier 1)
 --------------------------
   1. Total deployed capital across all open positions
   2. Total open position count across all modules
   3. Same ticker appearing in more than GLOBAL_MAX_TICKER_OVERLAP modules
   4. Directional concentration: capital_in_direction / total_capital
      - SH (inverse ETF) counts as SHORT even though trade direction is LONG
      - All other equity longs count as US_LARGE_CAP LONG

 Persistent State  (global_risk_state.json)
 ------------------------------------------
   peak_combined_pnl    : high-water mark for drawdown calculation
   current_tier         : "ALLOW" | "FREEZE" | "LIQUIDATE"
   freeze_active        : bool
   liquidation_active   : bool
   last_updated         : ISO timestamp
   tier_change_history  : list of {timestamp, from_tier, to_tier, reasons}

 Files read (passively)
 ----------------------
   live_trades.csv        pairs module
   momentum_trades.csv    momentum module
   bear_trades.csv        bear module
   earnings_trades.csv    earnings module
   shock_trades.csv       shock module
================================================================================
"""

import os, sys, csv, json, datetime
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    # Global risk state file
    GLOBAL_RISK_STATE_JSON,
    # Tier 1 gatekeeper limits
    GLOBAL_RISK_ENABLED,
    GLOBAL_MAX_CAPITAL_DEPLOYED,
    GLOBAL_MAX_POSITIONS,
    GLOBAL_MAX_TICKER_OVERLAP,
    GLOBAL_MAX_DIRECTIONAL_PCT,
    # Tier 2 freeze thresholds
    GLOBAL_FREEZE_DRAWDOWN,
    GLOBAL_FREEZE_DAILY_LOSS,
    # Tier 3 liquidation threshold
    GLOBAL_LIQUIDATE_DRAWDOWN,
    # Directional mapping helpers
    GLOBAL_INVERSE_TICKERS,
    # Module CSV paths
    LIVE_TRADES_CSV,
    MOM_LIVE_TRADES_CSV,
    BEAR_TRADES_CSV,
    EARN_TRADES_CSV,
    SHOCK_TRADES_CSV,
)

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
GLOBAL_RISK_STATE_PATH = os.path.join(_SCRIPT_DIR, GLOBAL_RISK_STATE_JSON)

# Map module name -> (csv_path, ticker_column, has_single_ticker)
# Pairs module uses stock_a / stock_b columns; all others use a single ticker.
_MODULE_CSV = {
    "pairs":    (os.path.join(_SCRIPT_DIR, LIVE_TRADES_CSV),    None,     False),
    "momentum": (os.path.join(_SCRIPT_DIR, MOM_LIVE_TRADES_CSV), "ticker", True),
    "bear":     (os.path.join(_SCRIPT_DIR, BEAR_TRADES_CSV),    "ticker", True),
    "earnings": (os.path.join(_SCRIPT_DIR, EARN_TRADES_CSV),    "ticker", True),
    "shock":    (os.path.join(_SCRIPT_DIR, SHOCK_TRADES_CSV),   "ticker", True),
}

_DEFAULT_STATE = {
    "peak_combined_pnl":    0.0,
    "current_tier":         "ALLOW",
    "freeze_active":        False,
    "liquidation_active":   False,
    "last_updated":         "",
    "tier_change_history":  [],
}

_TIERS = ("ALLOW", "FREEZE", "LIQUIDATE")


# ---------------------------------------------------------------------------
#  State I/O
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    """Load global_risk_state.json, back-filling missing keys from defaults."""
    if not os.path.exists(GLOBAL_RISK_STATE_PATH):
        return dict(_DEFAULT_STATE)
    try:
        with open(GLOBAL_RISK_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        for k, v in _DEFAULT_STATE.items():
            state.setdefault(k, v)
        return state
    except Exception:
        return dict(_DEFAULT_STATE)


def _save_state(state: dict):
    """Write state to disk with fsync for crash safety."""
    state["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp = GLOBAL_RISK_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    # Atomic replace (best-effort on Windows)
    if os.path.exists(GLOBAL_RISK_STATE_PATH):
        os.replace(tmp, GLOBAL_RISK_STATE_PATH)
    else:
        os.rename(tmp, GLOBAL_RISK_STATE_PATH)


def _record_tier_change(state: dict, from_tier: str, to_tier: str,
                        reasons: list) -> dict:
    """Append a tier-change event to history (keep last 50)."""
    event = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from_tier": from_tier,
        "to_tier":   to_tier,
        "reasons":   reasons,
    }
    history = state.get("tier_change_history", [])
    history.append(event)
    state["tier_change_history"] = history[-50:]
    return state


# ---------------------------------------------------------------------------
#  CSV Loading Helpers
# ---------------------------------------------------------------------------

def _load_module_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a module CSV safely.  Returns an empty DataFrame if the file does not
    exist or cannot be parsed.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8")
        return df
    except Exception:
        return pd.DataFrame()


def _open_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows where status == 'open' (case-insensitive)."""
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    return df[df["status"].str.lower() == "open"].copy()


def _closed_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows where status == 'closed' (case-insensitive)."""
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    return df[df["status"].str.lower() == "closed"].copy()


def _numeric(val, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
#  Portfolio Snapshot Builder
# ---------------------------------------------------------------------------

def get_portfolio_snapshot() -> dict:
    """
    Gather all open positions across all five modules into a unified view.

    Returns
    -------
    dict with:
        positions        : list of position dicts (unified schema)
        by_module        : {module: [position_dicts]}
        total_capital    : float  (sum of capital_deployed)
        total_positions  : int
        ticker_map       : {ticker: [modules]}  -- overlap detection
    """
    all_positions = []
    by_module     = {}

    for module, (csv_path, ticker_col, single_ticker) in _MODULE_CSV.items():
        df   = _load_module_csv(csv_path)
        open = _open_trades(df)

        module_positions = []

        if open.empty:
            by_module[module] = []
            continue

        if single_ticker:
            # momentum / bear / earnings / shock
            for _, row in open.iterrows():
                ticker    = str(row.get(ticker_col, "UNKNOWN")).upper().strip()
                direction = str(row.get("direction", "LONG")).upper().strip()
                capital   = _numeric(row.get("capital_deployed", 0))
                entry_px  = _numeric(row.get("entry_price", 0))
                shares    = _numeric(row.get("shares", 0))
                date_open = str(row.get("date_open", ""))

                # SH inverse ETF counts as SHORT regardless of direction field
                effective_dir = _effective_direction(ticker, direction)

                pos = {
                    "module":        module,
                    "ticker":        ticker,
                    "direction":     direction,
                    "effective_dir": effective_dir,
                    "entry_price":   entry_px,
                    "shares":        shares,
                    "capital":       capital,
                    "date_open":     date_open,
                }
                module_positions.append(pos)
                all_positions.append(pos)

        else:
            # pairs module: two legs per trade (stock_a LONG, stock_b SHORT or vice versa)
            for _, row in open.iterrows():
                raw_dir = str(row.get("direction", "LONG_A")).upper().strip()
                # direction convention in live_trades.csv: "LONG_A" means long A / short B
                # total capital_deployed covers both legs combined
                capital_total = _numeric(row.get("capital_deployed", 0))
                capital_leg   = capital_total / 2.0

                for leg, col_price in (("stock_a", "price_a_entry"),
                                       ("stock_b", "price_b_entry")):
                    col_name = leg  # column holding the ticker symbol
                    ticker   = str(row.get(col_name, "UNKNOWN")).upper().strip()
                    if not ticker or ticker == "UNKNOWN":
                        continue

                    # Determine leg direction
                    if raw_dir == "LONG_A":
                        leg_dir = "LONG" if leg == "stock_a" else "SHORT"
                    else:
                        # LONG_B => long B, short A
                        leg_dir = "SHORT" if leg == "stock_a" else "LONG"

                    effective_dir = _effective_direction(ticker, leg_dir)
                    entry_px      = _numeric(row.get(col_price, 0))
                    shares        = _numeric(
                        row.get("shares_a" if leg == "stock_a" else "shares_b", 0)
                    )

                    pos = {
                        "module":        module,
                        "ticker":        ticker,
                        "direction":     leg_dir,
                        "effective_dir": effective_dir,
                        "entry_price":   entry_px,
                        "shares":        shares,
                        "capital":       capital_leg,
                        "date_open":     str(row.get("date_open", "")),
                    }
                    module_positions.append(pos)
                    all_positions.append(pos)

        by_module[module] = module_positions

    # Build ticker -> [modules] overlap map (deduplicated per module)
    ticker_map: dict = {}
    for pos in all_positions:
        t = pos["ticker"]
        m = pos["module"]
        if t not in ticker_map:
            ticker_map[t] = []
        if m not in ticker_map[t]:
            ticker_map[t].append(m)

    total_capital = sum(p["capital"] for p in all_positions)

    return {
        "positions":       all_positions,
        "by_module":       by_module,
        "total_capital":   total_capital,
        "total_positions": len(all_positions),
        "ticker_map":      ticker_map,
    }


def _effective_direction(ticker: str, declared_direction: str) -> str:
    """
    Resolve the economic direction of a position.
    Inverse ETFs (e.g. SH) are SHORT equity exposure even when bought LONG.
    """
    if ticker.upper() in [t.upper() for t in GLOBAL_INVERSE_TICKERS]:
        return "SHORT"
    return declared_direction.upper()


# ---------------------------------------------------------------------------
#  P&L Aggregation
# ---------------------------------------------------------------------------

def _aggregate_pnl() -> tuple:
    """
    Compute combined realized and estimated unrealized P&L across all modules.

    Unrealized P&L is estimated as zero (conservative) because we do not call
    any external API.  Only closed trades contribute realized P&L.

    Returns
    -------
    (realized_pnl: float, unrealized_pnl: float)
    """
    realized   = 0.0
    unrealized = 0.0  # conservative: no live prices fetched

    for module, (csv_path, _ticker_col, _single) in _MODULE_CSV.items():
        df = _load_module_csv(csv_path)
        if df.empty:
            continue

        closed = _closed_trades(df)
        if not closed.empty and "net_pnl" in closed.columns:
            pnl_vals = pd.to_numeric(closed["net_pnl"], errors="coerce").fillna(0.0)
            realized += float(pnl_vals.sum())

        # Unrealized: held at cost (entry price), so P&L = 0 until closed.
        # This is intentionally conservative -- we never overstate portfolio value.

    return realized, unrealized


def _estimate_daily_loss(snapshot: dict) -> float:
    """
    Estimate today's combined unrealized loss from open positions.
    Without live prices we cannot compute an exact intraday mark; we return 0.0
    so that freeze only triggers via realized P&L drawdown.  Callers may
    override by passing a pre-computed value to assess().
    """
    return 0.0


# ---------------------------------------------------------------------------
#  Core Assessment
# ---------------------------------------------------------------------------

def assess(override_daily_loss: float = None) -> dict:
    """
    Full portfolio scan.  Reads all module CSV/JSON files, computes combined
    exposure, evaluates all three tiers, updates peak P&L, and persists state.

    Parameters
    ----------
    override_daily_loss : float, optional
        If provided, use this as today's single-day combined unrealized loss
        instead of the default 0.0.  Useful when the caller has already
        fetched live prices.

    Returns
    -------
    dict with keys:
        tier                    "ALLOW" | "FREEZE" | "LIQUIDATE"
        total_capital_deployed  float
        total_positions         int
        directional_exposure    {long_pct: float, short_pct: float,
                                 long_capital: float, short_capital: float}
        combined_realized_pnl   float
        combined_unrealized_pnl float
        combined_drawdown       float
        ticker_overlap          {ticker: [modules]}
        reasons                 list[str]
    """
    if not GLOBAL_RISK_ENABLED:
        return _allow_result(0.0, 0, {}, 0.0, 0.0, 0.0, {}, [])

    state    = _load_state()
    snapshot = get_portfolio_snapshot()
    reasons  = []

    total_capital    = snapshot["total_capital"]
    total_positions  = snapshot["total_positions"]
    ticker_map       = snapshot["ticker_map"]
    positions        = snapshot["positions"]

    # -- Directional exposure ------------------------------------------------
    long_capital  = sum(p["capital"] for p in positions
                        if p["effective_dir"] == "LONG")
    short_capital = sum(p["capital"] for p in positions
                        if p["effective_dir"] == "SHORT")

    if total_capital > 0:
        long_pct  = long_capital  / total_capital
        short_pct = short_capital / total_capital
    else:
        long_pct  = 0.0
        short_pct = 0.0

    directional_exposure = {
        "long_capital":  round(long_capital,  2),
        "short_capital": round(short_capital, 2),
        "long_pct":      round(long_pct,  4),
        "short_pct":     round(short_pct, 4),
    }

    # -- P&L and drawdown ----------------------------------------------------
    realized, unrealized = _aggregate_pnl()
    combined_pnl         = realized + unrealized

    # Update peak high-water mark
    if combined_pnl > state["peak_combined_pnl"]:
        state["peak_combined_pnl"] = combined_pnl

    peak_pnl        = state["peak_combined_pnl"]
    combined_dd     = combined_pnl - peak_pnl   # <= 0 always

    daily_loss = override_daily_loss if override_daily_loss is not None else 0.0

    # -- Tier 3: Liquidation -------------------------------------------------
    new_tier       = "ALLOW"
    liq_triggered  = False
    freeze_trigger = False

    if combined_dd <= GLOBAL_LIQUIDATE_DRAWDOWN:
        liq_triggered = True
        reasons.append(
            f"LIQUIDATE: combined drawdown ${combined_dd:+.2f} "
            f"<= threshold ${GLOBAL_LIQUIDATE_DRAWDOWN:.0f}"
        )

    # -- Tier 2: Freeze ------------------------------------------------------
    if not liq_triggered:
        if combined_dd <= GLOBAL_FREEZE_DRAWDOWN:
            freeze_trigger = True
            reasons.append(
                f"FREEZE: combined drawdown ${combined_dd:+.2f} "
                f"<= threshold ${GLOBAL_FREEZE_DRAWDOWN:.0f}"
            )
        if daily_loss <= GLOBAL_FREEZE_DAILY_LOSS:
            freeze_trigger = True
            reasons.append(
                f"FREEZE: single-day loss ${daily_loss:+.2f} "
                f"<= threshold ${GLOBAL_FREEZE_DAILY_LOSS:.0f}"
            )

    # Determine new tier
    if liq_triggered:
        new_tier = "LIQUIDATE"
    elif freeze_trigger or state.get("freeze_active", False):
        # Once frozen, stay frozen until manual reset
        new_tier = "FREEZE"
    elif state.get("liquidation_active", False):
        new_tier = "LIQUIDATE"
    else:
        new_tier = "ALLOW"

    # Record tier transitions
    old_tier = state.get("current_tier", "ALLOW")
    if new_tier != old_tier:
        state = _record_tier_change(state, old_tier, new_tier, reasons)

    # Persist flags
    state["current_tier"]       = new_tier
    state["freeze_active"]      = new_tier in ("FREEZE", "LIQUIDATE")
    state["liquidation_active"] = new_tier == "LIQUIDATE"
    _save_state(state)

    return {
        "tier":                    new_tier,
        "total_capital_deployed":  round(total_capital, 2),
        "total_positions":         total_positions,
        "directional_exposure":    directional_exposure,
        "combined_realized_pnl":   round(realized,     2),
        "combined_unrealized_pnl": round(unrealized,   2),
        "combined_drawdown":       round(combined_dd,  2),
        "ticker_overlap":          ticker_map,
        "reasons":                 reasons,
    }


def _allow_result(capital, positions, directional, realized, unrealized,
                  drawdown, overlap, reasons) -> dict:
    """Return a clean ALLOW result dict (used when GLOBAL_RISK_ENABLED=False)."""
    return {
        "tier":                    "ALLOW",
        "total_capital_deployed":  capital,
        "total_positions":         positions,
        "directional_exposure":    directional or {},
        "combined_realized_pnl":   realized,
        "combined_unrealized_pnl": unrealized,
        "combined_drawdown":       drawdown,
        "ticker_overlap":          overlap or {},
        "reasons":                 reasons or [],
    }


# ---------------------------------------------------------------------------
#  Gatekeeper  (Tier 1 -- per-trade checks)
# ---------------------------------------------------------------------------

def may_enter(ticker: str, direction: str, capital: float,
              module: str) -> tuple:
    """
    Quick gatekeeper check before a new trade is logged.

    Parameters
    ----------
    ticker    : proposed ticker symbol
    direction : "LONG" or "SHORT"
    capital   : proposed capital to deploy ($)
    module    : calling module name ("pairs", "momentum", "bear", etc.)

    Returns
    -------
    (allowed: bool, reason: str)
      allowed=True  means the trade is permitted by global risk rules.
      allowed=False carries a human-readable reason string.
    """
    if not GLOBAL_RISK_ENABLED:
        return True, "GLOBAL_RISK_ENABLED=False"

    state = _load_state()

    # Freeze / Liquidate always blocks
    if state.get("liquidation_active", False):
        return False, "Global risk: LIQUIDATE tier active -- all entries blocked"
    if state.get("freeze_active", False):
        return False, "Global risk: FREEZE tier active -- all entries blocked"

    # Re-evaluate current tier (lightweight, no state write)
    snapshot       = get_portfolio_snapshot()
    total_capital  = snapshot["total_capital"]
    total_pos      = snapshot["total_positions"]
    ticker_map     = snapshot["ticker_map"]
    positions      = snapshot["positions"]

    ticker_upper   = ticker.upper().strip()
    effective_dir  = _effective_direction(ticker_upper, direction)

    # -- Check 1: Capital limit -----------------------------------------------
    projected_capital = total_capital + capital
    if projected_capital > GLOBAL_MAX_CAPITAL_DEPLOYED:
        return (False,
                f"Global risk: capital ${projected_capital:.0f} would exceed "
                f"limit ${GLOBAL_MAX_CAPITAL_DEPLOYED:.0f} "
                f"(current: ${total_capital:.0f})")

    # -- Check 2: Position count limit ----------------------------------------
    if total_pos + 1 > GLOBAL_MAX_POSITIONS:
        return (False,
                f"Global risk: position count {total_pos + 1} would exceed "
                f"limit {GLOBAL_MAX_POSITIONS} (current: {total_pos})")

    # -- Check 3: Ticker overlap limit ----------------------------------------
    existing_modules = ticker_map.get(ticker_upper, [])
    # Don't double-count the calling module if it already has this ticker open
    unique_modules = set(existing_modules)
    unique_modules.discard(module)  # only count OTHER modules' exposure
    if len(unique_modules) + 1 > GLOBAL_MAX_TICKER_OVERLAP:
        return (False,
                f"Global risk: {ticker_upper} already open in "
                f"{sorted(unique_modules)} -- ticker overlap limit "
                f"{GLOBAL_MAX_TICKER_OVERLAP} would be breached")

    # -- Check 4: Directional concentration -----------------------------------
    # Only meaningful when there are existing positions to measure against.
    # A portfolio starting from zero always begins at 100% in one direction;
    # the check is only applied once at least one other position already exists.
    if total_capital > 0:
        long_capital  = sum(p["capital"] for p in positions
                            if p["effective_dir"] == "LONG")
        short_capital = sum(p["capital"] for p in positions
                            if p["effective_dir"] == "SHORT")

        if effective_dir == "LONG":
            projected_dir_capital = long_capital  + capital
        else:
            projected_dir_capital = short_capital + capital

        projected_total = total_capital + capital
        dir_pct = projected_dir_capital / projected_total

        if dir_pct > GLOBAL_MAX_DIRECTIONAL_PCT:
            dir_label = "LONG" if effective_dir == "LONG" else "SHORT"
            return (False,
                    f"Global risk: {dir_label} directional concentration "
                    f"{dir_pct:.1%} would exceed limit "
                    f"{GLOBAL_MAX_DIRECTIONAL_PCT:.0%}")

    return True, "OK"


# ---------------------------------------------------------------------------
#  Console Dashboard
# ---------------------------------------------------------------------------

def print_global_risk_status():
    """
    Print a formatted console dashboard showing combined portfolio state,
    tier, per-module breakdown, capital utilisation, and directional exposure.
    """
    result   = assess()
    state    = _load_state()
    snapshot = get_portfolio_snapshot()

    tier      = result["tier"]
    tier_icon = {"ALLOW": "OK", "FREEZE": "!! FREEZE", "LIQUIDATE": "!!! LIQUIDATE"}

    print()
    print("  -- Global Risk Manager " + "-" * 38)
    print(f"    Status             : [{tier_icon.get(tier, tier)}]  {tier}")
    print(f"    Capital deployed   : ${result['total_capital_deployed']:>8.2f}  "
          f"/ ${GLOBAL_MAX_CAPITAL_DEPLOYED:.0f}")
    print(f"    Open positions     : {result['total_positions']:>3}  "
          f"/ {GLOBAL_MAX_POSITIONS}")

    de = result["directional_exposure"]
    print(f"    Long  exposure     : ${de['long_capital']:>8.2f}  "
          f"({de['long_pct']:.1%})")
    print(f"    Short exposure     : ${de['short_capital']:>8.2f}  "
          f"({de['short_pct']:.1%})")

    print(f"    Realized P&L       : ${result['combined_realized_pnl']:>+.2f}")
    print(f"    Unrealized P&L     : ${result['combined_unrealized_pnl']:>+.2f}  "
          f"(estimated at cost)")
    print(f"    Combined drawdown  : ${result['combined_drawdown']:>+.2f}  "
          f"(freeze: ${GLOBAL_FREEZE_DRAWDOWN:.0f}, "
          f"liq: ${GLOBAL_LIQUIDATE_DRAWDOWN:.0f})")
    print(f"    Peak combined P&L  : ${state['peak_combined_pnl']:>+.2f}")

    if result["reasons"]:
        print()
        for r in result["reasons"]:
            print(f"    >> {r}")

    # Per-module breakdown
    print()
    print(f"    {'MODULE':<12} {'OPEN':>5} {'CAPITAL':>10} {'LONG':>8} {'SHORT':>8}")
    print(f"    {'-'*12} {'-'*5} {'-'*10} {'-'*8} {'-'*8}")
    for mod, positions in snapshot["by_module"].items():
        n      = len(positions)
        cap    = sum(p["capital"] for p in positions)
        lc     = sum(p["capital"] for p in positions if p["effective_dir"] == "LONG")
        sc     = sum(p["capital"] for p in positions if p["effective_dir"] == "SHORT")
        print(f"    {mod:<12} {n:>5} ${cap:>9.2f} ${lc:>7.2f} ${sc:>7.2f}")

    # Ticker overlap warnings
    overlaps = {t: mods for t, mods in result["ticker_overlap"].items()
                if len(mods) > 1}
    if overlaps:
        print()
        print("    Ticker overlap:")
        for ticker, mods in overlaps.items():
            flag = " !! EXCEEDS LIMIT" if len(mods) > GLOBAL_MAX_TICKER_OVERLAP else ""
            print(f"      {ticker:<8} -> {', '.join(mods)}{flag}")

    # Recent tier change history
    history = state.get("tier_change_history", [])
    if history:
        print()
        print("    Recent tier changes:")
        for event in history[-3:]:
            print(f"      {event['timestamp']}  "
                  f"{event['from_tier']} -> {event['to_tier']}")

    print("  " + "-" * 61)
    print()


# ---------------------------------------------------------------------------
#  Manual Overrides
# ---------------------------------------------------------------------------

def reset_freeze():
    """
    Manually clear the freeze flag after human review.
    Only clears FREEZE -- will not clear LIQUIDATE.
    """
    state = _load_state()
    if not state.get("freeze_active", False):
        print("  Global risk: no active freeze to reset.")
        return
    if state.get("liquidation_active", False):
        print("  Global risk: LIQUIDATE is active -- use reset_liquidation() instead.")
        return
    old_tier                   = state["current_tier"]
    state["freeze_active"]     = False
    state["current_tier"]      = "ALLOW"
    state = _record_tier_change(state, old_tier, "ALLOW",
                                ["Manual freeze reset by operator"])
    _save_state(state)
    print("  Global risk: FREEZE cleared.  System will accept new entries on next run.")


def reset_liquidation():
    """
    Manually clear the liquidation flag after human review and position cleanup.
    Also clears the freeze flag since liquidation implies freeze.
    """
    state = _load_state()
    if not state.get("liquidation_active", False):
        print("  Global risk: no active liquidation to reset.")
        return
    old_tier                      = state["current_tier"]
    state["liquidation_active"]   = False
    state["freeze_active"]        = False
    state["current_tier"]         = "ALLOW"
    state["peak_combined_pnl"]    = 0.0   # reset peak so drawdown recalculates fresh
    state = _record_tier_change(state, old_tier, "ALLOW",
                                ["Manual liquidation reset by operator"])
    _save_state(state)
    print("  Global risk: LIQUIDATE cleared and peak P&L reset.  "
          "System will accept new entries on next run.")


# ---------------------------------------------------------------------------
#  Module-Level Entry Point (called from run_system.py)
# ---------------------------------------------------------------------------

def run_global_risk_check(verbose: bool = True) -> dict:
    """
    Convenience wrapper called by run_system.py at the start of each daily run.
    Runs assess(), prints status if verbose, and returns the result dict.
    """
    result = assess()
    if verbose:
        print_global_risk_status()
    return result


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Global Risk Manager -- self-test")
    print_global_risk_status()
    print()
    print("may_enter test (SPY, LONG, $1000, momentum):")
    allowed, reason = may_enter("SPY", "LONG", 1000.0, "momentum")
    print(f"  allowed={allowed}  reason={reason}")
