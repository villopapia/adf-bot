"""
================================================================================
 BEAR TRACKER  -  Paper Trade Journal & Portfolio Risk Manager (Bear Module)
================================================================================
 Manages bear-market paper trades independently from the pairs and momentum
 systems.  On each daily run (called by run_system.py):

   Phase B0a -- check_bear_exits()
       For every open bear position, fetches current SPY/SH price data and
       closes the trade if any exit condition is met.

       Bounce exits (module="bounce"):
           sma_recovery  : SPY closes above 5-day SMA (prior bar)
           ibs_strength  : current-bar IBS > 0.80
           time_stop     : hold_days >= BOUNCE_MAX_HOLD
           (no trailing stop -- Connors research shows stops damage MR returns)

       Short exits (module="short"):
           spy_above_20sma : SPY closes above 20-day SMA (trend reversal)
           time_stop       : hold_days >= SHORT_MAX_HOLD (inverse ETF decay)

       Updates bear_state.json with realized P&L, consecutive losses, and
       evaluates the kill switch.

   Phase B0b -- log_bear_entry(signal, module, vix_scale, capitulation)
       Records a new bear signal as an open paper trade in bear_trades.csv
       using VIX-tiered position sizing with optional capitulation boost.

 Portfolio Risk Controls
 -----------------------
   BEAR_MAX_POSITIONS       : hard cap on simultaneous open trades
   BEAR_KILL_SWITCH_LOSSES  : kill switch fires after N live losses in a row
   BEAR_KILL_SWITCH_DD      : kill switch fires when live P&L drops this far
                              from its peak (e.g. -$800)

 Files produced / updated
 ------------------------
   bear_trades.csv   -- every bear paper trade (open + closed)
   bear_state.json   -- running P&L, consecutive losses, kill switch flag
================================================================================
"""

import os, sys, csv, json, datetime
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    BEAR_TRADES_CSV, BEAR_STATE_JSON,
    BEAR_MAX_POSITIONS, BEAR_KILL_SWITCH_LOSSES, BEAR_KILL_SWITCH_DD,
    BOUNCE_INSTRUMENT, BOUNCE_CAPITAL_BASE, BOUNCE_SLIPPAGE_PCT,
    BOUNCE_EXIT_SMA, BOUNCE_IBS_EXIT, BOUNCE_MAX_HOLD,
    BOUNCE_VIX_TIERS, BOUNCE_RSI_PERIOD,
    SHORT_INSTRUMENT, SHORT_CAPITAL_SCALE, SHORT_SLIPPAGE_PCT,
    SHORT_EXIT_SMA, SHORT_MAX_HOLD,
    CAPIT_BOOST_SCALE,
)

BEAR_TRADES_PATH = os.path.join(_SCRIPT_DIR, BEAR_TRADES_CSV)
BEAR_STATE_PATH  = os.path.join(_SCRIPT_DIR, BEAR_STATE_JSON)

BEAR_TRADE_HEADERS = [
    "trade_id", "date_open", "ticker", "module",
    "direction", "entry_price", "shares", "capital_deployed",
    "vix_at_entry", "vix_scale", "capitulation_boost",
    "bt_win_rate", "bt_profit_factor", "bt_total_pnl",
    "status",
    "date_close", "exit_price", "exit_reason",
    "hold_days", "gross_pnl", "net_pnl",
]

_DEFAULT_BEAR_STATE = {
    "consecutive_losses":  0,
    "consecutive_wins":    0,
    "total_realized_pnl":  0.0,
    "peak_pnl":            0.0,
    "total_trades":        0,
    "total_wins":          0,
    "kill_switch":         False,
    "kill_switch_reason":  "",
    "last_updated":        "",
}


# ------------------------------------------------------------------------------
#  CSV / JSON I/O
# ------------------------------------------------------------------------------

def _init_bear_trades_csv() -> bool:
    """Create bear_trades.csv with headers if missing. Returns True if new."""
    if not os.path.exists(BEAR_TRADES_PATH):
        with open(BEAR_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(BEAR_TRADE_HEADERS)
        return True
    return False


def load_bear_trades() -> pd.DataFrame:
    _init_bear_trades_csv()
    df = pd.read_csv(BEAR_TRADES_PATH, dtype=str, encoding="utf-8")
    return df


def _save_bear_trades(df: pd.DataFrame):
    """Rewrite bear_trades.csv from DataFrame with flush+fsync for crash safety."""
    tmp_path = BEAR_TRADES_PATH + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    with open(tmp_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(BEAR_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.remove(tmp_path)


def load_bear_state() -> dict:
    if not os.path.exists(BEAR_STATE_PATH):
        return dict(_DEFAULT_BEAR_STATE)
    try:
        with open(BEAR_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        for k, v in _DEFAULT_BEAR_STATE.items():
            state.setdefault(k, v)
        return state
    except Exception:
        return dict(_DEFAULT_BEAR_STATE)


def save_bear_state(state: dict):
    with open(BEAR_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------------------
#  Portfolio Queries
# ------------------------------------------------------------------------------

def get_open_bear_trades() -> list:
    """Return list of dicts for all open bear positions."""
    df = load_bear_trades()
    if df.empty:
        return []
    return df[df["status"] == "open"].to_dict("records")


def bear_open_count() -> int:
    return len(get_open_bear_trades())


def at_bear_cap() -> bool:
    """True if bear portfolio is at or above the position cap."""
    return bear_open_count() >= BEAR_MAX_POSITIONS


# ------------------------------------------------------------------------------
#  Kill Switch
# ------------------------------------------------------------------------------

def is_bear_kill_switch() -> tuple:
    """Check if the bear kill switch is active. Returns (bool, reason_str)."""
    state = load_bear_state()
    return state.get("kill_switch", False), state.get("kill_switch_reason", "")


def reset_bear_kill_switch():
    """Manually reset the bear kill switch after review."""
    state = load_bear_state()
    state["kill_switch"]        = False
    state["kill_switch_reason"] = ""
    state["consecutive_losses"] = 0
    save_bear_state(state)
    print("  Bear kill switch reset. System will accept new entries on next run.")


def _evaluate_bear_kill_switch(state: dict) -> dict:
    """Evaluate and set kill switch if thresholds are breached. No-op if already set."""
    if state["kill_switch"]:
        return state  # already triggered -- manual reset required
    reason = ""
    if state["consecutive_losses"] >= BEAR_KILL_SWITCH_LOSSES:
        reason = (f"{state['consecutive_losses']} consecutive losses "
                  f"(threshold: {BEAR_KILL_SWITCH_LOSSES})")
    drawdown = state["total_realized_pnl"] - state["peak_pnl"]
    if drawdown <= BEAR_KILL_SWITCH_DD:
        reason = (f"Bear drawdown ${drawdown:+.2f} "
                  f"(threshold: ${BEAR_KILL_SWITCH_DD:.0f})")
    if reason:
        state["kill_switch"]        = True
        state["kill_switch_reason"] = reason
    return state


# ------------------------------------------------------------------------------
#  Entry Logging
# ------------------------------------------------------------------------------

def log_bear_entry(signal: dict, module: str, vix_scale: float = 1.0,
                   capitulation: bool = False) -> str:
    """
    Record a new bear signal as an open paper trade.

    signal dict expected keys:
        ticker             : str
        live.price         : float   (SPY price for bounce, SH price for short)
        live.direction     : str     (always "LONG")
        bt.win_rate        : float
        bt.profit_factor   : float
        bt.total_pnl       : float

    module: "bounce" or "short"
    vix_scale: VIX-tier scaling factor (e.g. 1.0, 0.75, 0.50, 0.25)
    capitulation: True if capitulation boost applies (adds 50% to bounce size)

    Returns the trade_id assigned.
    """
    _init_bear_trades_csv()

    ticker = signal["ticker"]
    live   = signal["live"]
    bt     = signal["bt"]
    price  = float(live["price"])

    # Position sizing
    if module == "bounce":
        capital = BOUNCE_CAPITAL_BASE * vix_scale
        if capitulation:
            capital *= CAPIT_BOOST_SCALE
        slippage_pct = BOUNCE_SLIPPAGE_PCT
    else:  # "short"
        capital = BOUNCE_CAPITAL_BASE * SHORT_CAPITAL_SCALE * vix_scale
        slippage_pct = SHORT_SLIPPAGE_PCT

    shares = capital / price

    today    = datetime.date.today().strftime("%Y-%m-%d")
    trade_id = f"{today.replace('-', '')}_{module}_{ticker}"

    row = {
        "trade_id":          trade_id,
        "date_open":         today,
        "ticker":            ticker,
        "module":            module,
        "direction":         str(live.get("direction", "LONG")),
        "entry_price":       f"{price:.4f}",
        "shares":            f"{shares:.4f}",
        "capital_deployed":  f"{capital:.2f}",
        "vix_at_entry":      f"{vix_scale:.4f}",
        "vix_scale":         f"{vix_scale:.4f}",
        "capitulation_boost": str(capitulation),
        "bt_win_rate":       f"{float(bt['win_rate']):.1f}",
        "bt_profit_factor":  f"{float(bt['profit_factor']):.2f}",
        "bt_total_pnl":      f"{float(bt['total_pnl']):.2f}",
        "status":            "open",
        "date_close":        "",
        "exit_price":        "",
        "exit_reason":       "",
        "hold_days":         "",
        "gross_pnl":         "",
        "net_pnl":           "",
    }

    with open(BEAR_TRADES_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BEAR_TRADE_HEADERS)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    return trade_id


# ------------------------------------------------------------------------------
#  Exit Checking
# ------------------------------------------------------------------------------

def check_bear_exits() -> list:
    """
    For every open bear position, fetch current price data and close the trade
    if any exit condition is met.

    Bounce exits:
        sma_recovery  : SPY closes above prior-bar 5-day SMA
        ibs_strength  : current-bar IBS > 0.80
        time_stop     : hold_days >= BOUNCE_MAX_HOLD

    Short exits:
        spy_above_20sma : SPY closes above prior-bar 20-day SMA
        time_stop       : hold_days >= SHORT_MAX_HOLD

    Returns list of trade dicts that were closed this run.
    """
    import yfinance as yf

    _init_bear_trades_csv()
    trades_df = load_bear_trades()
    open_mask = trades_df["status"] == "open"

    if not open_mask.any():
        return []

    closed_today = []
    state        = load_bear_state()

    # Pre-fetch SPY data (needed for both bounce and short exits)
    try:
        spy_raw = yf.download("SPY", period="30d", auto_adjust=True, progress=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_high  = spy_raw[("High",  "SPY")]
            spy_low   = spy_raw[("Low",   "SPY")]
            spy_close = spy_raw[("Close", "SPY")]
        else:
            spy_high  = spy_raw["High"]
            spy_low   = spy_raw["Low"]
            spy_close = spy_raw["Close"]
    except Exception as e:
        print(f"    [BEAR-EXIT] Failed to fetch SPY data: {e}")
        return []

    for idx in trades_df[open_mask].index:
        row         = trades_df.loc[idx]
        ticker      = row["ticker"]
        module      = row["module"]
        entry_price = float(row["entry_price"])
        shares      = float(row["shares"])
        date_open   = str(row["date_open"])

        open_date = datetime.datetime.strptime(date_open, "%Y-%m-%d").date()
        hold_days = (datetime.date.today() - open_date).days

        try:
            exit_reason   = ""
            current_price = None

            if module == "bounce":
                # Bounce exits: SPY-based (we track SPY as the instrument)
                current_price = float(spy_close.iloc[-1])

                # Exit 1: SPY closes above prior-bar 5-day SMA
                sma_5 = spy_close.rolling(BOUNCE_EXIT_SMA).mean()
                if len(sma_5.dropna()) >= 2 and current_price > float(sma_5.iloc[-2]):
                    exit_reason = "sma_recovery"

                # Exit 2: current-bar IBS > 0.80
                if not exit_reason:
                    range_hl = float(spy_high.iloc[-1]) - float(spy_low.iloc[-1])
                    if range_hl > 0:
                        current_ibs = (float(spy_close.iloc[-1]) - float(spy_low.iloc[-1])) / range_hl
                    else:
                        current_ibs = 0.5
                    if current_ibs > BOUNCE_IBS_EXIT:
                        exit_reason = "ibs_strength"

                # Exit 3: time stop
                if not exit_reason and hold_days >= BOUNCE_MAX_HOLD:
                    exit_reason = "time_stop"

                # NOTE: no trailing stop, no price-based stop (intentional).
                # Connors research shows stops damage mean-reversion performance.

                slippage_pct = BOUNCE_SLIPPAGE_PCT

            elif module == "short":
                # Short exits: we're holding SH; exit driven by SPY regime
                try:
                    sh_raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
                    if isinstance(sh_raw.columns, pd.MultiIndex):
                        sh_close = sh_raw[("Close", ticker)]
                    else:
                        sh_close = sh_raw["Close"]
                    current_price = float(sh_close.iloc[-1])
                except Exception:
                    continue

                # Exit 1: SPY closes above prior-bar 20-day SMA (trend reversal)
                sma_20 = spy_close.rolling(SHORT_EXIT_SMA).mean()
                if len(sma_20.dropna()) >= 2:
                    spy_current = float(spy_close.iloc[-1])
                    if spy_current > float(sma_20.iloc[-2]):
                        exit_reason = "spy_above_20sma"

                # Exit 2: time stop (avoid inverse ETF decay)
                if not exit_reason and hold_days >= SHORT_MAX_HOLD:
                    exit_reason = "time_stop"

                slippage_pct = SHORT_SLIPPAGE_PCT

            else:
                continue  # unknown module -- skip

            if not exit_reason or current_price is None:
                continue

            # Close the trade
            gross    = shares * (current_price - entry_price)
            slippage = slippage_pct * shares * (entry_price + current_price)
            net      = gross - slippage

            trades_df.at[idx, "status"]     = "closed"
            trades_df.at[idx, "date_close"] = datetime.date.today().strftime("%Y-%m-%d")
            trades_df.at[idx, "exit_price"] = f"{current_price:.4f}"
            trades_df.at[idx, "exit_reason"] = exit_reason
            trades_df.at[idx, "hold_days"]  = str(hold_days)
            trades_df.at[idx, "gross_pnl"]  = f"{gross:.2f}"
            trades_df.at[idx, "net_pnl"]    = f"{net:.2f}"

            # Update running state
            state["total_trades"]       += 1
            state["total_realized_pnl"] += net

            if net > 0:
                state["total_wins"]        += 1
                state["consecutive_wins"]  += 1
                state["consecutive_losses"] = 0
            else:
                state["consecutive_losses"] += 1
                state["consecutive_wins"]    = 0

            if state["total_realized_pnl"] > state["peak_pnl"]:
                state["peak_pnl"] = state["total_realized_pnl"]

            state = _evaluate_bear_kill_switch(state)
            closed_today.append(trades_df.loc[idx].to_dict())

        except Exception as e:
            print(f"    [BEAR-EXIT] Error checking {ticker} ({module}): {e}")
            continue

    # Always persist state (even when no exits occurred)
    state["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_bear_trades(trades_df)
    save_bear_state(state)

    return closed_today


# ------------------------------------------------------------------------------
#  Console Display
# ------------------------------------------------------------------------------

def print_bear_portfolio_status():
    """Print a snapshot of current open bear positions and running P&L."""
    open_trades = get_open_bear_trades()
    state       = load_bear_state()
    today       = datetime.date.today()

    print("  -- Bear Module Portfolio " + "-" * 36)
    print(f"    Open positions : {len(open_trades)} / {BEAR_MAX_POSITIONS}")
    print(f"    Realized P&L   : ${state['total_realized_pnl']:+.2f}")
    print(f"    Total trades   : {state['total_trades']}")
    if state["total_trades"] > 0:
        wr = state["total_wins"] / state["total_trades"] * 100
        print(f"    Live win rate  : {wr:.1f}%")
    print(f"    Consec. losses : {state['consecutive_losses']}")

    if state.get("kill_switch"):
        print(f"\n    !! KILL SWITCH ACTIVE: {state['kill_switch_reason']}")

    if open_trades:
        print(f"\n    {'TICKER':<8} {'MODULE':<8} {'DIR':>5} {'ENTRY':>8} "
              f"{'CAPITAL':>8} {'CAPIT':>6} {'DAYS':>5}")
        print(f"    {'-'*8} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*5}")
        for t in open_trades:
            open_dt = datetime.datetime.strptime(t["date_open"], "%Y-%m-%d").date()
            days    = (today - open_dt).days
            capit   = "Y" if str(t.get("capitulation_boost", "False")).lower() == "true" else "N"
            print(f"    {t['ticker']:<8} {t['module']:<8} {'LONG':>5} "
                  f"${float(t['entry_price']):>7.2f} "
                  f"${float(t['capital_deployed']):>7.2f} "
                  f"{'YES' if capit == 'Y' else 'no':>6} "
                  f"{days:>5}")
    print("  " + "-" * 61)
    print()


def print_bear_performance_report():
    """
    Print a performance summary of all closed bear trades.
    Shows total P&L, win rate, average hold, best/worst trade,
    and a breakdown by module (bounce vs short).
    Only meaningful when there are at least 3 closed trades.
    """
    df = load_bear_trades()
    if df.empty:
        return

    closed   = df[df["status"] == "closed"].copy()
    n_closed = len(closed)
    if n_closed < 3:
        return

    closed["net_pnl"]          = pd.to_numeric(closed["net_pnl"],          errors="coerce")
    closed["gross_pnl"]        = pd.to_numeric(closed["gross_pnl"],         errors="coerce")
    closed["hold_days"]        = pd.to_numeric(closed["hold_days"],          errors="coerce")
    closed["bt_win_rate"]      = pd.to_numeric(closed["bt_win_rate"],        errors="coerce")
    closed["bt_profit_factor"] = pd.to_numeric(closed["bt_profit_factor"],   errors="coerce")

    pnls = closed["net_pnl"].dropna().values
    if len(pnls) < 3:
        return

    wins        = pnls[pnls > 0]
    losses      = pnls[pnls <= 0]
    live_wr     = len(wins) / len(pnls) * 100
    live_avg    = pnls.mean()
    total_pnl   = pnls.sum()
    avg_hold    = closed["hold_days"].dropna().mean()
    bt_wr_avg   = closed["bt_win_rate"].dropna().mean()
    best_trade  = float(pnls.max())
    worst_trade = float(pnls.min())

    avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    profit_factor = (wins.sum() / abs(losses.sum())
                     if len(losses) > 0 and losses.sum() != 0 else float("inf"))

    # Live Sharpe (rough annualisation)
    live_sharpe_str = "N/A"
    if len(pnls) >= 5:
        std_pnl       = pnls.std(ddof=1) if len(pnls) > 1 else 1e-12
        avg_hold_safe = max(avg_hold, 1.0) if not np.isnan(avg_hold) else 5.0
        live_sharpe   = (live_avg / std_pnl) * np.sqrt(252 / avg_hold_safe)
        live_sharpe_str = f"{live_sharpe:+.2f}"

    # Maximum consecutive losses in closed history
    max_consec  = 0
    curr_streak = 0
    for p in pnls:
        if p <= 0:
            curr_streak += 1
            max_consec   = max(max_consec, curr_streak)
        else:
            curr_streak = 0

    # System drawdown on closed trades
    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    max_dd  = float((cum_pnl - peak).min())

    # Module breakdown
    bounce_closed = closed[closed["module"] == "bounce"]
    short_closed  = closed[closed["module"] == "short"]
    n_bounce      = len(bounce_closed)
    n_short       = len(short_closed)

    b_pnls = bounce_closed["net_pnl"].dropna().values
    s_pnls = short_closed["net_pnl"].dropna().values

    b_wr  = (len(b_pnls[b_pnls > 0]) / len(b_pnls) * 100) if len(b_pnls) > 0 else 0.0
    s_wr  = (len(s_pnls[s_pnls > 0]) / len(s_pnls) * 100) if len(s_pnls) > 0 else 0.0
    b_pnl = float(b_pnls.sum()) if len(b_pnls) > 0 else 0.0
    s_pnl = float(s_pnls.sum()) if len(s_pnls) > 0 else 0.0

    # Edge verdict vs backtest
    if bt_wr_avg > 0 and live_wr >= bt_wr_avg * 0.8:
        verdict = "Edge holding"
    else:
        verdict = "Edge degrading -- review"

    print()
    print("  -- Bear Performance Report " + "-" * 34)
    print(f"    Closed trades      : {n_closed}")
    print(f"    Total net P&L      : ${total_pnl:>+.2f}")
    print(f"    Live win rate      : {live_wr:.1f}%")
    print(f"    BT avg win rate    : {bt_wr_avg:.1f}%")
    print(f"    Live avg P&L       : ${live_avg:>+.2f}")
    print(f"    Avg win            : ${avg_win:>+.2f}")
    print(f"    Avg loss           : ${avg_loss:>+.2f}")
    print(f"    Profit factor      : {profit_factor:.2f}")
    print(f"    Avg hold (days)    : {avg_hold:.1f}")
    print(f"    Live Sharpe        : {live_sharpe_str}")
    print(f"    Best trade         : ${best_trade:>+.2f}")
    print(f"    Worst trade        : ${worst_trade:>+.2f}")
    print(f"    Max consec. losses : {max_consec}")
    print(f"    Max drawdown       : ${max_dd:>+.2f}")
    print()
    print(f"    Module breakdown:")
    print(f"      Bounce : {n_bounce:>3} trades  win={b_wr:.0f}%  P&L=${b_pnl:>+.2f}")
    print(f"      Short  : {n_short:>3} trades  win={s_wr:.0f}%  P&L=${s_pnl:>+.2f}")
    print(f"    Verdict            : {verdict}")
    print("  " + "-" * 61)
    print()
