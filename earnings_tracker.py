"""
================================================================================
 EARNINGS TRACKER  -  Paper Trade Journal & Portfolio Risk Manager
================================================================================
 Manages pre-earnings paper trades independently from the pairs, momentum,
 and bear systems.  On each daily run (called by run_system.py):

   Phase E0 -- check_earn_exits()
       For every open earnings position, fetches current price and closes the
       trade if any exit condition is met:

           post_earnings  : today > earnings_date (capture the gap)
           stop_loss      : current_price < entry_price * (1 - EARN_STOP_PCT/100)
           time_stop      : hold_days >= EARN_MAX_HOLD (safety net)

       Updates earnings_state.json with realized P&L, consecutive losses, and
       evaluates the kill switch.

   Phase E1 -- log_earn_entry(signal)
       Records a new earnings signal as an open paper trade in
       earnings_trades.csv using fixed EARN_CAPITAL_PER_TRADE sizing.

 Portfolio Risk Controls
 -----------------------
   EARN_MAX_POSITIONS       : hard cap on simultaneous open trades
   EARN_KILL_SWITCH_LOSSES  : kill switch fires after N live losses in a row
   EARN_KILL_SWITCH_DD      : kill switch fires when live P&L drops this far
                              from its peak (e.g. -$500)

 Files produced / updated
 ------------------------
   earnings_trades.csv   -- every earnings paper trade (open + closed)
   earnings_state.json   -- running P&L, consecutive losses, kill switch flag
================================================================================
"""

import os, sys, csv, json, datetime
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    EARN_TRADES_CSV, EARN_STATE_JSON,
    EARN_MAX_POSITIONS, EARN_KILL_SWITCH_LOSSES, EARN_KILL_SWITCH_DD,
    EARN_CAPITAL_PER_TRADE, EARN_SLIPPAGE_PCT,
    EARN_STOP_PCT, EARN_MAX_HOLD,
)

EARN_TRADES_PATH = os.path.join(_SCRIPT_DIR, EARN_TRADES_CSV)
EARN_STATE_PATH  = os.path.join(_SCRIPT_DIR, EARN_STATE_JSON)

EARN_TRADE_HEADERS = [
    "trade_id", "date_open", "ticker", "direction",
    "entry_price", "shares", "capital_deployed",
    "earnings_date", "days_until_earnings",
    "beat_rate", "avg_eps_surprise_pct", "eps_trend",
    "status",
    "date_close", "exit_price", "exit_reason",
    "hold_days", "gross_pnl", "net_pnl",
]

_DEFAULT_EARN_STATE = {
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

def _init_earn_trades_csv() -> bool:
    """Create earnings_trades.csv with headers if missing. Returns True if new."""
    if not os.path.exists(EARN_TRADES_PATH):
        with open(EARN_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(EARN_TRADE_HEADERS)
        return True
    return False


def load_earn_trades() -> pd.DataFrame:
    _init_earn_trades_csv()
    df = pd.read_csv(EARN_TRADES_PATH, dtype=str, encoding="utf-8")
    return df


def _save_earn_trades(df: pd.DataFrame):
    """Rewrite earnings_trades.csv from DataFrame with flush+fsync for crash safety."""
    tmp_path = EARN_TRADES_PATH + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    with open(tmp_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(EARN_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.remove(tmp_path)


def load_earn_state() -> dict:
    if not os.path.exists(EARN_STATE_PATH):
        return dict(_DEFAULT_EARN_STATE)
    try:
        with open(EARN_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        for k, v in _DEFAULT_EARN_STATE.items():
            state.setdefault(k, v)
        return state
    except Exception:
        return dict(_DEFAULT_EARN_STATE)


def save_earn_state(state: dict):
    with open(EARN_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------------------
#  Portfolio Queries
# ------------------------------------------------------------------------------

def get_open_earn_trades() -> list:
    """Return list of dicts for all open earnings positions."""
    df = load_earn_trades()
    if df.empty:
        return []
    return df[df["status"] == "open"].to_dict("records")


def earn_open_count() -> int:
    return len(get_open_earn_trades())


def at_earn_cap() -> bool:
    """True if earnings portfolio is at or above the position cap."""
    return earn_open_count() >= EARN_MAX_POSITIONS


# ------------------------------------------------------------------------------
#  Kill Switch
# ------------------------------------------------------------------------------

def is_earn_kill_switch() -> tuple:
    """Check if the earnings kill switch is active. Returns (bool, reason_str)."""
    state = load_earn_state()
    return state.get("kill_switch", False), state.get("kill_switch_reason", "")


def reset_earn_kill_switch():
    """Manually reset the earnings kill switch after review."""
    state = load_earn_state()
    state["kill_switch"]        = False
    state["kill_switch_reason"] = ""
    state["consecutive_losses"] = 0
    save_earn_state(state)
    print("  Earnings kill switch reset. System will accept new entries on next run.")


def _evaluate_earn_kill_switch(state: dict) -> dict:
    """Evaluate and set kill switch if thresholds are breached. No-op if already set."""
    if state["kill_switch"]:
        return state  # already triggered -- manual reset required
    reason = ""
    if state["consecutive_losses"] >= EARN_KILL_SWITCH_LOSSES:
        reason = (f"{state['consecutive_losses']} consecutive losses "
                  f"(threshold: {EARN_KILL_SWITCH_LOSSES})")
    drawdown = state["total_realized_pnl"] - state["peak_pnl"]
    if drawdown <= EARN_KILL_SWITCH_DD:
        reason = (f"Earnings drawdown ${drawdown:+.2f} "
                  f"(threshold: ${EARN_KILL_SWITCH_DD:.0f})")
    if reason:
        state["kill_switch"]        = True
        state["kill_switch_reason"] = reason
    return state


# ------------------------------------------------------------------------------
#  Entry Logging
# ------------------------------------------------------------------------------

def log_earn_entry(signal: dict) -> str:
    """
    Record a new earnings signal as an open paper trade.

    signal dict expected keys:
        ticker                     : str
        live.price                 : float
        live.direction             : str   (always "LONG")
        live.earnings_date         : str   "YYYY-MM-DD"
        live.days_until_earnings   : int
        live.beat_rate             : float
        live.avg_eps_surprise_pct  : float
        live.eps_trend             : str
        bt.win_rate                : float

    Returns the trade_id assigned.
    """
    _init_earn_trades_csv()

    ticker = signal["ticker"]
    live   = signal["live"]
    price  = float(live["price"])

    capital  = EARN_CAPITAL_PER_TRADE
    shares   = capital / price

    today    = datetime.date.today().strftime("%Y-%m-%d")
    trade_id = f"{today.replace('-', '')}_earn_{ticker}"

    row = {
        "trade_id":             trade_id,
        "date_open":            today,
        "ticker":               ticker,
        "direction":            str(live.get("direction", "LONG")),
        "entry_price":          f"{price:.4f}",
        "shares":               f"{shares:.4f}",
        "capital_deployed":     f"{capital:.2f}",
        "earnings_date":        str(live.get("earnings_date", "")),
        "days_until_earnings":  str(live.get("days_until_earnings", "")),
        "beat_rate":            f"{float(live.get('beat_rate', 0)):.4f}",
        "avg_eps_surprise_pct": f"{float(live.get('avg_eps_surprise_pct', 0)):.2f}",
        "eps_trend":            str(live.get("eps_trend", "")),
        "status":               "open",
        "date_close":           "",
        "exit_price":           "",
        "exit_reason":          "",
        "hold_days":            "",
        "gross_pnl":            "",
        "net_pnl":              "",
    }

    with open(EARN_TRADES_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EARN_TRADE_HEADERS)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    return trade_id


# ------------------------------------------------------------------------------
#  Exit Checking
# ------------------------------------------------------------------------------

def check_earn_exits() -> list:
    """
    For every open earnings position, fetch current price and close if any
    exit condition is met:

        post_earnings  : today > earnings_date  (exit after earnings gap)
        stop_loss      : current_price < entry_price * (1 - EARN_STOP_PCT/100)
        time_stop      : hold_days >= EARN_MAX_HOLD

    Returns list of trade dicts that were closed this run.
    """
    import yfinance as yf

    _init_earn_trades_csv()
    trades_df = load_earn_trades()
    open_mask = trades_df["status"] == "open"

    if not open_mask.any():
        return []

    closed_today = []
    state        = load_earn_state()
    today        = datetime.date.today()

    for idx in trades_df[open_mask].index:
        row         = trades_df.loc[idx]
        ticker      = row["ticker"]
        entry_price = float(row["entry_price"])
        shares      = float(row["shares"])
        date_open   = str(row["date_open"])
        earn_date_str = str(row.get("earnings_date", ""))

        open_date = datetime.datetime.strptime(date_open, "%Y-%m-%d").date()
        hold_days = int(np.busday_count(open_date, today))

        # Parse earnings_date
        try:
            earnings_date = datetime.date.fromisoformat(earn_date_str)
        except (ValueError, TypeError):
            earnings_date = None

        try:
            # Fetch current price
            raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if raw is None or len(raw) == 0:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                close_series = raw[("Close", ticker)]
            else:
                close_series = raw["Close"]

            current_price = float(close_series.iloc[-1])

            exit_reason = ""

            # Exit 1: Post-earnings (today is after or on earnings date)
            if earnings_date is not None and today > earnings_date:
                exit_reason = "post_earnings"

            # Exit 2: Stop loss (price dropped before earnings)
            if not exit_reason:
                stop_level = entry_price * (1.0 - EARN_STOP_PCT / 100.0)
                if current_price < stop_level:
                    exit_reason = "stop_loss"

            # Exit 3: Time stop (safety net)
            if not exit_reason and hold_days >= EARN_MAX_HOLD:
                exit_reason = "time_stop"

            if not exit_reason:
                continue

            # Close the trade
            gross    = shares * (current_price - entry_price)
            slippage = EARN_SLIPPAGE_PCT * shares * (entry_price + current_price)
            net      = gross - slippage

            trades_df.at[idx, "status"]      = "closed"
            trades_df.at[idx, "date_close"]  = today.strftime("%Y-%m-%d")
            trades_df.at[idx, "exit_price"]  = f"{current_price:.4f}"
            trades_df.at[idx, "exit_reason"] = exit_reason
            trades_df.at[idx, "hold_days"]   = str(hold_days)
            trades_df.at[idx, "gross_pnl"]   = f"{gross:.2f}"
            trades_df.at[idx, "net_pnl"]     = f"{net:.2f}"

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

            state = _evaluate_earn_kill_switch(state)
            closed_today.append(trades_df.loc[idx].to_dict())

        except Exception as exc:
            print(f"    [EARN-EXIT] Error checking {ticker}: {exc}")
            continue

    # Always persist state (even when no exits occurred)
    state["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_earn_trades(trades_df)
    save_earn_state(state)

    return closed_today


# ------------------------------------------------------------------------------
#  Console Display
# ------------------------------------------------------------------------------

def print_earn_portfolio_status():
    """Print a snapshot of current open earnings positions and running P&L."""
    open_trades = get_open_earn_trades()
    state       = load_earn_state()
    today       = datetime.date.today()

    print("  -- Earnings Module Portfolio " + "-" * 32)
    print(f"    Open positions : {len(open_trades)} / {EARN_MAX_POSITIONS}")
    print(f"    Realized P&L   : ${state['total_realized_pnl']:+.2f}")
    print(f"    Total trades   : {state['total_trades']}")
    if state["total_trades"] > 0:
        wr = state["total_wins"] / state["total_trades"] * 100
        print(f"    Live win rate  : {wr:.1f}%")
    print(f"    Consec. losses : {state['consecutive_losses']}")

    if state.get("kill_switch"):
        print(f"\n    !! KILL SWITCH ACTIVE: {state['kill_switch_reason']}")

    if open_trades:
        print(f"\n    {'TICKER':<8} {'DIR':>5} {'ENTRY':>8} "
              f"{'CAPITAL':>8} {'EARN DATE':>12} {'BEAT%':>7} {'DAYS':>5}")
        print(f"    {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*12} {'-'*7} {'-'*5}")
        for t in open_trades:
            open_dt = datetime.datetime.strptime(t["date_open"], "%Y-%m-%d").date()
            days    = (today - open_dt).days
            beat    = float(t.get("beat_rate", 0)) * 100
            print(f"    {t['ticker']:<8} {'LONG':>5} "
                  f"${float(t['entry_price']):>7.2f} "
                  f"${float(t['capital_deployed']):>7.2f} "
                  f"{t.get('earnings_date', ''):>12} "
                  f"{beat:>6.0f}% "
                  f"{days:>5}")
    print("  " + "-" * 61)
    print()


def print_earn_performance_report():
    """
    Print a performance summary of all closed earnings trades.
    Only meaningful when there are at least 3 closed trades.
    """
    df = load_earn_trades()
    if df.empty:
        return

    closed   = df[df["status"] == "closed"].copy()
    n_closed = len(closed)
    if n_closed < 3:
        return

    closed["net_pnl"]   = pd.to_numeric(closed["net_pnl"],   errors="coerce")
    closed["gross_pnl"] = pd.to_numeric(closed["gross_pnl"], errors="coerce")
    closed["hold_days"] = pd.to_numeric(closed["hold_days"], errors="coerce")
    closed["beat_rate"] = pd.to_numeric(closed["beat_rate"], errors="coerce")

    pnls = closed["net_pnl"].dropna().values
    if len(pnls) < 3:
        return

    wins        = pnls[pnls > 0]
    losses      = pnls[pnls <= 0]
    live_wr     = len(wins) / len(pnls) * 100
    live_avg    = pnls.mean()
    total_pnl   = pnls.sum()
    avg_hold    = closed["hold_days"].dropna().mean()
    bt_wr_avg   = closed["beat_rate"].dropna().mean() * 100
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

    # Breakdown by exit reason
    by_reason = closed.groupby("exit_reason")["net_pnl"].agg(["count", "sum"]).reset_index()

    # Edge verdict vs historical beat rate
    if bt_wr_avg > 0 and live_wr >= bt_wr_avg * 0.8:
        verdict = "Edge holding"
    else:
        verdict = "Edge degrading -- review"

    print()
    print("  -- Earnings Performance Report " + "-" * 29)
    print(f"    Closed trades      : {n_closed}")
    print(f"    Total net P&L      : ${total_pnl:>+.2f}")
    print(f"    Live win rate      : {live_wr:.1f}%")
    print(f"    Hist beat rate avg : {bt_wr_avg:.1f}%")
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
    if not by_reason.empty:
        print(f"    Exit breakdown:")
        for _, row in by_reason.iterrows():
            print(f"      {row['exit_reason']:<16} : {int(row['count']):>3} trades  "
                  f"P&L=${float(row['sum']):>+.2f}")
    print(f"    Verdict            : {verdict}")
    print("  " + "-" * 61)
    print()
