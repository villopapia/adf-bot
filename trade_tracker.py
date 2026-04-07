"""
================================================================================
 TRADE TRACKER  —  Live Paper Trade Journal & Portfolio Risk Manager
================================================================================
 Tracks every diamond signal taken as a paper trade.
 On each daily run (called by run_system.py):

   Phase 0a — check_exits()
       For every open position, fetches current price data, recomputes the
       z-score, and closes the trade if exit conditions are met.
       Updates system_state.json with realized P&L, consecutive losses,
       and evaluates the kill switch.

   Phase 0b — log_entry(diamond)
       Records a new diamond signal as an open paper trade in live_trades.csv.
       Respects the portfolio cap (MAX_CONCURRENT_POSITIONS) and kill switch.

 Portfolio Risk Controls
 -----------------------
   MAX_CONCURRENT_POSITIONS : hard cap on simultaneous open trades
   MAX_CONSECUTIVE_LOSSES   : kill switch fires after N live losses in a row
   MAX_SYSTEM_DRAWDOWN      : kill switch fires when live P&L drops this much
                              from its peak (e.g. -$500)

 Files produced / updated
 ------------------------
   live_trades.csv      — every paper trade (open + closed)
   system_state.json    — running P&L, consecutive losses, kill switch flag
================================================================================
"""

import os, sys, csv, json, datetime
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    LIVE_TRADES_CSV, SYSTEM_STATE_JSON,
    MAX_CONCURRENT_POSITIONS, BORROW_COST_PCT,
    MAX_CONSECUTIVE_LOSSES, MAX_SYSTEM_DRAWDOWN,
    Z_STOP, MAX_HOLD, SLIPPAGE_PCT, CAPITAL_PER_TRADE,
    PORTFOLIO_CORR_MAX,
    KELLY_FRACTION, KELLY_MIN_SCALE, KELLY_MAX_SCALE,
)

LIVE_TRADES_PATH  = os.path.join(_SCRIPT_DIR, LIVE_TRADES_CSV)
SYSTEM_STATE_PATH = os.path.join(_SCRIPT_DIR, SYSTEM_STATE_JSON)

TRADE_HEADERS = [
    "trade_id", "date_open", "stock_a", "stock_b", "direction",
    "entry_z", "beta", "shares_a", "shares_b",
    "price_a_entry", "price_b_entry", "capital_deployed", "kelly_scale",
    "bt_expected_pnl", "bt_win_rate", "bt_profit_factor",
    "status",
    "date_close", "price_a_exit", "price_b_exit",
    "exit_reason", "hold_days", "exit_z", "gross_pnl", "net_pnl",
]

_DEFAULT_STATE = {
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


# ──────────────────────────────────────────────────────────────────────────────
#  CSV / JSON I/O
# ──────────────────────────────────────────────────────────────────────────────

def _init_trades_csv() -> bool:
    """Create live_trades.csv with headers if missing. Returns True if new."""
    if not os.path.exists(LIVE_TRADES_PATH):
        with open(LIVE_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(TRADE_HEADERS)
        return True
    return False


def load_trades() -> pd.DataFrame:
    _init_trades_csv()
    df = pd.read_csv(LIVE_TRADES_PATH, dtype=str)
    return df


def get_open_trades() -> list[dict]:
    df = load_trades()
    if df.empty:
        return []
    return df[df["status"] == "open"].to_dict("records")


def _save_new_trade(trade: dict):
    _init_trades_csv()
    with open(LIVE_TRADES_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([trade.get(h, "") for h in TRADE_HEADERS])


def _update_trade(trade_id: str, updates: dict):
    """Rewrite live_trades.csv updating the row matching trade_id."""
    df = load_trades()
    mask = df["trade_id"].astype(str) == str(trade_id)
    for k, v in updates.items():
        if k in df.columns:
            df.loc[mask, k] = v
    df.to_csv(LIVE_TRADES_PATH, index=False)


# ──────────────────────────────────────────────────────────────────────────────
#  System state
# ──────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not os.path.exists(SYSTEM_STATE_PATH):
        return dict(_DEFAULT_STATE)
    try:
        with open(SYSTEM_STATE_PATH, "r") as f:
            state = json.load(f)
        for k, v in _DEFAULT_STATE.items():
            state.setdefault(k, v)
        return state
    except Exception:
        return dict(_DEFAULT_STATE)


def save_state(state: dict):
    state["last_updated"] = str(datetime.date.today())
    with open(SYSTEM_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
#  Portfolio cap
# ──────────────────────────────────────────────────────────────────────────────

def open_position_count() -> int:
    return len(get_open_trades())


def at_portfolio_cap() -> bool:
    return open_position_count() >= MAX_CONCURRENT_POSITIONS


# ──────────────────────────────────────────────────────────────────────────────
#  Kill switch
# ──────────────────────────────────────────────────────────────────────────────

def is_kill_switch_triggered() -> tuple[bool, str]:
    state = load_state()
    return state.get("kill_switch", False), state.get("kill_switch_reason", "")


def reset_kill_switch():
    """Manually reset the kill switch (call when you've reviewed and decided to resume)."""
    state = load_state()
    state["kill_switch"]        = False
    state["kill_switch_reason"] = ""
    state["consecutive_losses"] = 0
    save_state(state)
    print("  Kill switch reset. System will accept new entries on next run.")


def _evaluate_kill_switch(state: dict) -> dict:
    if state["kill_switch"]:
        return state  # already triggered — manual reset required
    reason = ""
    if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        reason = (f"{state['consecutive_losses']} consecutive live losses "
                  f"(threshold: {MAX_CONSECUTIVE_LOSSES})")
    drawdown = state["total_realized_pnl"] - state["peak_pnl"]
    if drawdown <= MAX_SYSTEM_DRAWDOWN:
        reason = (f"System drawdown ${drawdown:+.2f} "
                  f"(threshold: ${MAX_SYSTEM_DRAWDOWN:.0f})")
    if reason:
        state["kill_switch"]        = True
        state["kill_switch_reason"] = reason
    return state


# ──────────────────────────────────────────────────────────────────────────────
#  Kelly criterion sizing
# ──────────────────────────────────────────────────────────────────────────────

def _kelly_capital(bt: dict) -> tuple[float, float]:
    """
    Compute Kelly-adjusted capital for a trade.

    Kelly formula:
        f* = win_rate - (1 - win_rate) / profit_factor
        f_kelly = f* * KELLY_FRACTION
        scale = clamp(f_kelly, KELLY_MIN_SCALE, KELLY_MAX_SCALE)
        adjusted_capital = CAPITAL_PER_TRADE * scale

    Returns (adjusted_capital, scale).
    Falls back to (CAPITAL_PER_TRADE, 1.0) if stats are missing or invalid.
    """
    try:
        win_rate      = bt.get("win_rate", 0) / 100.0   # convert % to fraction
        profit_factor = bt.get("profit_factor", 0)
        if win_rate <= 0 or profit_factor <= 0:
            return CAPITAL_PER_TRADE, 1.0
        f_star  = win_rate - (1.0 - win_rate) / profit_factor
        f_kelly = f_star * KELLY_FRACTION
        scale   = max(KELLY_MIN_SCALE, min(KELLY_MAX_SCALE, f_kelly))
        return CAPITAL_PER_TRADE * scale, round(scale, 4)
    except Exception:
        return CAPITAL_PER_TRADE, 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  Entry logging
# ──────────────────────────────────────────────────────────────────────────────

def log_entry(diamond: dict) -> str:
    """
    Record a new diamond signal as an open paper trade.
    Returns the trade_id assigned.
    """
    today    = str(datetime.date.today())
    a        = diamond["a"]
    b        = diamond["b"]
    live     = diamond["live"]
    bt       = diamond["bt"]
    kelly_capital, kelly_scale = _kelly_capital(bt)
    vol_scale = bt.get("vol_scale", 1.0) if isinstance(bt, dict) else 1.0
    half     = (kelly_capital * vol_scale) / 2.0
    shares_a = round(half / live["price_a"])
    shares_b = round(half / live["price_b"])
    capital  = shares_a * live["price_a"] + shares_b * live["price_b"]
    trade_id = f"{today.replace('-','')}_{a}_{b}"

    trade = {
        "trade_id":         trade_id,
        "date_open":        today,
        "stock_a":          a,
        "stock_b":          b,
        "direction":        live["direction"],
        "entry_z":          round(live["z"], 4),
        "beta":             round(live["beta"], 4),
        "shares_a":         shares_a,
        "shares_b":         shares_b,
        "price_a_entry":    round(live["price_a"], 4),
        "price_b_entry":    round(live["price_b"], 4),
        "capital_deployed": round(capital, 2),
        "kelly_scale":      kelly_scale,
        "bt_expected_pnl":  round(bt.get("avg_pnl", 0), 2),
        "bt_win_rate":      round(bt.get("win_rate", 0), 1),
        "bt_profit_factor": round(bt.get("profit_factor", 0), 2),
        "status":           "open",
        "date_close":       "", "price_a_exit":     "",
        "price_b_exit":     "", "exit_reason":      "",
        "hold_days":        "", "exit_z":           "",
        "gross_pnl":        "", "net_pnl":          "",
    }
    _save_new_trade(trade)
    return trade_id


# ──────────────────────────────────────────────────────────────────────────────
#  Exit checking
# ──────────────────────────────────────────────────────────────────────────────

def check_exits() -> list[dict]:
    """
    For every open position, fetch current price data, recompute the z-score,
    and close the trade if any exit condition is met.

    Exit conditions (mirror the backtest):
      LONG spread  : exit when z >= 0 (mean reversion achieved)
      SHORT spread : exit when z <= 0
      Stop         : |z| > Z_STOP
      Time         : hold_days >= MAX_HOLD

    P&L calculation includes slippage and short-leg borrow cost,
    matching the backtest model exactly.

    Returns list of trades that were closed this run.
    """
    open_trades = get_open_trades()
    if not open_trades:
        return []

    # Lazy import avoids circular dependency at module load time
    from master_signal import fetch_pair, compute_rolling_signals

    today  = datetime.date.today()
    closed = []
    state  = load_state()

    for trade in open_trades:
        a          = trade["stock_a"]
        b          = trade["stock_b"]
        tid        = trade["trade_id"]
        entry_date = datetime.date.fromisoformat(str(trade["date_open"]))
        hold_days  = (today - entry_date).days

        close = fetch_pair(a, b)
        if close is None:
            continue

        signals  = compute_rolling_signals(close, a, b)
        last_row = signals.iloc[-1]
        z_now    = last_row["z"]
        if np.isnan(z_now):
            continue

        direction   = trade["direction"]
        close_trade = False
        exit_reason = ""

        if direction == "LONG"  and z_now >= 0:
            close_trade, exit_reason = True, "target"
        elif direction == "SHORT" and z_now <= 0:
            close_trade, exit_reason = True, "target"
        if abs(z_now) > Z_STOP:
            close_trade, exit_reason = True, "stop"
        if hold_days >= MAX_HOLD:
            close_trade, exit_reason = True, "time"

        if not close_trade:
            continue

        # ── P&L (mirrors backtest model exactly) ─────────────────────────
        pa_entry = float(trade["price_a_entry"])
        pb_entry = float(trade["price_b_entry"])
        pa_exit  = float(last_row["price_a"])
        pb_exit  = float(last_row["price_b"])
        shares_a = float(trade["shares_a"])
        shares_b = float(trade["shares_b"])

        if direction == "LONG":
            gross  =  shares_a * (pa_exit - pa_entry) - shares_b * (pb_exit - pb_entry)
            avg_pb = (pb_entry + pb_exit) / 2.0
            borrow = shares_b * avg_pb * (BORROW_COST_PCT / 252) * hold_days
        else:
            gross  = -shares_a * (pa_exit - pa_entry) + shares_b * (pb_exit - pb_entry)
            avg_pa = (pa_entry + pa_exit) / 2.0
            borrow = shares_a * avg_pa * (BORROW_COST_PCT / 252) * hold_days

        entry_notional = shares_a * pa_entry + shares_b * pb_entry
        exit_notional  = shares_a * pa_exit  + shares_b * pb_exit
        slippage       = SLIPPAGE_PCT * (entry_notional + exit_notional)
        net_pnl        = gross - slippage - borrow

        updates = {
            "status":       "closed",
            "date_close":   str(today),
            "price_a_exit": round(pa_exit, 4),
            "price_b_exit": round(pb_exit, 4),
            "exit_reason":  exit_reason,
            "hold_days":    hold_days,
            "exit_z":       round(z_now, 4),
            "gross_pnl":    round(gross, 2),
            "net_pnl":      round(net_pnl, 2),
        }
        _update_trade(tid, updates)

        # ── Update system state ───────────────────────────────────────────
        state["total_trades"]        += 1
        state["total_realized_pnl"]  += net_pnl
        if net_pnl > 0:
            state["total_wins"]          += 1
            state["consecutive_losses"]   = 0
            state["consecutive_wins"]    += 1
            state["peak_pnl"] = max(state["peak_pnl"],
                                    state["total_realized_pnl"])
        else:
            state["consecutive_losses"] += 1
            state["consecutive_wins"]    = 0

        state = _evaluate_kill_switch(state)
        trade.update(updates)
        closed.append(trade)

    save_state(state)
    return closed


# ──────────────────────────────────────────────────────────────────────────────
#  Console display
# ──────────────────────────────────────────────────────────────────────────────

def print_portfolio_status():
    open_trades = get_open_trades()
    state       = load_state()
    today       = datetime.date.today()

    print("  -- Live Portfolio " + "-" * 43)
    if not open_trades:
        print("    No open positions.\n")
    else:
        print(f"    {'PAIR':<14} {'DIR':>5} {'Z-ENTRY':>8} "
              f"{'HOLD':>5} {'CAPITAL':>8}")
        print(f"    {'─'*14} {'─'*5} {'─'*8} {'─'*5} {'─'*8}")
        for t in open_trades:
            hold = (today - datetime.date.fromisoformat(
                        str(t["date_open"]))).days
            print(f"    {t['stock_a']}/{t['stock_b']:<8} "
                  f"{t['direction']:>5}  "
                  f"{float(t['entry_z']):>+8.2f}  "
                  f"{hold:>4}d  "
                  f"${float(t['capital_deployed']):>,.0f}")
        print()

    drawdown = state["total_realized_pnl"] - state["peak_pnl"]
    live_wr  = (state["total_wins"] / state["total_trades"] * 100
                if state["total_trades"] > 0 else 0.0)

    print(f"    Positions open   : {len(open_trades)} / {MAX_CONCURRENT_POSITIONS}")
    print(f"    Total live P&L   : ${state['total_realized_pnl']:>+.2f}")
    print(f"    Peak P&L         : ${state['peak_pnl']:>+.2f}")
    print(f"    System drawdown  : ${drawdown:>+.2f}  "
          f"(limit ${MAX_SYSTEM_DRAWDOWN:.0f})")
    if state["total_trades"] > 0:
        print(f"    Live trades      : {state['total_trades']}  "
              f"(WR {live_wr:.0f}%,  "
              f"{state['consecutive_losses']} consec. losses)")
    if state.get("kill_switch"):
        print(f"\n    !! KILL SWITCH ACTIVE: {state['kill_switch_reason']}")
    print("  " + "-" * 60)


# ──────────────────────────────────────────────────────────────────────────────
#  Portfolio correlation check
# ──────────────────────────────────────────────────────────────────────────────

def is_correlated_with_open(new_a: str, new_b: str,
                            threshold: float = PORTFOLIO_CORR_MAX) -> bool:
    """
    Check if a new pair's legs are highly correlated with any open position's legs.
    Returns True (blocked) if any open position leg has correlation > threshold
    with any leg of the new pair over the last 90 days.
    """
    import yfinance as yf

    open_trades = get_open_trades()
    if not open_trades:
        return False

    # Collect all unique tickers (open positions + new pair)
    open_tickers = set()
    for t in open_trades:
        open_tickers.add(t["stock_a"])
        open_tickers.add(t["stock_b"])
    new_tickers = {new_a, new_b}

    all_tickers = list(open_tickers | new_tickers)
    try:
        raw = yf.download(all_tickers, period="90d",
                          auto_adjust=True, progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].dropna()
        else:
            prices = raw[["Close"]].rename(
                columns={"Close": all_tickers[0]}).dropna()
    except Exception:
        return False  # cannot fetch — do not block

    if prices.empty or len(prices) < 20:
        return False

    for t in open_trades:
        open_legs = [t["stock_a"], t["stock_b"]]
        for ol in open_legs:
            for nl in [new_a, new_b]:
                if ol == nl:
                    continue  # same ticker is obviously correlated, skip
                if ol not in prices.columns or nl not in prices.columns:
                    continue
                try:
                    corr = float(prices[ol].corr(prices[nl]))
                    if abs(corr) > threshold:
                        return True
                except Exception:
                    continue
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  Live vs Backtest Performance Report
# ──────────────────────────────────────────────────────────────────────────────

def print_performance_report():
    """
    Print a live vs backtest comparison report.
    Reads live_trades.csv and compares live performance to backtest expectations.
    Only meaningful when there are at least 3 closed trades.
    """
    df = load_trades()
    if df.empty:
        return

    closed = df[df["status"] == "closed"].copy()
    n_closed = len(closed)
    if n_closed < 3:
        return

    # Parse numeric columns
    closed["net_pnl"]        = pd.to_numeric(closed["net_pnl"], errors="coerce")
    closed["bt_win_rate"]    = pd.to_numeric(closed["bt_win_rate"], errors="coerce")
    closed["bt_expected_pnl"] = pd.to_numeric(closed["bt_expected_pnl"], errors="coerce")

    pnls = closed["net_pnl"].dropna().values
    if len(pnls) < 3:
        return

    wins       = pnls[pnls > 0]
    live_wr    = len(wins) / len(pnls) * 100
    live_avg   = pnls.mean()
    bt_wr_avg  = closed["bt_win_rate"].dropna().mean()
    bt_pnl_avg = closed["bt_expected_pnl"].dropna().mean()

    # Live Sharpe (if enough trades)
    live_sharpe_str = "N/A"
    if len(pnls) >= 5:
        std_pnl = pnls.std(ddof=1) if len(pnls) > 1 else 1e-12
        live_sharpe = (live_avg / std_pnl) * np.sqrt(250 / 15)  # rough annualisation
        live_sharpe_str = f"{live_sharpe:+.2f}"

    # Consecutive losses
    max_consec_loss = 0
    curr_consec     = 0
    for p in pnls:
        if p <= 0:
            curr_consec += 1
            max_consec_loss = max(max_consec_loss, curr_consec)
        else:
            curr_consec = 0

    # System drawdown
    cum_pnl  = np.cumsum(pnls)
    peak     = np.maximum.accumulate(cum_pnl)
    max_dd   = float((cum_pnl - peak).min())

    # Verdict
    if bt_wr_avg > 0 and live_wr >= bt_wr_avg * 0.8:
        verdict = "Edge holding"
    else:
        verdict = "Edge degrading -- review"

    print()
    print("  -- Live vs Backtest Performance " + "-" * 29)
    print(f"    Closed trades      : {n_closed}")
    print(f"    Live win rate      : {live_wr:.1f}%")
    print(f"    Live avg P&L       : ${live_avg:>+.2f}")
    print(f"    BT avg win rate    : {bt_wr_avg:.1f}%")
    print(f"    BT avg expected P&L: ${bt_pnl_avg:>+.2f}")
    print(f"    Live Sharpe        : {live_sharpe_str}")
    print(f"    Max consec. losses : {max_consec_loss}")
    print(f"    System drawdown    : ${max_dd:>+.2f}")
    print(f"    Verdict            : {verdict}")
    print("  " + "-" * 60)
    print()
