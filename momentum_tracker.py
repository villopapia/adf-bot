"""
================================================================================
 MOMENTUM TRACKER  -  Paper Trade Journal & Portfolio Risk Manager
================================================================================
 Manages momentum paper trades independently from the pairs system.
 On each daily run (called by run_system.py):

   Phase M0a -- check_mom_exits()
       For every open momentum position, fetches current price data,
       recomputes the ATR trailing stop, and closes the trade if any
       exit condition is met (trail stop, take profit, time stop).
       Updates momentum_state.json with realized P&L, consecutive
       losses, and evaluates the kill switch.

   Phase M0b -- log_mom_entry(signal, vix_scale)
       Records a new momentum signal as an open paper trade in
       momentum_trades.csv using vol-targeted position sizing.

 Portfolio Risk Controls
 -----------------------
   MOM_MAX_POSITIONS        : hard cap on simultaneous open trades
   MOM_KILL_SWITCH_LOSSES   : kill switch fires after N live losses in a row
   MOM_KILL_SWITCH_DRAWDOWN : kill switch fires when live P&L drops this far
                              from its peak (e.g. -$800)

 Files produced / updated
 ------------------------
   momentum_trades.csv  -- every momentum paper trade (open + closed)
   momentum_state.json  -- running P&L, consecutive losses, kill switch flag
================================================================================
"""

import os, sys, csv, json, datetime
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    MOM_LIVE_TRADES_CSV, MOM_SYSTEM_STATE_JSON,
    MOM_MAX_POSITIONS, MOM_CAPITAL_PER_TRADE,
    MOM_SLIPPAGE_PCT, MOM_TIME_STOP_DAYS,
    MOM_ATR_PERIOD, MOM_ATR_TRAIL_MULT, MOM_ATR_TP_MULT,
    MOM_KILL_SWITCH_LOSSES, MOM_KILL_SWITCH_DRAWDOWN,
    MOM_PORTFOLIO_CORR_MAX,
    MOM_VOL_TARGET_ANN, MOM_VOL_FLOOR,
    MOM_VIX_TIERS,
)

MOM_TRADES_PATH = os.path.join(_SCRIPT_DIR, MOM_LIVE_TRADES_CSV)
MOM_STATE_PATH  = os.path.join(_SCRIPT_DIR, MOM_SYSTEM_STATE_JSON)

MOM_TRADE_HEADERS = [
    "trade_id", "date_open", "ticker", "direction",
    "entry_price", "shares", "capital_deployed",
    "vol_target_scale", "vix_scale",
    "atr_at_entry", "trail_stop_price", "take_profit_price",
    "bt_win_rate", "bt_profit_factor", "bt_total_pnl",
    "status",
    "date_close", "exit_price", "exit_reason",
    "hold_days", "peak_price", "gross_pnl", "net_pnl",
]

_DEFAULT_MOM_STATE = {
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

def _init_mom_trades_csv() -> bool:
    """Create momentum_trades.csv with headers if missing. Returns True if new."""
    if not os.path.exists(MOM_TRADES_PATH):
        with open(MOM_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(MOM_TRADE_HEADERS)
        return True
    return False


def load_mom_trades() -> pd.DataFrame:
    _init_mom_trades_csv()
    df = pd.read_csv(MOM_TRADES_PATH, dtype=str, encoding="utf-8")
    return df


def _save_mom_trades(df: pd.DataFrame):
    """Rewrite momentum_trades.csv from DataFrame with flush+fsync for crash safety."""
    tmp_path = MOM_TRADES_PATH + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    with open(tmp_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(MOM_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.remove(tmp_path)


def load_mom_state() -> dict:
    if not os.path.exists(MOM_STATE_PATH):
        return dict(_DEFAULT_MOM_STATE)
    try:
        with open(MOM_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        for k, v in _DEFAULT_MOM_STATE.items():
            state.setdefault(k, v)
        return state
    except Exception:
        return dict(_DEFAULT_MOM_STATE)


def save_mom_state(state: dict):
    with open(MOM_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------------------
#  Portfolio Queries
# ------------------------------------------------------------------------------

def get_open_mom_trades() -> list:
    """Return list of dicts for all open momentum positions."""
    df = load_mom_trades()
    if df.empty:
        return []
    return df[df["status"] == "open"].to_dict("records")


def mom_open_count() -> int:
    return len(get_open_mom_trades())


def at_mom_cap() -> bool:
    """True if momentum portfolio is at or above the position cap."""
    return mom_open_count() >= MOM_MAX_POSITIONS


# ------------------------------------------------------------------------------
#  Kill Switch
# ------------------------------------------------------------------------------

def is_mom_kill_switch() -> tuple:
    """Check if the momentum kill switch is active. Returns (bool, reason_str)."""
    state = load_mom_state()
    return state.get("kill_switch", False), state.get("kill_switch_reason", "")


def reset_mom_kill_switch():
    """Manually reset the momentum kill switch after review."""
    state = load_mom_state()
    state["kill_switch"]        = False
    state["kill_switch_reason"] = ""
    state["consecutive_losses"] = 0
    save_mom_state(state)
    print("  Momentum kill switch reset. System will accept new entries on next run.")


def _evaluate_mom_kill_switch(state: dict) -> dict:
    """Evaluate and set kill switch if thresholds are breached. No-op if already set."""
    if state["kill_switch"]:
        return state  # already triggered -- manual reset required
    reason = ""
    if state["consecutive_losses"] >= MOM_KILL_SWITCH_LOSSES:
        reason = (f"{state['consecutive_losses']} consecutive losses "
                  f"(threshold: {MOM_KILL_SWITCH_LOSSES})")
    drawdown = state["total_realized_pnl"] - state["peak_pnl"]
    if drawdown <= MOM_KILL_SWITCH_DRAWDOWN:
        reason = (f"Momentum drawdown ${drawdown:+.2f} "
                  f"(threshold: ${MOM_KILL_SWITCH_DRAWDOWN:.0f})")
    if reason:
        state["kill_switch"]        = True
        state["kill_switch_reason"] = reason
    return state


# ------------------------------------------------------------------------------
#  Entry Logging
# ------------------------------------------------------------------------------

def log_mom_entry(signal: dict, vix_scale: float = 1.0) -> str:
    """
    Record a new momentum signal as an open paper trade.

    signal dict expected keys:
        ticker          : str
        live.price      : float
        live.atr_14     : float
        live.vol_6mo    : float  (annualized realized vol)
        live.direction  : str    (always "LONG" for momentum)
        bt.win_rate     : float
        bt.profit_factor: float
        bt.total_pnl    : float

    Returns the trade_id assigned.
    """
    _init_mom_trades_csv()

    ticker = signal["ticker"]
    live   = signal["live"]
    bt     = signal["bt"]

    price   = float(live["price"])
    atr     = float(live["atr_14"])
    vol_6mo = float(live["vol_6mo"])

    # Vol-targeted sizing (Barroso & Santa-Clara 2015)
    realized       = max(vol_6mo, MOM_VOL_FLOOR)
    vol_scale      = MOM_VOL_TARGET_ANN / realized
    adjusted_scale = vol_scale * vix_scale
    adjusted_scale = min(max(adjusted_scale, 0.25), 2.0)   # clamp to [0.25, 2.0]
    capital        = MOM_CAPITAL_PER_TRADE * adjusted_scale
    shares         = capital / price

    # ATR-based stops
    trail_stop  = price - MOM_ATR_TRAIL_MULT * atr
    take_profit = price + MOM_ATR_TP_MULT   * atr

    today    = datetime.date.today().strftime("%Y-%m-%d")
    trade_id = f"{today.replace('-', '')}_{ticker}"

    row = {
        "trade_id":          trade_id,
        "date_open":         today,
        "ticker":            ticker,
        "direction":         str(live.get("direction", "LONG")),
        "entry_price":       f"{price:.4f}",
        "shares":            f"{shares:.4f}",
        "capital_deployed":  f"{capital:.2f}",
        "vol_target_scale":  f"{adjusted_scale:.4f}",
        "vix_scale":         f"{vix_scale:.4f}",
        "atr_at_entry":      f"{atr:.4f}",
        "trail_stop_price":  f"{trail_stop:.4f}",
        "take_profit_price": f"{take_profit:.4f}",
        "bt_win_rate":       f"{float(bt['win_rate']):.1f}",
        "bt_profit_factor":  f"{float(bt['profit_factor']):.2f}",
        "bt_total_pnl":      f"{float(bt['total_pnl']):.2f}",
        "status":            "open",
        "date_close":        "",
        "exit_price":        "",
        "exit_reason":       "",
        "hold_days":         "",
        "peak_price":        f"{price:.4f}",
        "gross_pnl":         "",
        "net_pnl":           "",
    }

    with open(MOM_TRADES_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MOM_TRADE_HEADERS)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    return trade_id


# ------------------------------------------------------------------------------
#  Exit Checking
# ------------------------------------------------------------------------------

def check_mom_exits() -> list:
    """
    For every open momentum position, fetch current price data, recompute the
    ATR trailing stop, and close the trade if any exit condition is met.

    Exit conditions:
      trail_stop  : current_price <= peak_price - ATR_TRAIL_MULT * ATR(14)
      take_profit : current_price >= entry_price + ATR_TP_MULT * ATR(14)
      time_stop   : hold_days >= MOM_TIME_STOP_DAYS

    Note: trailing stop is adaptive -- it ratchets up with peak price daily.

    Returns list of trade dicts that were closed this run.
    """
    import yfinance as yf

    _init_mom_trades_csv()
    trades_df = load_mom_trades()
    open_mask = trades_df["status"] == "open"

    if not open_mask.any():
        return []

    closed_today = []
    state        = load_mom_state()
    today        = datetime.date.today()

    for idx in trades_df[open_mask].index:
        row         = trades_df.loc[idx]
        ticker      = row["ticker"]
        entry_price = float(row["entry_price"])
        shares      = float(row["shares"])
        stored_peak = float(row["peak_price"])
        take_profit = float(row["take_profit_price"])
        date_open   = str(row["date_open"])

        try:
            raw = yf.download(ticker, period="30d", auto_adjust=True, progress=False)
            if raw is None or (hasattr(raw, "empty") and raw.empty):
                continue

            # Handle MultiIndex columns from yfinance batch downloads
            if isinstance(raw.columns, pd.MultiIndex):
                high  = raw[("High",  ticker)]
                low   = raw[("Low",   ticker)]
                close = raw[("Close", ticker)]
            else:
                high  = raw["High"]
                low   = raw["Low"]
                close = raw["Close"]

            if close.empty:
                continue

            current_price = float(close.iloc[-1])

            # Compute current ATR(14)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            current_atr = float(tr.rolling(MOM_ATR_PERIOD).mean().iloc[-1])

            # Update ratcheting peak price
            peak_price = max(stored_peak, current_price)
            trades_df.at[idx, "peak_price"] = f"{peak_price:.4f}"

            # Recompute adaptive trailing stop
            trail_stop = peak_price - MOM_ATR_TRAIL_MULT * current_atr
            trades_df.at[idx, "trail_stop_price"] = f"{trail_stop:.4f}"

            # Hold days
            open_date = datetime.datetime.strptime(date_open, "%Y-%m-%d").date()
            hold_days = (today - open_date).days

            # Evaluate exit conditions in priority order
            exit_reason = ""
            if current_price <= trail_stop:
                exit_reason = "trail_stop"
            elif current_price >= take_profit:
                exit_reason = "take_profit"
            elif hold_days >= MOM_TIME_STOP_DAYS:
                exit_reason = "time_stop"

            if not exit_reason:
                continue  # position still open -- carry updated peak/stop forward

            # Close the trade
            gross    = shares * (current_price - entry_price)
            slippage = MOM_SLIPPAGE_PCT * (shares * entry_price + shares * current_price)
            net      = gross - slippage

            trades_df.at[idx, "status"]     = "closed"
            trades_df.at[idx, "date_close"] = today.strftime("%Y-%m-%d")
            trades_df.at[idx, "exit_price"] = f"{current_price:.4f}"
            trades_df.at[idx, "exit_reason"] = exit_reason
            trades_df.at[idx, "hold_days"]  = str(hold_days)
            trades_df.at[idx, "gross_pnl"]  = f"{gross:.2f}"
            trades_df.at[idx, "net_pnl"]    = f"{net:.2f}"

            # Update running state
            state["total_trades"]       += 1
            state["total_realized_pnl"] += net

            if net > 0:
                state["total_wins"]          += 1
                state["consecutive_wins"]    += 1
                state["consecutive_losses"]   = 0
                if state["total_realized_pnl"] > state["peak_pnl"]:
                    state["peak_pnl"] = state["total_realized_pnl"]
            else:
                state["consecutive_losses"] += 1
                state["consecutive_wins"]    = 0

            state = _evaluate_mom_kill_switch(state)
            closed_today.append(trades_df.loc[idx].to_dict())

        except Exception as e:
            print(f"    [MOM-EXIT] Error checking {ticker}: {e}")
            continue

    # Always persist updated peak/stop values even for non-closed positions
    state["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_mom_trades(trades_df)
    save_mom_state(state)

    return closed_today


# ------------------------------------------------------------------------------
#  Correlation Check
# ------------------------------------------------------------------------------

def is_mom_correlated_with_open(new_ticker: str) -> bool:
    """
    Return True if new_ticker has abs(correlation) > MOM_PORTFOLIO_CORR_MAX
    with any currently open momentum position over the last 120 days.
    Blocks the entry to avoid concentrated bets on the same factor.
    """
    import yfinance as yf

    open_trades = get_open_mom_trades()
    if not open_trades:
        return False

    open_tickers = [t["ticker"] for t in open_trades]
    all_tickers  = list(set(open_tickers + [new_ticker]))

    try:
        raw = yf.download(all_tickers, period="120d",
                          auto_adjust=True, progress=False)
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return False

        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

        returns = closes.pct_change().dropna()
        if len(returns) < 20:
            return False

        for ot in open_tickers:
            if ot in returns.columns and new_ticker in returns.columns:
                corr = float(returns[ot].corr(returns[new_ticker]))
                if abs(corr) > MOM_PORTFOLIO_CORR_MAX:
                    return True

    except Exception:
        pass  # cannot fetch -- do not block

    return False


# ------------------------------------------------------------------------------
#  Console Display
# ------------------------------------------------------------------------------

def print_mom_portfolio_status():
    """Print a snapshot of current open momentum positions and running P&L."""
    open_trades = get_open_mom_trades()
    state       = load_mom_state()
    today       = datetime.date.today()

    print("  -- Momentum Portfolio " + "-" * 39)
    print(f"    Open positions : {len(open_trades)} / {MOM_MAX_POSITIONS}")
    print(f"    Realized P&L   : ${state['total_realized_pnl']:+.2f}")
    print(f"    Total trades   : {state['total_trades']}")
    if state["total_trades"] > 0:
        wr = state["total_wins"] / state["total_trades"] * 100
        print(f"    Live win rate  : {wr:.1f}%")
    print(f"    Consec. losses : {state['consecutive_losses']}")

    if state.get("kill_switch"):
        print(f"\n    !! KILL SWITCH ACTIVE: {state['kill_switch_reason']}")

    if open_trades:
        print(f"\n    {'TICKER':<8} {'DIR':>5} {'ENTRY':>8} {'PEAK':>8} "
              f"{'STOP':>8} {'TARGET':>8} {'DAYS':>5}")
        print(f"    {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
        for t in open_trades:
            open_dt  = datetime.datetime.strptime(t["date_open"], "%Y-%m-%d").date()
            hold     = (today - open_dt).days
            entry    = float(t["entry_price"])
            peak     = float(t["peak_price"])
            stop     = float(t["trail_stop_price"])
            target   = float(t["take_profit_price"])
            print(f"    {t['ticker']:<8} {'LONG':>5} "
                  f"${entry:>7.2f} "
                  f"${peak:>7.2f} "
                  f"${stop:>7.2f} "
                  f"${target:>7.2f} "
                  f"{hold:>5}")
    print("  " + "-" * 61)
    print()


def print_mom_performance_report():
    """
    Print a performance summary of all closed momentum trades.
    Shows total P&L, win rate, average hold, Sharpe, and best/worst trade.
    Only meaningful when there are at least 3 closed trades.
    """
    df = load_mom_trades()
    if df.empty:
        return

    closed   = df[df["status"] == "closed"].copy()
    n_closed = len(closed)
    if n_closed < 3:
        return

    closed["net_pnl"]         = pd.to_numeric(closed["net_pnl"],         errors="coerce")
    closed["gross_pnl"]       = pd.to_numeric(closed["gross_pnl"],       errors="coerce")
    closed["hold_days"]       = pd.to_numeric(closed["hold_days"],        errors="coerce")
    closed["bt_win_rate"]     = pd.to_numeric(closed["bt_win_rate"],      errors="coerce")
    closed["bt_profit_factor"]= pd.to_numeric(closed["bt_profit_factor"], errors="coerce")

    pnls = closed["net_pnl"].dropna().values
    if len(pnls) < 3:
        return

    wins         = pnls[pnls > 0]
    losses       = pnls[pnls <= 0]
    live_wr      = len(wins) / len(pnls) * 100
    live_avg     = pnls.mean()
    total_pnl    = pnls.sum()
    avg_hold     = closed["hold_days"].dropna().mean()
    bt_wr_avg    = closed["bt_win_rate"].dropna().mean()
    best_trade   = float(pnls.max())
    worst_trade  = float(pnls.min())

    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    profit_factor = (wins.sum() / abs(losses.sum())
                     if len(losses) > 0 and losses.sum() != 0 else float("inf"))

    # Live Sharpe (rough annualisation: 250 / avg_hold or 252)
    live_sharpe_str = "N/A"
    if len(pnls) >= 5:
        std_pnl = pnls.std(ddof=1) if len(pnls) > 1 else 1e-12
        avg_hold_safe = max(avg_hold, 1.0) if not np.isnan(avg_hold) else 21.0
        live_sharpe = (live_avg / std_pnl) * np.sqrt(252 / avg_hold_safe)
        live_sharpe_str = f"{live_sharpe:+.2f}"

    # Maximum consecutive losses in closed history
    max_consec = 0
    curr_streak = 0
    for p in pnls:
        if p <= 0:
            curr_streak += 1
            max_consec = max(max_consec, curr_streak)
        else:
            curr_streak = 0

    # System drawdown on closed trades
    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    max_dd  = float((cum_pnl - peak).min())

    # Edge verdict vs backtest
    if bt_wr_avg > 0 and live_wr >= bt_wr_avg * 0.8:
        verdict = "Edge holding"
    else:
        verdict = "Edge degrading -- review"

    print()
    print("  -- Momentum Performance Report " + "-" * 30)
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
    print(f"    Verdict            : {verdict}")
    print("  " + "-" * 61)
    print()
