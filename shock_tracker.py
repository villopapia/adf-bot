"""
================================================================================
 SHOCK TRACKER  -  Paper Trade Journal & Portfolio Risk Manager (Shock Module)
================================================================================
 Manages policy shock bounce paper trades independently from other modules.
 On each daily run (called by run_system.py):

   Phase S0 -- check_shock_exits()
       For every open shock position, fetches current price/VIX data and
       closes the trade if any exit condition is met:
           take_profit   : price >= entry * (1 + SHOCK_TAKE_PROFIT_PCT)
           stop_loss     : price <= entry * (1 - SHOCK_STOP_LOSS_PCT)
           vix_recovery  : VIX declined >= SHOCK_EXIT_VIX_DECLINE from peak
           time_stop     : hold_days >= SHOCK_MAX_HOLD

   Phase S1 -- log_shock_entry(signal, vix_scale)
       Records a new shock signal as an open paper trade in shock_trades.csv.

 Portfolio Risk Controls
 -----------------------
   SHOCK_MAX_POSITIONS       : hard cap on simultaneous open trades
   SHOCK_KILL_SWITCH_LOSSES  : kill switch fires after N live losses in a row
   SHOCK_KILL_SWITCH_DD      : kill switch fires at drawdown from peak

 Files produced / updated
 ------------------------
   shock_trades.csv   -- every shock paper trade (open + closed)
   shock_state.json   -- running P&L, consecutive losses, kill switch flag
================================================================================
"""

import os, sys, csv, json, datetime
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    SHOCK_TRADES_CSV, SHOCK_STATE_JSON,
    SHOCK_MAX_POSITIONS, SHOCK_KILL_SWITCH_LOSSES, SHOCK_KILL_SWITCH_DD,
    SHOCK_CAPITAL_PER_TRADE, SHOCK_SLIPPAGE_PCT, SHOCK_VIX_TIERS,
    SHOCK_TAKE_PROFIT_PCT, SHOCK_STOP_LOSS_PCT,
    SHOCK_MAX_HOLD, SHOCK_EXIT_VIX_DECLINE,
)

SHOCK_TRADES_PATH = os.path.join(_SCRIPT_DIR, SHOCK_TRADES_CSV)
SHOCK_STATE_PATH  = os.path.join(_SCRIPT_DIR, SHOCK_STATE_JSON)

SHOCK_TRADE_HEADERS = [
    "trade_id", "date_open", "ticker", "module",
    "direction", "entry_price", "shares", "capital_deployed",
    "vix_at_entry", "vix_scale", "shock_drop_pct", "vix_spike",
    "bt_win_rate", "bt_profit_factor", "bt_total_pnl",
    "status",
    "date_close", "exit_price", "exit_reason",
    "hold_days", "gross_pnl", "net_pnl",
]

_DEFAULT_SHOCK_STATE = {
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

def _init_shock_trades_csv() -> bool:
    if not os.path.exists(SHOCK_TRADES_PATH):
        with open(SHOCK_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SHOCK_TRADE_HEADERS)
        return True
    return False


def load_shock_trades() -> pd.DataFrame:
    _init_shock_trades_csv()
    df = pd.read_csv(SHOCK_TRADES_PATH, dtype=str, encoding="utf-8")
    return df


def _save_shock_trades(df: pd.DataFrame):
    tmp_path = SHOCK_TRADES_PATH + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    with open(tmp_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(SHOCK_TRADES_PATH, "w", newline="", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.remove(tmp_path)


def load_shock_state() -> dict:
    if not os.path.exists(SHOCK_STATE_PATH):
        return dict(_DEFAULT_SHOCK_STATE)
    try:
        with open(SHOCK_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        for k, v in _DEFAULT_SHOCK_STATE.items():
            state.setdefault(k, v)
        return state
    except Exception:
        return dict(_DEFAULT_SHOCK_STATE)


def save_shock_state(state: dict):
    with open(SHOCK_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------------------
#  Portfolio Queries
# ------------------------------------------------------------------------------

def get_open_shock_trades() -> list:
    df = load_shock_trades()
    if df.empty:
        return []
    return df[df["status"] == "open"].to_dict("records")


def shock_open_count() -> int:
    return len(get_open_shock_trades())


def at_shock_cap() -> bool:
    return shock_open_count() >= SHOCK_MAX_POSITIONS


# ------------------------------------------------------------------------------
#  Kill Switch
# ------------------------------------------------------------------------------

def is_shock_kill_switch() -> tuple:
    state = load_shock_state()
    return state.get("kill_switch", False), state.get("kill_switch_reason", "")


def reset_shock_kill_switch():
    state = load_shock_state()
    state["kill_switch"]        = False
    state["kill_switch_reason"] = ""
    state["consecutive_losses"] = 0
    save_shock_state(state)
    print("  Shock kill switch reset. System will accept new entries on next run.")


def _evaluate_shock_kill_switch(state: dict) -> dict:
    if state["kill_switch"]:
        return state
    reason = ""
    if state["consecutive_losses"] >= SHOCK_KILL_SWITCH_LOSSES:
        reason = (f"{state['consecutive_losses']} consecutive losses "
                  f"(threshold: {SHOCK_KILL_SWITCH_LOSSES})")
    drawdown = state["total_realized_pnl"] - state["peak_pnl"]
    if drawdown <= SHOCK_KILL_SWITCH_DD:
        reason = (f"Shock drawdown ${drawdown:+.2f} "
                  f"(threshold: ${SHOCK_KILL_SWITCH_DD:.0f})")
    if reason:
        state["kill_switch"]        = True
        state["kill_switch_reason"] = reason
    return state


# ------------------------------------------------------------------------------
#  Entry Logging
# ------------------------------------------------------------------------------

def log_shock_entry(signal: dict, vix_scale: float = 1.0) -> str:
    """
    Record a new shock signal as an open paper trade.

    signal dict expected keys:
        ticker             : str
        live.price         : float
        live.direction     : str (LONG)
        live.shock_return  : float
        live.vix_spike     : float
        live.vix_level     : float
        bt.win_rate        : float
        bt.profit_factor   : float
        bt.total_pnl       : float

    Returns the trade_id assigned.
    """
    _init_shock_trades_csv()

    ticker = signal["ticker"]
    live   = signal["live"]
    bt     = signal["bt"]
    price  = float(live["price"])

    capital = SHOCK_CAPITAL_PER_TRADE * vix_scale
    shares  = capital / price

    today    = datetime.date.today().strftime("%Y-%m-%d")
    trade_id = f"{today.replace('-', '')}_shock_{ticker}"

    row = {
        "trade_id":          trade_id,
        "date_open":         today,
        "ticker":            ticker,
        "module":            "shock",
        "direction":         str(live.get("direction", "LONG")),
        "entry_price":       f"{price:.4f}",
        "shares":            f"{shares:.4f}",
        "capital_deployed":  f"{capital:.2f}",
        "vix_at_entry":      f"{float(live.get('vix_level', 0)):.2f}",
        "vix_scale":         f"{vix_scale:.4f}",
        "shock_drop_pct":    f"{float(live.get('shock_return', 0))*100:.2f}",
        "vix_spike":         f"{float(live.get('vix_spike', 0)):.2f}",
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

    with open(SHOCK_TRADES_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHOCK_TRADE_HEADERS)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    return trade_id


# ------------------------------------------------------------------------------
#  Exit Checking
# ------------------------------------------------------------------------------

def check_shock_exits() -> list:
    """
    For every open shock position, fetch current price/VIX data and close
    if any exit condition is met.

    Exits:
        take_profit   : price >= entry * (1 + SHOCK_TAKE_PROFIT_PCT)
        stop_loss     : price <= entry * (1 - SHOCK_STOP_LOSS_PCT)
        vix_recovery  : VIX declined >= SHOCK_EXIT_VIX_DECLINE from entry VIX
        time_stop     : hold_days >= SHOCK_MAX_HOLD

    Returns list of trade dicts that were closed this run.
    """
    import yfinance as yf

    _init_shock_trades_csv()
    trades_df = load_shock_trades()
    open_mask = trades_df["status"] == "open"

    if not open_mask.any():
        return []

    closed_today = []
    state        = load_shock_state()

    # Fetch current VIX
    current_vix = None
    try:
        vix_raw = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_close = vix_raw[("Close", "^VIX")]
        else:
            vix_close = vix_raw["Close"]
        vix_close = vix_close.dropna()
        if not vix_close.empty:
            current_vix = float(vix_close.iloc[-1])
    except Exception:
        pass

    # Cache price data per ticker
    price_cache = {}

    for idx in trades_df[open_mask].index:
        row         = trades_df.loc[idx]
        ticker      = row["ticker"]
        entry_price = float(row["entry_price"])
        shares      = float(row["shares"])
        date_open   = str(row["date_open"])
        vix_at_entry = float(row.get("vix_at_entry", 0) or 0)

        open_date = datetime.datetime.strptime(date_open, "%Y-%m-%d").date()
        hold_days = (datetime.date.today() - open_date).days

        try:
            # Fetch current price
            if ticker not in price_cache:
                raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    price_cache[ticker] = float(raw[("Close", ticker)].dropna().iloc[-1])
                else:
                    price_cache[ticker] = float(raw["Close"].dropna().iloc[-1])

            current_price = price_cache[ticker]
            exit_reason   = ""

            # Exit 1: take profit
            tp_target = entry_price * (1.0 + SHOCK_TAKE_PROFIT_PCT)
            if current_price >= tp_target:
                exit_reason = "take_profit"

            # Exit 2: stop loss
            if not exit_reason:
                sl_target = entry_price * (1.0 - SHOCK_STOP_LOSS_PCT)
                if current_price <= sl_target:
                    exit_reason = "stop_loss"

            # Exit 3: VIX recovery
            if not exit_reason and current_vix is not None and vix_at_entry > 0:
                vix_decline = (vix_at_entry - current_vix) / vix_at_entry
                if vix_decline >= SHOCK_EXIT_VIX_DECLINE:
                    exit_reason = "vix_recovery"

            # Exit 4: time stop
            if not exit_reason and hold_days >= SHOCK_MAX_HOLD:
                exit_reason = "time_stop"

            if not exit_reason:
                continue

            # Close the trade
            gross    = shares * (current_price - entry_price)
            slippage = SHOCK_SLIPPAGE_PCT * shares * (entry_price + current_price)
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

            state = _evaluate_shock_kill_switch(state)
            closed_today.append(trades_df.loc[idx].to_dict())

        except Exception as e:
            print(f"    [SHOCK-EXIT] Error checking {ticker}: {e}")
            continue

    state["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_shock_trades(trades_df)
    save_shock_state(state)

    return closed_today


# ------------------------------------------------------------------------------
#  Console Display
# ------------------------------------------------------------------------------

def print_shock_portfolio_status():
    open_trades = get_open_shock_trades()
    state       = load_shock_state()
    today       = datetime.date.today()

    print("  -- Shock Module Portfolio " + "-" * 35)
    print(f"    Open positions : {len(open_trades)} / {SHOCK_MAX_POSITIONS}")
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
              f"{'CAPITAL':>8} {'DROP%':>7} {'DAYS':>5}")
        print(f"    {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*5}")
        for t in open_trades:
            open_dt = datetime.datetime.strptime(t["date_open"], "%Y-%m-%d").date()
            days    = (today - open_dt).days
            drop    = t.get("shock_drop_pct", "0")
            print(f"    {t['ticker']:<8} {'LONG':>5} "
                  f"${float(t['entry_price']):>7.2f} "
                  f"${float(t['capital_deployed']):>7.2f} "
                  f"{float(drop):>6.1f}% "
                  f"{days:>5}")
    print("  " + "-" * 61)
    print()


def print_shock_performance_report():
    df = load_shock_trades()
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

    live_sharpe_str = "N/A"
    if len(pnls) >= 5:
        std_pnl       = pnls.std(ddof=1) if len(pnls) > 1 else 1e-12
        avg_hold_safe = max(avg_hold, 1.0) if not np.isnan(avg_hold) else 5.0
        live_sharpe   = (live_avg / std_pnl) * np.sqrt(252 / avg_hold_safe)
        live_sharpe_str = f"{live_sharpe:+.2f}"

    max_consec  = 0
    curr_streak = 0
    for p in pnls:
        if p <= 0:
            curr_streak += 1
            max_consec   = max(max_consec, curr_streak)
        else:
            curr_streak = 0

    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    max_dd  = float((cum_pnl - peak).min())

    # Exit reason breakdown
    exit_reasons = closed["exit_reason"].value_counts()

    if bt_wr_avg > 0 and live_wr >= bt_wr_avg * 0.8:
        verdict = "Edge holding"
    else:
        verdict = "Edge degrading -- review"

    print()
    print("  -- Shock Performance Report " + "-" * 33)
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
    print(f"    Exit breakdown:")
    for reason, count in exit_reasons.items():
        print(f"      {reason:<15} : {count}")
    print(f"    Verdict            : {verdict}")
    print("  " + "-" * 61)
    print()
