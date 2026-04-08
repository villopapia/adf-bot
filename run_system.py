"""
================================================================================
 RUN SYSTEM  —  Professional Daily Orchestration
================================================================================
 One script to run daily.  Handles everything:

   1. System health check (data source, pair count, logs)
   2. Auto-refreshes candidate pairs weekly (via nightly_scanner.py)
   3. Runs master_signal.py (live z-score + backtest verification)
   4. Logs ALL signals to system_log.csv  (persistent, append-only)
   5. Logs per-pair math errors to error.log
   6. Full console output mirrored to timestamped log file

 Usage
 -----
   python run_system.py              — normal daily run
   python run_system.py --scan       — force scanner refresh today
   python run_system.py --scan-only  — run scanner only, skip signals

 Output Files
 ------------
   logs/YYYY-MM-DD_HHMMSS.log  — full console capture (ANSI-stripped)
   system_log.csv               — every signal ever found (verified + rejected)
   error.log                    — per-pair crash/math errors

================================================================================
"""

import os, sys, re, csv, datetime, argparse, traceback, subprocess

# Enable ANSI escape codes on Windows 10+
if sys.platform == "win32":
    os.system("")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    INPUT_CSV, SCANNER_INTERVAL_DAYS,
    LOG_DIR, SYSTEM_LOG, ERROR_LOG,
    Z_ENTRY, Z_EXIT, Z_STOP, MAX_HOLD,
    ROLLING_BETA_WIN, ROLLING_Z_WIN,
    CAPITAL_PER_TRADE, SLIPPAGE_PCT,
    MIN_WIN_RATE, MIN_PROFIT_FACTOR,
    MAX_CONCURRENT_POSITIONS,
    LOG_RETENTION_DAYS,
    MOMENTUM_ENABLED, MOM_ACTIVATION_MODE,
    MOM_MAX_POSITIONS, MOM_VIX_MAX_ENTRY,
    VIX_MAX_ENTRY,
    BEAR_MODULE_ENABLED, BEAR_VIX_ACTIVATE,
    BEAR_MAX_POSITIONS,
    LIVE_TRADING_ENABLED, ALPACA_SYNC_ON_STARTUP,
)
import trade_tracker
import momentum_tracker
import bear_tracker
import excel_tracker

# ── Broker (lazy init to avoid import errors if alpaca-py not installed) ────
_broker = None
def _get_broker():
    global _broker
    if _broker is None:
        from alpaca_broker import AlpacaBroker
        _broker = AlpacaBroker()
    return _broker

# ── Resolved Paths ───────────────────────────────────────────────────────────
LOG_DIR_PATH    = os.path.join(SCRIPT_DIR, LOG_DIR)
SYSTEM_LOG_PATH = os.path.join(SCRIPT_DIR, SYSTEM_LOG)
ERROR_LOG_PATH  = os.path.join(SCRIPT_DIR, ERROR_LOG)
CANDIDATES_PATH = os.path.join(SCRIPT_DIR, INPUT_CSV)

# ── ANSI Colors ──────────────────────────────────────────────────────────────
G   = "\033[92m"    # bright green
R   = "\033[91m"    # bright red
Y   = "\033[93m"    # yellow
CY  = "\033[96m"    # cyan
W   = "\033[97m"    # white
B   = "\033[1m"     # bold
D   = "\033[2m"     # dim
RST = "\033[0m"     # reset


# ══════════════════════════════════════════════════════════════════════════════
#  TEE LOGGER — mirror stdout + stderr to log file (strip ANSI for file)
# ══════════════════════════════════════════════════════════════════════════════
class TeeWriter:
    """File-like object that mirrors writes to a terminal and a log file."""
    def __init__(self, terminal, log_fh, ansi_re):
        self.terminal = terminal
        self._log     = log_fh
        self._ansi_re = ansi_re

    def write(self, msg):
        try:
            self.terminal.write(msg)
        except UnicodeEncodeError:
            self.terminal.write(msg.encode(self.terminal.encoding or "utf-8",
                                           errors="replace").decode(
                                           self.terminal.encoding or "utf-8"))
        self._log.write(self._ansi_re.sub("", msg))

    def flush(self):
        self.terminal.flush()
        self._log.flush()


def _setup_tee(log_path: str):
    """Redirect stdout + stderr through a shared log file. Returns cleanup fn."""
    orig_out = sys.stdout
    orig_err = sys.stderr
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh   = open(log_path, "w", encoding="utf-8")
    ansi_re  = re.compile(r"\033\[[0-9;]*m")
    sys.stdout = TeeWriter(orig_out, log_fh, ansi_re)
    sys.stderr = TeeWriter(orig_err, log_fh, ansi_re)

    def cleanup():
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_fh.close()

    return cleanup


# ══════════════════════════════════════════════════════════════════════════════
#  LOG RETENTION — clean up old .log files
# ══════════════════════════════════════════════════════════════════════════════
def _cleanup_old_logs(log_dir: str, days: int = LOG_RETENTION_DAYS):
    """Delete .log files in log_dir older than `days` days."""
    if not os.path.isdir(log_dir):
        return
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    for fname in os.listdir(log_dir):
        if not fname.endswith(".log"):
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM LOG  (system_log.csv — persistent record of all signals)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_LOG_HEADERS = [
    "timestamp", "stock_a", "stock_b", "status", "direction",
    "z_score", "beta", "bt_trades", "bt_win_rate",
    "bt_profit_factor", "bt_total_pnl", "bt_sharpe",
    "bt_avg_hold", "reject_reason",
]


def init_system_log() -> bool:
    """Create system_log.csv with headers if missing.  Returns True if new."""
    if not os.path.exists(SYSTEM_LOG_PATH):
        with open(SYSTEM_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SYSTEM_LOG_HEADERS)
        return True
    return False


def append_system_log(result: dict):
    """Append every signal (Verified + Rejected) to system_log.csv."""
    ts = result.get("timestamp", "")
    with open(SYSTEM_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for v in result.get("verified", []):
            w.writerow([
                ts, v["a"], v["b"], "DIAMOND", v["live"]["direction"],
                f"{v['live']['z']:+.4f}", f"{v['live']['beta']:.4f}",
                v["bt"]["n_trades"], f"{v['bt']['win_rate']:.1f}",
                f"{v['bt']['profit_factor']:.2f}",
                f"{v['bt']['total_pnl']:+.2f}",
                f"{v['bt']['sharpe']:+.2f}",
                f"{v['bt']['avg_hold']:.1f}", "",
            ])
        for r in result.get("rejected", []):
            w.writerow([
                ts, r["a"], r["b"], "REJECTED", r["live"]["direction"],
                f"{r['live']['z']:+.4f}", f"{r['live']['beta']:.4f}",
                r["bt"]["n_trades"], f"{r['bt']['win_rate']:.1f}",
                f"{r['bt']['profit_factor']:.2f}",
                f"{r['bt']['total_pnl']:+.2f}",
                f"{r['bt']['sharpe']:+.2f}",
                f"{r['bt']['avg_hold']:.1f}",
                r["bt"].get("reason", ""),
            ])
        # Ensure data persists to disk (crash safety)
        f.flush()
        os.fsync(f.fileno())


# ══════════════════════════════════════════════════════════════════════════════
#  LOCKFILE  (prevents concurrent runs from racing on shared files)
# ══════════════════════════════════════════════════════════════════════════════
LOCK_PATH       = os.path.join(SCRIPT_DIR, ".lock")
MAX_LOCK_AGE_S  = 14400          # 4 hours — stale lock auto-cleanup


def _acquire_lock() -> bool:
    """PID-based lockfile.  Returns False if another instance is active."""
    if os.path.exists(LOCK_PATH):
        age = (datetime.datetime.now() - datetime.datetime.fromtimestamp(
            os.path.getmtime(LOCK_PATH))).total_seconds()
        if age < MAX_LOCK_AGE_S:
            try:
                with open(LOCK_PATH, "r") as f:
                    old_pid = f.read().strip()
                print(f"  {R}\u2717 Another instance may be running "
                      f"(PID {old_pid}, lock {age/60:.0f}m old). "
                      f"Delete .lock to override.{RST}")
            except Exception:
                pass
            return False
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))
    return True


def _release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  SCANNER CHECK
# ══════════════════════════════════════════════════════════════════════════════
def should_run_scanner(force: bool) -> bool:
    if force:
        return True
    if not os.path.exists(CANDIDATES_PATH):
        return True
    if SCANNER_INTERVAL_DAYS <= 0:
        return False
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(CANDIDATES_PATH))
    return (datetime.datetime.now() - mod_time).days >= SCANNER_INTERVAL_DAYS


def run_scanner() -> bool:
    print(f"\n  {CY}{B}═══ PHASE 1 — Refreshing Candidates (nightly_scanner) ═══{RST}\n")
    try:
        import importlib
        mod = importlib.import_module("nightly_scanner")
        importlib.reload(mod)
        mod.main()
        print(f"\n  {G}✓ Scanner completed.{RST}\n")
        return True
    except Exception as e:
        print(f"\n  {R}✗ Scanner failed: {e}{RST}")
        traceback.print_exc()
        print(f"  Continuing with existing candidates…\n")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def print_header(now: datetime.datetime):
    print()
    print(f"  {CY}{B}╔══════════════════════════════════════════════════════════════╗{RST}")
    print(f"  {CY}{B}║     PAIRS TRADING SYSTEM — Daily Orchestration              ║{RST}")
    print(f"  {CY}{B}║     {now.strftime('%Y-%m-%d %H:%M:%S')}                                        ║{RST}")
    print(f"  {CY}{B}╚══════════════════════════════════════════════════════════════╝{RST}")
    print()


def print_health(pairs_loaded: int, scanner_age: str,
                 data_ok: bool, log_new: bool):
    src = f"{G}● ONLINE{RST}" if data_ok else f"{R}● OFFLINE{RST}"
    p_s = f"{G}{pairs_loaded}{RST}" if pairs_loaded > 0 else f"{R}0{RST}"
    l_s = f"{Y}Created{RST}" if log_new else f"{G}Exists{RST}"

    print(f"  {B}{W}── System Health ──────────────────────────────{RST}")
    print(f"    Data Source      : {src}")
    print(f"    Pairs Loaded     : {p_s}")
    print(f"    Scanner Age      : {scanner_age}")
    print(f"    System Log       : {l_s}")
    print(f"    Error Log        : {ERROR_LOG}")
    print(f"    Strategy         : z_entry={Z_ENTRY} | z_exit={Z_EXIT} "
          f"| z_stop={Z_STOP} | hold<={MAX_HOLD}d")
    print(f"    Quality Gates    : WR>{MIN_WIN_RATE}% | PF>{MIN_PROFIT_FACTOR}x "
          f"| P&L>$0")
    print(f"    Regime Gates     : Split-Half | ADF-90d | Recent Momentum")
    print(f"  {D}{'─' * 50}{RST}")
    print()


def print_verified_table(verified: list):
    if not verified:
        print(f"  {Y}No diamond signals today.{RST}\n")
        return

    print(f"  {G}{B}┌────────────────────────────────────────────────────────────────────────┐{RST}")
    print(f"  {G}{B}│  ◆  DIAMOND SIGNALS — Regime-Verified Trade Candidates              │{RST}")
    print(f"  {G}{B}└────────────────────────────────────────────────────────────────────────┘{RST}")
    print()
    print(f"    {B}{'PAIR':<14s} {'DIR':>5s} {'Z-SCORE':>8s} {'WR%':>6s} "
          f"{'PF':>7s} {'P&L':>10s} {'SHARPE':>8s} {'HOLD':>6s}{RST}")
    print(f"    {D}{'─'*14} {'─'*5} {'─'*8} {'─'*6} {'─'*7} {'─'*10} "
          f"{'─'*8} {'─'*6}{RST}")

    for v in verified:
        pair = f"{v['a']}/{v['b']}"
        d    = v["live"]["direction"]
        z    = v["live"]["z"]
        wr   = v["bt"]["win_rate"]
        pf   = v["bt"]["profit_factor"]
        pnl  = v["bt"]["total_pnl"]
        sh   = v["bt"]["sharpe"]
        hold = v["bt"]["avg_hold"]

        dc = G if d == "LONG" else R
        pc = G if pnl > 0 else R

        print(f"  {G}★{RST} {pair:<14s} {dc}{d:>5s}{RST} {z:>+8.2f} "
              f"{wr:>5.1f}% {pf:>6.2f}x {pc}${pnl:>+8.2f}{RST} "
              f"{sh:>+7.2f} {hold:>5.1f}d")
    print()


def print_rejected_summary(rejected: list):
    if not rejected:
        return
    print(f"  {D}── Rejected ({len(rejected)} pairs: "
          f"had live signal, failed regime gates) ──{RST}")
    for r in rejected:
        z   = r["live"]["z"]
        d   = r["live"]["direction"]
        wr  = r["bt"]["win_rate"]
        pf  = r["bt"]["profit_factor"]
        pnl = r["bt"]["total_pnl"]
        print(f"    {R}✗{RST} {r['a']:>6}/{r['b']:<6}  {d:>5}  "
              f"z={z:+.2f}  WR={wr:.0f}%  PF={pf:.2f}x  P&L=${pnl:+.0f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  DESKTOP NOTIFICATION  (Windows balloon — best-effort, no extra deps)
# ══════════════════════════════════════════════════════════════════════════════
def _notify_diamonds(diamonds: list):
    """Fire a Windows balloon toast when diamond signals are found."""
    if not diamonds or sys.platform != "win32":
        return
    n     = len(diamonds)
    title = f"\u25c6 {n} Diamond Signal{'s' if n > 1 else ''} Found"
    pairs = ", ".join(
        f"{d['a']}/{d['b']} {d['live']['direction']}" for d in diamonds
    )
    body  = pairs[:250]   # Windows balloon has a ~256 char body limit
    # Use Windows Forms balloon via PowerShell — no external packages needed
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Information; "
        "$n.Visible = $true; "
        f"$n.ShowBalloonTip(10000, '{title}', '{body}', "
        "[System.Windows.Forms.ToolTipIcon]::Info); "
        "Start-Sleep -Seconds 11; "
        "$n.Dispose()"
    )
    try:
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden",
             "-NonInteractive", "-Command", ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass   # notifications are best-effort; never block the run


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Daily trading system runner")
    parser.add_argument("--scan", action="store_true",
                        help="Force scanner refresh today")
    parser.add_argument("--scan-only", action="store_true",
                        help="Run scanner only, skip master signal")
    args = parser.parse_args()

    now = datetime.datetime.now()

    # ── Acquire lock (prevent concurrent runs) ────────────────────────
    if not _acquire_lock():
        return

    # ── Set up dual logging ──────────────────────────────────────────────
    log_name = now.strftime("%Y-%m-%d_%H%M%S") + ".log"
    log_path = os.path.join(LOG_DIR_PATH, log_name)
    tee_cleanup = _setup_tee(log_path)

    # ── Clean up old log files ─────────────────────────────────────────────
    _cleanup_old_logs(LOG_DIR_PATH)

    # ── Clear screen ─────────────────────────────────────────────────────
    os.system("cls" if os.name == "nt" else "clear")

    # ── Header ───────────────────────────────────────────────────────────
    print_header(now)

    # ── Init system_log.csv ──────────────────────────────────────────────
    log_new = init_system_log()

    # ── Pre-flight checks ────────────────────────────────────────────────
    scanner_age  = "N/A"
    pairs_loaded = 0
    data_ok      = False

    if os.path.exists(CANDIDATES_PATH):
        import pandas as pd
        mod_time = datetime.datetime.fromtimestamp(
            os.path.getmtime(CANDIDATES_PATH))
        scanner_age = f"{(now - mod_time).days}d old"
        try:
            pairs_loaded = len(pd.read_csv(CANDIDATES_PATH))
            data_ok = pairs_loaded > 0
        except Exception:
            pass

    print_health(pairs_loaded, scanner_age, data_ok, log_new)

    # ── Broker Startup Sync ───────────────────────────────────────────────
    if LIVE_TRADING_ENABLED:
        print(f"  {CY}{B}--- Broker Sync ---{RST}\n")
        try:
            broker = _get_broker()
            if broker.is_active:
                acct = broker.get_account_summary()
                if acct:
                    print(f"    Alpaca account : ${acct['equity']:,.2f} equity  "
                          f"| ${acct['buying_power']:,.2f} buying power")
                    print(f"    Mode           : "
                          f"{'PAPER' if acct.get('paper') else 'LIVE'}")
                if ALPACA_SYNC_ON_STARTUP:
                    drift = broker.sync_positions()
                    if drift["synced"]:
                        print(f"    Position sync  : {G}OK{RST}")
                    else:
                        print(f"    Position sync  : {Y}DRIFT DETECTED{RST}")
                        for o in drift.get("orphan_alpaca", []):
                            print(f"      {Y}! Alpaca has {o} "
                                  f"-- no paper trade{RST}")
                        for m in drift.get("missing_alpaca", []):
                            print(f"      {Y}! Paper trade {m} "
                                  f"-- no Alpaca position{RST}")
                print()
        except Exception as e:
            print(f"  {R}Broker sync failed: {e}{RST}\n")

    # ── Phase 0: Portfolio Update ─────────────────────────────────────────
    print(f"  {CY}{B}═══ PHASE 0 — Portfolio Update ═══{RST}\n")
    newly_closed = trade_tracker.check_exits()
    if newly_closed:
        print(f"  {G}Positions closed today:{RST}")
        for t in newly_closed:
            pnl   = float(t["net_pnl"])
            pc    = G if pnl > 0 else R
            icon  = "+" if pnl > 0 else "-"
            print(f"    {pc}{icon}{RST} {t['stock_a']}/{t['stock_b']}  "
                  f"{t['direction']}  exit={t['exit_reason']}  "
                  f"hold={t['hold_days']}d  "
                  f"P&L={pc}${pnl:>+.2f}{RST}")
        print()
    else:
        print(f"  {D}No positions closed today.{RST}\n")

    # Broker: close Alpaca positions for closed pairs trades
    if LIVE_TRADING_ENABLED and newly_closed:
        broker = _get_broker()
        if broker.is_active:
            for t in newly_closed:
                res = broker.close_position_pairs(
                    t["trade_id"], t["stock_a"], t["stock_b"])
                sc = G + "OK" if res["status"] == "closed" else R + res["status"]
                print(f"    [BROKER] {t['stock_a']}/{t['stock_b']} "
                      f"close: {sc}{RST}")

    trade_tracker.print_portfolio_status()
    trade_tracker.print_performance_report()

    ks_active, ks_reason = trade_tracker.is_kill_switch_triggered()
    if ks_active:
        print(f"\n  {R}{B}!! KILL SWITCH ACTIVE — no new entries will be accepted.{RST}")
        print(f"  {R}   Reason : {ks_reason}{RST}")
        print(f"  {D}   To reset: call trade_tracker.reset_kill_switch(){RST}\n")
        if LIVE_TRADING_ENABLED:
            broker = _get_broker()
            if broker.is_active:
                liq = broker.liquidate_all()
                print(f"  {R}   [BROKER] Emergency liquidation: "
                      f"{liq['positions_closed']} positions closed{RST}")

    # ── Phase 0b: Momentum Portfolio Update ───────────────────────────────
    if MOMENTUM_ENABLED:
        print(f"  {CY}{B}--- PHASE 0b — Momentum Portfolio Update ---{RST}\n")
        mom_closed = momentum_tracker.check_mom_exits()
        if mom_closed:
            print(f"  {G}Momentum positions closed today:{RST}")
            for t in mom_closed:
                pnl = float(t.get("net_pnl", 0))
                pc  = G if pnl > 0 else R
                icon = "+" if pnl > 0 else "-"
                print(f"    {pc}{icon}{RST} {t['ticker']}  "
                      f"exit={t.get('exit_reason','')}  "
                      f"hold={t.get('hold_days','')}d  "
                      f"P&L={pc}${pnl:>+.2f}{RST}")
            print()
        else:
            print(f"  {D}No momentum positions closed today.{RST}\n")

        # Broker: close Alpaca positions for closed momentum trades
        if LIVE_TRADING_ENABLED and mom_closed:
            broker = _get_broker()
            if broker.is_active:
                for t in mom_closed:
                    res = broker.close_position_single(
                        t.get("trade_id", ""), t["ticker"])
                    sc = G + "OK" if res["status"] == "closed" else R + res["status"]
                    print(f"    [BROKER] {t['ticker']} close: {sc}{RST}")

        momentum_tracker.print_mom_portfolio_status()
        momentum_tracker.print_mom_performance_report()

        mom_ks, mom_ks_reason = momentum_tracker.is_mom_kill_switch()
        if mom_ks:
            print(f"\n  {R}{B}!! MOMENTUM KILL SWITCH ACTIVE{RST}")
            print(f"  {R}   Reason : {mom_ks_reason}{RST}")
            print(f"  {D}   To reset: momentum_tracker.reset_mom_kill_switch(){RST}\n")
            if LIVE_TRADING_ENABLED:
                broker = _get_broker()
                if broker.is_active:
                    liq = broker.liquidate_all()
                    print(f"  {R}   [BROKER] Emergency liquidation: "
                          f"{liq['positions_closed']} positions closed{RST}")

    # ── Phase 0c: Bear Portfolio Update ────────────────────────────────
    if BEAR_MODULE_ENABLED:
        print(f"  {CY}{B}--- PHASE 0c — Bear Portfolio Update ---{RST}\n")
        bear_closed = bear_tracker.check_bear_exits()
        if bear_closed:
            print(f"  {G}Bear positions closed today:{RST}")
            for t in bear_closed:
                pnl = float(t.get("net_pnl", 0))
                pc  = G if pnl > 0 else R
                icon = "+" if pnl > 0 else "-"
                print(f"    {pc}{icon}{RST} {t['ticker']}  "
                      f"[{t.get('module', '')}]  "
                      f"exit={t.get('exit_reason', '')}  "
                      f"hold={t.get('hold_days', '')}d  "
                      f"P&L={pc}${pnl:>+.2f}{RST}")
            print()
        else:
            print(f"  {D}No bear positions closed today.{RST}\n")

        # Broker: close Alpaca positions for closed bear trades
        if LIVE_TRADING_ENABLED and bear_closed:
            broker = _get_broker()
            if broker.is_active:
                for t in bear_closed:
                    res = broker.close_position_single(
                        t.get("trade_id", ""), t["ticker"])
                    sc = G + "OK" if res["status"] == "closed" else R + res["status"]
                    print(f"    [BROKER] {t['ticker']} close: {sc}{RST}")

        bear_tracker.print_bear_portfolio_status()
        bear_tracker.print_bear_performance_report()

        bear_ks, bear_ks_reason = bear_tracker.is_bear_kill_switch()
        if bear_ks:
            print(f"\n  {R}{B}!! BEAR KILL SWITCH ACTIVE{RST}")
            print(f"  {R}   Reason : {bear_ks_reason}{RST}")
            print(f"  {D}   To reset: bear_tracker.reset_bear_kill_switch(){RST}\n")
            if LIVE_TRADING_ENABLED:
                broker = _get_broker()
                if broker.is_active:
                    liq = broker.liquidate_all()
                    print(f"  {R}   [BROKER] Emergency liquidation: "
                          f"{liq['positions_closed']} positions closed{RST}")

    # ── Excel Tracker: update prices & process close requests ────────────
    try:
        excel_tracker.update_all()
    except Exception as e:
        print(f"  {Y}Excel tracker update failed: {e}{RST}\n")

    # ── Phase 1: Scanner ─────────────────────────────────────────────────
    if should_run_scanner(args.scan or args.scan_only):
        run_scanner()
        # Re-check after scan
        if os.path.exists(CANDIDATES_PATH):
            try:
                import pandas as pd
                pairs_loaded = len(pd.read_csv(CANDIDATES_PATH))
                data_ok = pairs_loaded > 0
            except Exception:
                pass
    else:
        print(f"  {D}Scanner skipped (candidates {scanner_age}, "
              f"refresh every {SCANNER_INTERVAL_DAYS}d). "
              f"Use --scan to force.{RST}\n")

    if args.scan_only:
        print(f"  {Y}--scan-only set. Stopping here.{RST}\n")
        tee_cleanup()
        _release_lock()
        return

    if not data_ok:
        print(f"  {R}✗ No candidates available. Run with --scan first.{RST}\n")
        tee_cleanup()
        _release_lock()
        return

    # ── Phase 2: Master Signal ───────────────────────────────────────────
    print(f"  {CY}{B}═══ PHASE 2 — Master Signal (live + backtest) ═══{RST}\n")

    try:
        import importlib
        mod = importlib.import_module("master_signal")
        importlib.reload(mod)
        result = mod.main()
    except Exception as e:
        print(f"\n  {R}✗ Master signal failed: {e}{RST}")
        traceback.print_exc()
        result = None

    # ── Phase 3: Log & Dashboard ─────────────────────────────────────────
    if result:
        # Append to system_log.csv
        try:
            append_system_log(result)
            n_v = len(result.get("verified", []))
            n_r = len(result.get("rejected", []))
            print(f"\n  {G}✓ system_log.csv updated "
                  f"({n_v} diamond, {n_r} rejected){RST}")
        except Exception as e:
            print(f"\n  {R}✗ system_log.csv write failed: {e}{RST}")

        # ── Summary stats ────────────────────────────────────────────────
        total = (result.get("no_signal", 0)
                 + len(result.get("verified", []))
                 + len(result.get("rejected", []))
                 + result.get("errors", 0))

        print()
        print(f"  {B}{W}{'═' * 60}{RST}")
        print(f"  {B}{W}  DAILY RESULTS  (Standard + Split-Half + ADF90 + Momentum){RST}")
        print(f"  {B}{W}{'═' * 60}{RST}")
        print(f"    Pairs scanned   : {total}")
        print(f"    No live signal  : {result.get('no_signal', 0)}")
        print(f"    Download errors : {result.get('errors', 0)}")
        print(f"    {G}◆ Diamond{RST}       : "
              f"{G}{B}{len(result.get('verified', []))}{RST}")
        print(f"    {R}✗ Rejected{RST}      : "
              f"{len(result.get('rejected', []))}")
        print()

        # ── Bright verified table ────────────────────────────────────────
        print_verified_table(result.get("verified", []))

        # ── Auto-log diamond entries (portfolio cap + kill switch aware) ──
        diamonds = result.get("verified", [])
        if diamonds:
            ks_active, ks_reason = trade_tracker.is_kill_switch_triggered()
            logged, skipped_cap, skipped_ks, skipped_corr = [], [], [], []
            for d in diamonds:
                if ks_active:
                    skipped_ks.append(d)
                elif trade_tracker.at_portfolio_cap():
                    skipped_cap.append(d)
                elif trade_tracker.is_correlated_with_open(d["a"], d["b"]):
                    skipped_corr.append(d)
                else:
                    tid = trade_tracker.log_entry(d)
                    try:
                        excel_tracker.log_pair_entry(d)
                    except Exception:
                        pass
                    logged.append((d, tid))
                    # Broker: submit pairs entry to Alpaca
                    if LIVE_TRADING_ENABLED:
                        broker = _get_broker()
                        if broker.is_active:
                            pt = [t for t in trade_tracker.get_open_trades()
                                  if t["trade_id"] == tid]
                            if pt:
                                broker.submit_entry_pairs(tid, {
                                    "a": d["a"], "b": d["b"],
                                    "direction": d["live"]["direction"],
                                    "shares_a": float(pt[0]["shares_a"]),
                                    "shares_b": float(pt[0]["shares_b"]),
                                })
            if logged:
                print(f"  {G}Paper trades logged ({len(logged)}):{RST}")
                for d, tid in logged:
                    print(f"    {G}+{RST} {d['a']}/{d['b']}  "
                          f"{d['live']['direction']}  [{tid}]")
            if skipped_cap:
                print(f"  {Y}Skipped {len(skipped_cap)} signal(s) — "
                      f"portfolio at cap ({MAX_CONCURRENT_POSITIONS} positions){RST}")
            if skipped_ks:
                print(f"  {R}Skipped {len(skipped_ks)} signal(s) — "
                      f"kill switch active{RST}")
            if skipped_corr:
                print(f"  {Y}Skipped {len(skipped_corr)} signal(s) — "
                      f"correlated with open position(s){RST}")
                for d in skipped_corr:
                    print(f"    {Y}-{RST} {d['a']}/{d['b']}  "
                          f"{d['live']['direction']}")
            print()

        # ── Rejected summary ─────────────────────────────────────────────
        print_rejected_summary(result.get("rejected", []))

        # ── Desktop notification ──────────────────────────────────────────
        _notify_diamonds(result.get("verified", []))
    else:
        print(f"\n  {R}System encountered errors. "
              f"Check {ERROR_LOG} and log file.{RST}\n")

    # ── Phase 3: Momentum (conditional) ──────────────────────────────────
    if MOMENTUM_ENABLED:
        vix_level = result.get("vix") if result else None
        n_diamonds = len(result.get("verified", [])) if result else 0

        # Activation logic
        activate = False
        if MOM_ACTIVATION_MODE == "always":
            activate = True
        elif MOM_ACTIVATION_MODE == "vix_only":
            activate = (vix_level is not None and vix_level > VIX_MAX_ENTRY)
        else:  # "complement" — pairs idle or VIX elevated
            pairs_idle = (n_diamonds == 0)
            vix_elevated = (vix_level is not None and vix_level > VIX_MAX_ENTRY)
            activate = pairs_idle or vix_elevated

        if activate:
            print(f"  {CY}{B}--- PHASE 3 — Momentum Strategy ---{RST}\n")
            if vix_level is not None:
                print(f"  {D}VIX: {vix_level:.1f}  |  "
                      f"Pairs diamonds: {n_diamonds}  |  "
                      f"Mode: {MOM_ACTIVATION_MODE}{RST}\n")

            try:
                import importlib

                # Step 1: Scan for momentum candidates
                scanner_mod = importlib.import_module("momentum_scanner")
                importlib.reload(scanner_mod)
                candidates_df = scanner_mod.main(vix_level=vix_level)

                if candidates_df is not None and len(candidates_df) > 0:
                    # Get market regime from scanner
                    regime = scanner_mod.check_market_regime()

                    # Step 2: Run momentum signal engine
                    signal_mod = importlib.import_module("momentum_signal")
                    importlib.reload(signal_mod)
                    mom_result = signal_mod.main(
                        candidates_df=candidates_df,
                        vix_level=vix_level,
                        market_regime=regime,
                    )

                    if mom_result:
                        vix_scale = scanner_mod.get_vix_scale(vix_level)
                        mom_diamonds = mom_result.get("verified", [])
                        mom_rejected = mom_result.get("rejected", [])

                        # Summary
                        print()
                        print(f"  {B}{W}{'=' * 60}{RST}")
                        print(f"  {B}{W}  MOMENTUM RESULTS{RST}")
                        print(f"  {B}{W}{'=' * 60}{RST}")
                        print(f"    Candidates      : {len(candidates_df)}")
                        print(f"    No live signal  : {mom_result.get('no_signal', 0)}")
                        print(f"    Errors          : {mom_result.get('errors', 0)}")
                        print(f"    {G}* Mom Diamond{RST}   : "
                              f"{G}{B}{len(mom_diamonds)}{RST}")
                        print(f"    {R}x Rejected{RST}      : {len(mom_rejected)}")
                        print()

                        # Momentum diamond table
                        if mom_diamonds:
                            print(f"  {G}{B}-- MOMENTUM DIAMONDS --{RST}")
                            print(f"    {B}{'TICKER':<8} {'MOM':>8} {'WR%':>6} "
                                  f"{'PF':>7} {'P&L':>10} {'SHARPE':>8}{RST}")
                            print(f"    {D}{'-'*8} {'-'*8} {'-'*6} "
                                  f"{'-'*7} {'-'*10} {'-'*8}{RST}")
                            for v in mom_diamonds:
                                bt = v["bt"]
                                live = v["live"]
                                pnl = bt["total_pnl"]
                                pc = G if pnl > 0 else R
                                print(f"  {G}*{RST} {v['ticker']:<8} "
                                      f"{live['mom_score']:>+8.3f} "
                                      f"{bt['win_rate']:>5.1f}% "
                                      f"{bt['profit_factor']:>6.2f}x "
                                      f"{pc}${pnl:>+8.2f}{RST} "
                                      f"{bt['sharpe']:>+7.2f}")
                            print()

                        # Auto-log momentum entries
                        if mom_diamonds:
                            mom_ks, _ = momentum_tracker.is_mom_kill_switch()
                            m_logged, m_skip_cap, m_skip_ks, m_skip_corr = [], [], [], []
                            for d in mom_diamonds:
                                if mom_ks:
                                    m_skip_ks.append(d)
                                elif momentum_tracker.at_mom_cap():
                                    m_skip_cap.append(d)
                                elif momentum_tracker.is_mom_correlated_with_open(
                                        d["ticker"]):
                                    m_skip_corr.append(d)
                                else:
                                    tid = momentum_tracker.log_mom_entry(
                                        d, vix_scale=vix_scale)
                                    try:
                                        excel_tracker.log_momentum_entry(
                                            d, vix_scale=vix_scale)
                                    except Exception:
                                        pass
                                    m_logged.append((d, tid))
                                    # Broker: submit momentum entry
                                    if LIVE_TRADING_ENABLED:
                                        broker = _get_broker()
                                        if broker.is_active:
                                            pt = [t for t in
                                                  momentum_tracker.get_open_mom_trades()
                                                  if t["trade_id"] == tid]
                                            if pt:
                                                broker.submit_entry_single(
                                                    tid, d["ticker"],
                                                    float(pt[0]["shares"]),
                                                    "buy", "momentum")
                            if m_logged:
                                print(f"  {G}Momentum trades logged "
                                      f"({len(m_logged)}):{RST}")
                                for d, tid in m_logged:
                                    print(f"    {G}+{RST} {d['ticker']}  "
                                          f"LONG  [{tid}]")
                            if m_skip_cap:
                                print(f"  {Y}Skipped {len(m_skip_cap)} — "
                                      f"momentum at cap "
                                      f"({MOM_MAX_POSITIONS} positions){RST}")
                            if m_skip_ks:
                                print(f"  {R}Skipped {len(m_skip_ks)} — "
                                      f"momentum kill switch active{RST}")
                            if m_skip_corr:
                                print(f"  {Y}Skipped {len(m_skip_corr)} — "
                                      f"correlated with open position(s){RST}")
                            print()

                        # Rejected summary
                        if mom_rejected:
                            print(f"  {D}-- Momentum Rejected "
                                  f"({len(mom_rejected)}) --{RST}")
                            for r in mom_rejected[:5]:
                                print(f"    {R}x{RST} {r['ticker']:>6}  "
                                      f"{r['bt'].get('reason', '')}")
                            if len(mom_rejected) > 5:
                                print(f"    {D}... and "
                                      f"{len(mom_rejected) - 5} more{RST}")
                            print()
                else:
                    print(f"  {Y}No momentum candidates passed filters.{RST}\n")

            except Exception as e:
                print(f"\n  {R}Momentum phase failed: {e}{RST}")
                traceback.print_exc()
                print()
        else:
            print(f"\n  {D}Momentum skipped (pairs active, VIX normal).{RST}\n")

    # ── Phase 4: Bear Module (conditional) ────────────────────────────────
    if BEAR_MODULE_ENABLED:
        vix_level = result.get("vix") if result else None

        if vix_level is not None and vix_level > BEAR_VIX_ACTIVATE:
            print(f"  {CY}{B}--- PHASE 4 — Bear Market Module ---{RST}\n")
            print(f"  {D}VIX: {vix_level:.1f} > {BEAR_VIX_ACTIVATE} "
                  f"-- bear module active{RST}\n")

            try:
                import importlib
                bear_mod = importlib.import_module("bear_signal")
                importlib.reload(bear_mod)
                bear_result = bear_mod.main(vix_level=vix_level)

                if bear_result:
                    # Summary
                    n_bounce = len(bear_result.get("bounce_verified", []))
                    n_short  = len(bear_result.get("short_verified", []))
                    n_b_rej  = len(bear_result.get("bounce_rejected", []))
                    n_s_rej  = len(bear_result.get("short_rejected", []))

                    print()
                    print(f"  {B}{W}{'=' * 60}{RST}")
                    print(f"  {B}{W}  BEAR MODULE RESULTS{RST}")
                    print(f"  {B}{W}{'=' * 60}{RST}")
                    print(f"    Bounce diamonds : {G}{B}{n_bounce}{RST}")
                    print(f"    Bounce rejected : {n_b_rej}")
                    print(f"    Short diamonds  : {G}{B}{n_short}{RST}")
                    print(f"    Short rejected  : {n_s_rej}")
                    if bear_result.get("capitulation"):
                        print(f"    {Y}Capitulation boost ACTIVE{RST}")
                    print()

                    # Auto-log bounce diamonds
                    for d in bear_result.get("bounce_verified", []):
                        bear_ks, _ = bear_tracker.is_bear_kill_switch()
                        if bear_ks:
                            print(f"  {R}Skipped bounce -- "
                                  f"bear kill switch active{RST}")
                            break
                        if bear_tracker.at_bear_cap():
                            print(f"  {Y}Skipped bounce -- "
                                  f"bear at cap "
                                  f"({BEAR_MAX_POSITIONS} positions){RST}")
                            break
                        regime = bear_result["regime"]
                        tid = bear_tracker.log_bear_entry(
                            d, module="bounce",
                            vix_scale=regime["vix_scale"],
                            capitulation=bear_result.get(
                                "capitulation", False))
                        try:
                            excel_tracker.log_bear_entry(
                                d, module="bounce",
                                vix_scale=regime["vix_scale"])
                        except Exception:
                            pass
                        print(f"  {G}+ Bounce trade logged: "
                              f"{d['ticker']}  [{tid}]{RST}")
                        # Broker: submit bounce entry
                        if LIVE_TRADING_ENABLED:
                            broker = _get_broker()
                            if broker.is_active:
                                pt = [t for t in
                                      bear_tracker.get_open_bear_trades()
                                      if t["trade_id"] == tid]
                                if pt:
                                    broker.submit_entry_single(
                                        tid, d["ticker"],
                                        float(pt[0]["shares"]),
                                        "buy", "bear")

                    # Auto-log short diamonds
                    for d in bear_result.get("short_verified", []):
                        bear_ks, _ = bear_tracker.is_bear_kill_switch()
                        if bear_ks:
                            print(f"  {R}Skipped short -- "
                                  f"bear kill switch active{RST}")
                            break
                        if bear_tracker.at_bear_cap():
                            print(f"  {Y}Skipped short -- "
                                  f"bear at cap "
                                  f"({BEAR_MAX_POSITIONS} positions){RST}")
                            break
                        regime = bear_result["regime"]
                        tid = bear_tracker.log_bear_entry(
                            d, module="short",
                            vix_scale=regime["vix_scale"])
                        try:
                            excel_tracker.log_bear_entry(
                                d, module="short",
                                vix_scale=regime["vix_scale"])
                        except Exception:
                            pass
                        print(f"  {G}+ Short trade logged: "
                              f"{d['ticker']}  [{tid}]{RST}")
                        # Broker: submit short entry
                        if LIVE_TRADING_ENABLED:
                            broker = _get_broker()
                            if broker.is_active:
                                pt = [t for t in
                                      bear_tracker.get_open_bear_trades()
                                      if t["trade_id"] == tid]
                                if pt:
                                    broker.submit_entry_single(
                                        tid, d["ticker"],
                                        float(pt[0]["shares"]),
                                        "buy", "bear")

                    # Rejected summary
                    all_rejected = (bear_result.get("bounce_rejected", [])
                                    + bear_result.get("short_rejected", []))
                    if all_rejected:
                        print(f"\n  {D}-- Bear Rejected "
                              f"({len(all_rejected)}) --{RST}")
                        for r in all_rejected:
                            print(f"    {R}x{RST} {r['ticker']}  "
                                  f"{r['bt'].get('reason', '')}")
                    print()

            except Exception as e:
                print(f"\n  {R}Bear module failed: {e}{RST}")
                import traceback
                traceback.print_exc()
                print()
        else:
            vix_display = (f"{vix_level:.1f}" if vix_level is not None
                           else "N/A")
            print(f"\n  {D}Bear module skipped "
                  f"(VIX {vix_display} <= {BEAR_VIX_ACTIVATE}).{RST}\n")

    # ── Footer ───────────────────────────────────────────────────────────
    print(f"  {D}Log     → {log_path}{RST}")
    print(f"  {D}Journal → {SYSTEM_LOG_PATH}{RST}")
    print(f"  {D}Errors  → {ERROR_LOG_PATH}{RST}")
    print()

    # ── Restore stdout/stderr ───────────────────────────────────────────
    tee_cleanup()
    _release_lock()


if __name__ == "__main__":
    main()
