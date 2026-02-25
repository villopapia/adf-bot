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
)

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
        self.terminal.write(msg)
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

        # ── Rejected summary ─────────────────────────────────────────────
        print_rejected_summary(result.get("rejected", []))

        # ── Desktop notification ──────────────────────────────────────────
        _notify_diamonds(result.get("verified", []))
    else:
        print(f"\n  {R}System encountered errors. "
              f"Check {ERROR_LOG} and log file.{RST}\n")

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
