"""
═══════════════════════════════════════════════════════════════════════════════
 config.py — Centralized Configuration for Pairs Trading System
═══════════════════════════════════════════════════════════════════════════════
 Single source of truth for all strategy parameters, quality gates,
 file paths, and system settings.  Imported by master_signal.py,
 run_system.py, and nightly_scanner.py.
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Data Source ──────────────────────────────────────────────────────────────
INPUT_CSV          = "daily_candidates.csv"
LOOKBACK_YEARS     = 2

# ── Rolling Windows ─────────────────────────────────────────────────────────
ROLLING_BETA_WIN   = 60            # OLS regression window (days)
ROLLING_Z_WIN      = 20            # z-score lookback (days)

# ── Entry / Exit Rules ──────────────────────────────────────────────────────
Z_ENTRY            = 2.0           # entry threshold (both sides)
Z_EXIT             = 0.0           # mean-reversion target
Z_STOP             = 3.0           # stop-loss threshold
MAX_HOLD           = 30            # max holding period (days)

# ── Capital & Costs ─────────────────────────────────────────────────────────
CAPITAL_PER_TRADE  = 1000.0        # dollars per trade
SLIPPAGE_PCT       = 0.001         # 0.10% per leg

# ── Backtest Quality Gates ──────────────────────────────────────────────────
MIN_WIN_RATE       = 55.0          # minimum win rate (%)
MIN_PROFIT_FACTOR  = 1.3           # minimum profit factor
MIN_TOTAL_PNL      = 0.0           # minimum total P&L ($)
MIN_SHARPE         = 0.5           # minimum annualised Sharpe ratio
MAX_DRAWDOWN       = -200.0        # maximum allowable peak-to-trough drawdown ($)
MIN_TRADES         = 10            # minimum backtest trades for statistical validity
BETA_STABILITY_MAX = 0.30          # max hedge-ratio coefficient of variation (std/|mean|)
RECENT_CORR_WINDOW = 90            # days for recent correlation check
MIN_RECENT_CORR    = 0.65          # minimum recent correlation required (Gate D)

# ── Scanner ─────────────────────────────────────────────────────────────────
SCANNER_INTERVAL_DAYS = 7          # days between auto-refresh
CORRELATION_THRESHOLD = 0.80       # pre-filter correlation
TOP_N_PAIRS        = 100            # pairs to export

# ── Scanner — Advanced ──────────────────────────────────────────────────────
MISSING_THRESHOLD     = 0.10        # drop tickers with > 10% NaN values
COINT_PVAL_THRESH     = 0.05        # Engle-Granger cointegration gate
SCANNER_BATCH_SIZE    = 50          # tickers per yfinance batch call

# ── Retry / Resilience ─────────────────────────────────────────────────────
MAX_RETRIES        = 3             # max yfinance download attempts
RETRY_DELAY        = 2             # seconds between retries

# ── Forward-Robustness Filters ──────────────────────────────────────────────
# These reduce "past ≠ future" risk by detecting pairs that are already
# showing signs of regime change or deteriorating cointegration.
SPLIT_HALF_ENABLED   = True        # require BOTH halves profitable independently
SPLIT_HALF_MIN_PNL   = 20.0        # each half must clear this P&L floor ($), not just > $0
RECENT_ADF_WINDOW    = 90          # days for recent cointegration check
RECENT_ADF_PVAL      = 0.10        # max ADF p-value on recent spread
RECENT_MOMENTUM_N    = 5           # how many recent trades to check
                                   # reject if combined P&L of last N <= $0
VIX_MAX_ENTRY        = 25.0        # block new entries when VIX exceeds this level

# ── File Paths ──────────────────────────────────────────────────────────────
LOG_DIR            = "logs"
SYSTEM_LOG         = "system_log.csv"
ERROR_LOG          = "error.log"
