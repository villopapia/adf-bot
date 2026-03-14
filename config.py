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

# Kelly position sizing
KELLY_FRACTION     = 0.5           # fraction of full Kelly (0.5 = half-Kelly)
KELLY_MIN_SCALE    = 0.5           # minimum position as fraction of CAPITAL_PER_TRADE
KELLY_MAX_SCALE    = 2.0           # maximum position as fraction of CAPITAL_PER_TRADE

# ── Backtest Quality Gates ──────────────────────────────────────────────────
MIN_WIN_RATE       = 50.0          # minimum win rate (%)
MIN_PROFIT_FACTOR  = 1.3           # minimum profit factor
MIN_TOTAL_PNL      = 50.0          # minimum total P&L ($) over full backtest
MIN_SHARPE         = 0.5           # minimum annualised Sharpe ratio
MAX_DRAWDOWN       = -200.0        # maximum allowable peak-to-trough drawdown ($)
MIN_TRADES         = 10            # minimum backtest trades for statistical validity
BETA_STABILITY_MAX = 0.60          # max hedge-ratio coefficient of variation (std/|mean|)
MIN_SORTINO        = 0.5           # minimum annualised Sortino ratio
MIN_AVG_PNL        = 5.0           # minimum average net P&L per trade ($)
RECENT_CORR_WINDOW = 90            # days for recent correlation check
MIN_RECENT_CORR    = 0.65          # minimum recent correlation required (Gate D)

# ── Scanner ─────────────────────────────────────────────────────────────────
SCANNER_INTERVAL_DAYS = 3          # days between auto-refresh
CORRELATION_THRESHOLD = 0.80       # pre-filter correlation
TOP_N_PAIRS        = 150           # pairs to export
ENABLE_NASDAQ100   = True          # include Nasdaq-100 tickers in universe
ENABLE_HISTORICAL_CONSTITUENTS  = True   # include past S&P 500 members in universe
HISTORICAL_SECTOR_YFINANCE_MAX  = 20     # max yfinance fast_info lookups per scan run

# ── Scanner — Half-Life Filter ─────────────────────────────────────────────
MAX_HALF_LIFE      = 30            # reject pairs with half-life > this (days)
                                   # should match or be <= MAX_HOLD

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
RECENT_ADF_PVAL      = 0.20        # max ADF p-value on recent spread
RECENT_MOMENTUM_N    = 5           # how many recent trades to check
                                   # reject if combined P&L of last N <= $0
VIX_MAX_ENTRY        = 25.0        # block new entries when VIX exceeds this level
WALK_FORWARD_SPLIT   = 0.70        # train/test split for walk-forward validation (70/30)

# ── External APIs ────────────────────────────────────────────────────────────
# WARNING: do not commit real keys to a public repo — use a secrets.py instead
AV_API_KEY         = "KCUX8R91HIPN7296"  # Alpha Vantage key (DAILY_ADJUSTED needs paid plan)
AV_RATE_DELAY      = 12.0               # secs between AV calls (12 = free 5/min; 0.8 = paid)
FMP_API_KEY        = "VRe6WPJFJWPY7TZ2kz99DB8blWgx1YyR"  # Financial Modeling Prep key
EARNINGS_BLACKOUT_DAYS = 7             # block entry if earnings within this many days (±)

# ── Live Trade Tracking ───────────────────────────────────────────────────────
LIVE_TRADES_CSV          = "live_trades.csv"   # paper trade journal
SYSTEM_STATE_JSON        = "system_state.json" # running P&L, kill switch
MAX_CONCURRENT_POSITIONS = 10                  # max open pairs at any time
BORROW_COST_PCT          = 0.005               # annual short-leg borrow cost (0.5%)
MAX_CONSECUTIVE_LOSSES   = 5                   # kill switch: N live losses in a row
MAX_SYSTEM_DRAWDOWN      = -1500.0             # kill switch: peak-to-trough live P&L ($)
PORTFOLIO_CORR_MAX       = 0.7                 # block new entry if any leg corr > threshold with open

# ── File Paths ──────────────────────────────────────────────────────────────
LOG_DIR            = "logs"
SYSTEM_LOG         = "system_log.csv"
ERROR_LOG          = "error.log"
LOG_RETENTION_DAYS = 30             # delete .log files older than this many days
