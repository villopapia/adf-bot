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
LOOKBACK_YEARS     = 5

# ── Rolling Windows ─────────────────────────────────────────────────────────
ROLLING_BETA_WIN   = 60            # OLS regression window (days)
ROLLING_Z_WIN      = 20            # z-score lookback (days)

# ── Entry / Exit Rules ──────────────────────────────────────────────────────
Z_ENTRY            = 2.0           # entry threshold (both sides)
Z_EXIT             = 0.0           # mean-reversion target
Z_STOP             = 3.0           # stop-loss threshold
MAX_HOLD           = 20            # max holding period (days)

# ── Capital & Costs ─────────────────────────────────────────────────────────
CAPITAL_PER_TRADE  = 1000.0        # dollars per trade
SLIPPAGE_PCT       = 0.0015        # 0.15% round-trip for S&P 500 pairs

# Kelly position sizing
KELLY_FRACTION     = 0.5           # fraction of full Kelly (0.5 = half-Kelly)
KELLY_MIN_SCALE    = 0.5           # minimum position as fraction of CAPITAL_PER_TRADE
KELLY_MAX_SCALE    = 2.0           # maximum position as fraction of CAPITAL_PER_TRADE

# ── Backtest Quality Gates ──────────────────────────────────────────────────
MIN_WIN_RATE       = 55.0          # minimum win rate (%)
MIN_PROFIT_FACTOR  = 1.5           # minimum profit factor
MIN_TOTAL_PNL      = 50.0          # minimum total P&L ($) over full backtest
MIN_SHARPE         = 1.0           # minimum annualised Sharpe ratio
MAX_DRAWDOWN       = -200.0        # maximum allowable peak-to-trough drawdown ($)
MIN_TRADES         = 10            # minimum backtest trades for statistical validity
BETA_STABILITY_MAX = 0.60          # max hedge-ratio coefficient of variation (std/|mean|)
MIN_SORTINO        = 1.0           # minimum annualised Sortino ratio
MIN_AVG_PNL        = 8.0           # minimum average net P&L per trade ($)
MAX_SHARPE_OVERFIT      = 4.0       # reject if backtest Sharpe suspiciously high (overfitting)
MAX_PROFIT_FACTOR_OVERFIT = 5.0     # reject if PF suspiciously high (overfitting)
OOS_SHARPE_RATIO_MIN    = 0.25      # OOS Sharpe must be >= 25% of IS Sharpe
RECENT_CORR_WINDOW = 90            # days for recent correlation check
MIN_RECENT_CORR    = 0.70          # minimum recent correlation required (Gate D)
FDR_ALPHA              = 0.05       # Benjamini-Hochberg false discovery rate threshold

# ── Spread Volatility-Adjusted Sizing ──────────────────────────────────────
VOL_TARGET_DAILY        = 0.02      # target 2% daily volatility on spread P&L
VOL_LOOKBACK_DAYS       = 60        # window for realized spread volatility
VOL_SIZE_FLOOR          = 0.3       # never go below 30% of CAPITAL_PER_TRADE
VOL_SIZE_CAP            = 2.0       # never go above 200% of CAPITAL_PER_TRADE

# ── Scanner ─────────────────────────────────────────────────────────────────
SCANNER_INTERVAL_DAYS = 3          # days between auto-refresh
CORRELATION_THRESHOLD = 0.80       # pre-filter correlation
TOP_N_PAIRS        = 150           # pairs to export
ENABLE_NASDAQ100   = True          # include Nasdaq-100 tickers in universe
ENABLE_HISTORICAL_CONSTITUENTS  = True   # include past S&P 500 members in universe
HISTORICAL_SECTOR_YFINANCE_MAX  = 20     # max yfinance fast_info lookups per scan run

# ── Scanner — Half-Life Filter ─────────────────────────────────────────────
MAX_HALF_LIFE      = 20            # reject pairs with half-life > this (days)
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
SPLIT_HALF_MIN_PNL   = 30.0        # each half must clear this P&L floor ($), not just > $0
RECENT_ADF_WINDOW    = 90          # days for recent cointegration check
RECENT_ADF_PVAL      = 0.10        # max ADF p-value on recent spread
RECENT_MOMENTUM_N    = 5           # how many recent trades to check
                                   # reject if combined P&L of last N <= $0
VIX_MAX_ENTRY        = 25.0        # block new entries when VIX exceeds this level
WALK_FORWARD_SPLIT   = 0.70        # train/test split for walk-forward validation (70/30)
MIN_OOS_TRADES     = 5              # minimum OOS trades for walk-forward validity

# ── External APIs ────────────────────────────────────────────────────────────
# WARNING: do not commit real keys to a public repo — use a secrets.py instead
AV_API_KEY         = "KCUX8R91HIPN7296"  # Alpha Vantage key (DAILY_ADJUSTED needs paid plan)
AV_RATE_DELAY      = 12.0               # secs between AV calls (12 = free 5/min; 0.8 = paid)
FMP_API_KEY        = "VRe6WPJFJWPY7TZ2kz99DB8blWgx1YyR"  # Financial Modeling Prep key
EARNINGS_BLACKOUT_DAYS = 14            # block entry if earnings within this many days (±)

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

# ── Momentum Strategy ─────────────────────────────────────────────────────
# Activates when pairs trading is idle (no diamonds or VIX > 25)
MOMENTUM_ENABLED         = True          # master switch for momentum module
MOMENTUM_CANDIDATES_CSV  = "momentum_candidates.csv"  # scanner output

# ── Momentum — Formation & Holding Periods ─────────────────────────────────
MOM_FORMATION_DAYS       = 126           # 6-month lookback (trading days)
MOM_SKIP_DAYS            = 21            # skip most recent month (avoid reversal)
MOM_HOLDING_DAYS         = 21            # 1-month holding period
MOM_MIN_PRICE            = 10.0          # exclude penny stocks (< $10)
MOM_MIN_AVG_VOLUME       = 500_000       # min 500K shares avg daily volume (20d)

# ── Momentum — Entry Signals ──────────────────────────────────────────────
MOM_TOP_PCT              = 10            # top decile of 12-1 returns
MOM_52W_HIGH_PCT         = 0.90          # within 10% of 52-week high
MOM_VOLUME_RATIO_MIN     = 1.2           # recent volume >= 1.2x 50-day avg
MOM_RSI_MIN              = 50            # RSI(14) must be > 50 (uptrend)
MOM_RSI_MAX              = 80            # RSI(14) < 80 (not overbought)
MOM_ADX_MIN              = 20            # ADX(14) > 20 (trending, not range-bound)

# ── Momentum — Exit Rules ─────────────────────────────────────────────────
MOM_TRAILING_STOP_PCT    = 0.10          # 10% trailing stop from peak
MOM_TIME_STOP_DAYS       = 42            # max 2 months holding (forced exit)
MOM_TAKE_PROFIT_PCT      = 0.25          # take profit at +25%

# ── Momentum — Capital & Costs ────────────────────────────────────────────
MOM_CAPITAL_PER_TRADE    = 1000.0        # same as pairs for consistency
MOM_SLIPPAGE_PCT         = 0.0015        # 0.15% per trade
MOM_MAX_POSITIONS        = 5             # max concurrent momentum positions

# ── Momentum — Backtest Quality Gates ─────────────────────────────────────
MOM_MIN_TRADES           = 15            # minimum backtest trades
MOM_MIN_WIN_RATE         = 48.0          # lower than pairs (momentum wins fewer, bigger)
MOM_MIN_PROFIT_FACTOR    = 1.4           # lower than pairs
MOM_MIN_SHARPE           = 0.6           # annualised Sharpe on backtest
MOM_MIN_TOTAL_PNL        = 50.0          # minimum total P&L ($)
MOM_MAX_MOM_DRAWDOWN     = -300.0        # maximum drawdown ($)
MOM_MIN_AVG_GAIN_LOSS    = 1.5           # avg win / avg loss ratio > 1.5
MOM_WALK_FORWARD_SPLIT   = 0.70          # 70/30 train/test

# ── Momentum — Regime Filter ──────────────────────────────────────────────
MOM_VIX_MAX_ENTRY        = 40.0          # block when VIX > 40 (extreme panic)
MOM_VIX_SCALE_THRESHOLD  = 25.0          # above this VIX, reduce position size
MOM_VIX_SCALE_FACTOR     = 0.5           # multiply capital by this when VIX elevated

# ── Momentum — Activation Logic ──────────────────────────────────────────
MOM_ACTIVATION_MODE      = "complement"  # "complement" = only when pairs idle
                                          # "always" = run independently
                                          # "vix_only" = only when VIX > 25

# ── Momentum — News/Catalyst Detection ────────────────────────────────────
MOM_EARNINGS_BOOST       = True          # boost rank if positive earnings surprise
MOM_EARNINGS_SURPRISE_MIN = 5.0          # min +5% EPS surprise to count as catalyst
MOM_EARNINGS_BLACKOUT_PRE  = 7           # block entry if earnings upcoming within N days
MOM_EARNINGS_BLACKOUT_POST = 5           # block entry if earnings reported within last N days (gap already priced in)
MOM_EARNINGS_BEAT_RATE_MIN   = 0.75     # min beat rate to allow pre-earnings entry (75%)
MOM_EARNINGS_MIN_QUARTERS    = 3        # min quarters of historical data required
MOM_EARNINGS_ALLOW_DECLINING = False    # if True, allow entry even with declining EPS trend

# ── Momentum — Scanner ────────────────────────────────────────────────────
MOM_SCANNER_INTERVAL_DAYS = 1            # rescan daily (momentum changes fast)

# ── Momentum — Risk Controls ─────────────────────────────────────────────
MOM_KILL_SWITCH_LOSSES   = 4             # kill after 4 consecutive losses
MOM_KILL_SWITCH_DRAWDOWN = -800.0        # kill at -$800 system drawdown
MOM_PORTFOLIO_CORR_MAX   = 0.80          # block if new stock > 0.80 corr with open

# ── Momentum — Live Trade Tracking ────────────────────────────────────────
MOM_LIVE_TRADES_CSV      = "momentum_trades.csv"
MOM_SYSTEM_STATE_JSON    = "momentum_state.json"

# -- Momentum -- Volatility-Targeted Sizing (Barroso & Santa-Clara 2015) ──
MOM_VOL_TARGET_ANN       = 0.12          # 12% annualized target vol
MOM_VOL_LOOKBACK_DAYS    = 126           # 6-month realized vol window
MOM_VOL_FLOOR            = 0.05          # never assume vol < 5% annualized

# -- Momentum -- Absolute Momentum Filter (Antonacci 2014) ────────────────
MOM_ABSOLUTE_FILTER      = True          # require trailing return > 0

# -- Momentum -- ATR-Based Stops ──────────────────────────────────────────
MOM_ATR_PERIOD           = 14            # ATR lookback
MOM_ATR_TRAIL_MULT       = 3.0           # trailing stop = 3x ATR(14)
MOM_ATR_TP_MULT          = 5.0           # take profit = 5x ATR(14)

# -- Momentum -- Blended Formation (adaptive to vol regime) ───────────────
MOM_FORMATION_SHORT      = 63            # 3-month lookback
MOM_FORMATION_LONG       = 126           # 6-month lookback
MOM_FORMATION_BLEND_VIX  = 25.0          # above this VIX: blend 50/50

# -- Momentum -- 200d SMA / Market Regime ─────────────────────────────────
MOM_REQUIRE_ABOVE_200SMA = True          # stock must be above 200d SMA
MOM_MARKET_REGIME_EMA    = True          # S&P 500: 50d EMA > 200d EMA

# -- Momentum -- Graduated VIX Scaling (Daniel & Moskowitz 2016) ──────────
MOM_VIX_TIERS            = {25: 1.0, 30: 0.75, 35: 0.50, 40: 0.0}

# -- Momentum -- Inverse Volatility Weighting ─────────────────────────────
MOM_INVERSE_VOL_WEIGHT   = True          # weight positions by 1/vol

# -- Momentum -- Defensive Mode ───────────────────────────────────────────
MOM_DEFENSIVE_VIX        = 30.0          # above this: only low-vol stocks
MOM_DEFENSIVE_MAX_VOL    = 0.35          # base cap 35% ann vol (scales up with VIX)

# ══════════════════════════════════════════════════════════════════════════════
#  BEAR MARKET MODULE
# ══════════════════════════════════════════════════════════════════════════════
BEAR_MODULE_ENABLED      = True

# -- Regime Detection ---------------------------------------------------------
BEAR_VIX_ACTIVATE        = 25.0          # activate bear module when VIX > this

# -- Module 1: Mean-Reversion Bounce (PRIMARY - Connors RSI-2 + IBS) ----------
BOUNCE_INSTRUMENT        = "SPY"         # trade SPY for bounces
BOUNCE_RSI_PERIOD        = 2             # Connors RSI-2
BOUNCE_RSI_ENTRY         = 10            # buy when RSI(2) < 10
BOUNCE_IBS_ENTRY         = 0.20          # buy when IBS < 0.20
BOUNCE_EXIT_SMA          = 5             # exit when close > 5-day SMA
BOUNCE_IBS_EXIT          = 0.80          # exit when IBS > 0.80
BOUNCE_MAX_HOLD          = 5             # max 5 trading days
BOUNCE_CAPITAL_BASE      = 1000.0        # base capital per trade
BOUNCE_USE_STOP          = False         # NO individual stop loss (Connors research)
BOUNCE_SLIPPAGE_PCT      = 0.001         # 0.10% slippage
BOUNCE_VIX_TIERS         = {25: 1.0, 30: 0.75, 35: 0.50, 40: 0.25}

# -- Module 2: Trend Short via Inverse ETF (SECONDARY) ------------------------
SHORT_ENABLED            = True
SHORT_INSTRUMENT         = "SH"          # ProShares Short S&P500 (inverse ETF)
SHORT_SPY_BELOW_50SMA    = True          # SPY must be below 50d SMA
SHORT_SPY_BELOW_200SMA   = True          # SPY must be below 200d SMA
SHORT_MOM_LOOKBACK       = 20            # 20-day return lookback
SHORT_MOM_THRESHOLD      = -0.05         # require -5% return over lookback
SHORT_EXIT_SMA           = 20            # exit when SPY closes above 20d SMA
SHORT_MAX_HOLD           = 15            # max 15 days (avoid inverse ETF decay)
SHORT_CAPITAL_SCALE      = 0.50          # 50% of normal position size
SHORT_SLIPPAGE_PCT       = 0.001         # 0.10% slippage

# -- Capitulation Detector (enhances timing) -----------------------------------
CAPIT_VIX_SPIKE          = 40.0          # VIX spike threshold
CAPIT_VIX_DECLINE_PCT    = 0.20          # VIX must drop 20% from peak
CAPIT_BOOST_SCALE        = 1.50          # boost bounce size by 50% on capitulation

# -- Bear Module Risk Controls ------------------------------------------------
BEAR_MAX_POSITIONS       = 2             # max 2 concurrent bear positions
BEAR_KILL_SWITCH_LOSSES  = 4             # kill after 4 consecutive losses
BEAR_KILL_SWITCH_DD      = -800.0        # kill at -$800 drawdown

# -- Bear Module Trade Tracking -----------------------------------------------
BEAR_TRADES_CSV          = "bear_trades.csv"
BEAR_STATE_JSON          = "bear_state.json"

# -- Bear Module Backtest Quality Gates ----------------------------------------
BEAR_BT_MIN_TRADES       = 10
BEAR_BT_MIN_WIN_RATE     = 65.0          # higher bar for mean reversion
BEAR_BT_MIN_PROFIT_FACTOR = 1.5
BEAR_BT_MIN_SHARPE       = 0.5
BEAR_BT_MIN_TOTAL_PNL    = 30.0

# ── Multi-Instrument Bounce ──────────────────────────────────────────────
# Same RSI-2 + IBS strategy on additional ETFs (triples signal frequency)
BOUNCE_INSTRUMENTS       = ["SPY", "QQQ", "IWM"]

# ══════════════════════════════════════════════════════════════════════════════
#  EARNINGS STRATEGY MODULE
# ══════════════════════════════════════════════════════════════════════════════
EARN_MODULE_ENABLED          = True

# -- Entry Parameters --------------------------------------------------------
EARN_ENTRY_DAYS_BEFORE       = 7            # scan earnings up to N days ahead
EARN_MIN_BEAT_RATE           = 0.75         # minimum historical beat rate (75%)
EARN_MIN_QUARTERS            = 3            # minimum quarters of earnings data
EARN_ALLOW_DECLINING         = False        # allow entry with declining EPS trend
EARN_REQUIRE_ABOVE_200SMA    = True         # stock must be above 200d SMA

# -- Position Sizing ---------------------------------------------------------
EARN_CAPITAL_PER_TRADE       = 1000.0       # base capital per earnings trade
EARN_SLIPPAGE_PCT            = 0.001        # 0.10% slippage per side

# -- Exit Parameters ---------------------------------------------------------
EARN_STOP_PCT                = 5.0          # stop loss: exit if down 5% before earnings
EARN_MAX_HOLD                = 10           # max hold in trading days (safety net)

# -- Risk Controls -----------------------------------------------------------
EARN_MAX_POSITIONS           = 5            # max concurrent earnings trades
EARN_KILL_SWITCH_LOSSES      = 5            # kill after N consecutive losses
EARN_KILL_SWITCH_DD          = -500.0       # kill at drawdown from peak

# -- Trade Tracking ----------------------------------------------------------
EARN_TRADES_CSV              = "earnings_trades.csv"
EARN_STATE_JSON              = "earnings_state.json"

# ══════════════════════════════════════════════════════════════════════════════
#  POLICY SHOCK BOUNCE MODULE
# ══════════════════════════════════════════════════════════════════════════════
# Detects rapid market drawdowns (tariff announcements, geopolitical shocks)
# and enters long SPY/QQQ betting on the walkback/reversal pattern.
# Runs unconditionally (shocks happen in any VIX regime).
SHOCK_MODULE_ENABLED         = True

# -- Shock Detection Parameters -----------------------------------------------
SHOCK_INSTRUMENTS            = ["SPY", "QQQ"]       # instruments to trade the bounce
SHOCK_LOOKBACK_DAYS          = 3             # detect drop over this window
SHOCK_DROP_THRESHOLD         = -0.03         # -3% over lookback = shock detected
SHOCK_VIX_SPIKE_MIN          = 3.0           # VIX must jump >= 3 points in 3 days
SHOCK_COOLDOWN_DAYS          = 5             # min days between shock entries (same ticker)

# -- Entry Filters (avoid catching falling knives) ----------------------------
SHOCK_WAIT_DAYS              = 1             # wait N days after shock before entry
SHOCK_RSI_MAX                = 40            # RSI(14) must be < 40 (still oversold)
SHOCK_ABOVE_200SMA           = False         # do NOT require above 200d SMA (shocks break it)

# -- Exit Parameters ----------------------------------------------------------
SHOCK_TAKE_PROFIT_PCT        = 0.04          # take profit at +4% (partial recovery)
SHOCK_STOP_LOSS_PCT          = 0.03          # stop loss at -3% from entry
SHOCK_MAX_HOLD               = 10            # max hold 10 trading days
SHOCK_EXIT_VIX_DECLINE       = 0.15          # exit when VIX drops 15% from shock peak

# -- Position Sizing ----------------------------------------------------------
SHOCK_CAPITAL_PER_TRADE      = 1000.0        # base capital per trade
SHOCK_SLIPPAGE_PCT           = 0.001         # 0.10% slippage per side
SHOCK_VIX_TIERS              = {20: 1.0, 25: 0.75, 30: 0.50, 35: 0.25}

# -- Risk Controls -------------------------------------------------------------
SHOCK_MAX_POSITIONS          = 3             # max concurrent shock trades
SHOCK_KILL_SWITCH_LOSSES     = 4             # kill after N consecutive losses
SHOCK_KILL_SWITCH_DD         = -600.0        # kill at drawdown from peak

# -- Trade Tracking ------------------------------------------------------------
SHOCK_TRADES_CSV             = "shock_trades.csv"
SHOCK_STATE_JSON             = "shock_state.json"

# -- Backtest Quality Gates ----------------------------------------------------
SHOCK_BT_MIN_TRADES          = 8             # fewer events than other modules
SHOCK_BT_MIN_WIN_RATE        = 55.0          # majority of shocks do reverse
SHOCK_BT_MIN_PROFIT_FACTOR   = 1.3
SHOCK_BT_MIN_SHARPE          = 0.4
SHOCK_BT_MIN_TOTAL_PNL       = 20.0

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════════
# Cross-module portfolio risk layer.  Three escalation tiers:
#   Tier 1 (GATEKEEPER): block new entries when exposure limits are breached
#   Tier 2 (FREEZE):     halt all new entries/exits when drawdown is severe
#   Tier 3 (LIQUIDATE):  emergency close ALL positions on catastrophic loss
GLOBAL_RISK_ENABLED          = True

# -- Tier 1: Gatekeeper (blocks new entries) -----------------------------------
GLOBAL_MAX_CAPITAL_DEPLOYED  = 8000.0        # max $ across all modules combined
GLOBAL_MAX_POSITIONS         = 15            # max open positions across all modules
GLOBAL_MAX_TICKER_OVERLAP    = 2             # max times same ticker can appear across modules
GLOBAL_MAX_DIRECTIONAL_PCT   = 1.00          # 100% until bear/short positions exist
                                             # reduce to 0.80 once portfolio has mixed directions

# -- Tier 2: Freeze (halts all activity except existing stops) -----------------
GLOBAL_FREEZE_DRAWDOWN       = -2000.0       # combined realized + unrealized P&L from peak
GLOBAL_FREEZE_DAILY_LOSS     = -500.0        # single-day combined loss triggers freeze

# -- Tier 3: Liquidation (emergency close everything) -------------------------
GLOBAL_LIQUIDATE_DRAWDOWN    = -3500.0       # catastrophic loss -> close all positions
                                             # roughly sum of module kill switches

# -- Asset class mapping (Level 2 directional check) --------------------------
# Tickers are mapped to asset classes.  Modules long in the same asset class
# count as directional overlap even if the exact tickers differ.
GLOBAL_EQUITY_TICKERS        = {
    "US_LARGE_CAP": ["SPY", "QQQ", "IWM", "SH"],  # SH is inverse, handled separately
}
GLOBAL_INVERSE_TICKERS       = ["SH"]       # these count as SHORT equity exposure

# -- Correlation (Level 3, auto-activates when enough trades exist) -----------
GLOBAL_CORR_MIN_TRADES       = 30           # per module, before correlation kicks in
GLOBAL_CORR_WINDOW           = 20           # rolling trade-count window
GLOBAL_CORR_WARN_THRESHOLD   = 0.70         # warn when module-pair correlation > this
GLOBAL_CORR_BLOCK_THRESHOLD  = 0.85         # block entries when correlation > this

# -- State file ----------------------------------------------------------------
GLOBAL_RISK_STATE_JSON       = "global_risk_state.json"

# ══════════════════════════════════════════════════════════════════════════════
#  ALPACA BROKER INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
LIVE_TRADING_ENABLED     = False          # master switch (False = paper-only, no broker)
ALPACA_API_KEY           = ""             # set via secrets.py or env var ALPACA_API_KEY
ALPACA_SECRET_KEY        = ""             # set via secrets.py or env var ALPACA_SECRET_KEY
ALPACA_PAPER             = True           # True = paper trading endpoint
ALPACA_SYNC_ON_STARTUP   = True           # verify positions match paper state on startup
ALPACA_MAX_ORDER_RETRIES = 2              # retry failed order submissions
BROKER_STATE_JSON        = "broker_state.json"
