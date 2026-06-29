# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Python-based automated trading system that scans the S&P 500 + Nasdaq-100 universe daily for momentum and pre-earnings trade opportunities. It paper-trades signals through CSV journals and an Excel workbook, with optional Alpaca brokerage execution. Runs on Windows via Task Scheduler.

## Running the System

```bash
python run_system.py              # normal daily run
python run_system.py --scan       # force scanner refresh
python run_system.py --scan-only  # scanner only, skip signals
```

There are no tests, no linter, and no build step. The system is a flat directory of `.py` modules run directly.

## Architecture

**`run_system.py`** is the daily orchestrator. It runs phases in sequence:

1. **Broker sync** — reconcile Alpaca positions (if `LIVE_TRADING_ENABLED`)
2. **Exit phases (0b, 0d)** — check open momentum/earnings trades for stop/profit/time exits
3. **Excel tracker** — update live prices, process manual close requests from `paper_trades.xlsx`
4. **Global risk assessment** — cross-module portfolio check (ALLOW/FREEZE/LIQUIDATE tiers)
5. **Phase 3: Momentum signals** — scan → backtest → entry
6. **Phase 5: Earnings signals** — scan → filter → entry

Each active module follows a three-file pattern:

| Role | Momentum | Earnings |
|------|----------|----------|
| **Scanner** — universe scan, candidate ranking | `momentum_scanner.py` | `earnings_scanner.py` |
| **Signal** — walk-forward backtest, quality gates, live signal check | `momentum_signal.py` | `earnings_signal.py` |
| **Tracker** — paper trade journal, exits, kill switch, state persistence | `momentum_tracker.py` | `earnings_tracker.py` |

**Cross-cutting modules:**
- **`config.py`** — single source of truth for all parameters. Every tunable constant lives here.
- **`global_risk.py`** — reads all module CSV files passively (never imported by modules). Three tiers: ALLOW (gatekeeper checks per-trade), FREEZE (halt entries), LIQUIDATE (close everything).
- **`excel_tracker.py`** — writes `paper_trades.xlsx` with per-module sheets. Manual close: set Status to "CLOSE" in Excel; next run finalizes P&L.
- **`alpaca_broker.py`** — wraps `alpaca-py` for order execution. All methods are no-ops when `LIVE_TRADING_ENABLED=False`. Never crashes the main system (all methods wrapped in try/except).
- **`earnings_util.py`** — shared FMP earnings data fetch + blackout check, cached per-run.

**Disabled modules** (code preserved, config flags = False): `master_signal.py` / `nightly_scanner.py` (pairs), `bear_signal.py` / `bear_tracker.py`, `shock_signal.py` / `shock_tracker.py`. These produced zero trades and were disabled 2026-06-12.

## Data Flow

Signals flow: Scanner → Signal → Tracker (CSV + JSON) → Excel Tracker → Broker (optional)

**State files (JSON):** `momentum_state.json`, `earnings_state.json`, `global_risk_state.json` — track running P&L, kill switch flags, peak drawdown. Written atomically via `.tmp` + `os.replace`.

**Trade journals (CSV):** `momentum_trades.csv`, `earnings_trades.csv` — append-only logs of all trades with `status` column (open/closed).

**Logs:** `logs/YYYY-MM-DD_HHMMSS.log` — full console capture with ANSI stripped. Auto-cleaned after `LOG_RETENTION_DAYS`. `system_log.csv` — persistent record of all pairs signals. `error.log` — per-pair crash/math errors.

## Key Dependencies

`yfinance` (price data), `pandas`/`numpy` (data processing), `openpyxl` (Excel journal), `requests` (FMP API), `alpaca-py` (broker). No `requirements.txt` — install individually.

## External APIs

- **FMP (Financial Modeling Prep):** earnings calendar + historical earnings data. Key in `config.py` via `FMP_API_KEY`.
- **yfinance:** primary price/OHLCV source. No API key needed.
- **Alpaca:** paper/live order execution. Keys via env vars `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` or `config.py`.
- **Alpha Vantage:** free-tier fallback. Key in `config.py` via `AV_API_KEY`.

## Windows-Specific Constraints

- All file writes must use `encoding="utf-8"` — the default cp1252 encoding chokes on Unicode.
- Avoid Unicode box-drawing characters (`═`, `║`, `╔`, etc.) in dynamically generated strings — they fail under cp1252 terminal encoding. Use ASCII dashes: `"-" * 60`.
- `run_system.py` uses a PID-based `.lock` file to prevent concurrent runs (4-hour stale timeout).

## Conventions

- All parameters are centralized in `config.py`. Modules import constants from config; never hardcode thresholds.
- Module enable/disable via `*_MODULE_ENABLED` booleans in config.
- Each tracker module manages its own kill switch (consecutive losses or drawdown from peak). Reset functions: `momentum_tracker.reset_mom_kill_switch()`, `earnings_tracker.reset_earn_kill_switch()`, `global_risk.reset_freeze()`, `global_risk.reset_liquidation()`.
- Walk-forward backtests use a 70/30 train/test split. Signals observed at close of bar i−1 execute at bar i close (no lookahead).
- VIX-based position scaling uses tiered dicts (e.g., `MOM_VIX_TIERS = {25: 1.0, 30: 0.75, ...}`).
