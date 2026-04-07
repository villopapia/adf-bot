"""
================================================================================
 MOMENTUM SCANNER  — Daily Momentum Universe Scanner
================================================================================
 System  : Hybrid Topological-Statistical Arbitrage + Momentum
 Purpose : Scan S&P 500 + Nasdaq-100 universe daily, compute blended momentum
           scores, apply entry filters, and export ranked candidates to CSV.

 Pipeline
 --------
 Step 1  -> Fetch S&P 500 + Nasdaq-100 universe (with sectors)
 Step 2  -> Download OHLCV data (2 years, batched)
 Step 3  -> Clean data, align High/Low/Volume to valid Close columns
 Step 4  -> Check market regime (S&P 500 EMAs + VIX level)
 Step 5  -> Compute blended momentum scores (6m / 3m adaptive to VIX)
 Step 6  -> Compute technical indicators (RSI, ADX, ATR, SMA, vol)
 Step 7  -> Apply hard entry filters
 Step 8  -> Export momentum_candidates.csv, print summary

 Output
 ------
 momentum_candidates.csv  (ticker, sector, mom_score, mom_long, mom_short,
                            rsi, adx, atr, sma_200, price, high_52w_pct,
                            vol_ratio, vol_6mo)

================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import os
import sys
import datetime
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ------------------------------------------------------------------------------
# CONFIG IMPORTS
# ------------------------------------------------------------------------------
from config import (
    LOOKBACK_YEARS,
    MOMENTUM_CANDIDATES_CSV, SCANNER_BATCH_SIZE, MISSING_THRESHOLD,
    MOM_FORMATION_DAYS, MOM_FORMATION_SHORT, MOM_FORMATION_LONG,
    MOM_FORMATION_BLEND_VIX, MOM_SKIP_DAYS,
    MOM_MIN_PRICE, MOM_MIN_AVG_VOLUME, MOM_TOP_PCT,
    MOM_52W_HIGH_PCT, MOM_VOLUME_RATIO_MIN,
    MOM_RSI_MIN, MOM_RSI_MAX, MOM_ADX_MIN,
    MOM_ABSOLUTE_FILTER, MOM_REQUIRE_ABOVE_200SMA,
    MOM_MARKET_REGIME_EMA, MOM_ATR_PERIOD,
    MOM_VOL_LOOKBACK_DAYS, MOM_VIX_TIERS,
    MOM_DEFENSIVE_VIX, MOM_DEFENSIVE_MAX_VOL,
    MOM_VIX_MAX_ENTRY,
)

# ------------------------------------------------------------------------------
# NIGHTLY SCANNER IMPORTS
# ------------------------------------------------------------------------------
from nightly_scanner import (
    get_sp500_with_sectors,
    get_nasdaq100_tickers,
    bulk_download,
    clean_data,
)

# Resolve output path relative to script directory
_OUTPUT_CSV = os.path.join(_SCRIPT_DIR, MOMENTUM_CANDIDATES_CSV)


# ==============================================================================
#  SECTION 1 — OHLCV DOWNLOAD
# ==============================================================================

def download_ohlcv(tickers: list, years: int = LOOKBACK_YEARS) -> dict:
    """
    Download OHLCV data for all tickers in batched yfinance calls.

    Parameters
    ----------
    tickers : list of str
    years   : int — lookback in calendar years

    Returns
    -------
    dict with keys "High", "Low", "Close", "Volume".
    Each value is a DataFrame indexed by date, columns = tickers.
    Returns an empty dict if all batches fail.
    """
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=365 * years)

    n_tickers = len(tickers)
    n_batches = (n_tickers + SCANNER_BATCH_SIZE - 1) // SCANNER_BATCH_SIZE
    print(f"  [SCAN] Downloading OHLCV for {n_tickers} tickers "
          f"({start} -> {end}) in batches of {SCANNER_BATCH_SIZE} ...")

    high_frames   = []
    low_frames    = []
    close_frames  = []
    volume_frames = []

    for i in range(0, n_tickers, SCANNER_BATCH_SIZE):
        batch   = tickers[i : i + SCANNER_BATCH_SIZE]
        batch_n = i // SCANNER_BATCH_SIZE + 1
        print(f"  [DATA] Batch {batch_n}/{n_batches} ({len(batch)} tickers) ...",
              end=" ", flush=True)

        try:
            raw = yf.download(
                batch,
                start=str(start), end=str(end),
                auto_adjust=True, progress=False,
                threads=True,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                # Multi-ticker download returns MultiIndex columns (field, ticker)
                h = raw["High"]
                l = raw["Low"]
                c = raw["Close"]
                v = raw["Volume"]
            else:
                # Single-ticker download returns flat columns
                ticker_name = batch[0]
                h = raw[["High"]].rename(columns={"High":   ticker_name})
                l = raw[["Low"]].rename(columns={"Low":    ticker_name})
                c = raw[["Close"]].rename(columns={"Close":  ticker_name})
                v = raw[["Volume"]].rename(columns={"Volume": ticker_name})

            high_frames.append(h)
            low_frames.append(l)
            close_frames.append(c)
            volume_frames.append(v)
            print(f"OK ({c.shape[1]} cols)")

        except Exception as exc:
            print(f"FAILED ({exc})")

        time.sleep(0.5)

    if not close_frames:
        print("  [DATA] ERROR: all OHLCV batches failed.")
        return {}

    def _merge(frames: list) -> pd.DataFrame:
        df = pd.concat(frames, axis=1)
        return df.loc[:, ~df.columns.duplicated()]

    return {
        "High":   _merge(high_frames),
        "Low":    _merge(low_frames),
        "Close":  _merge(close_frames),
        "Volume": _merge(volume_frames),
    }


# ==============================================================================
#  SECTION 2 — TECHNICAL INDICATORS
# ==============================================================================

def compute_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute RSI for each column in prices.

    Uses standard Wilder smoothing (exponential moving average of gains/losses).
    No lookahead: each row uses only data up to bar i-1 implicitly via shift.

    Parameters
    ----------
    prices : pd.DataFrame — daily close prices, columns = tickers
    period : int — RSI lookback period (default 14)

    Returns
    -------
    pd.DataFrame — RSI values, same shape as input.
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing: EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_adx(highs: pd.DataFrame, lows: pd.DataFrame,
                closes: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute a simplified ADX approximation from OHLCV data.

    Uses a close-based proxy for Directional Movement:
      +DM = max(close - prev_close, 0)
      -DM = max(prev_close - close, 0)
    TR   = abs(close - prev_close)

    This is an approximation suitable for ranking/filtering purposes.
    Full Wilder ADX requires true high/low; when those are available they
    are used for TR but DM still uses the close-based proxy.

    Parameters
    ----------
    highs   : pd.DataFrame — daily highs
    lows    : pd.DataFrame — daily lows
    closes  : pd.DataFrame — daily closes
    period  : int

    Returns
    -------
    pd.DataFrame — ADX values indexed like closes, columns = tickers.
    """
    prev_close = closes.shift(1)

    # True Range using actual high/low
    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows  - prev_close).abs()
    tr  = np.maximum(tr1, np.maximum(tr2, tr3))

    # Directional movement (close-based approximation)
    diff = closes.diff()
    dm_pos = diff.clip(lower=0)
    dm_neg = (-diff).clip(lower=0)

    # Wilder smoothing
    alpha = 1.0 / period
    atr_smooth   = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    dm_pos_smooth = dm_pos.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    dm_neg_smooth = dm_neg.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    di_pos = 100.0 * dm_pos_smooth / atr_smooth.replace(0, np.nan)
    di_neg = 100.0 * dm_neg_smooth / atr_smooth.replace(0, np.nan)

    dx  = 100.0 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx


def compute_atr(highs: pd.DataFrame, lows: pd.DataFrame,
                closes: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) using Wilder smoothing.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = Wilder EMA(TR, period)

    Parameters
    ----------
    highs   : pd.DataFrame
    lows    : pd.DataFrame
    closes  : pd.DataFrame
    period  : int

    Returns
    -------
    pd.DataFrame — ATR values, same shape as closes.
    """
    prev_close = closes.shift(1)
    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows  - prev_close).abs()
    tr  = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = pd.DataFrame(tr).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    return atr


def compute_blended_momentum(closes: pd.DataFrame,
                             vix_level: "float | None") -> pd.DataFrame:
    """
    Compute blended momentum scores for each ticker at the latest date.

    Formation periods use skip-most-recent-month (MOM_SKIP_DAYS) to avoid
    short-term reversal. All indexing is backward-looking — no lookahead.

      mom_long  = price[t-skip] / price[t-formation_long]  - 1
      mom_short = price[t-skip] / price[t-formation_short] - 1

    Blending rule:
      VIX <= MOM_FORMATION_BLEND_VIX : score = mom_long  (pure 6-month)
      VIX >  MOM_FORMATION_BLEND_VIX : score = 0.5 * mom_long + 0.5 * mom_short

    Parameters
    ----------
    closes    : pd.DataFrame — daily close prices, columns = tickers
    vix_level : float or None

    Returns
    -------
    pd.DataFrame — one row per ticker with columns:
        mom_score, mom_long, mom_short
    """
    skip  = MOM_SKIP_DAYS
    f_long  = MOM_FORMATION_LONG
    f_short = MOM_FORMATION_SHORT

    # Shifted closes (no lookahead)
    price_now   = closes.shift(skip)
    price_long  = closes.shift(skip + f_long)
    price_short = closes.shift(skip + f_short)

    mom_long  = price_now / price_long.replace(0, np.nan)  - 1.0
    mom_short = price_now / price_short.replace(0, np.nan) - 1.0

    # Take only the latest valid row
    latest_long  = mom_long.iloc[-1]
    latest_short = mom_short.iloc[-1]

    blend = (
        vix_level is None or vix_level <= MOM_FORMATION_BLEND_VIX
    )
    if blend:
        mom_score = latest_long
    else:
        mom_score = 0.5 * latest_long + 0.5 * latest_short

    result = pd.DataFrame({
        "mom_score": mom_score,
        "mom_long":  latest_long,
        "mom_short": latest_short,
    })
    result.index.name = "ticker"
    return result


def compute_technical_indicators(closes: pd.DataFrame,
                                 highs: pd.DataFrame,
                                 lows: pd.DataFrame,
                                 volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators at the latest bar for each ticker.

    Indicators computed
    -------------------
    - RSI(14)
    - ADX(14)  — close-based approximation
    - ATR(14)  — in price units
    - sma_200  — 200-day simple moving average
    - price    — latest close price
    - high_52w_pct  — latest price / 252-day rolling max (proximity to 52-week high)
    - avg_volume_20d — 20-day average daily volume
    - avg_volume_50d — 50-day average daily volume
    - vol_ratio      — avg_volume_20d / avg_volume_50d
    - vol_6mo        — annualised 126-day realised volatility

    All calculations use data up to bar[-1] only — no lookahead.

    Parameters
    ----------
    closes  : pd.DataFrame
    highs   : pd.DataFrame
    lows    : pd.DataFrame
    volumes : pd.DataFrame

    Returns
    -------
    pd.DataFrame — indexed by ticker, one row per ticker.
    """
    rsi_df = compute_rsi(closes, period=14)
    adx_df = compute_adx(highs, lows, closes, period=MOM_ATR_PERIOD)
    atr_df = compute_atr(highs, lows, closes, period=MOM_ATR_PERIOD)

    sma_200_df    = closes.rolling(200, min_periods=150).mean()
    high_52w_df   = closes.rolling(252, min_periods=200).max()
    vol_20d_df    = volumes.rolling(20, min_periods=10).mean()
    vol_50d_df    = volumes.rolling(50, min_periods=30).mean()

    # Annualised realised volatility: std of log returns over vol lookback
    log_ret = np.log(closes / closes.shift(1))
    vol_6mo_df = log_ret.rolling(
        MOM_VOL_LOOKBACK_DAYS, min_periods=MOM_VOL_LOOKBACK_DAYS // 2
    ).std() * np.sqrt(252)

    # Extract latest bar values
    latest_close  = closes.iloc[-1]
    latest_rsi    = rsi_df.iloc[-1]
    latest_adx    = adx_df.iloc[-1]
    latest_atr    = atr_df.iloc[-1]
    latest_sma200 = sma_200_df.iloc[-1]
    latest_52w    = high_52w_df.iloc[-1]
    latest_v20    = vol_20d_df.iloc[-1]
    latest_v50    = vol_50d_df.iloc[-1]
    latest_vol6m  = vol_6mo_df.iloc[-1]

    high_52w_pct = latest_close / latest_52w.replace(0, np.nan)
    vol_ratio    = latest_v20 / latest_v50.replace(0, np.nan)

    result = pd.DataFrame({
        "price":         latest_close,
        "rsi":           latest_rsi,
        "adx":           latest_adx,
        "atr":           latest_atr,
        "sma_200":       latest_sma200,
        "high_52w_pct":  high_52w_pct,
        "avg_volume_20d": latest_v20,
        "avg_volume_50d": latest_v50,
        "vol_ratio":     vol_ratio,
        "vol_6mo":       latest_vol6m,
    })
    result.index.name = "ticker"
    return result


# ==============================================================================
#  SECTION 3 — MARKET REGIME
# ==============================================================================

def get_vix_scale(vix_level: "float | None") -> float:
    """
    Map VIX level to a position-size scaling factor using MOM_VIX_TIERS.

    MOM_VIX_TIERS = {25: 1.0, 30: 0.75, 35: 0.50, 40: 0.0}
    The scale equals the value at the highest threshold that vix_level exceeds.
    If vix_level > 40 (the key with scale 0.0), returns 0.0.
    If vix_level is None, returns 1.0 (full size).

    Parameters
    ----------
    vix_level : float or None

    Returns
    -------
    float — scaling factor in [0.0, 1.0]
    """
    if vix_level is None:
        return 1.0

    scale = 1.0
    for threshold in sorted(MOM_VIX_TIERS.keys()):
        if vix_level > threshold:
            scale = MOM_VIX_TIERS[threshold]
    return scale


def check_market_regime() -> dict:
    """
    Fetch S&P 500 (^GSPC) and VIX (^VIX) from yfinance and assess regime.

    Regime classification
    ---------------------
    - "bull" : 50d EMA > 200d EMA  (golden cross)
    - "bear" : 50d EMA <= 200d EMA (death cross or below)

    Returns
    -------
    dict with keys:
        regime            : str  "bull" or "bear"
        vix               : float or None
        vix_scale         : float
        sp500_above_200ema : bool
    """
    print("  [REGIME] Fetching S&P 500 and VIX data ...")

    result = {
        "regime": "bull",
        "vix": None,
        "vix_scale": 1.0,
        "sp500_above_200ema": True,
    }

    # --- S&P 500 regime ---
    try:
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=365 * LOOKBACK_YEARS)
        sp_raw = yf.download(
            "^GSPC", start=str(start), end=str(end),
            auto_adjust=True, progress=False,
        )
        if isinstance(sp_raw.columns, pd.MultiIndex):
            sp_close = sp_raw["Close"].iloc[:, 0]
        else:
            sp_close = sp_raw["Close"]

        sp_close = sp_close.dropna()

        if len(sp_close) >= 200:
            ema50  = sp_close.ewm(span=50,  adjust=False).mean()
            ema200 = sp_close.ewm(span=200, adjust=False).mean()

            latest_price  = float(sp_close.iloc[-1])
            latest_ema50  = float(ema50.iloc[-1])
            latest_ema200 = float(ema200.iloc[-1])

            ema_bull = latest_ema50 > latest_ema200
            above_200 = latest_price > latest_ema200

            result["sp500_above_200ema"] = above_200

            # EMA cross is lagging — override with faster signals
            # Price below 200d EMA = bearish regardless of 50/200 cross
            # VIX override happens below after VIX fetch
            if not above_200:
                result["regime"] = "bear"
            elif ema_bull:
                result["regime"] = "bull"
            else:
                result["regime"] = "bear"

            print(f"  [REGIME] S&P 500 price={latest_price:.1f}  "
                  f"EMA50={latest_ema50:.1f}  EMA200={latest_ema200:.1f}  "
                  f"above_200EMA={'Y' if above_200 else 'N'}")
        else:
            print("  [REGIME] Insufficient S&P 500 data for regime check.")

    except Exception as exc:
        print(f"  [REGIME] S&P 500 fetch failed ({exc}); assuming bull.")

    # --- VIX ---
    try:
        vix_raw = yf.download(
            "^VIX", period="5d",
            auto_adjust=True, progress=False,
        )
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_close = vix_raw["Close"].iloc[:, 0]
        else:
            vix_close = vix_raw["Close"]

        vix_close = vix_close.dropna()

        if len(vix_close) > 0:
            vix_val = float(vix_close.iloc[-1])
            result["vix"] = vix_val
            result["vix_scale"] = get_vix_scale(vix_val)

            # VIX overrides EMA-based regime — VIX > 25 is NOT a bull market
            if vix_val > 30:
                result["regime"] = "stressed"
            elif vix_val > 25:
                result["regime"] = "elevated"
            # else keep EMA-based regime

            print(f"  [REGIME] VIX={vix_val:.2f}  scale={result['vix_scale']:.2f}  "
                  f"-> {result['regime'].upper()}")
        else:
            print("  [REGIME] VIX data unavailable.")

    except Exception as exc:
        print(f"  [REGIME] VIX fetch failed ({exc}); VIX set to None.")

    return result


# ==============================================================================
#  SECTION 4 — ENTRY FILTERS
# ==============================================================================

def apply_entry_filters(candidates: pd.DataFrame,
                        vix_level: "float | None") -> pd.DataFrame:
    """
    Apply all hard entry filters to the merged candidates DataFrame.

    Filters applied in order
    ------------------------
    1. price >= MOM_MIN_PRICE (default $10)
    2. avg_volume_20d >= MOM_MIN_AVG_VOLUME (default 500K shares)
    3. mom_score in top MOM_TOP_PCT percentile (default top 10%)
    4. high_52w_pct >= MOM_52W_HIGH_PCT (default 0.90 = within 10% of 52w high)
    5. vol_ratio >= MOM_VOLUME_RATIO_MIN (default 1.2)
    6. RSI in [MOM_RSI_MIN, MOM_RSI_MAX] (default 50-80)
    7. ADX >= MOM_ADX_MIN (default 20)
    8. If MOM_ABSOLUTE_FILTER: mom_score > 0
    9. If MOM_REQUIRE_ABOVE_200SMA: price > sma_200

    Defensive mode (vix > MOM_DEFENSIVE_VIX):
    10. vol_6mo <= MOM_DEFENSIVE_MAX_VOL (default 0.25 = 25% annualised)

    Parameters
    ----------
    candidates : pd.DataFrame — merged momentum + indicators, indexed by ticker
    vix_level  : float or None

    Returns
    -------
    pd.DataFrame — filtered candidates sorted by mom_score descending.
    """
    df = candidates.copy()
    initial_count = len(df)

    def _log_filter(label: str, mask: pd.Series) -> None:
        removed = (~mask).sum()
        if removed > 0:
            print(f"  [FILTER] {label}: removed {removed} "
                  f"({len(df[mask])} remain)")

    # 1. Minimum price
    mask_price = df["price"] >= MOM_MIN_PRICE
    _log_filter(f"price >= {MOM_MIN_PRICE}", mask_price)
    df = df[mask_price]

    # 2. Minimum average volume
    mask_vol = df["avg_volume_20d"] >= MOM_MIN_AVG_VOLUME
    _log_filter(f"avg_vol_20d >= {MOM_MIN_AVG_VOLUME:,}", mask_vol)
    df = df[mask_vol]

    # 3. Top percentile momentum
    if len(df) > 0:
        threshold = df["mom_score"].quantile(1.0 - MOM_TOP_PCT / 100.0)
        mask_top = df["mom_score"] >= threshold
        _log_filter(f"top {MOM_TOP_PCT}% mom_score", mask_top)
        df = df[mask_top]

    # 4. 52-week high proximity
    mask_52w = df["high_52w_pct"] >= MOM_52W_HIGH_PCT
    _log_filter(f"52w_high_pct >= {MOM_52W_HIGH_PCT}", mask_52w)
    df = df[mask_52w]

    # 5. Volume ratio (recent vs 50d baseline)
    mask_vr = df["vol_ratio"] >= MOM_VOLUME_RATIO_MIN
    _log_filter(f"vol_ratio >= {MOM_VOLUME_RATIO_MIN}", mask_vr)
    df = df[mask_vr]

    # 6. RSI range
    mask_rsi = (df["rsi"] >= MOM_RSI_MIN) & (df["rsi"] <= MOM_RSI_MAX)
    _log_filter(f"RSI in [{MOM_RSI_MIN}, {MOM_RSI_MAX}]", mask_rsi)
    df = df[mask_rsi]

    # 7. ADX minimum (confirms trend strength)
    mask_adx = df["adx"] >= MOM_ADX_MIN
    _log_filter(f"ADX >= {MOM_ADX_MIN}", mask_adx)
    df = df[mask_adx]

    # 8. Absolute momentum filter
    if MOM_ABSOLUTE_FILTER:
        mask_abs = df["mom_score"] > 0.0
        _log_filter("mom_score > 0 (absolute momentum)", mask_abs)
        df = df[mask_abs]

    # 9. Price above 200d SMA
    if MOM_REQUIRE_ABOVE_200SMA:
        mask_sma = df["price"] > df["sma_200"]
        _log_filter("price > SMA200", mask_sma)
        df = df[mask_sma]

    # 10. Defensive mode volatility cap (scaled to VIX regime)
    if vix_level is not None and vix_level > MOM_DEFENSIVE_VIX:
        # Scale vol cap up with VIX — fixed 0.25 is too tight when VIX is 30+
        # Base cap 0.25 at VIX=30, scale up: +0.05 per 5 VIX points above 30
        vol_cap = MOM_DEFENSIVE_MAX_VOL + max(0, (vix_level - 30.0)) * 0.01
        vol_cap = min(vol_cap, 0.50)  # never exceed 50% ann vol
        mask_def = df["vol_6mo"] <= vol_cap
        _log_filter(
            f"defensive: vol_6mo <= {vol_cap:.2f} "
            f"(VIX={vix_level:.1f} > {MOM_DEFENSIVE_VIX})",
            mask_def,
        )
        df = df[mask_def]

    print(f"  [FILTER] {initial_count} -> {len(df)} after all filters.")
    return df.sort_values("mom_score", ascending=False)


# ==============================================================================
#  SECTION 5 — MAIN PIPELINE
# ==============================================================================

def main(vix_level: "float | None" = None) -> "pd.DataFrame | None":
    """
    Full momentum scanner pipeline.

    Steps
    -----
    1.  Print header
    2.  Build universe: S&P 500 + Nasdaq-100, deduplicated
    3.  Download OHLCV data (2 years)
    4.  Clean Close prices; align H/L/V to surviving tickers
    5.  Check market regime (EMAs + VIX) — override vix_level if provided
    6.  Block if VIX > MOM_VIX_MAX_ENTRY
    7.  Warn (do not block) if regime is "bear" and MOM_MARKET_REGIME_EMA is True
    8.  Compute blended momentum scores
    9.  Compute technical indicators
    10. Merge into candidates DataFrame; attach sector
    11. Drop rows with too many NaN indicators
    12. Apply entry filters
    13. Export to momentum_candidates.csv
    14. Print summary

    Parameters
    ----------
    vix_level : float or None
        If provided, overrides the fetched VIX reading (useful for testing).

    Returns
    -------
    pd.DataFrame of filtered candidates, or None if pipeline is blocked.
    """
    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  MOMENTUM SCANNER")
    print(f"  Run date : {datetime.date.today()}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Step 1 - Universe
    # ------------------------------------------------------------------
    print("\n[STEP 1] Building universe ...")
    try:
        sp500_df   = get_sp500_with_sectors()
        ndx100_df  = get_nasdaq100_tickers()
        universe   = pd.concat([sp500_df, ndx100_df], ignore_index=True)
        universe   = universe.drop_duplicates(subset="Symbol", keep="first")
        tickers    = universe["Symbol"].tolist()
        sector_map = universe.set_index("Symbol")["Sector"].to_dict()
        print(f"  [SCAN] Universe: {len(tickers)} unique tickers "
              f"(S&P 500 + Nasdaq-100).")
    except Exception as exc:
        print(f"  [SCAN] ERROR building universe: {exc}")
        return None

    # ------------------------------------------------------------------
    # Step 2 - OHLCV download
    # ------------------------------------------------------------------
    print("\n[STEP 2] Downloading OHLCV data ...")
    ohlcv = download_ohlcv(tickers, years=LOOKBACK_YEARS)
    if not ohlcv:
        print("  [SCAN] ERROR: OHLCV download completely failed. Aborting.")
        return None

    # ------------------------------------------------------------------
    # Step 3 - Clean data
    # ------------------------------------------------------------------
    print("\n[STEP 3] Cleaning data ...")
    closes_raw = ohlcv["Close"]
    closes     = clean_data(closes_raw, missing_thresh=MISSING_THRESHOLD)

    valid_tickers = closes.columns.tolist()

    # Align H/L/V to the same surviving tickers
    def _align(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in valid_tickers if c in df.columns]
        aligned = df[cols].reindex(closes.index).ffill(limit=2)
        return aligned

    highs   = _align(ohlcv["High"])
    lows    = _align(ohlcv["Low"])
    volumes = _align(ohlcv["Volume"])

    print(f"  [DATA] {len(valid_tickers)} tickers after cleaning.")

    # ------------------------------------------------------------------
    # Step 4 - Market regime
    # ------------------------------------------------------------------
    print("\n[STEP 4] Checking market regime ...")
    regime_info = check_market_regime()

    # Allow caller to override VIX (e.g. for testing)
    if vix_level is not None:
        regime_info["vix"] = vix_level
        regime_info["vix_scale"] = get_vix_scale(vix_level)
        print(f"  [REGIME] VIX overridden to {vix_level:.2f}")

    effective_vix = regime_info["vix"]

    # ------------------------------------------------------------------
    # Step 5 - VIX gate
    # ------------------------------------------------------------------
    if effective_vix is not None and effective_vix > MOM_VIX_MAX_ENTRY:
        print(f"\n  [SCAN] BLOCKED: VIX={effective_vix:.2f} > "
              f"MOM_VIX_MAX_ENTRY={MOM_VIX_MAX_ENTRY}. "
              f"No momentum entries in extreme panic.")
        return None

    # ------------------------------------------------------------------
    # Step 6 - Bear market warning
    # ------------------------------------------------------------------
    if MOM_MARKET_REGIME_EMA and regime_info["regime"] == "bear":
        print(f"  [REGIME] WARNING: S&P 500 in BEAR regime "
              f"(50d EMA < 200d EMA). Momentum signals less reliable.")

    # ------------------------------------------------------------------
    # Step 7 - Blended momentum scores
    # ------------------------------------------------------------------
    print("\n[STEP 5] Computing blended momentum scores ...")
    try:
        mom_df = compute_blended_momentum(closes, vix_level=effective_vix)
        print(f"  [SCAN] Momentum scores computed for {len(mom_df)} tickers.")
    except Exception as exc:
        print(f"  [SCAN] ERROR computing momentum: {exc}")
        return None

    # ------------------------------------------------------------------
    # Step 8 - Technical indicators
    # ------------------------------------------------------------------
    print("\n[STEP 6] Computing technical indicators ...")
    try:
        tech_df = compute_technical_indicators(closes, highs, lows, volumes)
        print(f"  [SCAN] Technical indicators computed for {len(tech_df)} tickers.")
    except Exception as exc:
        print(f"  [SCAN] ERROR computing indicators: {exc}")
        return None

    # ------------------------------------------------------------------
    # Step 9 - Merge
    # ------------------------------------------------------------------
    print("\n[STEP 7] Merging candidates ...")
    candidates = mom_df.join(tech_df, how="inner")
    candidates.index.name = "ticker"

    # Attach sector
    candidates["sector"] = candidates.index.map(
        lambda t: sector_map.get(t, "Unknown")
    )

    # Drop rows where critical indicators are NaN
    critical_cols = ["mom_score", "price", "rsi", "adx", "sma_200",
                     "high_52w_pct", "vol_ratio", "vol_6mo"]
    before_drop = len(candidates)
    candidates = candidates.dropna(subset=critical_cols)
    dropped = before_drop - len(candidates)
    if dropped > 0:
        print(f"  [DATA] Dropped {dropped} tickers with missing indicators.")

    print(f"  [SCAN] {len(candidates)} candidates before entry filters.")

    # ------------------------------------------------------------------
    # Step 10 - Entry filters
    # ------------------------------------------------------------------
    print("\n[STEP 8] Applying entry filters ...")
    filtered = apply_entry_filters(candidates, vix_level=effective_vix)

    # ------------------------------------------------------------------
    # Step 11 - Export CSV
    # ------------------------------------------------------------------
    output_cols = [
        "sector", "mom_score", "mom_long", "mom_short",
        "rsi", "adx", "atr", "sma_200", "price",
        "high_52w_pct", "vol_ratio", "vol_6mo",
    ]
    # Keep only columns that exist in filtered
    export_cols = [c for c in output_cols if c in filtered.columns]
    export_df   = filtered[export_cols].reset_index()
    # reset_index() names the new column after the index name ("ticker").
    # Guard against the rare case where index.name was not set.
    if "ticker" not in export_df.columns and "index" in export_df.columns:
        export_df.rename(columns={"index": "ticker"}, inplace=True)

    try:
        export_df.to_csv(_OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\n  [SCAN] Exported {len(export_df)} candidates -> {_OUTPUT_CSV}")
    except Exception as exc:
        print(f"  [SCAN] ERROR writing CSV: {exc}")

    # ------------------------------------------------------------------
    # Step 12 - Summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"  MOMENTUM SCAN COMPLETE")
    print(f"  Regime   : {regime_info['regime'].upper()}")
    print(f"  VIX      : {effective_vix if effective_vix is not None else 'N/A'}")
    print(f"  VIX scale: {regime_info['vix_scale']:.2f}")
    print(f"  Candidates passed: {len(filtered)}")

    if len(filtered) > 0:
        print(f"\n  Top 5 momentum candidates:")
        print(f"  {'Ticker':<8} {'Sector':<28} {'MomScore':>9} {'RSI':>6} {'ADX':>6}")
        print(f"  {'-'*7:<8} {'-'*27:<28} {'-'*9:>9} {'-'*6:>6} {'-'*6:>6}")
        for row in filtered.head(5).itertuples():
            ticker  = row.Index
            sector  = str(getattr(row, "sector", ""))[:27]
            score   = getattr(row, "mom_score", float("nan"))
            rsi_val = getattr(row, "rsi",       float("nan"))
            adx_val = getattr(row, "adx",       float("nan"))
            print(f"  {ticker:<8} {sector:<28} {score:>9.3f} "
                  f"{rsi_val:>6.1f} {adx_val:>6.1f}")
    else:
        print("  No candidates passed all filters today.")

    print("-" * 60)

    # Ensure 'ticker' is a column (not just the index) for downstream consumers
    if "ticker" not in filtered.columns and filtered.index.name == "ticker":
        filtered = filtered.reset_index()
    return filtered


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
