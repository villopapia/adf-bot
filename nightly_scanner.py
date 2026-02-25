"""
================================================================================
 SCRIPT A — NIGHTLY UNIVERSE SCANNER  (v2 — Cointegration + Sector)
================================================================================
 System  : Hybrid Topological-Statistical Arbitrage
 Purpose : Scan the S&P 500 universe, filter to same-sector pairs, test for
           cointegration (not just correlation), and export the Top-50
           candidates ranked by cointegration strength.

 v2 Upgrades over v1
 --------------------
 • Ranks by Engle-Granger cointegration p-value (not raw correlation)
 • Same-sector pairing — shared fundamentals boost mean-reversion odds
 • Correlation pre-filter loosened to 0.80 to widen the funnel

 Pipeline
 --------
 Step 1 → Fetch current S&P 500 tickers + GICS Sector from Wikipedia
 Step 2 → Bulk-download 2 years of daily Close via yfinance
 Step 3 → Clean: drop tickers with > 10 % missing values
 Step 4 → Pre-filter: keep same-sector pairs with correlation > 0.80
 Step 5 → Engle-Granger cointegration test on pre-filtered pairs
 Step 6 → Rank by cointegration p-value, keep Top 50, write CSV

 Output
 ------
 daily_candidates.csv   (Stock_A, Stock_B, Sector, Correlation, Coint_pval)

================================================================================
 SCHEDULING
================================================================================

 WINDOWS — Task Scheduler
 ─────────────────────────────────────────────────────────────────
 1. Open Task Scheduler  (Win+R → taskschd.msc)
 2. Create Basic Task → Name: "Nightly Pair Scanner"
    • Trigger : Daily, 01:00 AM  (well after US market close)
    • Action  : Start a Program
        Program   : C:\\path\\to\\python.exe
        Arguments : "C:\\Users\\User\\Desktop\\adf bot\\nightly_scanner.py"
        Start in  : "C:\\Users\\User\\Desktop\\adf bot"
 3. Check "Run whether user is logged on or not".

 LINUX / WSL — cron
 ─────────────────────────────────────────────────────────────────
 0 1 * * 1-5  /usr/bin/python3 /path/to/nightly_scanner.py >> scanner.log 2>&1

================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, datetime, os, sys, time, io
from collections import defaultdict
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (centralised in config.py — single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import (
    LOOKBACK_YEARS, CORRELATION_THRESHOLD, TOP_N_PAIRS,
    INPUT_CSV, MISSING_THRESHOLD, COINT_PVAL_THRESH,
    SCANNER_BATCH_SIZE,
)

CORR_PRE_FILTER = CORRELATION_THRESHOLD
OUTPUT_CSV      = INPUT_CSV
BATCH_SIZE      = SCANNER_BATCH_SIZE


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FETCH S&P 500 TICKERS  +  SECTORS
# ══════════════════════════════════════════════════════════════════════════════

def get_sp500_with_sectors() -> pd.DataFrame:
    """
    Scrape the S&P 500 table from Wikipedia.
    Returns a DataFrame with columns ['Symbol', 'Sector'].
    Falls back to a hardcoded snapshot if the request fails.
    """
    url = ("https://en.wikipedia.org/wiki/"
           "List_of_S%26P_500_companies")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0][["Symbol", "GICS Sector"]].copy()
        df.columns = ["Symbol", "Sector"]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df = df.drop_duplicates("Symbol")
        print(f"[STEP 1] Scraped {len(df)} S&P 500 tickers + sectors "
              f"from Wikipedia.")
        return df

    except Exception as exc:
        print(f"[STEP 1] Wikipedia scrape failed ({exc}); using hardcoded list.")
        return _hardcoded_sp500_with_sectors()


def _hardcoded_sp500_with_sectors() -> pd.DataFrame:
    """Fallback — representative tickers grouped by major GICS Sector."""
    sector_map = {
        "Financials": (
            "AXP BAC BK BLK BRK-B C CFG CMA COF FITB GS HBAN JPM KEY MS "
            "NTRS PNC RF SCHW STT SYF TFC USB WFC ZION"
        ),
        "Information Technology": (
            "AAPL ACN ADBE ADI AMAT AMD ANET APH AVGO CDNS CRM CSCO CTSH "
            "ENPH FFIV FISV FTNT GPN HPE HPQ IBM INTC INTU IT KEYS KLAC "
            "LRCX MA MCHP MPWR MSFT MSI MU NXPI ON ORCL PANW PYPL QCOM "
            "SNPS STX TEL TXN V WDC"
        ),
        "Utilities": (
            "AEE AEP AES ATO CEG CMS CNP D DTE DUK ED EIX ES ETR EVRG "
            "EXC FE LNT NEE NI NRG PCG PEG PNW PPL SO SRE WEC XEL"
        ),
        "Health Care": (
            "ABBV ABT ALGN AMGN BAX BDX BIIB BIO BSX CAH CI CNC CRL CVS "
            "DXCM EW GILD HCA HOLX HUM IDXX ILMN INCY IQV ISRG JNJ LH "
            "LLY MCK MDT MRK MRNA PFE REGN RMD STE SYK TMO UHS UNH "
            "VRTX WAT ZBH ZTS"
        ),
        "Industrials": (
            "ALLE AME AON BA CAT CHRW CMI CPRT CSX CTAS DAL DE DOV EMR "
            "ETN EXPD FAST FDX GD GE GWW HWM IR ITW J JBHT JCI LHX LMT "
            "MAS MMM NOC NSC ODFL OTIS PCAR PH PNR PWR ROK ROL ROP RSG "
            "RTX SNA TDG TDY TXT UNP UPS URI WAB WM"
        ),
        "Consumer Discretionary": (
            "AMZN APTV AZO BBY BKNG BWA CCL CMG CZR DG DHI DLTR DPZ DRI "
            "EBAY ETSY EXPE F GM GRMN GPC HAS HD KMX LEN LKQ LOW LVS "
            "MAR MCD MGM MHK NCLH NKE NVR ORLY PHM POOL PVH RL RCL "
            "ROST SBUX TJX TPR TSLA TKO ULTA VFC WHR WYNN YUM"
        ),
        "Communication Services": (
            "CHTR CMCSA DIS EA FOX FOXA GOOG GOOGL LYV META MTCH "
            "NFLX NWS NWSA OMC T TMUS TTWO VZ WBD"
        ),
        "Consumer Staples": (
            "ADM BF-B CAG CHD CL CLX COST CPB EL GIS HRL HSY KDP KHC "
            "KMB KO KR MDLZ MKC MNST MO PEP PG PM SJM SYY TAP TSN WMT"
        ),
        "Energy": (
            "APA BKR COP CTRA CVX DVN EOG EQT FANG HAL KMI MPC "
            "OKE OXY PSX SLB TRGP VLO WMB XOM"
        ),
        "Real Estate": (
            "AMT ARE AVB BXP CCI CPT CBRE EQIX EQR ESS FRT INVH "
            "IRM KIM MAA O PLD PSA REG SBAC SPG UDR VTR VICI WELL"
        ),
        "Materials": (
            "ALB AMCR APD AVY BG CE CF DD DOW ECL EMN FCX FMC IFF IP "
            "LIN LYB MLM NEM NUE PKG PPG SHW VMC"
        ),
    }

    rows = []
    for sector, tickers_str in sector_map.items():
        for t in tickers_str.split():
            rows.append({"Symbol": t, "Sector": sector})

    df = pd.DataFrame(rows).drop_duplicates("Symbol")
    print(f"[STEP 1] Using hardcoded list of {len(df)} tickers with sectors.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — BULK DOWNLOAD PRICE DATA
# ══════════════════════════════════════════════════════════════════════════════

def bulk_download(tickers: list[str], years: int) -> pd.DataFrame:
    """Download daily Close for all tickers in batched calls."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=365 * years)

    print(f"\n[STEP 2] Downloading {len(tickers)} tickers  "
          f"({start} → {end}) in batches of {BATCH_SIZE} …")

    frames = []
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch   = tickers[i : i + BATCH_SIZE]
        batch_n = i // BATCH_SIZE + 1
        print(f"  Batch {batch_n}/{n_batches}  "
              f"({len(batch)} tickers) …", end=" ", flush=True)

        try:
            raw = yf.download(
                batch,
                start=str(start), end=str(end),
                auto_adjust=True, progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})

            frames.append(close)
            print(f"OK  ({close.shape[1]} cols)")
        except Exception as exc:
            print(f"FAILED  ({exc})")

        time.sleep(0.5)

    if not frames:
        raise RuntimeError("All download batches failed.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    print(f"[STEP 2] Raw price matrix: {prices.shape[0]} days × "
          f"{prices.shape[1]} tickers.\n")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(prices: pd.DataFrame,
               missing_thresh: float = MISSING_THRESHOLD) -> pd.DataFrame:
    """Drop tickers with > threshold NaN fraction, then ffill residuals."""
    n_rows   = len(prices)
    frac_nan = prices.isna().sum() / n_rows
    bad_cols = frac_nan[frac_nan > missing_thresh].index.tolist()

    prices = prices.drop(columns=bad_cols)
    prices = prices.ffill(limit=2).dropna(axis=1)

    print(f"[STEP 3] Dropped {len(bad_cols)} tickers (>{missing_thresh*100:.0f}% "
          f"missing).  Remaining: {prices.shape[1]} tickers.\n")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — SAME-SECTOR CORRELATION PRE-FILTER
# ══════════════════════════════════════════════════════════════════════════════

def sector_corr_prefilter(prices: pd.DataFrame,
                          sector_df: pd.DataFrame,
                          corr_thresh: float = CORR_PRE_FILTER
                          ) -> list[tuple[str, str, str, float]]:
    """
    For each sector group, compute the correlation matrix and keep pairs
    with ρ ≥ corr_thresh.  Returns list of (A, B, Sector, corr).
    """
    print("[STEP 4] Same-sector correlation pre-filter …")

    available = set(prices.columns)
    sec_map   = (sector_df[sector_df["Symbol"].isin(available)]
                 .set_index("Symbol")["Sector"].to_dict())

    # Group tickers by sector
    groups = defaultdict(list)
    for sym, sec in sec_map.items():
        groups[sec].append(sym)

    candidates = []
    for sector, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        sub  = prices[members]
        corr = sub.corr()
        n    = len(members)
        for i in range(n):
            for j in range(i + 1, n):
                rho = corr.iloc[i, j]
                if rho >= corr_thresh:
                    candidates.append((members[i], members[j],
                                       sector, round(rho, 6)))

    print(f"         Same-sector pairs with ρ ≥ {corr_thresh}: "
          f"{len(candidates)}\n")
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — ENGLE-GRANGER COINTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════════════

def cointegration_filter(prices: pd.DataFrame,
                         candidates: list[tuple[str, str, str, float]],
                         pval_thresh: float = COINT_PVAL_THRESH
                         ) -> pd.DataFrame:
    """
    Run Engle-Granger cointegration on each candidate pair.
    Keep only pairs with coint p-value < threshold.
    """
    n_cand = len(candidates)
    print(f"[STEP 5] Running Engle-Granger cointegration on "
          f"{n_cand} pairs …")

    results = []
    tested  = 0

    for idx, (a, b, sector, rho) in enumerate(candidates):
        if (idx + 1) % 200 == 0:
            print(f"         … {idx+1}/{n_cand}", flush=True)
        try:
            # Use strict dropna per-pair — matches master_signal.py data hygiene.
            # The ffill in clean_data() was for correlation pre-filtering only;
            # cointegration tests should see only real observed prices.
            pair_data = prices[[a, b]].dropna()
            _, pval, _ = coint(pair_data[a], pair_data[b])
            tested += 1
            if pval < pval_thresh:
                results.append({
                    "Stock_A":      a,
                    "Stock_B":      b,
                    "Sector":       sector,
                    "Correlation":  rho,
                    "Coint_pval":   round(pval, 6),
                })
        except Exception:
            pass

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("Coint_pval").reset_index(drop=True)

    print(f"         Tested: {tested}  |  "
          f"Cointegrated (p < {pval_thresh}): {len(df)}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — EXPORT CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(df_pairs: pd.DataFrame, top_n: int = TOP_N_PAIRS,
               path: str = OUTPUT_CSV) -> pd.DataFrame:
    """Write the top-N cointegrated pairs to CSV."""
    df_out = df_pairs.head(top_n).copy()
    df_out.to_csv(path, index=False)
    print(f"[STEP 6] Saved → {path}  ({len(df_out)} rows)\n")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("╔" + "═" * 62 + "╗")
    print("║  NIGHTLY SCANNER v2 — Cointegration + Sector Filtering      ║")
    print(f"║  Run : {ts}                                ║")
    print("╚" + "═" * 62 + "╝")
    print()

    # 1. Get ticker universe with sectors
    sector_df = get_sp500_with_sectors()

    # 2. Bulk download
    all_tickers = sector_df["Symbol"].tolist()
    prices = bulk_download(all_tickers, LOOKBACK_YEARS)

    # 3. Clean
    prices = clean_data(prices)

    # 4. Same-sector correlation pre-filter
    candidates = sector_corr_prefilter(prices, sector_df)

    # 5. Cointegration test
    df_coint = cointegration_filter(prices, candidates)

    if len(df_coint) == 0:
        print("  ⚠  No cointegrated pairs found. CSV will be empty.")
        print("     Consider lowering CORR_PRE_FILTER or COINT_PVAL_THRESH.\n")
        # Write empty CSV so live_validator doesn't crash on missing file
        pd.DataFrame(columns=["Stock_A","Stock_B","Sector",
                               "Correlation","Coint_pval"]).to_csv(
            OUTPUT_CSV, index=False)
        return pd.DataFrame()

    # 6. Export top N
    df_out = export_csv(df_coint)

    # Preview
    print("  Top 10 Preview:")
    print("  " + "-" * 72)
    for _, row in df_out.head(10).iterrows():
        print(f"    {row['Stock_A']:>6} / {row['Stock_B']:<6}  "
              f"Sector: {row['Sector']:<28}  "
              f"ρ={row['Correlation']:.3f}  "
              f"coint p={row['Coint_pval']:.4f}")
    print()

    print("  \u26a0  NOTE: Universe is current S&P 500 constituents.")
    print("     Historical backtest results may exhibit survivorship bias.\n")

    return df_out


if __name__ == "__main__":
    main()
