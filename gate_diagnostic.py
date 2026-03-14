"""
Gate-by-gate diagnostic: runs every pair in daily_candidates.csv through the
backtest engine and reports WHICH gates each pair fails, plus aggregate stats.

Usage:  python gate_diagnostic.py
"""
import csv
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    INPUT_CSV, MIN_TRADES, MIN_WIN_RATE, MIN_PROFIT_FACTOR, MIN_TOTAL_PNL,
    MIN_SHARPE, MAX_DRAWDOWN, BETA_STABILITY_MAX, MIN_SORTINO, MIN_AVG_PNL,
    SPLIT_HALF_ENABLED, SPLIT_HALF_MIN_PNL, RECENT_MOMENTUM_N,
    RECENT_ADF_WINDOW, RECENT_ADF_PVAL, RECENT_CORR_WINDOW, MIN_RECENT_CORR,
    WALK_FORWARD_SPLIT, MIN_TRADES as MT,
)
from master_signal import fetch_pair, compute_rolling_signals, backtest_pair

from statsmodels.tsa.stattools import adfuller


def check_recent_adf(signals):
    """Gate C: recent ADF on last 90 days of spread."""
    spread = signals["spread"].dropna()
    recent = spread.iloc[-RECENT_ADF_WINDOW:]
    if len(recent) < 30:
        return False, 1.0
    try:
        pval = adfuller(recent, maxlag=5, autolag="AIC")[1]
    except Exception:
        return False, 1.0
    return pval < RECENT_ADF_PVAL, pval


def check_recent_corr(close, a, b):
    """Gate D: recent correlation on last 90 days."""
    recent = close.iloc[-RECENT_CORR_WINDOW:]
    if len(recent) < 30:
        return False, 0.0
    corr = recent[a].corr(recent[b])
    return corr >= MIN_RECENT_CORR, corr


def main():
    # Read candidates
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pairs = [(r["Stock_A"], r["Stock_B"]) for r in reader]

    total = len(pairs)
    print(f"Diagnosing {total} pairs from {INPUT_CSV}\n")

    # Gate failure counters
    gate_names = [
        "Data/Download",
        f"Min Trades (<{MIN_TRADES})",
        f"Win Rate (<{MIN_WIN_RATE}%)",
        f"Profit Factor (<{MIN_PROFIT_FACTOR}x)",
        f"Total P&L (<=${MIN_TOTAL_PNL})",
        f"Sharpe (<{MIN_SHARPE})",
        f"Max Drawdown (<${MAX_DRAWDOWN})",
        f"Beta CV (>{BETA_STABILITY_MAX})",
        f"Sortino (<{MIN_SORTINO})",
        f"Avg P&L (<${MIN_AVG_PNL})",
        "Split-Half (Gate A)",
        "Recent Momentum (Gate B)",
        f"Recent ADF p>{RECENT_ADF_PVAL} (Gate C)",
        f"Recent Corr <{MIN_RECENT_CORR} (Gate D)",
        "Walk-Forward OOS (Gate F)",
    ]
    gate_fail = {g: 0 for g in gate_names}
    gate_fail_pairs = {g: [] for g in gate_names}

    passed_all = []
    pair_results = []

    for idx, (a, b) in enumerate(pairs, 1):
        label = f"{a}/{b}"
        sys.stdout.write(f"\r  [{idx}/{total}] {label:<20}")
        sys.stdout.flush()

        # Fetch data
        close = fetch_pair(a, b)
        if close is None:
            gate_fail["Data/Download"] += 1
            gate_fail_pairs["Data/Download"].append(label)
            pair_results.append((label, ["Data/Download"], {}))
            continue

        signals = compute_rolling_signals(close, a, b)
        bt = backtest_pair(signals)

        failures = []

        # Standard gates
        if bt["n_trades"] < MIN_TRADES:
            k = f"Min Trades (<{MIN_TRADES})"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        if bt["win_rate"] < MIN_WIN_RATE:
            k = f"Win Rate (<{MIN_WIN_RATE}%)"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        if bt["profit_factor"] < MIN_PROFIT_FACTOR:
            k = f"Profit Factor (<{MIN_PROFIT_FACTOR}x)"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        if bt["total_pnl"] <= MIN_TOTAL_PNL:
            k = f"Total P&L (<=${MIN_TOTAL_PNL})"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        if bt["sharpe"] < MIN_SHARPE:
            k = f"Sharpe (<{MIN_SHARPE})"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        if bt["max_dd"] < MAX_DRAWDOWN:
            k = f"Max Drawdown (<${MAX_DRAWDOWN})"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        # Beta CV is now a soft warning, not a hard gate
        # if bt["beta_cv"] > BETA_STABILITY_MAX:
        #     k = f"Beta CV (>{BETA_STABILITY_MAX})"
        #     gate_fail[k] += 1
        #     gate_fail_pairs[k].append(label)
        #     failures.append(k)

        if bt["sortino"] < MIN_SORTINO:
            k = f"Sortino (<{MIN_SORTINO})"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        if bt["avg_pnl"] < MIN_AVG_PNL:
            k = f"Avg P&L (<${MIN_AVG_PNL})"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        # Gate A: Split-Half
        if "H1 P&L" in bt.get("reason", "") or "H2 P&L" in bt.get("reason", "") or "Split-Half" in bt.get("reason", ""):
            k = "Split-Half (Gate A)"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        # Gate B: Recent Momentum
        if "Momentum" in bt.get("reason", ""):
            k = "Recent Momentum (Gate B)"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        # Gate C: Recent ADF
        adf_pass, adf_pval = check_recent_adf(signals)
        if not adf_pass:
            k = f"Recent ADF p>{RECENT_ADF_PVAL} (Gate C)"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        # Gate D: Recent Correlation
        corr_pass, corr_val = check_recent_corr(close, a, b)
        if not corr_pass:
            k = f"Recent Corr <{MIN_RECENT_CORR} (Gate D)"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        # Gate F: Walk-Forward
        if "Walk-Forward" in bt.get("reason", ""):
            k = "Walk-Forward OOS (Gate F)"
            gate_fail[k] += 1
            gate_fail_pairs[k].append(label)
            failures.append(k)

        if not failures:
            passed_all.append(label)

        pair_results.append((label, failures, bt))

    print("\r" + " " * 60)

    # ── Summary Report ──────────────────────────────────────────────
    print("=" * 65)
    print("  GATE-BY-GATE FAILURE REPORT")
    print("=" * 65)
    print(f"  Total pairs analysed: {total}")
    print(f"  Passed ALL gates:     {len(passed_all)}")
    print("-" * 65)
    print(f"  {'Gate':<40} {'Fail':>5} {'Rate':>7}")
    print("-" * 65)

    # Sort by failure count descending
    sorted_gates = sorted(gate_fail.items(), key=lambda x: -x[1])
    for gate, count in sorted_gates:
        rate = count / total * 100 if total > 0 else 0
        bar = "#" * int(rate / 2)
        print(f"  {gate:<40} {count:>5} {rate:>5.1f}%  {bar}")

    print("-" * 65)

    # ── Distribution of failure counts ──────────────────────────────
    print("\n  FAILURE COUNT DISTRIBUTION")
    print("-" * 40)
    fail_counts = {}
    for label, failures, bt in pair_results:
        n = len(failures)
        fail_counts[n] = fail_counts.get(n, 0) + 1
    for n in sorted(fail_counts.keys()):
        print(f"    {n} gates failed: {fail_counts[n]:>4} pairs")

    # ── "Almost passed" pairs (1-2 failures) ────────────────────────
    print("\n  NEAR-MISS PAIRS (failed 1-2 gates only)")
    print("-" * 65)
    near_misses = [(l, f, b) for l, f, b in pair_results if 0 < len(f) <= 2]
    if near_misses:
        for label, failures, bt in near_misses:
            wr = bt.get("win_rate", 0)
            pnl = bt.get("total_pnl", 0)
            bcv = bt.get("beta_cv", 0)
            sh = bt.get("sharpe", 0)
            print(f"  {label:<15} WR={wr:.0f}% PnL=${pnl:+.0f} BetaCV={bcv:.2f} Sharpe={sh:+.2f}")
            for f in failures:
                print(f"    FAILED: {f}")
    else:
        print("  (none)")

    # ── Pairs that passed all gates ─────────────────────────────────
    if passed_all:
        print(f"\n  PASSED ALL GATES ({len(passed_all)}):")
        for p in passed_all:
            print(f"    {p}")

    # ── Key stats for top pairs by total P&L ────────────────────────
    print("\n  TOP 10 PAIRS BY TOTAL P&L (regardless of gates)")
    print("-" * 65)
    scored = [(l, f, b) for l, f, b in pair_results if b.get("n_trades", 0) > 0]
    scored.sort(key=lambda x: -x[2].get("total_pnl", 0))
    for label, failures, bt in scored[:10]:
        n_fail = len(failures)
        print(f"  {label:<15} PnL=${bt['total_pnl']:>+8.0f}  WR={bt['win_rate']:.0f}%  "
              f"Sharpe={bt['sharpe']:+.2f}  BetaCV={bt['beta_cv']:.2f}  "
              f"Fails={n_fail}")

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
