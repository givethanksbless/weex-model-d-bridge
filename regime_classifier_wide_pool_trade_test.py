"""
regime_classifier_wide_pool_trade_test.py

Widens Part 4's trade-outcome test (regime_classifier_v2.py) the same way
the spot leg widened its flash-crash test when it was too thin (Part 11 of
forex-script-automation-addendum-2026-08-20.md: 4 symbols -> 56 symbols,
pooled, real Wilson CI). Part 4 tested only AUDIOUSDT/LPTUSDT/FLUXUSDT and
got a real but statistically thin (n=25 vs 11) result that went the wrong
direction. This script asks the same question -- does the classifier's
BULL-like flag predict weaker real trades? -- across every cached symbol
that has all three regime files and ISN'T part of the reference basket
(COMPUSDT/CRVUSDT/ONTUSDT/CELRUSDT are excluded: they ARE what the
classifier is built from, so testing their own trades against it would be
circular).

Uses the SAME fitted classifier as regime_classifier_v2.py (tier-1 basket,
centroids fit on TRAIN windows only) and the SAME locked, unmodified signal
engine to generate real trades. Restricted to held-out entries only (same
per-regime test-cutoff timestamps as v2), so nothing here was used to fit
the classifier.

Performance note: v2's classify_at_time() reloads and fully recomputes ATR
over the whole series on every call, which does not scale to pooling
thousands of trades across 88 symbols. This script precomputes the
reference basket's bars/ATR arrays ONCE per regime and does fast in-memory
window lookups instead -- verified to produce IDENTICAL results to
classify_at_time() on a sample before being trusted (see sanity checks).

Run: python3 regime_classifier_wide_pool_trade_test.py
"""
import os
import bisect
import math
import regime_classifier_v2 as R

EXCLUDE_BASKET = set(R.REFERENCE_SYMBOLS)  # COMPUSDT, CRVUSDT, ONTUSDT, CELRUSDT


def discover_wide_pool():
    files = os.listdir(R.DATA_DIR)
    by_symbol = {}
    for f in files:
        for suf in ("cache", "calm2023", "bull2024h1"):
            suffix_tag = f"_5m_binance_{suf}.json"
            if f.endswith(suffix_tag):
                sym = f[: -len(suffix_tag)].upper()
                by_symbol.setdefault(sym, set()).add(suf)
    return sorted(s for s, sufs in by_symbol.items() if sufs == {"cache", "calm2023", "bull2024h1"} and s not in EXCLUDE_BASKET)


def precompute_basket(suffix):
    """Precompute bars45/atr/closes ONCE per regime for the reference basket,
    so per-trade classification is fast array lookups, not file reloads."""
    per_symbol = {}
    for sym in R.REFERENCE_SYMBOLS:
        bars5 = R.load_bars(sym, suffix)
        bars45 = R.resample_45m(bars5)
        atr = R.compute_atr(bars45)
        closes = [b["c"] for b in bars45]
        ts = [b["t"] for b in bars45]
        per_symbol[sym] = {"bars45": bars45, "atr": atr, "closes": closes, "ts": ts}
    return per_symbol


def fast_classify_at_time(precomputed, t_ms, norm_stats, centroids, window_days=R.WINDOW_DAYS):
    win_bars = int(round(window_days * R.BARS_PER_DAY_45M))
    feats = []
    for sym, d in precomputed.items():
        ts = d["ts"]
        end_idx = bisect.bisect_right(ts, t_ms) - 1
        if end_idx < 0:
            return None
        start_idx = end_idx - win_bars + 1
        if start_idx < 0:
            return None
        atr, closes = d["atr"], d["closes"]
        seg_atr = [atr[i] / closes[i] * 100.0 for i in range(start_idx, end_idx + 1) if atr[i] is not None and closes[i] > 0]
        if not seg_atr:
            return None
        c0, c1 = closes[start_idx], closes[end_idx]
        if c0 <= 0:
            return None
        feats.append({"ret_pct": (c1 / c0 - 1.0) * 100.0, "atr_pct": sum(seg_atr) / len(seg_atr)})
    ret_pct = sum(f["ret_pct"] for f in feats) / len(feats)
    atr_pct = sum(f["atr_pct"] for f in feats) / len(feats)
    return R.classify(ret_pct, atr_pct, norm_stats, centroids)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - adj) / denom, (centre + adj) / denom)


def run_sanity_checks():
    print("=== SANITY CHECKS ===")
    # verify fast_classify_at_time matches v2's own classify_at_time exactly
    # on a real sample -- this MUST hold before trusting the faster version.
    all_windows = {}
    for suffix, label in R.REGIMES:
        all_windows[label] = R.basket_rolling_features(suffix, R.REFERENCE_SYMBOLS, R.WINDOW_DAYS, R.STEP_DAYS)
    norm_stats, centroids, train_split, test_split = R.build_classifier(all_windows, R.TRAIN_FRAC)

    suffix, label = "cache", "CRASH_2025-26"
    precomputed = precompute_basket(suffix)
    sample_ts = precomputed[R.REFERENCE_SYMBOLS[0]]["ts"]
    mismatches = 0
    checked = 0
    for idx in range(3000, len(sample_ts), 700):  # spot-check across the series
        t = sample_ts[idx]
        slow = R.classify_at_time(t, suffix, R.REFERENCE_SYMBOLS, norm_stats, centroids)
        fast = fast_classify_at_time(precomputed, t, norm_stats, centroids)
        checked += 1
        if slow is None or fast is None:
            if slow != fast:
                mismatches += 1
            continue
        if slow["nearest"] != fast["nearest"] or abs(slow["distances"][slow["nearest"]] - fast["distances"][fast["nearest"]]) > 1e-6:
            mismatches += 1
    assert checked > 0, "no sample points checked"
    assert mismatches == 0, f"fast_classify_at_time disagrees with classify_at_time on {mismatches}/{checked} samples"
    print(f"  fast_classify_at_time matches classify_at_time exactly on {checked} spot-checked timestamps -- OK")

    # Wilson CI sanity: textbook value, 10/20 at 95% -> [29.93%, 70.07%] (used
    # elsewhere in this project, e.g. ninth_batch_stability_check.py per the
    # spot-leg addendum's own verification note)
    lo, hi = wilson_ci(10, 20)
    assert abs(lo - 0.2993) < 0.001 and abs(hi - 0.7007) < 0.001, f"Wilson CI sanity failed: {lo}, {hi}"
    print(f"  Wilson CI(10/20) = [{lo*100:.2f}%, {hi*100:.2f}%], matches known textbook value -- OK")
    print("=== SANITY CHECKS PASSED ===\n")
    return norm_stats, centroids, test_split


def main():
    norm_stats, centroids, test_split = run_sanity_checks()
    cutoffs = {label: (test_split[label][0]["t_start"] if test_split[label] else None) for label in centroids}

    pool = discover_wide_pool()
    print(f"=== Wide pool: {len(pool)} symbols with all 3 regime files, excluding the 4 reference-basket coins ===\n")

    precomputed_by_suffix = {suffix: precompute_basket(suffix) for suffix, label in R.REGIMES}

    bull_rs, other_rs = [], []
    per_symbol_rows = []
    n_symbols_with_trades = 0

    for i, sym in enumerate(pool):
        sym_bull_rs, sym_other_rs = [], []
        for suffix, label in R.REGIMES:
            bars5 = R.load_bars(sym, suffix)
            if not bars5:
                continue
            bars45 = R.resample_45m(bars5)
            trades = R.detect_trades(bars45, sym)
            cutoff = cutoffs[label]
            if cutoff is None:
                continue
            held_out = [t for t in trades if t["entry_t"] >= cutoff]
            for t in held_out:
                res = fast_classify_at_time(precomputed_by_suffix[suffix], t["entry_t"], norm_stats, centroids)
                if res is None:
                    continue
                if res["nearest"] == "BULL_2024H1":
                    bull_rs.append(t["r"])
                    sym_bull_rs.append(t["r"])
                else:
                    other_rs.append(t["r"])
                    sym_other_rs.append(t["r"])
        if sym_bull_rs or sym_other_rs:
            n_symbols_with_trades += 1
            per_symbol_rows.append((sym, len(sym_bull_rs), len(sym_other_rs),
                                     sum(sym_bull_rs) / len(sym_bull_rs) if sym_bull_rs else None,
                                     sum(sym_other_rs) / len(sym_other_rs) if sym_other_rs else None))
        if (i + 1) % 20 == 0:
            print(f"  ...processed {i+1}/{len(pool)} symbols")

    print(f"\n=== Done. {n_symbols_with_trades}/{len(pool)} symbols produced at least one held-out trade. ===\n")

    print(f"{'symbol':10s} {'n_BULL':>7s} {'avgR_BULL':>10s} {'n_other':>8s} {'avgR_other':>11s}")
    for sym, nb, no, ab, ao in per_symbol_rows:
        ab_s = f"{ab:+.2f}" if ab is not None else "n/a"
        ao_s = f"{ao:+.2f}" if ao is not None else "n/a"
        print(f"{sym:10s} {nb:7d} {ab_s:>10s} {no:8d} {ao_s:>11s}")

    n_bull, n_other = len(bull_rs), len(other_rs)
    wins_bull = sum(1 for r in bull_rs if r == R.TP_RR)
    wins_other = sum(1 for r in other_rs if r == R.TP_RR)
    avg_bull = sum(bull_rs) / n_bull if n_bull else float("nan")
    avg_other = sum(other_rs) / n_other if n_other else float("nan")
    wr_bull = wins_bull / n_bull if n_bull else float("nan")
    wr_other = wins_other / n_other if n_other else float("nan")
    lo_b, hi_b = wilson_ci(wins_bull, n_bull) if n_bull else (float("nan"), float("nan"))
    lo_o, hi_o = wilson_ci(wins_other, n_other) if n_other else (float("nan"), float("nan"))

    print(f"\n=== POOLED RESULT ({n_symbols_with_trades} symbols, held-out trades only, wide pool) ===")
    print(f"  BULL-flagged (Model D -> HALF_SIZE):      n={n_bull:5d}  avg R={avg_bull:+.3f}  win rate={100*wr_bull:.1f}%  95% CI [{100*lo_b:.1f}%, {100*hi_b:.1f}%]")
    print(f"  CRASH/CALM-flagged (Model D -> FULL_SIZE): n={n_other:5d}  avg R={avg_other:+.3f}  win rate={100*wr_other:.1f}%  95% CI [{100*lo_o:.1f}%, {100*hi_o:.1f}%]")
    if n_bull > 0 and n_other > 0:
        ci_overlap = not (hi_b < lo_o or hi_o < lo_b)
        print(f"  Win-rate 95% CIs {'OVERLAP -- difference not statistically clear at this sample size' if ci_overlap else 'DO NOT OVERLAP -- a real, statistically distinguishable difference'}.")
        if avg_bull < avg_other:
            print(f"  Direction: BULL-flagged trades DID average worse ({avg_bull:+.3f} vs {avg_other:+.3f}) -- consistent with Model D's premise, at this sample size.")
        else:
            print(f"  Direction: BULL-flagged trades did NOT average worse ({avg_bull:+.3f} vs {avg_other:+.3f}) -- still does not support Model D's premise, reported honestly.")
    print("\nRecommendation-building result only, not a live/adopted system.")


if __name__ == "__main__":
    main()
