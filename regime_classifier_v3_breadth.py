"""
regime_classifier_v3_breadth.py

Adds a THIRD feature to regime_classifier_v2.py's classifier: cross-sectional
BREADTH -- what fraction of a wide, independent basket of coins are strongly
trending up (trailing 60-day return > +20%) at the same time. This is a
genuinely different kind of signal than return%/ATR% on one basket: it's a
macro/structural trait (broad-based participation) rather than a single
basket's own chart shape, prompted directly by the user's question of
whether there's something uniquely bullish that a narrower price-action lens
would miss.

Real exploration before building this (ad hoc, not saved as a script):
median breadth (%%pool with 60d return > +20%%) was 11% inside BULL_2024H1
vs 6% inside CALM_2023, with a much wider top quartile (76% vs 41%) --
a real, if partial, separation that return%/ATR% alone don't fully capture,
since BULL and CALM's median RETURN values were shown earlier (Part 2/4) to
sometimes point the "wrong" way (CALM_2023 had a higher median 60d return
than BULL_2024H1 for BTC/ETH).

BREADTH BASKET, and why it's a separate basket from the return/ATR
reference basket: computed from all symbols with cached data across all 3
regimes MINUS AUDIOUSDT/LPTUSDT/FLUXUSDT (the coins actually being scored in
the trade-outcome test below) -- so a coin's own trades are never scored
against a breadth signal partly built from its own price action. This
basket DOES include the tier-1 return/ATR reference symbols (COMPUSDT,
CRVUSDT, ONTUSDT, CELRUSDT) plus everything from the wide pool used in
regime_classifier_wide_pool_trade_test.py except the 3 tier-2 coins --
89 symbols in total.

Scope note, stated honestly: this version is validated specifically against
the tier-2 (AUDIOUSDT/LPTUSDT/FLUXUSDT) trade-outcome question -- the actual
thing Model D needs -- rather than immediately re-running the full 88-symbol
wide-pool test from Part 7. If this improves the tier-2 result, widening it
further is a reasonable next step; if it doesn't, there's no point paying
that extra compute first.

Run: python3 regime_classifier_v3_breadth.py
"""
import os
import bisect
import regime_classifier_v2 as R
import regime_classifier_wide_pool_trade_test as W

TIER2 = set(R.TIER2_SYMBOLS)  # AUDIOUSDT, LPTUSDT, FLUXUSDT -- excluded from the breadth basket
BREADTH_RET_THRESHOLD = 20.0  # %, matches the real exploration that motivated this feature


def breadth_basket_symbols():
    wide = set(W.discover_wide_pool())  # excludes tier-1 (return/ATR reference basket) already
    universe = wide | set(R.REFERENCE_SYMBOLS)  # add tier-1 back in -- fine for breadth, not circular here
    return sorted(universe - TIER2)


def precompute_breadth_basket(suffix, symbols):
    """closes + timestamps only -- breadth needs trailing return, not ATR."""
    per_symbol = {}
    for sym in symbols:
        bars5 = R.load_bars(sym, suffix)
        if not bars5:
            continue
        bars45 = R.resample_45m(bars5)
        if len(bars45) < 10:
            continue
        per_symbol[sym] = {"closes": [b["c"] for b in bars45], "ts": [b["t"] for b in bars45]}
    return per_symbol


def breadth_at(precomputed, t_ms, window_days=R.WINDOW_DAYS, threshold=BREADTH_RET_THRESHOLD):
    win_bars = int(round(window_days * R.BARS_PER_DAY_45M))
    n_above, n_total = 0, 0
    for sym, d in precomputed.items():
        ts, closes = d["ts"], d["closes"]
        end_idx = bisect.bisect_right(ts, t_ms) - 1
        if end_idx < 0:
            continue
        start_idx = end_idx - win_bars + 1
        if start_idx < 0:
            continue
        c0, c1 = closes[start_idx], closes[end_idx]
        if c0 <= 0:
            continue
        ret_pct = (c1 / c0 - 1.0) * 100.0
        n_total += 1
        if ret_pct > threshold:
            n_above += 1
    if n_total == 0:
        return None
    return n_above / n_total


def build_3feature_windows(suffix, breadth_precomputed):
    """Combine tier-1 return%/ATR% (unchanged from v2) with breadth% at the
    same window end-times."""
    base = R.basket_rolling_features(suffix, R.REFERENCE_SYMBOLS, R.WINDOW_DAYS, R.STEP_DAYS)
    out = []
    for w in base:
        b = breadth_at(breadth_precomputed, w["t_end"])
        if b is None:
            continue
        out.append({"t_start": w["t_start"], "t_end": w["t_end"], "ret_pct": w["ret_pct"], "atr_pct": w["atr_pct"], "breadth": b})
    return out


def zscore3(w, stats):
    rm, rs, am, asd, bm, bs = stats
    return (R.zscore(w["ret_pct"], rm, rs), R.zscore(w["atr_pct"], am, asd), R.zscore(w["breadth"], bm, bs))


def euclid3(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def build_classifier_3f(all_windows, train_frac=R.TRAIN_FRAC):
    train_split, test_split = {}, {}
    for label, ws in all_windows.items():
        ws_sorted = sorted(ws, key=lambda w: w["t_start"])
        cut = int(round(len(ws_sorted) * train_frac))
        train_split[label] = ws_sorted[:cut]
        test_split[label] = ws_sorted[cut:]
    pooled = [w for ws in train_split.values() for w in ws]
    rets = [w["ret_pct"] for w in pooled]
    atrs = [w["atr_pct"] for w in pooled]
    breadths = [w["breadth"] for w in pooled]
    stats = (sum(rets) / len(rets), R._std(rets), sum(atrs) / len(atrs), R._std(atrs), sum(breadths) / len(breadths), R._std(breadths))
    centroids = {}
    for label, ws in train_split.items():
        zs = [zscore3(w, stats) for w in ws]
        centroids[label] = tuple(sum(z[i] for z in zs) / len(zs) for i in range(3))
    return stats, centroids, train_split, test_split


def classify3(ret_pct, atr_pct, breadth, stats, centroids):
    z = zscore3({"ret_pct": ret_pct, "atr_pct": atr_pct, "breadth": breadth}, stats)
    dists = {label: euclid3(z, c) for label, c in centroids.items()}
    nearest = min(dists, key=dists.get)
    action = "HALF_SIZE" if nearest == "BULL_2024H1" else "FULL_SIZE"
    return {"nearest": nearest, "distances": dists, "model_d_action": action}


def run_sanity_checks():
    print("=== SANITY CHECKS ===")
    # breadth_at hand-computed check: 3 synthetic symbols, known returns
    class FakeBars:
        pass
    precomputed = {
        "A": {"ts": [0, 1000], "closes": [100.0, 130.0]},  # +30% -> above 20% threshold
        "B": {"ts": [0, 1000], "closes": [100.0, 110.0]},  # +10% -> below
        "C": {"ts": [0, 1000], "closes": [100.0, 121.0]},  # +21% -> above
    }
    # win_bars=2 (start_idx=0, end_idx=1) so the window spans both synthetic bars.
    window_days_for_2_bars = 2.0 / R.BARS_PER_DAY_45M
    b = breadth_at(precomputed, 1000, window_days=window_days_for_2_bars, threshold=20.0)
    n_above = 0
    for sym, d in precomputed.items():
        c0, c1 = d["closes"][0], d["closes"][1]
        ret = (c1 / c0 - 1.0) * 100.0
        if ret > 20.0:
            n_above += 1
    expected = n_above / 3
    assert abs(expected - 2 / 3) < 1e-9, "hand-check itself is wrong"
    assert b is not None and abs(b - expected) < 1e-9, f"breadth_at sanity failed: got {b}, expected {expected}"
    print(f"  breadth_at() on 3 synthetic symbols (A +30%%, B +10%%, C +21%%, threshold 20%%): got {b:.4f}, expected {expected:.4f} -- OK")

    assert abs(euclid3((0, 0, 0), (1, 2, 2)) - 3.0) < 1e-9, "euclid3 sanity failed"
    print("  euclid3 hand-computed check (0,0,0)->(1,2,2) = 3.0 -- OK")

    print("=== SANITY CHECKS PASSED ===\n")


def main():
    run_sanity_checks()

    symbols = breadth_basket_symbols()
    print(f"=== Breadth basket: {len(symbols)} symbols (wide pool + tier-1, excluding tier-2) ===\n")

    print("=== Building 3-feature (return%, ATR%, breadth%) archetype windows ===")
    all_windows = {}
    breadth_precomputed_by_suffix = {}
    for suffix, label in R.REGIMES:
        bp = precompute_breadth_basket(suffix, symbols)
        breadth_precomputed_by_suffix[suffix] = bp
        all_windows[label] = build_3feature_windows(suffix, bp)
        print(f"  {label:14s} {len(all_windows[label]):4d} windows")

    stats, centroids, train_split, test_split = build_classifier_3f(all_windows)
    print(f"\nCentroids (z_ret, z_atr, z_breadth):")
    for label, c in centroids.items():
        print(f"  {label:14s} ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})")

    print("\n=== Calendar-label held-out accuracy, 3-feature classifier ===")
    total, correct = 0, 0
    per_label = {l: [0, 0] for l in centroids}
    for true_label, ws in test_split.items():
        for w in ws:
            res = classify3(w["ret_pct"], w["atr_pct"], w["breadth"], stats, centroids)
            total += 1
            per_label[true_label][1] += 1
            if res["nearest"] == true_label:
                correct += 1
                per_label[true_label][0] += 1
    print(f"  Overall: {correct}/{total} = {100.0*correct/total:.1f}%")
    for l, (c, n) in per_label.items():
        print(f"  {l:14s} {c}/{n} = {100.0*c/n:.1f}%")

    cutoffs = {label: (test_split[label][0]["t_start"] if test_split[label] else None) for label in centroids}

    print("\n=== PRIMARY TEST: does the 3-feature BULL flag predict weaker tier-2 (AUDIO/LPT/FLUX) trades? ===")
    tier1_precomputed_by_suffix = {suffix: W.precompute_basket(suffix) for suffix, _ in R.REGIMES}

    bull_rs, other_rs = [], []
    per_symbol_summary = []
    for sym in R.TIER2_SYMBOLS:
        sym_bull, sym_other = [], []
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
            tier1_pre = tier1_precomputed_by_suffix[suffix]
            breadth_pre = breadth_precomputed_by_suffix[suffix]
            win_bars = int(round(R.WINDOW_DAYS * R.BARS_PER_DAY_45M))
            for t in held_out:
                # Pull raw ret_pct/atr_pct directly from the precomputed tier-1
                # arrays (same reference basket as v2), then combine with breadth.
                feats = []
                for rsym, d in tier1_pre.items():
                    ts = d["ts"]
                    end_idx = bisect.bisect_right(ts, t["entry_t"]) - 1
                    if end_idx < 0:
                        continue
                    start_idx = end_idx - win_bars + 1
                    if start_idx < 0:
                        continue
                    atrv, closes = d["atr"], d["closes"]
                    seg = [atrv[i] / closes[i] * 100.0 for i in range(start_idx, end_idx + 1) if atrv[i] is not None and closes[i] > 0]
                    if not seg:
                        continue
                    c0, c1 = closes[start_idx], closes[end_idx]
                    if c0 <= 0:
                        continue
                    feats.append(((c1 / c0 - 1.0) * 100.0, sum(seg) / len(seg)))
                if not feats:
                    continue
                ret_pct = sum(f[0] for f in feats) / len(feats)
                atr_pct = sum(f[1] for f in feats) / len(feats)
                b = breadth_at(breadth_pre, t["entry_t"])
                if b is None:
                    continue
                res = classify3(ret_pct, atr_pct, b, stats, centroids)
                if res["nearest"] == "BULL_2024H1":
                    bull_rs.append(t["r"])
                    sym_bull.append(t["r"])
                else:
                    other_rs.append(t["r"])
                    sym_other.append(t["r"])
        avg_b = sum(sym_bull) / len(sym_bull) if sym_bull else None
        avg_o = sum(sym_other) / len(sym_other) if sym_other else None
        per_symbol_summary.append((sym, len(sym_bull), len(sym_other), avg_b, avg_o))

    print(f"  {'symbol':10s} {'n_BULL':>7s} {'avgR_BULL':>10s} {'n_other':>8s} {'avgR_other':>11s}")
    for sym, nb, no, ab, ao in per_symbol_summary:
        ab_s = f"{ab:+.2f}" if ab is not None else "n/a"
        ao_s = f"{ao:+.2f}" if ao is not None else "n/a"
        print(f"  {sym:10s} {nb:7d} {ab_s:>10s} {no:8d} {ao_s:>11s}")

    n_bull, n_other = len(bull_rs), len(other_rs)
    avg_bull = sum(bull_rs) / n_bull if n_bull else float("nan")
    avg_other = sum(other_rs) / n_other if n_other else float("nan")
    print(f"\n  POOLED (tier-2 only, held-out): BULL-flagged n={n_bull} avg R={avg_bull:+.3f}  |  other-flagged n={n_other} avg R={avg_other:+.3f}")
    if n_bull > 0 and n_other > 0:
        if avg_bull < avg_other:
            print(f"  -> BULL-flagged trades DID average worse with the 3-feature (breadth-added) classifier -- supports Model D's premise.")
        else:
            print(f"  -> BULL-flagged trades still did NOT average worse -- breadth did not fix the core problem, reported honestly.")
    else:
        print("  -> Not enough trades in one or both buckets to draw a conclusion.")

    print("\nRecommendation-building result only, not a live/adopted system.")


if __name__ == "__main__":
    main()
