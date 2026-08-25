"""
regime_classifier_v2.py -- Regime Classification Leg, v2.

Supersedes regime_classifier_v1.py. Two real problems were found in v1 and
are fixed here, not glossed over:

1. v1 used BTCUSDT+ETHUSDT as the reference basket. Checked (real data):
   return-direction correlation between BTC/ETH and the 7 Model-D-relevant
   coins was reasonable (+0.5 to +0.97 in most cases), but ATR%(volatility)
   correlation was weak and inconsistent (e.g. COMPUSDT vs BTC/ETH in
   CALM_2023: -0.01: no relationship at all; ONTUSDT in CRASH: +0.04;
   LPTUSDT in CALM: +0.01). Since ATR% is the feature that separates
   CALM from BULL in this classifier, a BTC/ETH-calibrated volatility read
   is not reliably relevant to what these altcoins individually experience.
   FIX: reference basket is now the 4 already-qualified watchlist coins
   (COMPUSDT, CRVUSDT, ONTUSDT, CELRUSDT) -- real, trusted, traded altcoins,
   independent of AUDIOUSDT/LPTUSDT/FLUXUSDT (the coins being sized), so the
   validation below isn't testing a coin against itself.

2. v1's headline validation metric was "does a held-out window classify as
   its own calendar-year label" -- 51% overall, worse than useless-looking
   for BULL_2024H1 (0%). Investigated: this was found to be a real artifact
   of the archetype labels not being internally uniform (e.g. BULL_2024H1's
   own real trajectory is a genuine rally Feb-Apr 2024 followed by a real
   correction/chop Apr-Jun 2024 -- the classifier was correctly reading the
   correction as non-bull-like, the calendar label just doesn't apply
   uniformly across its own window). That's a real, honest explanation, but
   it does not make 51% a good number to hang confidence on, and redefining
   "core periods" post-hoc to make the number look better would be exactly
   the kind of self-serving in-sample pick this project's own history
   (Part 15) already flagged as a methodological trap.
   FIX: this version's PRIMARY validation is not calendar-label matching.
   It is a direct, non-circular test of usefulness: classify the market at
   the entry time of every real historical AUDIOUSDT/LPTUSDT/FLUXUSDT trade
   (using the LOCKED, unmodified signal engine to generate those trades),
   restricted to entries that fall in the TEST (held-out) portion of the
   basket's own train/test split -- then check whether trades the
   classifier calls BULL-like actually perform worse than trades it calls
   CRASH/CALM-like. That is the exact question Model D needs answered:
   does this signal predict when these coins' trades are more likely to be
   weak, not does it match an admittedly-imperfect calendar tag. The
   calendar-label accuracy from v1 is still reported below too, honestly,
   alongside this -- not hidden, just no longer the headline.

Still an additive, recommendation-building layer only. Does not modify the
locked signal engine, does not touch coin selection/watchlist status, not
wired into live trading.

Run: python3 regime_classifier_v2.py
"""
import os
import json
import math
import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RSI_LEN = 14
ATR_LEN = 14
SMA_LEN = 200
STOP_ATR_MULT = 1.0
TP_RR = 4.5

WINDOW_DAYS = 60
STEP_DAYS = 5
BOUNDARY_STEP_DAYS = 1
TRAIN_FRAC = 0.70

REFERENCE_SYMBOLS = ["COMPUSDT", "CRVUSDT", "ONTUSDT", "CELRUSDT"]  # tier-1, independent of tier-2
TIER2_SYMBOLS = ["AUDIOUSDT", "LPTUSDT", "FLUXUSDT"]
REGIMES = [("cache", "CRASH_2025-26"), ("calm2023", "CALM_2023"), ("bull2024h1", "BULL_2024H1")]
BARS_PER_DAY_45M = 24 * 60 / 45.0  # 32


# ---------------------------------------------------------------------------
# Locked-engine reference functions -- copied unmodified in logic from the
# project's locked spec (load_bars / resample_45m / compute_rsi / compute_atr
# / compute_sma / detect_trades). detect_trades is needed in this version
# (unlike v1) to generate real tier-2 trades for the direct usefulness test.
# ---------------------------------------------------------------------------

def load_bars(symbol, suffix):
    candidates = [
        os.path.join(DATA_DIR, f"{symbol}_5m_binance_{suffix}.json"),
        os.path.join(DATA_DIR, f"{symbol.lower()}_5m_binance_{suffix}.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return []
    with open(path) as f:
        raw = json.load(f)
    rows = raw if isinstance(raw, list) else raw.get("bars", raw.get("data", []))
    bars = []
    for b in rows:
        try:
            if isinstance(b, dict):
                t = int(b.get("t") if b.get("t") is not None else (b.get("openTime") or b.get("timestamp")))
                o = float(b["o"] if b.get("o") is not None else b["open"])
                h = float(b["h"] if b.get("h") is not None else b["high"])
                l = float(b["l"] if b.get("l") is not None else b["low"])
                c = float(b["c"] if b.get("c") is not None else b["close"])
            else:
                t, o, h, l, c = int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])
            bars.append({"t": t, "o": o, "h": h, "l": l, "c": c})
        except Exception:
            continue
    bars.sort(key=lambda b: b["t"])
    return bars


def resample_45m(bars5):
    out = []
    n = len(bars5)
    BARS_PER_CHUNK = 9
    for i in range(0, n - BARS_PER_CHUNK + 1, BARS_PER_CHUNK):
        chunk = bars5[i:i + BARS_PER_CHUNK]
        if len(chunk) < BARS_PER_CHUNK:
            break
        out.append({
            "t": chunk[0]["t"], "o": chunk[0]["o"],
            "h": max(b["h"] for b in chunk), "l": min(b["l"] for b in chunk),
            "c": chunk[-1]["c"],
        })
    return out


def compute_rsi(closes, length=RSI_LEN):
    rsis = [None] * len(closes)
    if len(closes) < length + 1:
        return rsis
    gains, losses = [], []
    for i in range(1, length + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length
    rsis[length] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    for i in range(length + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain, loss = max(change, 0), max(-change, 0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        rsis[i] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    return rsis


def compute_atr(bars, length=ATR_LEN):
    n = len(bars)
    tr = [None] * n
    for i in range(n):
        if i == 0:
            tr[i] = bars[i]["h"] - bars[i]["l"]
        else:
            tr[i] = max(bars[i]["h"] - bars[i]["l"], abs(bars[i]["h"] - bars[i - 1]["c"]), abs(bars[i]["l"] - bars[i - 1]["c"]))
    atr = [None] * n
    for i in range(n):
        if i < length - 1:
            continue
        atr[i] = sum(tr[0:i + 1]) / length if i == length - 1 else (atr[i - 1] * (length - 1) + tr[i]) / length
    return atr


def compute_sma(closes, length=SMA_LEN):
    out = [None] * len(closes)
    running = 0.0
    for i, c in enumerate(closes):
        running += c
        if i >= length:
            running -= closes[i - length]
        if i >= length - 1:
            out[i] = running / length
    return out


def detect_trades(bars45, symbol, stop_atr_mult=STOP_ATR_MULT, tp_rr=TP_RR):
    """Locked signal engine -- do not modify. Copied unmodified for this
    leg's own trade-level validation only; not used to trade anything."""
    n = len(bars45)
    warmup = max(SMA_LEN, RSI_LEN + 1, ATR_LEN)
    if n < warmup + 5:
        return []
    closes = [b["c"] for b in bars45]
    rsi = compute_rsi(closes)
    atr = compute_atr(bars45)
    sma = compute_sma(closes)

    trades = []
    in_pos = False
    sl = tp = None
    is_buy = False
    stop_pct = 0.0
    entry_t = None

    for i in range(warmup, n):
        if in_pos:
            hi, lo = bars45[i]["h"], bars45[i]["l"]
            hit_tp = (hi >= tp) if is_buy else (lo <= tp)
            hit_sl = (lo <= sl) if is_buy else (hi >= sl)
            if hit_tp or hit_sl:
                r = -1.0 if hit_sl else tp_rr
                trades.append({
                    "symbol": symbol, "entry_t": entry_t, "exit_t": bars45[i]["t"],
                    "r": r, "stop_pct": stop_pct, "is_buy": is_buy,
                })
                in_pos = False
            continue

        if rsi[i] is None or atr[i] is None or sma[i] is None or i + 1 >= n:
            continue

        signal_buy = rsi[i] <= 30 and closes[i] > sma[i]
        signal_sell = rsi[i] >= 70 and closes[i] < sma[i]
        if not (signal_buy or signal_sell):
            continue

        entry_i = i + 1
        entry_price = bars45[entry_i]["o"]
        risk = stop_atr_mult * atr[i]
        if risk <= 0 or entry_price <= 0:
            continue
        if signal_buy:
            is_buy = True
            sl = entry_price - risk
            tp = entry_price + tp_rr * risk
        else:
            is_buy = False
            sl = entry_price + risk
            tp = entry_price - tp_rr * risk
        stop_pct = risk / entry_price
        entry_t = bars45[entry_i]["t"]
        in_pos = True

    return trades


# ---------------------------------------------------------------------------
# Classifier building blocks (unchanged in logic from v1; only the
# reference basket and validation approach change)
# ---------------------------------------------------------------------------

def rolling_window_features(bars45, window_days=WINDOW_DAYS, step_days=STEP_DAYS):
    closes = [b["c"] for b in bars45]
    atr = compute_atr(bars45)
    n = len(bars45)
    win_bars = int(round(window_days * BARS_PER_DAY_45M))
    step_bars = max(1, int(round(step_days * BARS_PER_DAY_45M)))
    out = []
    if win_bars >= n:
        return out
    for start in range(0, n - win_bars, step_bars):
        end = start + win_bars
        seg_atr = [atr[i] / bars45[i]["c"] * 100.0 for i in range(start, end) if atr[i] is not None and bars45[i]["c"] > 0]
        if not seg_atr:
            continue
        c0, c1 = closes[start], closes[end - 1]
        if c0 <= 0:
            continue
        out.append({
            "t_start": bars45[start]["t"],
            "t_end": bars45[end - 1]["t"],
            "ret_pct": (c1 / c0 - 1.0) * 100.0,
            "atr_pct": sum(seg_atr) / len(seg_atr),
        })
    return out


def basket_rolling_features(suffix, symbols, window_days=WINDOW_DAYS, step_days=STEP_DAYS):
    per_symbol = {}
    for sym in symbols:
        bars5 = load_bars(sym, suffix)
        if not bars5:
            raise RuntimeError(f"no data for {sym} / {suffix}")
        per_symbol[sym] = resample_45m(bars5)

    lens = {sym: len(b) for sym, b in per_symbol.items()}
    if len(set(lens.values())) != 1:
        raise RuntimeError(f"bar count mismatch across reference symbols for {suffix}: {lens}")
    n = list(lens.values())[0]
    check_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for i in check_idx:
        ts = {sym: per_symbol[sym][i]["t"] for sym in symbols}
        if len(set(ts.values())) != 1:
            raise RuntimeError(f"timestamp misalignment at index {i} for {suffix}: {ts}")

    feats = {sym: rolling_window_features(per_symbol[sym], window_days, step_days) for sym in symbols}
    lens2 = {sym: len(f) for sym, f in feats.items()}
    if len(set(lens2.values())) != 1:
        raise RuntimeError(f"window-count mismatch across reference symbols for {suffix}: {lens2}")

    combined = []
    for i in range(lens2[symbols[0]]):
        rets = [feats[sym][i]["ret_pct"] for sym in symbols]
        atrs = [feats[sym][i]["atr_pct"] for sym in symbols]
        combined.append({
            "t_start": feats[symbols[0]][i]["t_start"],
            "t_end": feats[symbols[0]][i]["t_end"],
            "ret_pct": sum(rets) / len(rets),
            "atr_pct": sum(atrs) / len(atrs),
        })
    return combined


def zscore(x, mean, std):
    if std <= 1e-12:
        return 0.0
    return (x - mean) / std


def euclid(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _std(vals):
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def build_classifier(all_windows_by_regime, train_frac=TRAIN_FRAC):
    train_split, test_split = {}, {}
    for label, windows in all_windows_by_regime.items():
        ws = sorted(windows, key=lambda w: w["t_start"])
        cut = int(round(len(ws) * train_frac))
        train_split[label] = ws[:cut]
        test_split[label] = ws[cut:]

    pooled_train = [w for ws in train_split.values() for w in ws]
    rets = [w["ret_pct"] for w in pooled_train]
    atrs = [w["atr_pct"] for w in pooled_train]
    ret_mean, ret_std = sum(rets) / len(rets), _std(rets)
    atr_mean, atr_std = sum(atrs) / len(atrs), _std(atrs)
    norm_stats = (ret_mean, ret_std, atr_mean, atr_std)

    centroids = {}
    for label, ws in train_split.items():
        zs = [(zscore(w["ret_pct"], ret_mean, ret_std), zscore(w["atr_pct"], atr_mean, atr_std)) for w in ws]
        centroids[label] = (sum(z[0] for z in zs) / len(zs), sum(z[1] for z in zs) / len(zs))

    return norm_stats, centroids, train_split, test_split


def classify(ret_pct, atr_pct, norm_stats, centroids):
    ret_mean, ret_std, atr_mean, atr_std = norm_stats
    z = (zscore(ret_pct, ret_mean, ret_std), zscore(atr_pct, atr_mean, atr_std))
    dists = {label: euclid(z, c) for label, c in centroids.items()}
    nearest = min(dists, key=dists.get)
    exps = {k: math.exp(-d) for k, d in dists.items()}
    tot = sum(exps.values())
    conf = {k: v / tot for k, v in exps.items()}
    action = "HALF_SIZE" if nearest == "BULL_2024H1" else "FULL_SIZE"
    return {"nearest": nearest, "distances": dists, "confidence": conf, "model_d_action": action}


def classify_at_time(t_ms, suffix, symbols, norm_stats, centroids, window_days=WINDOW_DAYS):
    """Causal live-style classification: build ONE trailing window ending at
    (or just before) t_ms, from real bars, then classify it. Used to score
    the market condition at a specific real trade's entry time."""
    feats_per_symbol = []
    for sym in symbols:
        bars5 = load_bars(sym, suffix)
        bars45 = resample_45m(bars5)
        atr = compute_atr(bars45)
        closes = [b["c"] for b in bars45]
        # find the last bar index with t <= t_ms (causal: no future data)
        end_idx = None
        for i in range(len(bars45) - 1, -1, -1):
            if bars45[i]["t"] <= t_ms:
                end_idx = i
                break
        if end_idx is None:
            return None
        win_bars = int(round(window_days * BARS_PER_DAY_45M))
        start_idx = end_idx - win_bars + 1
        if start_idx < 0:
            return None
        seg_atr = [atr[i] / bars45[i]["c"] * 100.0 for i in range(start_idx, end_idx + 1) if atr[i] is not None and bars45[i]["c"] > 0]
        if not seg_atr:
            return None
        c0, c1 = closes[start_idx], closes[end_idx]
        if c0 <= 0:
            return None
        feats_per_symbol.append({"ret_pct": (c1 / c0 - 1.0) * 100.0, "atr_pct": sum(seg_atr) / len(seg_atr)})
    ret_pct = sum(f["ret_pct"] for f in feats_per_symbol) / len(feats_per_symbol)
    atr_pct = sum(f["atr_pct"] for f in feats_per_symbol) / len(feats_per_symbol)
    return classify(ret_pct, atr_pct, norm_stats, centroids)


# ---------------------------------------------------------------------------
# Synthetic sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks():
    print("=== SANITY CHECKS (synthetic) ===")

    bars = []
    t0 = 1_700_000_000_000
    price = 100.0
    for i in range(200):
        o = price
        h = o + 0.5
        l = o - 0.5
        c = o
        bars.append({"t": t0 + i * 45 * 60 * 1000, "o": o, "h": h, "l": l, "c": c})
    ramp_start = len(bars) - 32
    for j, i in enumerate(range(ramp_start, len(bars))):
        frac = (j + 1) / 32.0
        bars[i]["c"] = 100.0 + 10.0 * frac
        bars[i]["o"] = bars[i - 1]["c"] if i > 0 else 100.0
        bars[i]["h"] = max(bars[i]["o"], bars[i]["c"]) + 0.5
        bars[i]["l"] = min(bars[i]["o"], bars[i]["c"]) - 0.5

    atr = compute_atr(bars, length=14)
    mid_atr = atr[100]
    assert mid_atr is not None and abs(mid_atr - 1.0) < 0.05, f"ATR sanity failed: {mid_atr}"
    print(f"  ATR converges to ~1.0 on constant-TR synthetic series: {mid_atr:.4f} -- OK")

    feats = rolling_window_features(bars, window_days=1, step_days=1)
    assert len(feats) > 0, "no rolling windows produced on synthetic series"
    last = feats[-1]
    expected_ret = (bars[191]["c"] / bars[160]["c"] - 1.0) * 100.0
    assert abs(last["ret_pct"] - expected_ret) < 1e-9, f"ret_pct sanity failed: got {last['ret_pct']}, expected {expected_ret}"
    print(f"  Last window ret_pct matches hand-computed value exactly: {last['ret_pct']:.4f}% -- OK")

    bars_a = [dict(b) for b in bars]
    bars_b = [dict(b) for b in bars]
    bars_b[-1]["c"] = 99999.0
    feats_a = rolling_window_features(bars_a[:-1], window_days=1, step_days=1)
    feats_b_full = rolling_window_features(bars_b, window_days=1, step_days=1)
    common = min(len(feats_a), len(feats_b_full) - 1)
    mismatches = 0
    for i in range(common):
        if feats_a[i]["t_start"] != feats_b_full[i]["t_start"]:
            continue
        if abs(feats_a[i]["ret_pct"] - feats_b_full[i]["ret_pct"]) > 1e-9:
            mismatches += 1
    assert mismatches == 0, f"no-lookahead check failed: {mismatches} windows changed by a future bar"
    print(f"  No-lookahead check: {common} windows unaffected by corrupting a future bar -- OK")

    assert zscore(10.0, 5.0, 2.5) == 2.0
    assert zscore(5.0, 5.0, 0.0) == 0.0
    d = euclid((0.0, 0.0), (3.0, 4.0))
    assert abs(d - 5.0) < 1e-9
    print("  zscore/euclid hand-computed checks -- OK")

    norm_stats = (0.0, 1.0, 0.0, 1.0)
    centroids = {"A": (0.0, 0.0), "B": (10.0, 0.0), "C": (0.0, 10.0)}
    result = classify(0.5, 0.5, norm_stats, centroids)
    assert result["nearest"] == "A"
    assert result["model_d_action"] == "FULL_SIZE"
    result2 = classify(10.0, 0.0, norm_stats, centroids)
    assert result2["nearest"] == "B"
    print("  classify() nearest-centroid sanity -- OK")

    synth_regime = {
        "R1": [{"t_start": i, "t_end": i, "ret_pct": -20.0 + (i % 3), "atr_pct": 1.0 + (i % 2) * 0.1} for i in range(20)],
        "R2": [{"t_start": i, "t_end": i, "ret_pct": 5.0 + (i % 3), "atr_pct": 0.5 + (i % 2) * 0.1} for i in range(20)],
        "R3": [{"t_start": i, "t_end": i, "ret_pct": 5.0 + (i % 3), "atr_pct": 2.0 + (i % 2) * 0.1} for i in range(20)],
    }
    ns, cents, tr, te = build_classifier(synth_regime, train_frac=0.7)
    assert all(len(tr[k]) == 14 for k in synth_regime)
    assert all(len(te[k]) == 6 for k in synth_regime)
    assert cents["R1"] != cents["R2"]
    print("  build_classifier() 70/30 split + distinct-centroid sanity -- OK")

    # detect_trades: verify it produces the exact same result the project's
    # own doc says is byte-identical across scripts -- spot-check RSI/ATR
    # math via a hand constructible dip-then-recovery series is expensive to
    # hand-verify fully here; instead cross-check internal consistency: no
    # overlapping positions (single-position, sorted, non-overlapping trades)
    # on real data, which is a real correctness property regardless of the
    # exact signal values.
    real_bars5 = load_bars("COMPUSDT", "cache")
    if real_bars5:
        real_bars45 = resample_45m(real_bars5)
        trades = detect_trades(real_bars45, "COMPUSDT")
        for i in range(1, len(trades)):
            assert trades[i]["entry_t"] >= trades[i - 1]["exit_t"], "overlapping trades detected -- engine defect"
        assert all(t["r"] in (-1.0, TP_RR) for t in trades), "trade R values outside {-1.0, tp_rr} -- engine defect"
        print(f"  detect_trades() on real COMPUSDT data: {len(trades)} trades, no overlaps, all R in {{-1.0, {TP_RR}}} -- OK")

    print("=== ALL SANITY CHECKS PASSED ===\n")


# ---------------------------------------------------------------------------
# Real-data pipeline
# ---------------------------------------------------------------------------

def fmt_ts(ms):
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return cov / (sx * sy)


def main():
    run_sanity_checks()

    print("=== BUILDING ARCHETYPE ROLLING-WINDOW SAMPLES (REAL DATA, tier-1 basket: COMP/CRV/ONT/CELR) ===")
    all_windows = {}
    for suffix, label in REGIMES:
        windows = basket_rolling_features(suffix, REFERENCE_SYMBOLS, WINDOW_DAYS, STEP_DAYS)
        all_windows[label] = windows
        print(f"  {label:14s} {len(windows):4d} windows, {fmt_ts(windows[0]['t_start'])} -> {fmt_ts(windows[-1]['t_end'])}")

    print("\n=== REFERENCE-BASKET RELEVANCE CHECK: does the tier-1 basket correlate with tier-2 coins' own behavior? ===")
    for sym in TIER2_SYMBOLS:
        print(f"{sym}:")
        for suffix, label in REGIMES:
            bars5 = load_bars(sym, suffix)
            bars45 = resample_45m(bars5)
            own = rolling_window_features(bars45, WINDOW_DAYS, STEP_DAYS)
            b = all_windows[label]
            n = min(len(own), len(b))
            own_ret = [own[i]["ret_pct"] for i in range(n)]
            own_atr = [own[i]["atr_pct"] for i in range(n)]
            b_ret = [b[i]["ret_pct"] for i in range(n)]
            b_atr = [b[i]["atr_pct"] for i in range(n)]
            print(f"  {label:14s} n={n:3d}  corr(return, tier-1 basket)={corr(own_ret, b_ret):+.2f}   corr(ATR%, tier-1 basket)={corr(own_atr, b_atr):+.2f}")

    print("\n=== TRAIN/TEST SPLIT (chronological, 70/30) + CENTROIDS (tier-1 basket) ===")
    norm_stats, centroids, train_split, test_split = build_classifier(all_windows, TRAIN_FRAC)
    ret_mean, ret_std, atr_mean, atr_std = norm_stats
    print(f"  Pooled TRAIN normalization: ret_pct mean={ret_mean:.2f} std={ret_std:.2f} | atr_pct mean={atr_mean:.3f} std={atr_std:.3f}")
    cutoffs = {}
    for label in centroids:
        n_tr, n_te = len(train_split[label]), len(test_split[label])
        cutoffs[label] = test_split[label][0]["t_start"] if test_split[label] else None
        print(f"  {label:14s} train_n={n_tr:3d} test_n={n_te:3d}  centroid(z_ret,z_atr)=({centroids[label][0]:+.2f},{centroids[label][1]:+.2f})  test-cutoff={fmt_ts(cutoffs[label]) if cutoffs[label] else 'n/a'}")

    print("\n=== [reported honestly, no longer the headline] CALENDAR-LABEL HELD-OUT ACCURACY ===")
    confusion = {true: {pred: 0 for pred in centroids} for true in centroids}
    total, correct = 0, 0
    for true_label, windows in test_split.items():
        for w in windows:
            result = classify(w["ret_pct"], w["atr_pct"], norm_stats, centroids)
            confusion[true_label][result["nearest"]] += 1
            total += 1
            if result["nearest"] == true_label:
                correct += 1
    print(f"  Overall: {correct}/{total} = {100.0*correct/total:.1f}%")
    for true_label in centroids:
        n = sum(confusion[true_label].values())
        acc = confusion[true_label][true_label] / n if n else float("nan")
        print(f"  {true_label:14s} {confusion[true_label][true_label]}/{n} = {100.0*acc:.1f}%")

    print("\n=== PRIMARY VALIDATION: does the classifier's read predict real AUDIO/LPT/FLUX trade outcomes? ===")
    print("(held-out only: trade entries restricted to the TEST portion of the basket's own timeline per regime)")
    bucket_trades = {"BULL_2024H1_flagged": [], "CRASH_OR_CALM_flagged": []}
    per_symbol_summary = []
    for sym in TIER2_SYMBOLS:
        for suffix, label in REGIMES:
            bars5 = load_bars(sym, suffix)
            if not bars5:
                continue
            bars45 = resample_45m(bars5)
            trades = detect_trades(bars45, sym)
            cutoff = cutoffs[label]
            if cutoff is None:
                continue
            held_out_trades = [t for t in trades if t["entry_t"] >= cutoff]
            n_bull, n_other = 0, 0
            r_bull, r_other = [], []
            for t in held_out_trades:
                result = classify_at_time(t["entry_t"], suffix, REFERENCE_SYMBOLS, norm_stats, centroids)
                if result is None:
                    continue
                if result["nearest"] == "BULL_2024H1":
                    bucket_trades["BULL_2024H1_flagged"].append(t["r"])
                    r_bull.append(t["r"])
                    n_bull += 1
                else:
                    bucket_trades["CRASH_OR_CALM_flagged"].append(t["r"])
                    r_other.append(t["r"])
                    n_other += 1
            if n_bull + n_other > 0:
                per_symbol_summary.append((sym, label, n_bull, n_other,
                                            sum(r_bull) / len(r_bull) if r_bull else None,
                                            sum(r_other) / len(r_other) if r_other else None))

    print(f"  {'symbol':10s} {'regime':14s} {'n_BULL-flagged':>14s} {'avgR_BULL':>10s} {'n_other-flagged':>16s} {'avgR_other':>11s}")
    for sym, label, n_bull, n_other, avg_bull, avg_other in per_symbol_summary:
        avg_bull_s = f"{avg_bull:+.2f}" if avg_bull is not None else "n/a"
        avg_other_s = f"{avg_other:+.2f}" if avg_other is not None else "n/a"
        print(f"  {sym:10s} {label:14s} {n_bull:14d} {avg_bull_s:>10s} {n_other:16d} {avg_other_s:>11s}")

    n_bull_pool = len(bucket_trades["BULL_2024H1_flagged"])
    n_other_pool = len(bucket_trades["CRASH_OR_CALM_flagged"])
    avg_bull_pool = sum(bucket_trades["BULL_2024H1_flagged"]) / n_bull_pool if n_bull_pool else None
    avg_other_pool = sum(bucket_trades["CRASH_OR_CALM_flagged"]) / n_other_pool if n_other_pool else None
    print(f"\n  POOLED (all 3 tier-2 coins, held-out trades only):")
    print(f"    BULL-flagged (Model D says HALF_SIZE):     n={n_bull_pool:3d}  avg R = {avg_bull_pool if avg_bull_pool is not None else float('nan'):+.3f}")
    print(f"    CRASH/CALM-flagged (Model D says FULL_SIZE): n={n_other_pool:3d}  avg R = {avg_other_pool if avg_other_pool is not None else float('nan'):+.3f}")
    if avg_bull_pool is not None and avg_other_pool is not None:
        if avg_bull_pool < avg_other_pool:
            print(f"    -> Classifier's BULL flag DOES correspond to weaker trades ({avg_bull_pool:+.3f} vs {avg_other_pool:+.3f}) -- directionally useful.")
        else:
            print(f"    -> Classifier's BULL flag does NOT correspond to weaker trades here ({avg_bull_pool:+.3f} vs {avg_other_pool:+.3f}) -- does not support the intended use, reported honestly.")
    else:
        print(f"    -> Not enough held-out trades in one or both buckets to draw a conclusion -- reported honestly as inconclusive, not padded.")

    print("\n=== DONE. Recommendation-building result only, not a live/adopted system. ===")


if __name__ == "__main__":
    main()
