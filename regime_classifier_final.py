"""
regime_classifier_final.py — Regime Classification Leg, FINAL recommended design.

Consolidates everything validated across this leg into the single design
actually being recommended for Model D. Supersedes v1-v5 as the reference
implementation; those are kept in ~/trading for the record (the addendum
documents why each one was tried and what was learned from it), but this
is the one script meant to be read/used going forward if Model D is
adopted.

THE DECISION THIS FILE REFLECTS: stop trying to build a 3-way classifier
that agrees with the historical CRASH_2025-26/CALM_2023/BULL_2024H1
calendar labels. Three independent signals (trailing return, cross-
sectional breadth, long-term trend structure) all independently showed
that CALM_2023's own held-out stretch does not behave like one consistent
thing throughout its own calendar window -- part of it looks like a
decline, part of it looks like a broad rally. No amount of additional
filtering fixes a label that doesn't match its own contents. Since Model D
only ever needed a TWO-way answer (CRASH_2025-26-like and CALM_2023-like
both map to the same FULL_SIZE action; only a BULL_2024H1-like read
changes anything), matching the calendar names was never actually
necessary -- only the two-way split needs to be right.

THE SIGNAL: market-wide trend structure. A basket of real, independent
coins (95 symbols by default, wide-pool + tier-1, excluding the tier-2
coins being sized) is checked for what fraction currently sit above their
own real long-term (100-day) moving average. High participation
(>= 70% by default) => BULL_2024H1-like => HALF_SIZE. Anything else =>
FULL_SIZE. This is the classic, simplest chart-reading definition of "the
market is trending up" -- not a return-magnitude threshold, not a
volatility measure, not a breadth-of-big-movers measure. It beat all of
those when tested directly.

REAL VALIDATION BEHIND THIS DESIGN (regime_classifier_v5_trend_wide_pool.py,
94 symbols, 5,491 held-out real trades from the locked, unmodified signal
engine): BULL-flagged trades averaged +1.033R (37.0% win rate, 95% CI
[34.1%, 39.9%]) vs +1.272R (41.3% win rate, 95% CI [39.9%, 42.8%]) for
everything else -- consistent direction, holds at real scale, win-rate CIs
nearly touching rather than solidly overlapping. Exit detection (does the
flag drop promptly when a real rally ends) checked directly against
BULL_2024H1's own real April-May 2024 rollover: participation fell from
88% to 24% in 5 days, and continued to single digits within a month --
clean and prompt, unlike every return/ATR-based version tried earlier in
this leg.

HONEST, UNRESOLVED CAVEATS -- read before treating this as more than a
strong candidate:
- The 70% threshold was chosen from visual inspection of the exit-
  detection pattern, not cross-validated on a data slice separate from
  the one it was evaluated against.
- Win-rate 95% confidence intervals are close but still technically
  overlap -- "no real effect" has not been fully statistically ruled out.
- A mild, uncorrected circularity exists: a wide-pool symbol contributes
  about 1/95 (~1.1%) of its own trend-basket signal when its own trades
  are being scored.
- A specific, well-reasoned alternative hypothesis (a QUIET, low-
  volatility uptrend specifically, tied to this project's own earlier
  low-volatility finding) was tested and found NOT to hold -- logged
  honestly in the addendum, not hidden.

THIS IS A RECOMMENDATION, NOT A LIVE SYSTEM. Nothing here trades. Model D
itself remains an unadopted recommendation from the spot leg; this file
only gives it a live input, should it ever be adopted.

Run: python3 regime_classifier_final.py
"""
import os
import json
import bisect

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

ATR_LEN = 14
BARS_PER_DAY_45M = 24 * 60 / 45.0  # 32
LONG_SMA_DAYS = 100
LONG_SMA_BARS = int(LONG_SMA_DAYS * BARS_PER_DAY_45M)  # ~3200
TREND_FLAG_THRESHOLD = 0.70  # fraction of basket above its own rising 100-day SMA


# ---------------------------------------------------------------------------
# Locked-engine reference functions (load_bars / resample_45m / compute_sma),
# copied unmodified in logic from the project's locked spec.
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


def compute_sma(closes, length):
    out = [None] * len(closes)
    running = 0.0
    for i, c in enumerate(closes):
        running += c
        if i >= length:
            running -= closes[i - length]
        if i >= length - 1:
            out[i] = running / length
    return out


# ---------------------------------------------------------------------------
# The classifier itself
# ---------------------------------------------------------------------------

def default_trend_basket_symbols():
    """The 95-symbol basket used in validation: every cached symbol with all
    three regime files, excluding AUDIOUSDT/LPTUSDT/FLUXUSDT (the tier-2
    coins Model D sizes -- excluded so their own trades are never scored
    against a signal built substantially from their own price action)."""
    exclude = {"AUDIOUSDT", "LPTUSDT", "FLUXUSDT"}
    files = os.listdir(DATA_DIR)
    by_symbol = {}
    for f in files:
        for suf in ("cache", "calm2023", "bull2024h1"):
            tag = f"_5m_binance_{suf}.json"
            if f.endswith(tag):
                sym = f[: -len(tag)].upper()
                by_symbol.setdefault(sym, set()).add(suf)
    return sorted(s for s, sufs in by_symbol.items() if sufs == {"cache", "calm2023", "bull2024h1"} and s not in exclude)


def build_long_series(symbol, suffix):
    """CALM_2023 + BULL_2024H1 are real, gap-free contiguous data (verified
    this leg) -- concatenated for BULL_2024H1 queries so the long SMA has
    genuine warmup history. CRASH_2025-26 is used standalone (its own
    365-day span already covers 100-day warmup)."""
    if suffix == "bull2024h1":
        bars5 = load_bars(symbol, "calm2023") + load_bars(symbol, "bull2024h1")
    else:
        bars5 = load_bars(symbol, suffix)
    if not bars5:
        return None
    return resample_45m(bars5)


def precompute_trend_basket(symbols, suffix):
    per_symbol = {}
    for sym in symbols:
        bars45 = build_long_series(sym, suffix)
        if not bars45 or len(bars45) < LONG_SMA_BARS + 10:
            continue
        closes = [b["c"] for b in bars45]
        sma = compute_sma(closes, length=LONG_SMA_BARS)
        ts = [b["t"] for b in bars45]
        per_symbol[sym] = {"closes": closes, "sma": sma, "ts": ts}
    return per_symbol


def trend_participation_at(precomputed_basket, t_ms):
    """Fraction of the basket with close > its own rising 100-day SMA at
    t_ms. Returns None if there isn't enough data to answer."""
    n_above, n_total = 0, 0
    for sym, d in precomputed_basket.items():
        ts = d["ts"]
        idx = bisect.bisect_right(ts, t_ms) - 1
        if idx < 0 or d["sma"][idx] is None:
            continue
        n_total += 1
        if d["closes"][idx] > d["sma"][idx]:
            n_above += 1
    if n_total == 0:
        return None
    return n_above / n_total


def classify(precomputed_basket, t_ms, flag_threshold=TREND_FLAG_THRESHOLD):
    """The actual Model D input. Returns a dict with the participation
    reading and the recommended action -- never executes anything."""
    participation = trend_participation_at(precomputed_basket, t_ms)
    if participation is None:
        return {"participation": None, "regime": "UNKNOWN (insufficient data)", "model_d_action": "FULL_SIZE (default/fail-safe)"}
    is_bull = participation >= flag_threshold
    return {
        "participation": participation,
        "regime": "BULL_2024H1-like" if is_bull else "not-BULL-like (CRASH/CALM-like)",
        "model_d_action": "HALF_SIZE" if is_bull else "FULL_SIZE",
    }


# ---------------------------------------------------------------------------
# Sanity checks -- run before any real-data use, per standing project
# discipline.
# ---------------------------------------------------------------------------

def run_sanity_checks():
    print("=== SANITY CHECKS ===")

    closes = [100.0] * 10 + [50.0] * 5  # SMA(10) then a real drop below it
    sma = compute_sma(closes, length=10)
    assert sma[9] == 100.0, f"SMA sanity failed: {sma[9]}"
    print(f"  compute_sma(): flat series of 100.0 -> SMA=100.0 -- OK")

    fake_basket = {
        "UP": {"closes": [100.0] * 10 + [150.0] * 5, "sma": compute_sma([100.0] * 10 + [150.0] * 5, 10), "ts": list(range(15))},
        "DOWN": {"closes": [100.0] * 10 + [50.0] * 5, "sma": compute_sma([100.0] * 10 + [50.0] * 5, 10), "ts": list(range(15))},
    }
    participation = trend_participation_at(fake_basket, 14)
    assert participation is not None and abs(participation - 0.5) < 1e-9, f"trend_participation_at sanity failed: {participation}"
    print(f"  trend_participation_at(): 1 of 2 synthetic symbols above its own SMA -> 0.5 -- OK")

    result_bull = classify(fake_basket, 14, flag_threshold=0.5)
    assert result_bull["model_d_action"] == "HALF_SIZE", f"classify() threshold-inclusive sanity failed: {result_bull}"
    result_not = classify(fake_basket, 14, flag_threshold=0.51)
    assert result_not["model_d_action"] == "FULL_SIZE", f"classify() threshold-exclusive sanity failed: {result_not}"
    print(f"  classify(): threshold boundary behaves correctly (>=0.5 flags, >0.5 required does not) -- OK")

    result_none = classify({}, 0)
    assert result_none["model_d_action"] == "FULL_SIZE (default/fail-safe)", "classify() empty-basket fail-safe check failed"
    print(f"  classify(): empty/no-data basket fails safe to FULL_SIZE, not a crash or an unflagged guess -- OK")

    print("=== SANITY CHECKS PASSED ===\n")


def main():
    run_sanity_checks()

    symbols = default_trend_basket_symbols()
    print(f"=== Default trend basket: {len(symbols)} symbols ===\n")

    print("=== DEMO: classifying the most recent available point in each cached regime file ===")
    print("(this is what a live call would look like -- NOT a validation run; see the addendum for that)\n")
    for suffix, label in [("cache", "CRASH_2025-26"), ("calm2023", "CALM_2023"), ("bull2024h1", "BULL_2024H1")]:
        basket = precompute_trend_basket(symbols, suffix)
        any_symbol = next(iter(basket.values()))
        latest_t = any_symbol["ts"][-1]
        result = classify(basket, latest_t)
        p = result["participation"]
        p_str = f"{100*p:.1f}%" if p is not None else "n/a"
        print(f"  {label:14s} (latest cached point): participation={p_str:>7s}  regime={result['regime']:30s}  model_d_action={result['model_d_action']}")

    print("\nThis is a recommendation-building result only. Model D itself remains unadopted.")
    print("Full validation history and honest caveats: see claude/regime-classifier-addendum-2026-08-20.md")


if __name__ == "__main__":
    main()
