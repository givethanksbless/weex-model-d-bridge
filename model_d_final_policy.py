"""
model_d_final_policy.py

THE BRIDGE DELIVERABLE. Two legs of this project each independently did the
validation work behind Model D, but neither had ever assembled the result
into one single, callable, end-to-end script:

1. SPOT LEG (forex-script-automation-addendum-2026-08-20-part4.md, Parts
   30-31): ADOPTED Model D's full policy -- tier assignment, live regime
   read, per-trade quality refinement, concurrency-scaled sizing -- as the
   live standing policy for the 10-coin watchlist. Confirmed on a genuinely
   held-out CV-B split: n=380 HALF_SIZE (32.9% win, 95% CI [28.4%,37.8%])
   vs n=2,378 FULL_SIZE (41.5% win, 95% CI [39.5%,43.5%]), gap +8.6
   percentage points, CIs do not overlap.

2. REGIME-CLASSIFIER LEG (this chat -- regime-classifier-addendum-2026-08-20.md
   Part 13): independently re-derived the EXACT SAME refined design (same
   6.0-point RSI / 0.70% ATR thresholds, same AND-combination), confirming
   Part 30-31's result wasn't specific to one chat's implementation.

Both legs had the underlying decision already made. The actual gap -- what
this file closes -- is the spot leg's own flagged "Next step" (part4.md):
"the previously-discussed offer to consolidate the full adopted policy
(tier assignment -> live regime read -> per-trade quality check -> base
size -> concurrency-scaled final size) into one callable reference script."

PIPELINE (five stages, matching the adopted policy exactly, nothing added
or re-decided):
  1. TIER ASSIGNMENT -- 7 tier-1 coins always get full base size. 3 tier-2
     coins (AUDIOUSDT/LPTUSDT/FLUXUSDT) get a regime-aware base size.
  2. LIVE REGIME READ -- regime_classifier_final.py's trend-basket
     participation (95-symbol independent basket, each symbol's own real
     100-day SMA, >=70% participation = BULL-like).
  3. PER-TRADE QUALITY CHECK (tier-2 only, only consulted if step 2 flags
     BULL) -- RSI-at-entry within 6 points of its own 30/70 trigger, OR the
     traded symbol's own ATR%-at-entry >= 0.70%.
  4. BASE SIZE -- tier-1: always $20 (2%). tier-2: $10 (1%) only if BOTH
     step 2 AND step 3 fire; otherwise $20.
  5. CONCURRENCY-SCALED FINAL SIZE -- the already-adopted sweep-line soft
     cap (concurrency_scaled_sizing_test.py: $100 total concurrent risk,
     $2 floor), applied identically to all 10 coins on top of whichever
     base size steps 1-4 assigned.

NOT included by default: a news/event-driven systemwide override. The
parallel news/event historical-validation leg (news-event-historical-
validation-addendum-2026-08-23.md) validated a fast 24h/72h high/low
BTCUSDT drawdown metric as a real, clean crash-detection signal for 2 of 3
tested historical events (Terra: 2.8-5.2 days lead; FTX: ~6.1 days lead,
cross-validated 3 ways; Oct 2025: genuine unresolved null) -- but that
leg's own explicit conclusion is "not enough to justify building or wiring
anything into live trading" (only 3 hit events, zero live news-
classification pipeline built, only 2 false-positive comparables tested).
Implemented below as fast_drawdown_crash_check() -- a real, working,
independently callable function, reusing that leg's exact validated
methodology -- but NOT consulted by assign_position_size() unless
USE_NEWS_CRASH_OVERRIDE is explicitly set True. Off by default on purpose.

The locked signal engine, coin selection, and the concurrency-scaling
formula itself are all UNCHANGED -- this file assembles already-adopted
pieces into one place, it does not re-decide any of them. THIS IS A
REFERENCE IMPLEMENTATION, NOT WIRED INTO LIVE TRADING. Nothing here
executes real orders.

Run: python3 model_d_final_policy.py
"""
import os
import json
import bisect

import regime_classifier_final as RF  # already-adopted live regime read, reused as-is

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RSI_LEN = 14
ATR_LEN = 14
SMA_LEN = 200
STOP_ATR_MULT = 1.0
TP_RR = 4.5

# --- Adopted watchlist (spot leg, Part 32: "THE 10-COIN GOAL IS REACHED") ---
TIER1_SYMBOLS = ["COMPUSDT", "CRVUSDT", "ONTUSDT", "CELRUSDT", "ILVUSDT", "SYSUSDT", "XNOUSDT"]
TIER2_SYMBOLS = ["AUDIOUSDT", "LPTUSDT", "FLUXUSDT"]
ALL_SYMBOLS = TIER1_SYMBOLS + TIER2_SYMBOLS
REGIMES = [("cache", "CRASH_2025-26"), ("calm2023", "CALM_2023"), ("bull2024h1", "BULL_2024H1")]

# --- Adopted Model D refined trigger (spot leg Part 31 / this leg Part 13) ---
# These are the MIXED long+short pool numbers -- kept unchanged so every
# existing backtest/report in this file's history stays byte-identical.
TREND_FLAG_THRESHOLD = 0.70
RSI_EXTREMITY_THRESHOLD = 6.0
OWN_ATR_THRESHOLD = 0.70

# --- LONGS-ONLY re-validated thresholds (this leg, Part 16, 2026-08-25) ---
# User is not shorting -- the mixed-pool numbers above were fit on 63.6% short
# trades and don't hold at the same values once restricted to longs only.
# CV-B confirmed on longs-only wide pool: +6.9pp gap (weaker CI than the
# mixed-pool's +8.6pp, but held). USE THESE for anything that only ever
# trades long (e.g. the live WEEX bridge) -- do not use the mixed-pool
# numbers above for a longs-only system.
RSI_EXTREMITY_THRESHOLD_LONGS = 4.0
OWN_ATR_THRESHOLD_LONGS = 0.80

# --- Adopted position-sizing layer (spot leg Part 13) ---
START_BALANCE = 1000.0
BASE_RISK_PCT = 2.0
BASE_RISK_DOLLARS = START_BALANCE * BASE_RISK_PCT / 100.0          # $20 (tier-1 always; tier-2 FULL_SIZE)
HALF_RISK_DOLLARS = BASE_RISK_DOLLARS / 2.0                        # $10 (tier-2 HALF_SIZE)
MIN_STOP_PCT_FLOOR = 0.0005
MAX_CONCURRENT_RISK_PCT = 10.0
MAX_CONCURRENT_RISK_DOLLARS = START_BALANCE * MAX_CONCURRENT_RISK_PCT / 100.0   # $100
MIN_RISK_DOLLARS_FLOOR_PCT = 0.2
MIN_RISK_DOLLARS_FLOOR = START_BALANCE * MIN_RISK_DOLLARS_FLOOR_PCT / 100.0     # $2

# --- News/event leg's validated crash metric (NOT wired in by default) ---
USE_NEWS_CRASH_OVERRIDE = False  # see docstring -- that leg's own leg explicitly says not ready
NEWS_CRASH_SYMBOL = "BTCUSDT"    # matches the news leg's own tested benchmark exactly
NEWS_CRASH_DRAWDOWN_PCT = -15.0  # same magnitude used throughout this project's regime work

# --- COMPUSDT CRASH-regime blocklist (2026-08-25, addendum Part 18) ---
# Diagnostic finding, NOT CV-confirmed (only one real CRASH episode exists in cached
# data -- see compusdt_regime_diagnostic.py): COMPUSDT's own win rate in the
# CRASH_2025-26 window collapsed to 15.6% (n=32 longs) vs. 40-50% for every other
# tier-1 peer in that same window, and held up (18.8% / 12.5%) across an EARLY/LATE
# split of that window -- not a one-off cluster. Every other tier-1 coin stayed
# normal in the same window, so this reads as COMPUSDT-specific, not a project-wide
# CRASH effect. Adopted as a precautionary, explicitly flagged exception to the
# general "no per-coin customization" policy (Part 14) -- narrow, evidence-backed,
# and asymmetric (costs nothing but CRASH-window trades on one coin; the downside of
# being wrong the other way already happened once). NOT extended to any other coin
# without its own equivalent diagnostic.
CRASH_BLOCKLIST_SYMBOLS = ()  # REVERTED Part 20 (2026-08-25): COMPUSDT was blocklisted in
                              # Part 18 on ONE crash window's evidence (15.6-20.0% win rate
                              # vs peers' 40-50%). Genuine second independent slow-crash episode
                              # (SLOWCRASH_2022, real LUNA/FTX bear market) FAILED to confirm --
                              # COMPUSDT was tied-best of tier-1 there (50.0%, n=10), while a
                              # DIFFERENT coin (CRVUSDT, 6.7%, n=15) showed the collapse instead.
                              # Conclusion: single-crash-window per-coin collapses look like small-
                              # sample noise that migrates between coins, not a stable per-coin
                              # trait. Mechanism kept in code (fail-safe, tested) in case a future
                              # symbol earns real CV-confirmed evidence -- just nothing blocklisted
                              # right now. See addendum Part 20.
CRASH_RET_THRESHOLD = -15.0  # tier-1 60-day return threshold, reused unchanged from
                             # regime_decision_engine.py's still-valid price-only
                             # CRASH-like definition (that file's HALF 1 logic, not superseded)


# ---------------------------------------------------------------------------
# Locked-engine reference functions -- copied unmodified in logic, as every
# other script in this project does. Do not modify.
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


def detect_trades_with_features(bars45, symbol, stop_atr_mult=STOP_ATR_MULT, tp_rr=TP_RR):
    """Locked signal engine, UNCHANGED control flow, with the two per-trade
    quality features (Step 3) captured alongside each trade. Verified
    byte-identical (entry_t/exit_t/r) to the plain locked engine below
    before being trusted -- same discipline as regime_classifier_v8."""
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
    feat = None

    for i in range(warmup, n):
        if in_pos:
            hi, lo = bars45[i]["h"], bars45[i]["l"]
            hit_tp = (hi >= tp) if is_buy else (lo <= tp)
            hit_sl = (lo <= sl) if is_buy else (hi >= sl)
            if hit_tp or hit_sl:
                r = -1.0 if hit_sl else tp_rr
                trades.append({
                    "symbol": symbol, "entry_t": entry_t, "exit_t": bars45[i]["t"],
                    "r": r, "stop_pct": stop_pct, "is_buy": is_buy, **feat,
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
            rsi_extremity = 30.0 - rsi[i]
        else:
            is_buy = False
            sl = entry_price + risk
            tp = entry_price - tp_rr * risk
            rsi_extremity = rsi[i] - 70.0
        stop_pct = risk / entry_price
        entry_t = bars45[entry_i]["t"]
        own_atr_pct = atr[i] / closes[i] * 100.0 if closes[i] > 0 else None
        feat = {"rsi_extremity": rsi_extremity, "own_atr_pct": own_atr_pct}
        in_pos = True

    return trades


def detect_trades_locked(bars45, symbol, stop_atr_mult=STOP_ATR_MULT, tp_rr=TP_RR):
    """Plain locked engine, no extra fields -- used only to verify
    detect_trades_with_features() produces an identical trade sequence."""
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
    entry_t = None
    for i in range(warmup, n):
        if in_pos:
            hi, lo = bars45[i]["h"], bars45[i]["l"]
            hit_tp = (hi >= tp) if is_buy else (lo <= tp)
            hit_sl = (lo <= sl) if is_buy else (hi >= sl)
            if hit_tp or hit_sl:
                r = -1.0 if hit_sl else tp_rr
                trades.append({"entry_t": entry_t, "exit_t": bars45[i]["t"], "r": r})
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
        entry_t = bars45[entry_i]["t"]
        in_pos = True
    return trades


# ---------------------------------------------------------------------------
# COMPUSDT CRASH blocklist check (2026-08-25) -- a narrow, explicitly-flagged
# exception to the "no per-coin customization" policy, see Part 18.
# ---------------------------------------------------------------------------

def is_crash_blocked(symbol, tier1_60d_return_pct, blocklist=CRASH_BLOCKLIST_SYMBOLS):
    """Fails safe like every other classifier in this project: if the tier-1
    60-day return isn't available, this does NOT block -- missing data never
    triggers the risk-reducing action, only a confirmed reading does.
    `blocklist` defaults to the live CRASH_BLOCKLIST_SYMBOLS (currently empty,
    see Part 20) -- overridable only for testing the mechanism itself."""
    if symbol not in blocklist:
        return False, None
    if tier1_60d_return_pct is None:
        return False, "tier1 60d return unavailable -- fail-safe, not blocked"
    if tier1_60d_return_pct <= CRASH_RET_THRESHOLD:
        return True, (f"CRASH-like regime detected (tier1 60d return {tier1_60d_return_pct:.1f}% "
                       f"<= {CRASH_RET_THRESHOLD}%) -- {symbol} blocklisted, see addendum Part 18")
    return False, (f"tier1 60d return {tier1_60d_return_pct:.1f}% > {CRASH_RET_THRESHOLD}% "
                    f"-- not CRASH-like, {symbol} not blocked")


# ---------------------------------------------------------------------------
# STAGE 1+4: tier assignment + base size
# ---------------------------------------------------------------------------

def base_size_for_trade_from_regime(symbol, final_regime, rsi_extremity, own_atr_pct,
                                     rsi_threshold=RSI_EXTREMITY_THRESHOLD, atr_threshold=OWN_ATR_THRESHOLD):
    """SYNC POINT with regime_decision_engine.py. That script is the single
    source of truth for 'which regime are we in' (price signal, optionally
    news-adjusted) -- this function consumes its output directly instead of
    re-deriving bull-or-not from raw trend_participation, so there is only
    ever ONE regime determination, not two that could disagree. Tier-1 is
    unaffected either way (always full); tier-2 only changes behavior when
    final_regime == 'BULL-like', exactly matching the already-adopted
    CRASH-like/CALM-like -> FULL_SIZE, BULL-like -> HALF_SIZE(if quality
    condition also fires) policy. No new action types introduced -- CRASH
    and CALM still resolve identically, per the spot leg's own adopted
    design, not re-decided here."""
    if symbol in TIER1_SYMBOLS:
        return BASE_RISK_DOLLARS, f"TIER1 (always full, regardless of regime={final_regime})"
    if symbol not in TIER2_SYMBOLS:
        raise ValueError(f"{symbol} is in neither tier -- not part of the adopted 10-coin watchlist")

    if final_regime != "BULL-like":
        return BASE_RISK_DOLLARS, f"TIER2, regime={final_regime} (not BULL-like) -> FULL_SIZE"

    quality_bad = (rsi_extremity is not None and rsi_extremity <= rsi_threshold) or \
                  (own_atr_pct is not None and own_atr_pct >= atr_threshold)
    if quality_bad:
        return HALF_RISK_DOLLARS, "TIER2, regime=BULL-like AND per-trade quality condition fired -> HALF_SIZE"
    return BASE_RISK_DOLLARS, "TIER2, regime=BULL-like but per-trade quality condition did NOT fire -> FULL_SIZE"


def base_size_for_trade(symbol, trend_participation, rsi_extremity, own_atr_pct):
    """Steps 1, 2(read-in), 3, 4 combined into one decision. trend_participation
    is the already-computed Step 2 regime read (None if unavailable -- fails
    safe to FULL_SIZE, same as regime_classifier_final.py). rsi_extremity/
    own_atr_pct are the Step 3 per-trade quality features at this trade's
    entry (only consulted for tier-2, and only if Step 2 already flagged
    BULL-like -- matches the adopted AND-combination exactly)."""
    if symbol in TIER1_SYMBOLS:
        return BASE_RISK_DOLLARS, "TIER1 (always full)"
    if symbol not in TIER2_SYMBOLS:
        raise ValueError(f"{symbol} is in neither tier -- not part of the adopted 10-coin watchlist")

    if trend_participation is None:
        return BASE_RISK_DOLLARS, "TIER2, regime read unavailable -> fail-safe FULL_SIZE"

    regime_bull = trend_participation >= TREND_FLAG_THRESHOLD
    if not regime_bull:
        return BASE_RISK_DOLLARS, f"TIER2, regime not BULL-like (participation={trend_participation:.3f}) -> FULL_SIZE"

    quality_bad = (rsi_extremity is not None and rsi_extremity <= RSI_EXTREMITY_THRESHOLD) or \
                  (own_atr_pct is not None and own_atr_pct >= OWN_ATR_THRESHOLD)
    if quality_bad:
        return HALF_RISK_DOLLARS, f"TIER2, regime BULL-like AND per-trade quality condition fired -> HALF_SIZE"
    return BASE_RISK_DOLLARS, f"TIER2, regime BULL-like but per-trade quality condition did NOT fire -> FULL_SIZE"


# ---------------------------------------------------------------------------
# STAGE 5: concurrency-scaled final size (adopted, unchanged formula from
# concurrency_scaled_sizing_test.py, generalized to accept a variable base
# size per trade instead of one fixed $20 for everyone).
# ---------------------------------------------------------------------------

def run_concurrency_scaled_portfolio(trades_with_base_size, cost_pct=0.0030,
                                      max_concurrent_dollars=MAX_CONCURRENT_RISK_DOLLARS,
                                      floor_dollars=MIN_RISK_DOLLARS_FLOOR):
    """trades_with_base_size: list of trade dicts, each already carrying its
    own 'base_dollars' (from base_size_for_trade). Sweep-line over entry-time
    order -- unchanged sizing math from the already-adopted concurrency test,
    just reading base_dollars per-trade instead of a single global constant."""
    ordered = sorted(trades_with_base_size, key=lambda x: x["entry_t"])
    open_positions = []  # list of (exit_t, assigned_dollars)
    balance = START_BALANCE
    wins = 0
    scaled_count = 0
    scale_factors = []
    max_concurrent_seen = 0
    per_trade_log = []

    for t in ordered:
        entry_t = t["entry_t"]
        open_positions = [p for p in open_positions if p[0] > entry_t]
        currently_open_dollars = sum(p[1] for p in open_positions)
        available = max_concurrent_dollars - currently_open_dollars
        base_dollars = t["base_dollars"]
        assigned = max(floor_dollars, min(base_dollars, available))
        if assigned < base_dollars - 1e-9:
            scaled_count += 1
        scale_factors.append(assigned / base_dollars)
        max_concurrent_seen = max(max_concurrent_seen, len(open_positions) + 1)
        open_positions.append((t["exit_t"], assigned))

        effective_stop = max(t.get("stop_pct", 0.0), MIN_STOP_PCT_FLOOR)
        cost_in_r = cost_pct / effective_stop
        r_net = t["r"] - cost_in_r
        balance += assigned * r_net
        if r_net > 0:
            wins += 1
        per_trade_log.append({**t, "assigned_dollars": assigned})

    n = len(ordered)
    pct = (balance - START_BALANCE) / START_BALANCE * 100.0
    wr = (wins / n * 100.0) if n else 0.0
    avg_scale = (sum(scale_factors) / len(scale_factors)) if scale_factors else 1.0
    return {
        "final_balance": balance, "return_pct": pct, "n_trades": n, "win_rate_pct": wr,
        "scaled_count": scaled_count, "avg_scale_factor": avg_scale,
        "max_concurrent_seen": max_concurrent_seen, "per_trade_log": per_trade_log,
    }


# ---------------------------------------------------------------------------
# News/event leg's validated fast-drawdown crash metric -- implemented,
# callable, but NOT wired into base_size_for_trade unless explicitly opted
# into. See docstring at the top of this file for why.
# ---------------------------------------------------------------------------

def fast_drawdown_crash_check(symbol=NEWS_CRASH_SYMBOL, suffix="cache", window_hours=72,
                               drawdown_pct=NEWS_CRASH_DRAWDOWN_PCT):
    """Reuses the news/event leg's own validated methodology exactly: a
    high/low-based (not close-only) trailing drawdown over a short window,
    which that leg found gave real, clean lead time for 2 of 3 tested
    historical crashes (Terra, FTX) where the 60-day regime metric was
    either too slow or contaminated by a pre-existing trend. Returns the
    worst trailing drawdown currently in view and whether it crosses the
    same -15% magnitude used throughout this project's regime work."""
    bars5 = load_bars(symbol, suffix)
    if not bars5:
        return {"available": False}
    bars45 = resample_45m(bars5)
    if len(bars45) < 2:
        return {"available": False}
    window_bars = int(round(window_hours * 60 / 45))
    worst_dd = 0.0
    for i in range(1, len(bars45)):
        start = max(0, i - window_bars)
        ref_high = max(b["h"] for b in bars45[start:i + 1])
        dd = (bars45[i]["l"] / ref_high - 1.0) * 100.0
        worst_dd = min(worst_dd, dd)
    flagged = worst_dd <= drawdown_pct
    return {"available": True, "worst_drawdown_pct": worst_dd, "flagged": flagged,
            "note": "validated on 2/3 historical events (Terra, FTX); Oct 2025 remains an unresolved null -- see news-event-historical-validation-addendum-2026-08-23.md"}


# ---------------------------------------------------------------------------
# The full, callable pipeline entry point
# ---------------------------------------------------------------------------

def assign_position_size(symbol, entry_t, trend_basket, rsi_extremity=None, own_atr_pct=None,
                          open_positions=None, use_news_override=USE_NEWS_CRASH_OVERRIDE,
                          tier1_60d_return_pct=None):
    """THE callable reference function -- everything the adopted Model D
    policy needs to size one trade, end to end. Returns a full, auditable
    breakdown (not just a number), matching this project's standing logging
    discipline. open_positions, if provided, is a list of (exit_t,
    assigned_dollars) for currently-open trades, used to apply Stage 5's
    concurrency scaling to this single trade in a live/incremental context.
    tier1_60d_return_pct, if provided, is checked against the COMPUSDT CRASH
    blocklist (Part 18) before anything else -- optional, fails safe to
    'not blocked' if omitted, same discipline as every other classifier here."""
    blocked, block_reason = is_crash_blocked(symbol, tier1_60d_return_pct)
    if blocked:
        return {
            "symbol": symbol, "entry_t": entry_t, "tier": "TIER1" if symbol in TIER1_SYMBOLS else "TIER2",
            "regime_participation": None, "rsi_extremity": rsi_extremity, "own_atr_pct": own_atr_pct,
            "base_dollars": 0.0, "final_dollars": 0.0, "reason": block_reason, "news_check": None,
        }

    participation = RF.trend_participation_at(trend_basket, entry_t) if trend_basket else None
    base_dollars, reason = base_size_for_trade(symbol, participation, rsi_extremity, own_atr_pct)

    news_flag = None
    if use_news_override:
        news_flag = fast_drawdown_crash_check()
        if news_flag.get("flagged"):
            base_dollars = MIN_RISK_DOLLARS_FLOOR
            reason += " | OVERRIDDEN: news-leg fast-drawdown crash check fired (opt-in, use_news_override=True)"

    final_dollars = base_dollars
    if open_positions is not None:
        currently_open = sum(p[1] for p in open_positions if p[0] > entry_t)
        available = MAX_CONCURRENT_RISK_DOLLARS - currently_open
        final_dollars = max(MIN_RISK_DOLLARS_FLOOR, min(base_dollars, available))

    return {
        "symbol": symbol, "entry_t": entry_t, "tier": "TIER1" if symbol in TIER1_SYMBOLS else "TIER2",
        "regime_participation": participation, "rsi_extremity": rsi_extremity, "own_atr_pct": own_atr_pct,
        "base_dollars": base_dollars, "final_dollars": final_dollars, "reason": reason, "news_check": news_flag,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks():
    print("=== SANITY CHECKS ===")

    # 1. detect_trades_with_features matches the plain locked engine exactly
    mismatches, checked = 0, 0
    for sym in ("COMPUSDT", "AUDIOUSDT"):
        for suffix, label in REGIMES:
            bars5 = load_bars(sym, suffix)
            if not bars5:
                continue
            bars45 = resample_45m(bars5)
            plain = detect_trades_locked(bars45, sym)
            featured = detect_trades_with_features(bars45, sym)
            checked += 1
            if len(plain) != len(featured):
                mismatches += 1
                continue
            for a, b in zip(plain, featured):
                if a["entry_t"] != b["entry_t"] or a["exit_t"] != b["exit_t"] or a["r"] != b["r"]:
                    mismatches += 1
                    break
    assert checked > 0 and mismatches == 0, f"detect_trades_with_features mismatch: {mismatches}/{checked}"
    print(f"  detect_trades_with_features() matches the plain locked engine exactly on {checked} real symbol/regime combinations -- OK")

    # 2. base_size_for_trade: all 4 tier-2 branches, plus tier-1
    assert base_size_for_trade("COMPUSDT", 0.99, 0.0, 0.0)[0] == BASE_RISK_DOLLARS, "tier-1 sanity failed"
    assert base_size_for_trade("AUDIOUSDT", 0.50, 0.0, 0.0)[0] == BASE_RISK_DOLLARS, "tier-2/not-bull sanity failed"
    assert base_size_for_trade("AUDIOUSDT", 0.80, 20.0, 0.10)[0] == BASE_RISK_DOLLARS, "tier-2/bull-but-quality-clean sanity failed"
    assert base_size_for_trade("AUDIOUSDT", 0.80, 3.0, 0.10)[0] == HALF_RISK_DOLLARS, "tier-2/bull+RSI-quality sanity failed"
    assert base_size_for_trade("AUDIOUSDT", 0.80, 20.0, 0.90)[0] == HALF_RISK_DOLLARS, "tier-2/bull+ATR-quality sanity failed"
    assert base_size_for_trade("AUDIOUSDT", None, 0.0, 0.0)[0] == BASE_RISK_DOLLARS, "fail-safe sanity failed"
    print("  base_size_for_trade(): all 4 tier-2 branches + tier-1 + fail-safe verified against hand-picked inputs -- OK")

    # base_size_for_trade_from_regime (the regime_decision_engine.py sync point)
    # must agree with base_size_for_trade on every case where the two encode
    # the same underlying regime determination.
    cases = [
        ("COMPUSDT", "CALM-like", 0.0, 0.0),
        ("AUDIOUSDT", "CALM-like", 0.0, 0.0),
        ("AUDIOUSDT", "CRASH-like", 0.0, 0.0),
        ("AUDIOUSDT", "BULL-like", 20.0, 0.10),
        ("AUDIOUSDT", "BULL-like", 3.0, 0.10),
        ("AUDIOUSDT", "BULL-like", 20.0, 0.90),
    ]
    for sym, regime, rsi_ex, atrp in cases:
        participation_equiv = 0.80 if regime == "BULL-like" else 0.50
        via_regime, _ = base_size_for_trade_from_regime(sym, regime, rsi_ex, atrp)
        via_participation, _ = base_size_for_trade(sym, participation_equiv, rsi_ex, atrp)
        assert via_regime == via_participation, f"sync mismatch for {sym}/{regime}: {via_regime} vs {via_participation}"
    print("  base_size_for_trade_from_regime(): agrees with base_size_for_trade() on every equivalent case -- OK (the sync point with regime_decision_engine.py holds)")

    # 3. concurrency scaling: a single trade should get its full base size;
    # two overlapping trades that together exceed the cap should split, with
    # the floor enforced.
    single = run_concurrency_scaled_portfolio([{"entry_t": 0, "exit_t": 10, "r": 4.5, "stop_pct": 0.01, "base_dollars": 20.0}], cost_pct=0.0)
    assert single["per_trade_log"][0]["assigned_dollars"] == 20.0, "single-trade concurrency sanity failed"
    print("  run_concurrency_scaled_portfolio(): a lone trade gets its full base size, unaffected -- OK")

    crowded = [
        {"entry_t": 0, "exit_t": 100, "r": 4.5, "stop_pct": 0.01, "base_dollars": 20.0},
        {"entry_t": 1, "exit_t": 100, "r": 4.5, "stop_pct": 0.01, "base_dollars": 20.0},
        {"entry_t": 2, "exit_t": 100, "r": 4.5, "stop_pct": 0.01, "base_dollars": 20.0},
        {"entry_t": 3, "exit_t": 100, "r": 4.5, "stop_pct": 0.01, "base_dollars": 20.0},
        {"entry_t": 4, "exit_t": 100, "r": 4.5, "stop_pct": 0.01, "base_dollars": 20.0},
        {"entry_t": 5, "exit_t": 100, "r": 4.5, "stop_pct": 0.01, "base_dollars": 20.0},  # 6th trade: only $0 of $100 cap left, should floor at $2
    ]
    res = run_concurrency_scaled_portfolio(crowded, cost_pct=0.0, max_concurrent_dollars=100.0, floor_dollars=2.0)
    sixth = res["per_trade_log"][5]
    assert sixth["assigned_dollars"] == 2.0, f"crowded-portfolio floor sanity failed: {sixth['assigned_dollars']}"
    assert res["scaled_count"] == 1, f"crowded-portfolio scaled_count sanity failed: {res['scaled_count']}"
    print("  run_concurrency_scaled_portfolio(): 6th overlapping $20 trade against a $100 cap floors correctly at $2 -- OK")

    # 4. is_crash_blocked: mechanism tested generically against a fake blocklist entry
    #    (CRASH_BLOCKLIST_SYMBOLS is empty as of Part 20 -- the COMPUSDT exception was
    #    reverted after failing a genuine second-crash-window confirm test; see Part 20).
    #    Verifies the fail-safe LOGIC still works correctly even with nothing blocklisted.
    _fake_blocklist = ("FAKESYM",)
    assert is_crash_blocked("FAKESYM", -20.0, blocklist=_fake_blocklist)[0] is True, "crash-blocked (past threshold) sanity failed"
    assert is_crash_blocked("FAKESYM", -10.0, blocklist=_fake_blocklist)[0] is False, "not-crash (above threshold) sanity failed"
    assert is_crash_blocked("FAKESYM", None, blocklist=_fake_blocklist)[0] is False, "missing-data fail-safe sanity failed"
    assert is_crash_blocked("CRVUSDT", -50.0, blocklist=_fake_blocklist)[0] is False, "non-blocklisted-symbol sanity failed"
    assert is_crash_blocked("COMPUSDT", -50.0)[0] is False, "COMPUSDT should NOT be blocked -- blocklist reverted Part 20"
    print("  is_crash_blocked(): mechanism verified generically (blocked/not-blocked/missing-data-fail-safe/non-blocklisted) -- OK")
    print("  confirmed: CRASH_BLOCKLIST_SYMBOLS is empty -- COMPUSDT exception reverted (Part 20), nothing currently blocklisted -- OK")

    # 5. fast_drawdown_crash_check runs and returns a well-formed result on real data
    check = fast_drawdown_crash_check()
    assert check.get("available"), "fast_drawdown_crash_check produced no result on real BTCUSDT data"
    assert isinstance(check["worst_drawdown_pct"], float) and check["worst_drawdown_pct"] <= 0.0, "fast_drawdown_crash_check sanity failed"
    print(f"  fast_drawdown_crash_check(): ran on real BTCUSDT/cache data, worst 72h drawdown={check['worst_drawdown_pct']:.2f}% -- OK")

    print("=== SANITY CHECKS PASSED ===\n")


# ---------------------------------------------------------------------------
# Full historical demo: the adopted policy, all 10 coins, all 3 regimes,
# assembled end-to-end for the first time in one script.
# ---------------------------------------------------------------------------

def main():
    run_sanity_checks()

    print(f"=== ADOPTED MODEL D POLICY -- {len(TIER1_SYMBOLS)} tier-1 + {len(TIER2_SYMBOLS)} tier-2 = {len(ALL_SYMBOLS)} coins ===")
    print(f"Tier-1 (always full ${BASE_RISK_DOLLARS:.0f}): {', '.join(TIER1_SYMBOLS)}")
    print(f"Tier-2 (regime-aware ${HALF_RISK_DOLLARS:.0f}/${BASE_RISK_DOLLARS:.0f}): {', '.join(TIER2_SYMBOLS)}")
    print(f"News-leg crash override: {'ON (opted in)' if USE_NEWS_CRASH_OVERRIDE else 'OFF (default -- see docstring)'}\n")

    print("=== Precomputing the live regime-read trend basket (95 symbols, once per regime source) ===")
    trend_symbols = RF.default_trend_basket_symbols()
    trend_by_suffix = {suffix: RF.precompute_trend_basket(trend_symbols, suffix) for suffix, _ in REGIMES}

    print("=== Pooling real trades for all 10 adopted coins, across all 3 cached regimes, with per-trade features ===")
    all_trades = []
    for sym in ALL_SYMBOLS:
        for suffix, label in REGIMES:
            bars5 = load_bars(sym, suffix)
            if not bars5:
                continue
            bars45 = resample_45m(bars5)
            trades = detect_trades_with_features(bars45, sym)
            for t in trades:
                participation = RF.trend_participation_at(trend_by_suffix[suffix], t["entry_t"])
                base_dollars, reason = base_size_for_trade(sym, participation, t.get("rsi_extremity"), t.get("own_atr_pct"))
                all_trades.append({**t, "base_dollars": base_dollars, "reason": reason, "regime_label": label})
    print(f"  {len(all_trades)} real trades pooled across all 10 coins / 3 regimes\n")

    n_tier1 = sum(1 for t in all_trades if t["symbol"] in TIER1_SYMBOLS)
    n_tier2_half = sum(1 for t in all_trades if t["symbol"] in TIER2_SYMBOLS and t["base_dollars"] == HALF_RISK_DOLLARS)
    n_tier2_full = sum(1 for t in all_trades if t["symbol"] in TIER2_SYMBOLS and t["base_dollars"] == BASE_RISK_DOLLARS)
    print(f"  tier-1 trades (always full): {n_tier1}")
    print(f"  tier-2 trades flagged HALF_SIZE: {n_tier2_half}")
    print(f"  tier-2 trades FULL_SIZE: {n_tier2_full}\n")

    print("=== STAGE 5: running the full pooled portfolio through the adopted concurrency-scaled sizing layer ===")
    result = run_concurrency_scaled_portfolio(all_trades, cost_pct=0.0030)
    print(f"  n_trades={result['n_trades']}  win_rate={result['win_rate_pct']:.1f}%  return={result['return_pct']:+.1f}%")
    print(f"  max concurrent open positions observed: {result['max_concurrent_seen']}")
    print(f"  trades sized below their own base size: {result['scaled_count']} ({100*result['scaled_count']/result['n_trades']:.1f}%)")
    print(f"  average size as a fraction of base: {100*result['avg_scale_factor']:.1f}%")

    print("\n=== DEMO: assign_position_size(), the live callable entry point, on the latest cached point per regime ===")
    for suffix, label in REGIMES:
        basket = trend_by_suffix[suffix]
        any_symbol_data = next(iter(basket.values())) if basket else None
        if not any_symbol_data:
            continue
        latest_t = any_symbol_data["ts"][-1]
        for sym in ("COMPUSDT", "AUDIOUSDT"):
            out = assign_position_size(sym, latest_t, basket, rsi_extremity=4.0, own_atr_pct=0.30)
            print(f"  {label:14s} {sym:10s} participation={out['regime_participation']}  base=${out['base_dollars']:.0f}  final=${out['final_dollars']:.0f}  ({out['reason']})")

    print("\nThis is a consolidated REFERENCE implementation of the already-adopted policy. Not wired into live trading.")
    print("Full validation history: regime-classifier-addendum-2026-08-20.md (this leg) and")
    print("forex-script-automation-addendum-2026-08-20-part4.md Parts 30-32 (spot leg).")


if __name__ == "__main__":
    main()
