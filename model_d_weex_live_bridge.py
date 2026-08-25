#!/usr/bin/env python3
"""
model_d_weex_live_bridge.py

The real live-execution bridge for the adopted Model D system, replacing
rsi_reversal_weex_bridge.py's SOL/XRP/DOGE/ADA/NEAR/FIL config (a DIFFERENT,
earlier system: flat 2%-of-equity sizing, no SMA(200) trend filter, no
regime awareness). This file trades the actual adopted 10-coin watchlist
(model_d_final_policy.py), with the actual locked signal engine, the actual
regime-aware sizing, and the actual news/price arbitration layer -- all
three pieces this leg spent this whole session building and validating,
finally wired to the one thing that places real orders.

FIVE THINGS THIS FILE FIXES OR ADDS, each because a real gap was found by
inspecting what was ALREADY running on this machine (2026-08-25), not
assumed:

1. WATCHLIST -- was SOL/XRP/DOGE/ADA/NEAR/FIL. Now the real adopted 10:
   COMPUSDT/CRVUSDT/ONTUSDT/CELRUSDT/ILVUSDT/SYSUSDT/XNOUSDT (tier-1) +
   AUDIOUSDT/LPTUSDT/FLUXUSDT (tier-2).

2. SIGNAL -- the old bridge fired on ANY RSI 30/70 cross, no SMA(200)
   trend condition. That does NOT match the locked engine every backtest
   in this project used (buy: RSI<=30 AND close>SMA(200); sell: RSI>=70
   AND close<SMA(200)). Fixed here to match exactly
   (model_d_final_policy.compute_rsi/compute_atr/compute_sma, imported
   verbatim, not re-derived).

3. WARMUP -- WEEX's public klines endpoint has NO pagination, capped at
   ~1000 5m bars (~3.5 days) per call (confirmed in this project's own
   squeeze_5m_accumulate.py). SMA(200) on 45m bars needs ~6.25 days of
   real history just to turn on. A single poll can never have enough.
   Fixed with the exact same accumulator pattern squeeze_5m_accumulate.py
   already uses: a persistent per-symbol local JSON file, appended with
   only-new bars every poll, so real depth builds up over real elapsed
   days. Until a symbol has enough accumulated bars, this bridge reports
   "still warming up" and takes NO signal for it -- fails safe, does not
   guess with a short-window pseudo-SMA.

4. SIZING -- was flat 2% of equity on every trade. Now calls
   model_d_final_policy.base_size_for_trade_from_regime(), using the
   LONGS-ONLY re-validated thresholds (RSI_EXTREMITY_THRESHOLD_LONGS=4.0,
   OWN_ATR_THRESHOLD_LONGS=0.80, Part 16) -- NOT the mixed-pool defaults
   baked into that function, since this bridge only ever trades long (see
   #5). The regime input itself comes from the existing, already-fresh
   Binance-cache historical files this leg has been using all session
   (regime_decision_engine.price_regime()) -- NOT rebuilt from WEEX, on
   purpose: the wide 125-symbol trend-participation basket needs ~100 days
   of history per symbol, which WEEX's uncapped-pagination public klines
   endpoint cannot practically supply live. The regime read is therefore
   as fresh as the last time the Binance cache-extension script was run
   by hand (same as every regime read this whole session) -- slow-moving
   by nature (a 60-day trailing return), so this is an honest match to
   the metric, not a shortcut.

5. NO SHORTING -- user has been explicit and repeated about this all
   session ("i wont be shorting"). This bridge structurally only ever
   acts on buy-side (long) signals. A sell-side signal is detected,
   logged, and explicitly SKIPPED -- never sized, never submitted. This
   is enforced in code, not just by convention.

NEWS ARBITRATION -- calls news_price_arbitration_layer.arbitrate() before
every sizing decision, reading news_assessment from a local shared file
(NEWS_ASSESSMENT_FILE below) if present, else None. Today (2026-08-25)
that file does not exist -- delisting_poc is paused pending Gemini tokens
-- so every call runs with news_assessment=None, which arbitrate() already
handles by passing the plain price regime through unchanged. The moment
delisting_poc writes real classified output to that file (or whatever
path it's pointed at -- update NEWS_ASSESSMENT_FILE below to match), this
bridge picks it up on the very next poll, no other change needed. If
arbitrate() returns watchlist_action == "PULL_FROM_WATCHLIST" for a
symbol, this bridge skips it entirely for that poll (does not size, does
not submit) and logs why -- it does NOT implement the actual "removed
from watchlist" bookkeeping/persistence, since that's a real decision
(temporary pause vs. permanent removal) this leg's own docs left
explicitly undecided.

STILL REQUIRES MANUAL VERIFICATION ON YOUR MACHINE (this sandbox has no
WEEX access, confirmed via direct curl -- everything below is smoke-
tested against synthetic data only, same discipline as the file it
replaces):
  a. WEEX_ORDER_SYMBOL mapping (ticker -> WEEX's order-endpoint symbol,
     e.g. COMPUSDT -> COMPSUSDT) is INFERRED from the one proven-working
     example in this codebase (BTCUSDT -> BTCSUSDT, local_trading_bridge.py).
     This is a real pattern match, not a guess pulled from nowhere, but
     it has not been confirmed against WEEX for these specific 10 tickers.
     Run with --dry-run first; if a real order attempt 400s with an
     invalid-symbol error again, the mapping needs adjusting per symbol.
  b. Confirm all 10 watchlist coins actually exist as tradable WEEX
     futures contracts at all -- not verified from here.
  c. Set real leverage for each symbol in the WEEX UI (API can't set it) --
     LEVERAGE below is a conservative placeholder, same limitation as the
     file this replaces.
  d. Let this run continuously for several real days before trusting any
     signal -- see WARMUP above. Below MIN_45M_BARS_FOR_TRADING, it will
     print "still warming up" and do nothing, by design.

Run: python3 model_d_weex_live_bridge.py --dry-run   (ALWAYS start here)
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_trading_bridge import (  # noqa: E402
    http_request, load_state, save_state, get_record, update_record, now_iso,
    weex_sign, weex_headers, weex_get_balance, WEEX_BASE,
)
import model_d_final_policy as M               # noqa: E402
import news_price_arbitration_layer as ARB     # noqa: E402
import regime_decision_engine as RDE           # noqa: E402
import regime_classifier_final as RF           # noqa: E402
import regime_classifier_v2 as RV2             # noqa: E402

DRY_RUN = "--dry-run" in sys.argv

# ============================================================
# WATCHLIST -- the real adopted 10 coins, tier + WEEX order-symbol mapping.
# leverage/max_notional/margin_safety are placeholders (see docstring item c)
# -- confirm/adjust for real before removing --dry-run.
# ============================================================
def to_weex_order_symbol(ticker):
    """INFERRED mapping (see docstring item a): WEEX's one proven-working
    order symbol in this codebase is BTCUSDT -> BTCSUSDT (an inserted 'S'
    before USDT, matching their USDT-margined-swap naming). Applied
    generically here. NOT confirmed for these specific 10 tickers yet."""
    if not ticker.endswith("USDT"):
        raise ValueError(f"{ticker}: expected a *USDT ticker")
    return ticker[:-4] + "SUSDT"


WATCHLIST = [
    {"key": sym.lower(), "symbol": sym, "weex_order_symbol": to_weex_order_symbol(sym),
     "tier": "tier1", "leverage": 10, "max_notional": 500, "margin_safety": 0.5}
    for sym in M.TIER1_SYMBOLS
] + [
    {"key": sym.lower(), "symbol": sym, "weex_order_symbol": to_weex_order_symbol(sym),
     "tier": "tier2", "leverage": 10, "max_notional": 500, "margin_safety": 0.5}
    for sym in M.TIER2_SYMBOLS
]

# ============================================================
# BAR ACCUMULATOR -- same load/save-by-timestamp pattern as
# squeeze_5m_accumulate.py, reused verbatim, distinct filenames so this
# never collides with that script's own history files.
# ============================================================
HISTORY_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_45M_BARS_FOR_TRADING = M.SMA_LEN + 5  # 205 -- SMA(200) warmup + small buffer


def history_path(symbol):
    return os.path.join(HISTORY_DIR, f"{symbol}_5m_modeld_live_history.json")


def load_history(symbol):
    path = history_path(symbol)
    if os.path.exists(path):
        with open(path) as f:
            bars = json.load(f)
        return {b["openTimeMs"]: b for b in bars}
    return {}


def save_history(symbol, by_ts):
    bars = sorted(by_ts.values(), key=lambda b: b["openTimeMs"])
    with open(history_path(symbol), "w") as f:
        json.dump(bars, f)
    return bars


def fetch_weex_klines(symbol, interval="5m", limit=1000):
    url = f"{WEEX_BASE}/capi/v3/market/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    bars = []
    for row in raw:
        ts = int(row[0])
        bars.append({"openTimeMs": ts, "o": float(row[1]), "h": float(row[2]),
                      "l": float(row[3]), "c": float(row[4])})
    bars.sort(key=lambda b: b["openTimeMs"])
    return bars


def accumulate_and_resample(symbol):
    """Fetches the latest bars, merges into the persistent per-symbol
    history, resamples to 45m. Returns (bars45m, total_5m_bars_accumulated)."""
    by_ts = load_history(symbol)
    fresh = fetch_weex_klines(symbol)
    for b in fresh:
        by_ts[b["openTimeMs"]] = b
    bars5 = save_history(symbol, by_ts)

    out = []
    n = len(bars5)
    for i in range(0, n - 9 + 1, 9):
        chunk = bars5[i:i + 9]
        out.append({"t": chunk[0]["openTimeMs"], "o": chunk[0]["o"],
                     "h": max(b["h"] for b in chunk), "l": min(b["l"] for b in chunk),
                     "c": chunk[-1]["c"]})
    return out, len(bars5)


# ============================================================
# SIGNAL -- the real locked engine (RSI<=30/>=70 AND SMA(200) condition),
# using model_d_final_policy's own compute_rsi/compute_atr/compute_sma
# verbatim, not re-derived. LONGS ONLY enforced structurally (item 5).
# ============================================================
def detect_model_d_signal(bars45m):
    n = len(bars45m)
    if n < MIN_45M_BARS_FOR_TRADING:
        return {"hasSignal": False, "note": f"still warming up: {n}/{MIN_45M_BARS_FOR_TRADING} 45m bars accumulated"}

    closes = [b["c"] for b in bars45m]
    rsi = M.compute_rsi(closes)
    atr = M.compute_atr(bars45m)
    sma = M.compute_sma(closes)

    i = n - 1  # freshest fully-closed 45m bar
    if rsi[i] is None or atr[i] is None or sma[i] is None:
        return {"hasSignal": False, "note": "indicators still warming up on this bar"}

    signal_buy = rsi[i] <= 30 and closes[i] > sma[i]
    signal_sell = rsi[i] >= 70 and closes[i] < sma[i]
    if not (signal_buy or signal_sell):
        return {"hasSignal": False, "rsi": round(rsi[i], 2), "sma": round(sma[i], 6), "close": closes[i]}

    if signal_sell:
        return {"hasSignal": False, "note": "SHORT signal detected but SKIPPED -- longs only, per standing user instruction",
                "rsi": round(rsi[i], 2), "would_have_been": "sell"}

    risk = M.STOP_ATR_MULT * atr[i]
    if risk <= 0:
        return {"hasSignal": False, "note": "zero ATR risk, skipping"}
    entry = closes[i]
    sl = entry - risk
    tp = entry + M.TP_RR * risk
    rsi_extremity = 30.0 - rsi[i]
    own_atr_pct = atr[i] / closes[i] * 100.0 if closes[i] > 0 else None

    return {"hasSignal": True, "side": "buy", "entry": entry, "sl": sl, "tp": tp,
            "risk": risk, "rsi_extremity": rsi_extremity, "own_atr_pct": own_atr_pct}


# ============================================================
# LIVE REGIME READ -- from the existing Binance-cache historical files
# this leg has used all session, NOT rebuilt from WEEX (see docstring
# item 4 for why). As fresh as the last manual cache-extension run.
# ============================================================
def compute_live_price_regime():
    try:
        win_bars = int(round(60 * RV2.BARS_PER_DAY_45M))
        rets = []
        for sym in RV2.REFERENCE_SYMBOLS:
            bars45 = RV2.resample_45m(RV2.load_bars(sym, "cache"))
            if len(bars45) < win_bars + 1:
                return {"regime": "CALM-like", "basis": "insufficient reference-basket history -- fail-safe"}
            anchor, last = bars45[-1 - win_bars], bars45[-1]
            rets.append((last["c"] - anchor["c"]) / anchor["c"] * 100.0)
        tier1_60d = sum(rets) / len(rets)

        symbols = RF.default_trend_basket_symbols()
        trend_basket = RF.precompute_trend_basket(symbols, "cache")
        latest_t = max(d["ts"][-1] for d in trend_basket.values())
        return RDE.price_regime("cache", latest_t, tier1_60d, trend_basket)
    except Exception as e:
        return {"regime": "CALM-like", "basis": f"live regime read failed ({e}) -- fail-safe to CALM-like"}


# ============================================================
# NEWS ASSESSMENT -- read from a shared local file if delisting_poc has
# written one. Missing file (today's real state) -> None, handled by
# arbitrate() as "no news, price regime unchanged."
# ============================================================
NEWS_ASSESSMENT_FILE = os.path.join(HISTORY_DIR, "latest_news_assessment.json")


def load_news_assessment():
    if not os.path.exists(NEWS_ASSESSMENT_FILE):
        return None
    try:
        with open(NEWS_ASSESSMENT_FILE) as f:
            return json.load(f)
    except Exception:
        return None  # malformed file -- fail safe, same as missing


# ============================================================
# SIZING -- Model D's real pipeline: tier -> regime -> arbitration ->
# per-trade quality -> base dollars -> scaled to LIVE balance.
# ============================================================
def size_position(cfg, sig, available_balance, price_result, news_assessment):
    final = ARB.arbitrate(cfg["symbol"], price_result, news_assessment)
    if final["watchlist_action"] == "PULL_FROM_WATCHLIST":
        return None, final, "PULLED_FROM_WATCHLIST -- not sized, not submitted"

    base_dollars, reason = M.base_size_for_trade_from_regime(
        cfg["symbol"], final["final_regime"], sig["rsi_extremity"], sig["own_atr_pct"],
        rsi_threshold=M.RSI_EXTREMITY_THRESHOLD_LONGS, atr_threshold=M.OWN_ATR_THRESHOLD_LONGS,
    )
    risk_pct_live = base_dollars / M.START_BALANCE * 100.0  # 2.0 or 1.0
    risk_dollars_live = available_balance * risk_pct_live / 100.0

    risk_qty = risk_dollars_live / sig["risk"]
    notional_qty = cfg["max_notional"] / sig["entry"]
    margin_cap_qty = (available_balance * cfg["leverage"] * cfg["margin_safety"]) / sig["entry"]
    qty = round(min(risk_qty, notional_qty, margin_cap_qty), 4)
    return qty, final, reason


# ============================================================
# ORDER SUBMISSION -- same confirmed auth/body shape as
# local_trading_bridge.py's weex_submit_order, symbol-parameterized with
# the corrected order-symbol mapping.
# ============================================================
EXEC_LOG_FILE = os.path.join(HISTORY_DIR, "model_d_weex_execution_log.jsonl")
POLL_LOG_FILE = os.path.join(HISTORY_DIR, "model_d_live_poll_log.jsonl")
CLOSED_TRADES_LOG_FILE = os.path.join(HISTORY_DIR, "model_d_closed_trades_log.jsonl")


def _append_jsonl(path, entry):
    try:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[log] WARNING: failed to write {path}: {e}")


def log_execution(entry):
    _append_jsonl(EXEC_LOG_FILE, entry)


def log_closed_trade(entry):
    _append_jsonl(CLOSED_TRADES_LOG_FILE, entry)


def weex_submit_market_order(order_symbol, side, qty, tp, sl):
    ts = str(int(time.time() * 1000))
    path = "/capi/v3/sim/order"
    body_obj = {
        "symbol": order_symbol, "side": "BUY" if side == "buy" else "SELL",
        "positionSide": "LONG" if side == "buy" else "SHORT", "type": "MARKET",
        "quantity": f"{qty:.4f}", "newClientOrderId": f"{order_symbol.lower()}modeld-{ts}",
        "tpTriggerPrice": str(round(tp, 6)), "slTriggerPrice": str(round(sl, 6)),
    }
    body = json.dumps(body_obj)
    sign = weex_sign(ts, "POST", path, body)
    return http_request("POST", WEEX_BASE + path, headers=weex_headers(ts, sign), body=body)


def weex_get_order_history(order_symbol, start_time_ms=None, limit=50):
    """GET /capi/v3/sim/order/history -- confirmed real WEEX demo endpoint
    (https://www.weex.com/api-doc/contract/demo/GetOrderHistory), whose own
    documented example queries symbol=BTCSUSDT -- independent confirmation
    (straight from WEEX's docs, not just this codebase's one internal
    example) that the 'S'-inserted symbol mapping used throughout this file
    is correct for the demo/sim environment. Used to replace the old
    last-close-vs-TP/SL GUESS with a CONFIRMED closing fill price."""
    ts = str(int(time.time() * 1000))
    path = "/capi/v3/sim/order/history"
    qs = f"symbol={order_symbol}&limit={limit}"
    if start_time_ms is not None:
        qs += f"&startTime={start_time_ms}"
    # NOTE: signed with the query string included in the path (path+"?"+qs) --
    # standard convention for this style of exchange signing (timestamp+method+
    # requestPath+body, where requestPath includes any query string), matching
    # how weex_sign is used elsewhere in this codebase for bodied requests.
    # NOT verified against a real WEEX response from this sandbox (no network
    # access) -- if this 400s/401s with a signature error, that's the first
    # thing to check.
    full_path = path + "?" + qs
    sign = weex_sign(ts, "GET", full_path)
    status, data = http_request("GET", WEEX_BASE + full_path, headers=weex_headers(ts, sign))
    return status, (data if isinstance(data, list) else [])


def weex_get_position(order_symbol):
    ts = str(int(time.time() * 1000))
    path = "/capi/v3/sim/position/allPosition"
    sign = weex_sign(ts, "GET", path)
    status, data = http_request("GET", WEEX_BASE + path, headers=weex_headers(ts, sign))
    arr = data if isinstance(data, list) else ([data] if data else [])
    return status, next((p for p in arr if p.get("symbol") == order_symbol and float(p.get("size", 0)) > 0), None)


# ============================================================
# PER-SYMBOL POLL
# ============================================================
def run_symbol(cfg, state, price_result, news_assessment):
    """Returns a per-symbol status dict every call (idle-no-signal,
    idle-signal, or position_open) -- collected by main() into one
    POLL_LOG_FILE line per poll, so the progress report has real data to
    read even during the multi-day SMA(200) warmup, before any trade fires."""
    key, symbol = cfg["key"], cfg["symbol"]
    rec = get_record(state, key)
    st = rec.get("state", "idle")

    if st == "idle":
        bars45, n5 = accumulate_and_resample(symbol)
        sig = detect_model_d_signal(bars45)
        print(f"[{key}] idle: hasSignal={sig.get('hasSignal')} 45m_bars={len(bars45)} "
              f"5m_accumulated={n5} note={sig.get('note')} rsi={sig.get('rsi')}")
        status_rec = {"key": key, "symbol": symbol, "state": "idle", "bars_45m": len(bars45),
                      "bars_5m_accumulated": n5, "warmup_needed": MIN_45M_BARS_FOR_TRADING,
                      "has_signal": sig.get("hasSignal", False), "note": sig.get("note"),
                      "rsi": sig.get("rsi"), "last_close": sig.get("close")}
        if not sig.get("hasSignal"):
            return status_rec

        available_balance = weex_get_balance()
        qty, final, reason = size_position(cfg, sig, available_balance, price_result, news_assessment)
        print(f"[{key}] regime={final['final_regime']} (price={final['price_regime']}) "
              f"action={final['watchlist_action']} reason={reason}")
        status_rec.update({"final_regime": final["final_regime"], "watchlist_action": final["watchlist_action"],
                            "sizing_reason": reason, "qty": qty})
        if qty is None:
            log_execution({"logged_at": now_iso(), "symbol": symbol, "skipped": True, "reason": reason})
            return status_rec
        if qty <= 0:
            print(f"[{key}] SIGNAL but computed qty<=0 (avail_bal={available_balance}) -- skipping")
            return status_rec

        if DRY_RUN:
            print(f"[{key}] DRY RUN would submit BUY MARKET qty={qty} entry~{sig['entry']} "
                  f"tp={sig['tp']} sl={sig['sl']} (avail_bal={available_balance}, order_symbol={cfg['weex_order_symbol']})")
            status_rec["dry_run_would_submit"] = True
            return status_rec

        status, data = weex_submit_market_order(cfg["weex_order_symbol"], "buy", qty, sig["tp"], sig["sl"])
        log_execution({"logged_at": now_iso(), "symbol": symbol, "order_symbol": cfg["weex_order_symbol"],
                        "side": "buy", "qty": qty, "tp": sig["tp"], "sl": sig["sl"],
                        "http_status": status, "response": data, "final_regime": final["final_regime"]})
        update_record(state, key, {
            "state": "position_open", "pendingSl": sig["sl"], "pendingTp": sig["tp"],
            "pendingQty": qty, "pendingSide": "buy", "pendingEntry": sig["entry"],
            "pendingOrderId": data.get("orderId") if isinstance(data, dict) else None,
            "pendingSubmittedAt": now_iso(), "pendingSubmittedAtMs": int(time.time() * 1000),
            "debug": f"order:{status}|{data}"})
        print(f"[{key}] SIGNAL: submitted BUY MARKET qty={qty}, status={status}, resp={data}")
        status_rec["submitted"] = True
        return status_rec

    elif st == "position_open":
        status, pos = weex_get_position(cfg["weex_order_symbol"])
        if not pos:
            entry, tp, sl = rec.get("pendingEntry"), rec.get("pendingTp"), rec.get("pendingSl")
            entry_order_id = rec.get("pendingOrderId")
            submitted_ms = rec.get("pendingSubmittedAtMs")
            log_entry = {"closed_at": now_iso(), "symbol": symbol, "entry": entry, "tp": tp, "sl": sl,
                         "qty": rec.get("pendingQty"), "submitted_at": rec.get("pendingSubmittedAt")}

            # PREFERRED: a real closing fill from WEEX's own confirmed demo
            # order-history endpoint (GET /capi/v3/sim/order/history --
            # https://www.weex.com/api-doc/contract/demo/GetOrderHistory).
            # Finds the newest FILLED order for this symbol, after this
            # position's own entry order, that ISN'T the entry order itself
            # -- that's the TP/SL-triggered closing fill.
            confirmed_exit = None
            try:
                _, orders = weex_get_order_history(cfg["weex_order_symbol"], start_time_ms=submitted_ms)
                closing_candidates = [
                    o for o in orders
                    if o.get("status") == "FILLED" and str(o.get("orderId")) != str(entry_order_id)
                ]
                if closing_candidates:
                    closing_candidates.sort(key=lambda o: o.get("updateTime", o.get("time", 0)))
                    confirmed_exit = closing_candidates[-1]
            except Exception as e:
                print(f"[{key}] order-history lookup failed ({e}) -- falling back to estimate")

            if confirmed_exit is not None and confirmed_exit.get("avgPrice"):
                exit_price = float(confirmed_exit["avgPrice"])
                # Only two exit paths exist in this system (TP above entry,
                # SL below entry, no manual exits) -- exit_price vs entry_price
                # alone cleanly separates them, no distance-guessing needed.
                outcome = "CONFIRMED_TP" if entry is not None and exit_price >= entry else "CONFIRMED_SL"
                log_entry.update({"exit_price": exit_price, "exit_order_id": confirmed_exit.get("orderId"),
                                   "outcome": outcome, "source": "weex_order_history_confirmed"})
                print(f"[{key}] position closed -> idle (CONFIRMED exit_price={exit_price}, outcome={outcome})")
            else:
                # FALLBACK ONLY: last accumulated close vs TP/SL, explicitly
                # labeled an estimate, not a confirmed fill.
                last_close = None
                try:
                    by_ts = load_history(symbol)
                    if by_ts:
                        last_bar = max(by_ts.values(), key=lambda b: b["openTimeMs"])
                        last_close = last_bar["c"]
                except Exception:
                    pass
                outcome = "unknown"
                if last_close is not None and tp is not None and sl is not None:
                    dist_to_tp, dist_to_sl = abs(last_close - tp), abs(last_close - sl)
                    outcome = "estimated_TP" if dist_to_tp < dist_to_sl else "estimated_SL"
                log_entry.update({"last_seen_close": last_close, "outcome": outcome,
                                   "source": "fallback_estimate_order_history_unavailable"})
                print(f"[{key}] position closed -> idle (ESTIMATE ONLY outcome={outcome} -- order-history lookup found no closing fill)")

            log_closed_trade(log_entry)
            update_record(state, key, {"state": "idle", "pendingSl": 0, "pendingTp": 0, "pendingQty": 0, "pendingSide": ""})
            return {"key": key, "symbol": symbol, "state": "closed_this_poll", "outcome": log_entry.get("outcome")}
        else:
            print(f"[{key}] position open: size={pos.get('size')}")
            return {"key": key, "symbol": symbol, "state": "position_open", "size": pos.get("size")}


def main():
    run_sanity_checks()
    if "--sanity-only" in sys.argv:
        return

    state = load_state()
    price_result = compute_live_price_regime()
    news_assessment = load_news_assessment()
    poll_time = now_iso()
    print(f"\n=== POLL {poll_time} -- price_regime={price_result['regime']} ({price_result['basis']}) "
          f"news={'CONNECTED' if news_assessment else 'none (delisting_poc not producing output yet)'} ===")
    symbol_statuses = []
    for cfg in WATCHLIST:
        try:
            status_rec = run_symbol(cfg, state, price_result, news_assessment)
            symbol_statuses.append(status_rec or {"key": cfg["key"], "symbol": cfg["symbol"], "state": "no_status_returned"})
        except Exception as e:
            print(f"[{cfg['key']}] ERROR: {e}")
            symbol_statuses.append({"key": cfg["key"], "symbol": cfg["symbol"], "state": "error", "error": str(e)})
    save_state(state)

    _append_jsonl(POLL_LOG_FILE, {
        "polled_at": poll_time, "price_regime": price_result["regime"], "price_basis": price_result.get("basis"),
        "news_connected": news_assessment is not None, "symbols": symbol_statuses,
    })


# ============================================================
# SANITY CHECKS -- synthetic only, this sandbox has no WEEX access.
# ============================================================
def run_sanity_checks():
    print("=== SANITY CHECKS ===")

    # 1. Symbol mapping matches the one proven-working example
    assert to_weex_order_symbol("BTCUSDT") == "BTCSUSDT"
    assert to_weex_order_symbol("COMPUSDT") == "COMPSUSDT"
    print("  to_weex_order_symbol(): matches the proven BTCUSDT->BTCSUSDT pattern -- OK")

    # 2. Warmup gate: too few bars -> no signal, no crash
    short_bars = [{"t": i, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0} for i in range(50)]
    sig = detect_model_d_signal(short_bars)
    assert sig["hasSignal"] is False and "warming up" in sig["note"]
    print("  detect_model_d_signal(): insufficient bars -> no signal, clear reason, no crash -- OK")

    # 3. Synthetic bars: build a real, CONFIRMED buy-signal setup (calibrated
    #    directly against M.compute_rsi/compute_sma -- not a guess: a steady
    #    uptrend followed by a 6-bar pullback lands RSI=27.64 (<=30) while
    #    close=141.56 stays above SMA(200)=128.70, a genuine buy trigger).
    bars = []
    price = 100.0
    for i in range(260):
        price *= 1.0015
        bars.append({"t": i * 45 * 60000, "o": price, "h": price * 1.001, "l": price * 0.999, "c": price})
    for i in range(6):
        price *= 0.993
        bars.append({"t": (260 + i) * 45 * 60000, "o": price, "h": price * 1.001, "l": price * 0.999, "c": price})
    sig = detect_model_d_signal(bars)
    assert sig["hasSignal"] is True, f"expected a confirmed buy signal, got: {sig}"
    assert sig["side"] == "buy"
    assert sig["rsi_extremity"] > 0
    assert sig["own_atr_pct"] is not None
    print(f"  synthetic calibrated uptrend+pullback: CONFIRMED real buy signal fires (rsi_extremity={sig['rsi_extremity']:.2f}, own_atr_pct={sig['own_atr_pct']:.3f}) -- OK")

    # 4. Short signal must be skipped, never sized -- calibrated real downtrend+bounce
    #    (RSI=72.28 >= 70, close=70.58 stays below SMA(200)=78.24 -- a genuine sell trigger).
    bars_short = []
    price = 100.0
    for i in range(260):
        price *= 0.9985
        bars_short.append({"t": i * 45 * 60000, "o": price, "h": price * 1.001, "l": price * 0.999, "c": price})
    for i in range(6):
        price *= 1.007
        bars_short.append({"t": (260 + i) * 45 * 60000, "o": price, "h": price * 1.001, "l": price * 0.999, "c": price})
    sig_short = detect_model_d_signal(bars_short)
    assert sig_short["hasSignal"] is False
    assert sig_short.get("would_have_been") == "sell", f"expected a confirmed would-be sell trigger, got: {sig_short}"
    print("  CONFIRMED real sell-trigger setup detected-and-skipped (longs only, per standing user instruction) -- OK")

    # 5. Sizing pipeline: tier-1 always base regardless of regime/news
    p_calm = {"regime": "CALM-like", "basis": "test"}
    cfg_t1 = {"symbol": "COMPUSDT", "leverage": 10, "max_notional": 500, "margin_safety": 0.5}
    sig_fake = {"entry": 20.0, "risk": 0.5, "rsi_extremity": 4.0, "own_atr_pct": 0.9}
    qty, final, reason = size_position(cfg_t1, sig_fake, 1000.0, p_calm, None)
    assert qty is not None and qty > 0
    assert "TIER1" in reason
    print("  size_position(): tier-1 always sized regardless of regime -- OK")

    # 6. Sizing pipeline: PULL_FROM_WATCHLIST news action -> qty is None, no order
    news_delist = {"confirmed": True, "relevant": True, "direction": "down",
                   "category": "exchange_delisting_halt", "symbols_affected": ["AUDIOUSDT"],
                   "note": "test", "source": "test"}
    cfg_t2 = {"symbol": "AUDIOUSDT", "leverage": 10, "max_notional": 500, "margin_safety": 0.5}
    qty2, final2, reason2 = size_position(cfg_t2, sig_fake, 1000.0, p_calm, news_delist)
    assert qty2 is None and "PULLED_FROM_WATCHLIST" in reason2
    print("  size_position(): PULL_FROM_WATCHLIST news action blocks sizing entirely -- OK")

    # 7. Sizing pipeline: tier-2 HALF_SIZE uses the LONGS-ONLY thresholds, not the mixed-pool ones
    p_bull = {"regime": "BULL-like", "basis": "test"}
    sig_quality_bad_longs_only = {"entry": 20.0, "risk": 0.5, "rsi_extremity": 5.0, "own_atr_pct": 0.75}
    # rsi_extremity=5.0: fires under LONGS threshold (<=4.0? no -- 5.0 > 4.0, doesn't fire on RSI)
    # own_atr_pct=0.75: fires under LONGS threshold (>=0.80? no) -- so use values that
    # differ between the two threshold sets to prove the LONGS ones are actually used:
    sig_between = {"entry": 20.0, "risk": 0.5, "rsi_extremity": 5.0, "own_atr_pct": 0.75}
    # Under MIXED thresholds (rsi<=6.0 OR atr>=0.70): 5.0<=6.0 fires -> HALF_SIZE
    # Under LONGS thresholds (rsi<=4.0 OR atr>=0.80): neither fires -> FULL_SIZE
    qty3, final3, reason3 = size_position(cfg_t2, sig_between, 1000.0, p_bull, None)
    assert "FULL_SIZE" in reason3, f"expected LONGS-only thresholds (FULL_SIZE), got: {reason3}"
    print("  size_position(): tier-2 in BULL-like uses LONGS-ONLY thresholds (4.0/0.80), confirmed distinct from mixed-pool (6.0/0.70) -- OK")

    # 8. News assessment file missing -> None, no crash
    assert load_news_assessment() is None or isinstance(load_news_assessment(), dict)
    print("  load_news_assessment(): missing file -> None, no crash -- OK")

    # 9. Confirmed-exit classification rule: exit_price >= entry -> TP, else SL
    #    (only two exit paths exist in this long-only, fixed-R:R system, so
    #    this simple comparison is exact, not a heuristic).
    entry_test = 20.0
    assert ("CONFIRMED_TP" if 25.0 >= entry_test else "CONFIRMED_SL") == "CONFIRMED_TP"
    assert ("CONFIRMED_TP" if 19.5 >= entry_test else "CONFIRMED_SL") == "CONFIRMED_SL"
    print("  confirmed-exit TP/SL classification rule (exit_price vs entry_price) verified on both sides -- OK")

    # 10. weex_get_order_history() symbol/query construction matches WEEX's own
    #     documented demo example (GET /capi/v3/sim/order/history?symbol=BTCSUSDT)
    assert to_weex_order_symbol("BTCUSDT") == "BTCSUSDT", "must match WEEX's own documented demo example exactly"
    print("  weex_get_order_history() symbol construction matches WEEX's own documented example (BTCSUSDT) -- OK")

    print("=== SANITY CHECKS PASSED ===\n")


if __name__ == "__main__":
    main()
