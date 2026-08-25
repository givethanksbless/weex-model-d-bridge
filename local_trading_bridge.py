#!/usr/bin/env python3
"""
local_trading_bridge.py

Local (off-Make.com) execution bridge for the paper/demo squeeze-breakout
account. This is a faithful line-for-line port of the JavaScript logic
currently running in 9 Make.com scenarios (TSLA/BA/AAPL/DIS/T on Alpaca
equities, USDJPY/EURUSD on OANDA, ETHUSD on Alpaca crypto, BTCSUSDT on WEEX
demo futures), extracted directly from the live scenario blueprints on
2026-08-18 so nothing is re-derived or approximated.

WHY THIS EXISTS: standing rule as of 2026-08-18 - Make.com is reserved
ONLY for live (real-money) trades. Every paper/demo bridge runs here
instead, at zero ongoing Make.com cost regardless of poll frequency or
symbol count. See the "Trade" Claude Project doc
(claude/forex-script-automation.md) for full history/context.

STATE: replaces Make's shared data store (id 132458) with a single local
JSON file (BRIDGE_STATE_FILE), one record per symbol key, same field
names Make used (state/pendingSl/pendingTp/pendingQty/pendingSide/
pendingOrderId/pendingSubmittedAt/ocoOrderId/slOrderId/tpOrderId), so the
state machine logic transfers directly.

NOT NETWORK-TESTED: this cloud sandbox cannot reach Alpaca/OANDA/WEEX.
Run this yourself (python3 local_trading_bridge.py --dry-run first, then
for real) before trusting it, then set up as a local scheduled task via
the desktop app's "Run this task -> On your computer" picker, same
pattern as squeeze_alert_scanner.py.

CUTOVER SAFETY: do NOT deactivate the equivalent Make scenario for a
symbol until you've confirmed (via scenarios_get / the data store) that
its state is "idle" with no open order or position. Running both the
Make bridge and this local bridge on the same symbol at the same time
risks double-entering a position.
"""

import json
import os
import sys
import time
import hmac
import hashlib
import base64
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta, timezone

# ============================================================
# CREDENTIALS - loaded from local_creds.py, which you fill in
# yourself directly on this machine. Never paste real secrets
# into a chat conversation.
# ============================================================
from local_creds import (
    ALPACA_KEY_ID, ALPACA_SECRET_KEY,
    OANDA_TOKEN, OANDA_ACCOUNT_ID,
    WEEX_ACCESS_KEY, WEEX_PASSPHRASE, WEEX_SECRET,
)
ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE = "https://data.alpaca.markets"
OANDA_BASE = "https://api-fxpractice.oanda.com"
WEEX_BASE = "https://api-contract.weex.com"

BRIDGE_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge_state.json")
WEEX_EXEC_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weex_execution_log.jsonl")

# ============================================================
# WATCHLIST - one entry per bridge, mirrors the 9 Make scenarios exactly.
# broker: "alpaca_equity" | "alpaca_crypto" | "oanda" | "weex"
# ============================================================
WATCHLIST = [
    {"key": "tsla", "broker": "alpaca_equity", "symbol": "TSLA",
     "min_sl": 0.30, "max_sl": 8.0, "account_bal": 100000, "risk_pct": 7.0,
     "leverage": 2.0, "min_len": 6, "max_wait": 6, "trend_filter": False,
     "market_hours_only": True},
    {"key": "ba2", "broker": "alpaca_equity", "symbol": "BA",
     "min_sl": 0.30, "max_sl": 8.0, "account_bal": 100000, "risk_pct": 7.0,
     "leverage": 2.0, "min_len": 6, "max_wait": 6, "trend_filter": False,
     "market_hours_only": True},
    {"key": "aapl2", "broker": "alpaca_equity", "symbol": "AAPL",
     "min_sl": 0.30, "max_sl": 8.0, "account_bal": 100000, "risk_pct": 7.0,
     "leverage": 2.0, "min_len": 6, "max_wait": 6, "trend_filter": False,
     "market_hours_only": True},
    {"key": "dis2", "broker": "alpaca_equity", "symbol": "DIS",
     "min_sl": 0.30, "max_sl": 8.0, "account_bal": 100000, "risk_pct": 7.0,
     "leverage": 2.0, "min_len": 6, "max_wait": 6, "trend_filter": False,
     "market_hours_only": True},
    {"key": "t2", "broker": "alpaca_equity", "symbol": "T",
     "min_sl": 0.30, "max_sl": 8.0, "account_bal": 100000, "risk_pct": 7.0,
     "leverage": 2.0, "min_len": 6, "max_wait": 6, "trend_filter": False,
     "market_hours_only": True},
    {"key": "usdjpy", "broker": "oanda", "instrument": "USD_JPY", "decimals": 3,
     "min_sl_acct": 0.0003, "max_sl_acct": 0.008, "account_bal": 100000,
     "risk_pct": 7.0, "leverage": 2.0, "min_len": 6, "max_wait": 6,
     "trend_filter": False},
    {"key": "eurusd", "broker": "oanda", "instrument": "EUR_USD", "decimals": 5,
     "min_sl_acct": 0.0003, "max_sl_acct": 0.008, "account_bal": 100000,
     "risk_pct": 7.0, "leverage": 2.0, "min_len": 6, "max_wait": 6,
     "trend_filter": False},
    {"key": "ethusd", "broker": "alpaca_crypto", "symbol": "ETH/USD",
     "min_sl": 13.0, "max_sl": 111.0, "account_bal": 100665, "risk_pct": 7.0,
     "leverage": 2.0, "min_len": 6, "max_wait": 6, "trend_filter": True,
     "slope_lookback": 10, "slope_atr_mult": 0.5},
    {"key": "weexbtc", "broker": "weex", "symbol": "BTCSUSDT", "alpaca_symbol": "BTC/USD",
     "min_sl": 225.0, "max_sl": 1900.0, "account_bal": 20000, "risk_pct": 5.0,
     "max_notional": 20000, "leverage": 20, "margin_safety": 0.5,
     "min_len": 3, "max_wait": 15, "trend_filter": True,
     "slope_lookback": 10, "slope_atr_mult": 0.5},
]

# Shared constants across every bridge (verbatim from the Make Code modules)
TP_RR = 2.0
KC_MULT = 1.5
COIL_MAX = 3.2
BB_LEN = 20
ATR_LEN = 14
EMA_LEN = 200

DRY_RUN = "--dry-run" in sys.argv


# ============================================================
# HTTP helper (stdlib only, no pip install needed)
# ============================================================
def http_request(method, url, headers=None, body=None, timeout=20):
    headers = headers or {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8") if not isinstance(body, (bytes, str)) else (
            body.encode("utf-8") if isinstance(body, str) else body
        )
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.getcode()
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        status = e.code
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {"_raw": raw}
    return status, parsed


# ============================================================
# STATE (local replacement for the Make data store, id 132458)
# ============================================================
def load_state():
    if os.path.exists(BRIDGE_STATE_FILE):
        with open(BRIDGE_STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(BRIDGE_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_record(state, key):
    return state.get(key, {"state": "idle"})


def update_record(state, key, fields):
    rec = state.get(key, {})
    rec.update(fields)
    state[key] = rec
    save_state(state)
    return rec


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================
# SQUEEZE-BREAKOUT SIGNAL DETECTION
# Faithful port of the Code module shared (with parameters) across
# all 9 Make scenarios. bars: list of dicts with t/o/h/l/c (chronological,
# oldest first, only fully-closed bars).
# ============================================================
def sma(arr, i, length):
    if i < length - 1:
        return None
    return sum(arr[i - length + 1:i + 1]) / length


def stdev_unbiased(arr, i, length):
    if i < length - 1:
        return None
    m = sma(arr, i, length)
    s = sum((arr[k] - m) ** 2 for k in range(i - length + 1, i + 1))
    return (s / (length - 1)) ** 0.5


def detect_signal(bars, cfg, conv_factor=1.0, available_balance=None):
    n = len(bars)
    min_len = cfg.get("min_len", 6)
    max_wait = cfg.get("max_wait", 6)
    trend_filter = cfg.get("trend_filter", False)
    slope_lookback = cfg.get("slope_lookback", 10)
    slope_atr_mult = cfg.get("slope_atr_mult", 0.5)
    use_acct_risk = "min_sl_acct" in cfg  # forex path (risk expressed in account currency)

    if n < EMA_LEN + 5:
        return {"hasSignal": False, "note": f"insufficient bars: {n}"}

    closes = [b["c"] for b in bars]
    highs = [b["h"] for b in bars]
    lows = [b["l"] for b in bars]

    tr = [None] * n
    for i in range(n):
        if i == 0:
            tr[i] = highs[i] - lows[i]
        else:
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    atr = [None] * n
    for i in range(n):
        if i < ATR_LEN - 1:
            continue
        if i == ATR_LEN - 1:
            atr[i] = sum(tr[0:i + 1]) / ATR_LEN
        else:
            atr[i] = (atr[i - 1] * (ATR_LEN - 1) + tr[i]) / ATR_LEN

    ema200 = [None] * n
    ema_alpha = 2 / (EMA_LEN + 1)
    for i in range(n):
        if i < EMA_LEN - 1:
            continue
        if i == EMA_LEN - 1:
            ema200[i] = sum(closes[0:i + 1]) / EMA_LEN
        else:
            ema200[i] = closes[i] * ema_alpha + ema200[i - 1] * (1 - ema_alpha)

    u_bb = [None] * n
    l_bb = [None] * n
    u_kc = [None] * n
    l_kc = [None] * n
    sqz_on = [None] * n
    for i in range(n):
        basis = sma(closes, i, BB_LEN)
        dev = stdev_unbiased(closes, i, BB_LEN)
        if basis is None or dev is None or atr[i] is None:
            continue
        u_bb[i] = basis + 2.0 * dev
        l_bb[i] = basis - 2.0 * dev
        u_kc[i] = basis + KC_MULT * atr[i]
        l_kc[i] = basis - KC_MULT * atr[i]
        sqz_on[i] = u_bb[i] < u_kc[i] and l_bb[i] > l_kc[i]

    run_hi = run_lo = None
    run_len = 0
    a_entry = a_sl = a_tp = a_qty = None
    a_is_buy = False
    a_filled = False
    a_active = False
    a_rel_bar = None
    last_signal = None

    start = EMA_LEN
    for i in range(start, n):
        if sqz_on[i] is None or sqz_on[i - 1] is None:
            continue
        prev_sqz = sqz_on[i - 1]

        if sqz_on[i]:
            if not prev_sqz:
                run_hi, run_lo, run_len = highs[i], lows[i], 1
            else:
                run_hi = max(run_hi, highs[i])
                run_lo = min(run_lo, lows[i])
                run_len += 1

        if a_active and not a_filled and not sqz_on[i]:
            if (a_is_buy and highs[i] >= a_entry) or (not a_is_buy and lows[i] <= a_entry):
                a_filled = True

        cleared = False
        if a_active:
            if a_filled:
                if (a_is_buy and (highs[i] >= a_tp or lows[i] <= a_sl)) or \
                   (not a_is_buy and (lows[i] <= a_tp or highs[i] >= a_sl)):
                    a_active = a_filled = False
                    a_entry = a_sl = a_tp = a_qty = None
                    a_rel_bar = None
                    cleared = True
            else:
                if a_rel_bar is None and not sqz_on[i] and prev_sqz:
                    a_rel_bar = i
                sl_before_entry = (lows[i] <= a_sl) if a_is_buy else (highs[i] >= a_sl)
                expired = a_rel_bar is not None and i > a_rel_bar + max_wait
                if a_rel_bar is not None and (sl_before_entry or expired):
                    a_active = a_filled = False
                    a_entry = a_sl = a_tp = a_qty = None
                    a_rel_bar = None
                    cleared = True

        if not cleared and sqz_on[i] and run_len >= min_len and run_hi is not None and not a_filled:
            coil_hi, coil_lo = run_hi, run_lo
            rng = coil_hi - coil_lo
            coil_atr = rng / atr[i] if atr[i] > 0 else 99.0
            is_buy = closes[i] >= ema200[i]
            entry = coil_hi if is_buy else coil_lo
            slv0 = coil_lo if is_buy else coil_hi
            risk0 = abs(entry - slv0)

            if use_acct_risk:
                min_sl_acct = cfg["min_sl_acct"]
                max_sl_acct = cfg["max_sl_acct"]
                risk_acct0 = risk0 * conv_factor
                use_floor = risk_acct0 < min_sl_acct
                slv = (entry - (min_sl_acct / conv_factor)) if (use_floor and is_buy) else \
                      (entry + (min_sl_acct / conv_factor)) if use_floor else slv0
                risk = (min_sl_acct / conv_factor) if use_floor else risk0
                risk_acct = risk * conv_factor
                passes_risk = risk_acct >= min_sl_acct and risk_acct <= max_sl_acct
            else:
                min_sl, max_sl = cfg["min_sl"], cfg["max_sl"]
                slv = (entry - min_sl) if (risk0 < min_sl and is_buy) else \
                      (entry + min_sl) if risk0 < min_sl else slv0
                risk = min_sl if risk0 < min_sl else risk0
                passes_risk = risk >= min_sl and risk <= max_sl

            trend_ok = True
            if trend_filter:
                ema_prev = ema200[i - slope_lookback] if (i - slope_lookback >= 0) else None
                signed_slope = (ema200[i] - ema_prev) if (ema_prev is not None and ema200[i] is not None) else None
                signed_ratio = (signed_slope / atr[i]) if (signed_slope is not None and atr[i] > 0) else None
                trend_ok = signed_ratio is not None and (
                    (signed_ratio >= slope_atr_mult) if is_buy else (signed_ratio <= -slope_atr_mult)
                )

            if coil_atr <= COIL_MAX and passes_risk and trend_ok:
                tp = entry + TP_RR * risk if is_buy else entry - TP_RR * risk
                risk_denominator = risk_acct if use_acct_risk else risk
                risk_qty = cfg["account_bal"] * cfg["risk_pct"] / 100.0 / risk_denominator

                if "max_notional" in cfg:  # WEEX: notional cap is a flat dollar cap, not leverage*balance
                    notional_qty = cfg["max_notional"] / entry
                else:
                    notional_qty = cfg["account_bal"] * cfg["leverage"] / entry

                caps = [risk_qty, notional_qty]
                if "margin_safety" in cfg and available_balance is not None:
                    margin_cap_qty = (available_balance * cfg["leverage"] * cfg["margin_safety"]) / entry
                    caps.append(margin_cap_qty)

                qty = round(min(caps) * 10000) / 10000

                first_arm = not a_active
                a_entry, a_sl, a_tp, a_qty, a_is_buy, a_active, a_rel_bar = entry, slv, tp, qty, is_buy, True, None
                if first_arm:
                    last_signal = {"side": "buy" if is_buy else "sell", "entry": entry, "sl": slv,
                                    "tp": tp, "qty": qty, "barIndex": i, "time": bars[i]["t"]}
            else:
                if a_active and not a_filled:
                    a_active = a_filled = False
                    a_entry = a_sl = a_tp = a_qty = None
                    a_rel_bar = None

    last_idx = n - 1
    is_new_signal = last_signal is not None and last_signal["barIndex"] == last_idx
    if not is_new_signal:
        return {"hasSignal": False, "lastBarTime": bars[last_idx]["t"], "barsProcessed": n}

    return {
        "hasSignal": True,
        "side": last_signal["side"],
        "entry": last_signal["entry"],
        "sl": last_signal["sl"],
        "tp": last_signal["tp"],
        "qty": last_signal["qty"],
        "lastBarTime": bars[last_idx]["t"],
        "barsProcessed": n,
    }


# ============================================================
# BAR FETCHING
# ============================================================
def fetch_alpaca_equity_bars(symbol):
    start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    url = (f"{ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars?"
           f"timeframe=30Min&limit=700&adjustment=raw&feed=iex&start={start}")
    status, data = http_request("GET", url, headers={
        "APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY})
    if status != 200:
        raise RuntimeError(f"Alpaca bars fetch failed ({status}): {data}")
    raw = data.get("bars", []) or []
    now_ms = time.time() * 1000
    bars = []
    for b in raw:
        t = b["t"]
        start_ms = datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() * 1000
        if start_ms + 30 * 60 * 1000 <= now_ms:
            bars.append({"t": t, "o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]), "c": float(b["c"])})
    return bars


def fetch_alpaca_crypto_bars(symbol):
    start = (datetime.now(timezone.utc) - timedelta(days=20)).strftime("%Y-%m-%d")
    url = (f"{ALPACA_DATA_BASE}/v1beta3/crypto/us/bars?"
           f"symbols={urllib.parse.quote(symbol)}&timeframe=30Min&limit=700&start={start}")
    status, data = http_request("GET", url, headers={
        "APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY})
    if status != 200:
        raise RuntimeError(f"Alpaca crypto bars fetch failed ({status}): {data}")
    raw = (data.get("bars", {}) or {}).get(symbol, [])
    now_ms = time.time() * 1000
    bars = []
    for b in raw:
        t = b["t"]
        start_ms = datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() * 1000
        if start_ms + 30 * 60 * 1000 <= now_ms:
            bars.append({"t": t, "o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]), "c": float(b["c"])})
    return bars


def fetch_oanda_pricing(instrument):
    url = f"{OANDA_BASE}/v3/accounts/{OANDA_ACCOUNT_ID}/pricing?instruments={instrument}"
    status, data = http_request("GET", url, headers={
        "Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"})
    if status != 200:
        raise RuntimeError(f"OANDA pricing fetch failed ({status}): {data}")
    prices = data.get("prices", [])
    if not prices:
        return 1.0
    return float(prices[0].get("quoteHomeConversionFactors", {}).get("positiveUnits", 1.0))


def fetch_oanda_candles(instrument):
    url = f"{OANDA_BASE}/v3/instruments/{instrument}/candles?granularity=M30&count=700&price=M"
    status, data = http_request("GET", url, headers={
        "Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"})
    if status != 200:
        raise RuntimeError(f"OANDA candles fetch failed ({status}): {data}")
    raw = data.get("candles", [])
    bars = []
    for c in raw:
        if not c.get("complete"):
            continue
        mid = c["mid"]
        bars.append({"t": c["time"], "o": float(mid["o"]), "h": float(mid["h"]),
                      "l": float(mid["l"]), "c": float(mid["c"])})
    return bars


# ============================================================
# WEEX signing + calls
# ============================================================
def weex_sign(ts, method, path, body=""):
    sign_str = f"{ts}{method}{path}{body}"
    return base64.b64encode(hmac.new(WEEX_SECRET.encode(), sign_str.encode(), hashlib.sha256).digest()).decode()


def weex_headers(ts, sign):
    return {"ACCESS-KEY": WEEX_ACCESS_KEY, "ACCESS-SIGN": sign,
            "ACCESS-PASSPHRASE": WEEX_PASSPHRASE, "ACCESS-TIMESTAMP": ts}


def weex_get_balance():
    ts = str(int(time.time() * 1000))
    path = "/capi/v3/sim/balance"
    sign = weex_sign(ts, "GET", path)
    status, data = http_request("GET", WEEX_BASE + path, headers=weex_headers(ts, sign))
    if status != 200:
        raise RuntimeError(f"WEEX balance fetch failed ({status}): {data}")
    if isinstance(data, list):
        data = data[0] if data else {}
    return float(data.get("availableBalance", 0))


def log_weex_execution(side, qty, signal_price, tp, sl, status, data):
    # Appends one line per WEEX order submission. Captures the reference
    # price the signal was based on (signal_price) alongside whatever the
    # exchange actually returns, so real execution cost (fill price vs
    # signal price, plus fee) can be measured later instead of assumed
    # from a cost tier. Best-effort -- a logging failure must never break
    # a live order submission.
    try:
        entry = {
            "logged_at": now_iso(),
            "side": side,
            "qty": qty,
            "signal_price": signal_price,
            "tp": tp,
            "sl": sl,
            "http_status": status,
            "response": data,
        }
        with open(WEEX_EXEC_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[weex execution log] WARNING: failed to log order: {e}")


def weex_submit_order(side, qty, tp, sl, signal_price=None):
    ts = str(int(time.time() * 1000))
    path = "/capi/v3/sim/order"
    body_obj = {
        "symbol": "BTCSUSDT",
        "side": "BUY" if side == "buy" else "SELL",
        "positionSide": "LONG" if side == "buy" else "SHORT",
        "type": "MARKET",
        "quantity": f"{qty:.4f}",
        "newClientOrderId": f"weexbtc-{ts}",
        "tpTriggerPrice": str(tp),
        "slTriggerPrice": str(sl),
    }
    body = json.dumps(body_obj)
    sign = weex_sign(ts, "POST", path, body)
    status, data = http_request("POST", WEEX_BASE + path, headers=weex_headers(ts, sign), body=body)
    log_weex_execution(side, qty, signal_price, tp, sl, status, data)
    return status, data


def weex_get_all_positions():
    ts = str(int(time.time() * 1000))
    path = "/capi/v3/sim/position/allPosition"
    sign = weex_sign(ts, "GET", path)
    status, data = http_request("GET", WEEX_BASE + path, headers=weex_headers(ts, sign))
    return status, data


# ============================================================
# ALPACA order helpers (equity)
# ============================================================
def alpaca_headers():
    return {"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}


def alpaca_submit_stop_order(symbol, qty, side, stop_price):
    body = {"symbol": symbol, "qty": str(qty), "side": side, "type": "stop",
            "stop_price": str(stop_price), "time_in_force": "day"}
    return http_request("POST", f"{ALPACA_PAPER_BASE}/v2/orders", headers=alpaca_headers(), body=body)


def alpaca_submit_oco_exit(symbol, qty, exit_side, tp, sl):
    body = {"symbol": symbol, "qty": str(qty), "side": exit_side, "type": "limit",
            "time_in_force": "gtc", "order_class": "oco",
            "take_profit": {"limit_price": str(tp)}, "stop_loss": {"stop_price": str(sl)}}
    return http_request("POST", f"{ALPACA_PAPER_BASE}/v2/orders", headers=alpaca_headers(), body=body)


def alpaca_get_order(order_id):
    return http_request("GET", f"{ALPACA_PAPER_BASE}/v2/orders/{order_id}", headers=alpaca_headers())


def alpaca_cancel_order(order_id):
    return http_request("DELETE", f"{ALPACA_PAPER_BASE}/v2/orders/{order_id}", headers=alpaca_headers())


def alpaca_get_position(symbol):
    return http_request("GET", f"{ALPACA_PAPER_BASE}/v2/positions/{urllib.parse.quote(symbol)}",
                         headers=alpaca_headers())


def alpaca_submit_stop_limit_order(symbol, qty, side, stop_price, limit_price):
    body = {"symbol": symbol, "qty": str(qty), "side": side, "type": "stop_limit",
            "stop_price": str(stop_price), "limit_price": str(limit_price), "time_in_force": "gtc"}
    return http_request("POST", f"{ALPACA_PAPER_BASE}/v2/orders", headers=alpaca_headers(), body=body)


def alpaca_submit_limit_order(symbol, qty, side, limit_price):
    body = {"symbol": symbol, "qty": str(qty), "side": side, "type": "limit",
            "limit_price": str(limit_price), "time_in_force": "gtc"}
    return http_request("POST", f"{ALPACA_PAPER_BASE}/v2/orders", headers=alpaca_headers(), body=body)


# ============================================================
# OANDA order helpers
# ============================================================
def oanda_headers():
    return {"Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"}


def oanda_submit_stop_order(instrument, units_signed, price, decimals, tp, sl):
    fmt = lambda x: f"{x:.{decimals}f}"
    body = {"order": {"type": "STOP", "instrument": instrument, "units": str(units_signed),
                       "price": fmt(price), "timeInForce": "GTC",
                       "takeProfitOnFill": {"price": fmt(tp)},
                       "stopLossOnFill": {"price": fmt(sl)}}}
    return http_request("POST", f"{OANDA_BASE}/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
                         headers=oanda_headers(), body=body)


def oanda_get_pending_orders():
    return http_request("GET", f"{OANDA_BASE}/v3/accounts/{OANDA_ACCOUNT_ID}/pendingOrders",
                         headers=oanda_headers())


def oanda_get_open_trades():
    return http_request("GET", f"{OANDA_BASE}/v3/accounts/{OANDA_ACCOUNT_ID}/openTrades",
                         headers=oanda_headers())


def oanda_cancel_order(order_id):
    return http_request("PUT", f"{OANDA_BASE}/v3/accounts/{OANDA_ACCOUNT_ID}/orders/{order_id}/cancel",
                         headers=oanda_headers())


# ============================================================
# PER-BROKER ROUTE HANDLERS
# Each mirrors the idle / pending_entry / position_open routing
# of its Make scenario exactly.
# ============================================================
def run_equity(cfg, state):
    key, symbol = cfg["key"], cfg["symbol"]
    rec = get_record(state, key)
    st = rec.get("state", "idle")

    if st == "idle":
        if cfg.get("market_hours_only"):
            now_et_hour = datetime.now(timezone.utc).hour  # rough guard; real cron restricts firing window too
        bars = fetch_alpaca_equity_bars(symbol)
        sig = detect_signal(bars, cfg)
        print(f"[{key}] idle: {sig.get('hasSignal')} bars={sig.get('barsProcessed')}")
        if sig.get("hasSignal"):
            if DRY_RUN:
                print(f"[{key}] DRY RUN would submit {sig['side']} stop qty={sig['qty']} @ {sig['entry']}")
                return
            status, data = alpaca_submit_stop_order(symbol, sig["qty"], sig["side"], sig["entry"])
            if status not in (200, 201):
                print(f"[{key}] ORDER SUBMIT FAILED ({status}): {data}")
                return
            update_record(state, key, {
                "state": "pending_entry", "pendingSl": sig["sl"], "pendingTp": sig["tp"],
                "pendingQty": sig["qty"], "pendingSide": sig["side"], "pendingOrderId": data.get("id"),
                "pendingSubmittedAt": now_iso()})
            print(f"[{key}] SIGNAL: submitted {sig['side']} stop order {data.get('id')}")

    elif st == "pending_entry":
        status, data = alpaca_get_order(rec["pendingOrderId"])
        if status != 200:
            print(f"[{key}] pending_entry: order status check failed ({status}): {data}")
            return
        order_status = data.get("status", "")
        if order_status in ("filled", "partially_filled"):
            exit_side = "sell" if rec["pendingSide"] == "buy" else "buy"
            status2, data2 = alpaca_submit_oco_exit(symbol, rec["pendingQty"], exit_side, rec["pendingTp"], rec["pendingSl"])
            if status2 not in (200, 201):
                print(f"[{key}] OCO EXIT SUBMIT FAILED ({status2}): {data2}")
                return
            update_record(state, key, {"state": "position_open", "ocoOrderId": data2.get("id")})
            print(f"[{key}] FILLED: entered position, OCO exit {data2.get('id')}")
        elif order_status in ("expired", "canceled", "rejected", "done_for_day", "replaced"):
            alpaca_cancel_order(rec["pendingOrderId"])
            update_record(state, key, {"state": "idle"})
            print(f"[{key}] entry expired/cancelled, reset to idle")

    elif st == "position_open":
        status, data = alpaca_get_position(symbol)
        if status == 404:
            update_record(state, key, {"state": "idle"})
            print(f"[{key}] position closed, reset to idle")


def run_alpaca_crypto(cfg, state):
    key, symbol = cfg["key"], cfg["symbol"]
    rec = get_record(state, key)
    st = rec.get("state", "idle")

    if st == "idle":
        bars = fetch_alpaca_crypto_bars(symbol)
        sig = detect_signal(bars, cfg)
        print(f"[{key}] idle: {sig.get('hasSignal')} bars={sig.get('barsProcessed')}")
        if sig.get("hasSignal"):
            entry_buf = round(sig["entry"] * 0.0015, 2)
            limit_price = sig["entry"] + entry_buf if sig["side"] == "buy" else sig["entry"] - entry_buf
            if DRY_RUN:
                print(f"[{key}] DRY RUN would submit {sig['side']} stop_limit qty={sig['qty']} stop={sig['entry']} limit={limit_price}")
                return
            status, data = alpaca_submit_stop_limit_order(symbol, sig["qty"], sig["side"], sig["entry"], limit_price)
            if status not in (200, 201):
                print(f"[{key}] ORDER SUBMIT FAILED ({status}): {data}")
                return
            update_record(state, key, {
                "state": "pending_entry", "pendingSl": sig["sl"], "pendingTp": sig["tp"],
                "pendingQty": sig["qty"], "pendingSide": sig["side"], "pendingOrderId": data.get("id"),
                "pendingSubmittedAt": now_iso()})
            print(f"[{key}] SIGNAL: submitted {sig['side']} stop_limit order {data.get('id')}")

    elif st == "pending_entry":
        status, data = alpaca_get_order(rec["pendingOrderId"])
        if status != 200:
            print(f"[{key}] pending_entry: order status check failed ({status}): {data}")
            return
        order_status = data.get("status", "")
        if order_status in ("filled", "partially_filled"):
            exit_side = "sell" if rec["pendingSide"] == "buy" else "buy"
            s1, d1 = alpaca_submit_limit_order(symbol, rec["pendingQty"], exit_side, rec["pendingTp"])
            sl_limit = round(rec["pendingSl"] * (0.9985 if rec["pendingSide"] == "buy" else 1.0015), 2)
            s2, d2 = alpaca_submit_stop_limit_order(symbol, rec["pendingQty"], exit_side, rec["pendingSl"], sl_limit)
            if s1 not in (200, 201) or s2 not in (200, 201):
                print(f"[{key}] TP/SL EXIT SUBMIT FAILED: tp=({s1},{d1}) sl=({s2},{d2})")
                return
            update_record(state, key, {"state": "position_open", "tpOrderId": d1.get("id"),
                                        "slOrderId": d2.get("id"), "pendingSl": 0, "pendingTp": 0})
            print(f"[{key}] FILLED: TP order {d1.get('id')}, SL order {d2.get('id')}")
        elif order_status in ("expired", "canceled", "rejected", "done_for_day", "replaced"):
            alpaca_cancel_order(rec["pendingOrderId"])
            update_record(state, key, {"state": "idle"})
            print(f"[{key}] entry expired/cancelled, reset to idle")

    elif st == "position_open":
        s1, d1 = alpaca_get_order(rec["tpOrderId"])
        s2, d2 = alpaca_get_order(rec["slOrderId"])
        tp_status = d1.get("status", "") if s1 == 200 else ""
        sl_status = d2.get("status", "") if s2 == 200 else ""
        filled = ("filled", "partially_filled")
        if tp_status in filled:
            alpaca_cancel_order(rec["slOrderId"])
            update_record(state, key, {"state": "idle"})
            print(f"[{key}] TP filled, cancelled SL leg, reset to idle")
        elif sl_status in filled:
            alpaca_cancel_order(rec["tpOrderId"])
            update_record(state, key, {"state": "idle"})
            print(f"[{key}] SL filled, cancelled TP leg, reset to idle")


def run_oanda(cfg, state):
    key, instrument, decimals = cfg["key"], cfg["instrument"], cfg["decimals"]
    rec = get_record(state, key)
    st = rec.get("state", "idle")

    if st == "idle":
        conv_factor = fetch_oanda_pricing(instrument)
        bars = fetch_oanda_candles(instrument)
        sig = detect_signal(bars, cfg, conv_factor=conv_factor)
        print(f"[{key}] idle: {sig.get('hasSignal')} bars={sig.get('barsProcessed')} conv={conv_factor}")
        if sig.get("hasSignal"):
            units = sig["qty"] if sig["side"] == "buy" else -sig["qty"]
            if DRY_RUN:
                print(f"[{key}] DRY RUN would submit {sig['side']} STOP units={units} @ {sig['entry']}")
                return
            status, data = oanda_submit_stop_order(instrument, units, sig["entry"], decimals, sig["tp"], sig["sl"])
            if status not in (200, 201):
                print(f"[{key}] ORDER SUBMIT FAILED ({status}): {data}")
                return
            order_id = data.get("orderCreateTransaction", {}).get("id")
            update_record(state, key, {
                "state": "pending_entry", "pendingSl": sig["sl"], "pendingTp": sig["tp"],
                "pendingQty": sig["qty"], "pendingSide": sig["side"], "pendingOrderId": order_id,
                "pendingSubmittedAt": now_iso()})
            print(f"[{key}] SIGNAL: submitted {sig['side']} STOP order {order_id}")

    elif st == "pending_entry":
        s1, d1 = oanda_get_pending_orders()
        s2, d2 = oanda_get_open_trades()
        orders = d1.get("orders", []) if s1 == 200 else []
        trades = d2.get("trades", []) if s2 == 200 else []
        pending_id = str(rec.get("pendingOrderId", ""))
        still_pending = any(str(o.get("id")) == pending_id for o in orders)

        if not still_pending:
            match = next((t for t in trades if t.get("instrument") == instrument), None)
            if match:
                update_record(state, key, {"state": "position_open", "ocoOrderId": match["id"],
                                            "pendingOrderId": "", "pendingSubmittedAt": ""})
                print(f"[{key}] FILLED: trade {match['id']}")
            else:
                update_record(state, key, {"state": "idle", "ocoOrderId": None})
                print(f"[{key}] order vanished (no matching trade), reset to idle")
        else:
            submitted_at = rec.get("pendingSubmittedAt", "")
            if submitted_at:
                submitted_dt = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
                elapsed = (datetime.now(timezone.utc) - submitted_dt).total_seconds()
                max_wait_s = cfg.get("max_wait", 6) * 30 * 60
                if elapsed > max_wait_s:
                    oanda_cancel_order(pending_id)
                    update_record(state, key, {"state": "idle", "ocoOrderId": None})
                    print(f"[{key}] entry expired ({elapsed:.0f}s > {max_wait_s}s), cancelled, reset to idle")

    elif st == "position_open":
        status, data = oanda_get_open_trades()
        trades = data.get("trades", []) if status == 200 else []
        trade_id = str(rec.get("ocoOrderId", ""))
        still_open = any(str(t.get("id")) == trade_id for t in trades)
        if not still_open:
            update_record(state, key, {"state": "idle"})
            print(f"[{key}] position closed, reset to idle")


def run_weex(cfg, state):
    key = cfg["key"]
    rec = get_record(state, key)
    st = rec.get("state", "idle")

    if st == "idle":
        bars = fetch_alpaca_crypto_bars(cfg["alpaca_symbol"])
        available_balance = weex_get_balance()
        sig = detect_signal(bars, cfg, available_balance=available_balance)
        print(f"[{key}] idle: {sig.get('hasSignal')} bars={sig.get('barsProcessed')} avail_bal={available_balance}")
        if sig.get("hasSignal"):
            if DRY_RUN:
                print(f"[{key}] DRY RUN would submit {sig['side']} MARKET qty={sig['qty']} tp={sig['tp']} sl={sig['sl']}")
                return
            status, data = weex_submit_order(sig["side"], sig["qty"], sig["tp"], sig["sl"], signal_price=sig["entry"])
            update_record(state, key, {
                "state": "position_open", "pendingSl": sig["sl"], "pendingTp": sig["tp"],
                "pendingQty": sig["qty"], "pendingSide": sig["side"],
                "pendingOrderId": data.get("orderId") if isinstance(data, dict) else None,
                "pendingSubmittedAt": now_iso(), "debug": f"order:{status}|{data}"})
            print(f"[{key}] SIGNAL: submitted {sig['side']} market order, status={status}")

    elif st == "position_open":
        status, data = weex_get_all_positions()
        arr = data if isinstance(data, list) else ([data] if data else [])
        pos = next((p for p in arr if p.get("symbol") == "BTCSUSDT" and float(p.get("size", 0)) > 0), None)
        if not pos:
            update_record(state, key, {"state": "idle", "pendingSl": 0, "pendingTp": 0, "pendingQty": 0,
                                        "pendingSide": "", "pendingOrderId": "", "pendingSubmittedAt": ""})
            print(f"[{key}] position closed, reset to idle")


# ============================================================
# MAIN
# ============================================================
def main():
    state = load_state()
    for cfg in WATCHLIST:
        try:
            if cfg["broker"] == "alpaca_equity":
                run_equity(cfg, state)
            elif cfg["broker"] == "alpaca_crypto":
                run_alpaca_crypto(cfg, state)
            elif cfg["broker"] == "oanda":
                run_oanda(cfg, state)
            elif cfg["broker"] == "weex":
                run_weex(cfg, state)
        except Exception as e:
            print(f"[{cfg['key']}] ERROR: {e}")


if __name__ == "__main__":
    main()
