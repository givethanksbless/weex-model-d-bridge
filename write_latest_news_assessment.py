#!/usr/bin/env python3
"""
write_latest_news_assessment.py

Translates delisting_poc's news_event_reports.json output into the exact
single-object schema news_price_arbitration_layer.arbitrate() expects,
and writes it to latest_news_assessment.json.

CONFIRMED schema (read directly from arbitrate()/_validate_news_assessment()
source on 2026-08-25, not guessed):
  confirmed: bool
  relevant: bool
  direction: "up" | "down" | "neutral" -- only "up"/"down" trigger anything
  category: exactly one of "macro/systemic", "exchange_delisting_halt",
            "exchange_operational" -- anything else is informational-only
            (arbitrate()'s own fail-safe branch, does not error)
  symbols_affected: list[str] of watchlist tickers (e.g. "COMPUSDT") OR
                     the literal string "systemwide" (not ["systemwide"])
  note: str
  source: str

news_event_reports.json's own category taxonomy (5 values, confirmed from
a real 35-item sample on 2026-08-25) does NOT line up 1:1 with the above --
this mapping is a deliberate judgment call, documented inline per category,
not a mechanical rename except for "macro/geopolitical shock" and
"stablecoin/peg failure" -> "macro/systemic".

KNOWN GAP, confirmed from real data, not hypothetical: the one live sample
seen so far (a "regulatory shock" item) had a scoped-sounding posture
("...in affected symbols") but an EMPTY symbols_affected list. This script
fails safe on that -- no determinable scope means no fire, same discipline
_symbol_in_scope() itself uses -- rather than guess which coin was meant.
Fix upstream in delisting_poc's extraction step if scoped events should
actually take effect.
"""
import json
import sys

# Model D's actual 10-coin watchlist, exact ticker form arbitrate() expects.
WATCHLIST_SYMBOLS = (
    "COMPUSDT", "CRVUSDT", "ONTUSDT", "CELRUSDT", "ILVUSDT",
    "SYSUSDT", "XNOUSDT", "AUDIOUSDT", "LPTUSDT", "FLUXUSDT",
)
# Base symbol (no USDT suffix) -> full watchlist ticker, for matching
# whatever raw symbol text the news classifier emits (e.g. "AUDIO" -> "AUDIOUSDT").
BASE_TO_TICKER = {t[:-4]: t for t in WATCHLIST_SYMBOLS}

# --- Category mapping: news_event_reports.json's 5 categories -> arbitrate()'s 3 ---
CATEGORY_MAP = {
    "macro/geopolitical shock": "macro/systemic",
    "stablecoin/peg failure": "macro/systemic",       # matches arbitrate.py's own
                                                        # docstring precedent (USDC/SVB)
    "regulatory shock": "macro/systemic",              # broad policy; scope comes
                                                        # from symbols_affected/posture
    "exchange/counterparty failure": "exchange_delisting_halt",
    # JUDGMENT CALL: pull-from-watchlist (conservative -- stop trading it
    # entirely) rather than just crash-sizing it under "macro/systemic".
    # Flip this to "macro/systemic" if you'd rather keep trading a coin
    # through an active exploit under crash-regime sizing instead.
    "hack/exploit": "exchange_delisting_halt",
}
# NOTE: nothing currently maps to "exchange_operational" -- this classifier
# doesn't produce that kind of event today. Add a mapping here if it does later.

NO_ACTION_POSTURE = "No action -- log only, do not fire"


def _posture_is_market_wide(posture):
    return "market-wide" in posture


def translate_one(item):
    """Returns a dict matching arbitrate()'s schema, or None if this item
    shouldn't produce a fire (either genuinely inert, or scope couldn't be
    determined -- both fail safe to 'do nothing' rather than guess)."""
    posture = item.get("recommended_posture", "")
    if posture == NO_ACTION_POSTURE:
        return None

    raw_category = item.get("category")
    category = CATEGORY_MAP.get(raw_category, raw_category)
    # If unmapped, pass the raw string through -- arbitrate() itself fails
    # safe on any category it doesn't recognize (informational only, no
    # action, no crash), so this is safe, just inert. Keeps a record of
    # what came through unmapped rather than silently dropping it.

    confirmed = item.get("confidence_tier") == "LIKELY"
    relevant = True  # NO_ACTION_POSTURE already filtered out above

    # Every posture value seen in this classifier's current output is
    # defensive/risk-reducing -- there is no "up"/recovery posture yet.
    direction = "down"

    if _posture_is_market_wide(posture):
        symbols_affected = "systemwide"
    else:
        raw_symbols = item.get("symbols_affected") or []
        mapped = []
        for s in raw_symbols:
            s_norm = str(s).strip().upper()
            if s_norm in WATCHLIST_SYMBOLS:
                mapped.append(s_norm)
            elif s_norm in BASE_TO_TICKER:
                mapped.append(BASE_TO_TICKER[s_norm])
            # else: not one of the 10 watchlist coins -- drop it, arbitrate()
            # only ever cares about watchlist symbols anyway.
        symbols_affected = mapped
        if not symbols_affected:
            # Posture implies a scoped symbol but none could be determined
            # (empty list, or nothing on it matched the watchlist). Confirmed
            # this actually happens in real data (see module docstring) --
            # fail safe to no-op rather than guess which coin was meant.
            return None

    return {
        "confirmed": confirmed,
        "relevant": relevant,
        "direction": direction,
        "category": category,
        "symbols_affected": symbols_affected,
        "note": item.get("evidence", {}).get("reasoning") or item.get("event_id", ""),
        "source": item.get("evidence", {}).get("source") or item.get("trigger_source", "unknown"),
        "timestamp": item.get("trigger_timestamp"),
        "_event_id": item.get("event_id"),  # extra, ignored by arbitrate() --
                                              # kept here for your own audit trail
        "_raw_category": raw_category,       # ditto -- lets you see what this
                                              # was before translation
    }


def pick_latest(reports_path):
    with open(reports_path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("items", [])

    candidates = []
    for item in items:
        if item.get("live_or_historical") != "live":
            continue  # historical events are already priced in -- don't
                        # re-fire a stale override on every poll
        translated = translate_one(item)
        if translated is not None:
            candidates.append(translated)

    if not candidates:
        return None

    # arbitrate() only accepts one object at a time -- most recent live,
    # actionable, mappable event wins.
    candidates.sort(key=lambda c: c.get("timestamp") or "", reverse=True)
    return candidates[0]


def main():
    reports_path = sys.argv[1] if len(sys.argv) > 1 else "news_event_reports.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "latest_news_assessment.json"

    latest = pick_latest(reports_path)
    if latest is None:
        print(
            "No live, actionable, watchlist-mappable event found -- leaving "
            f"{out_path} untouched. A missing file already means 'no news "
            "connected' to the bridge, handled safely."
        )
        return

    with open(out_path, "w") as f:
        json.dump(latest, f, indent=2)
    print(f"Wrote {out_path}:")
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    main()
