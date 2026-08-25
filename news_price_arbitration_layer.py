"""
news_price_arbitration_layer.py

THE missing piece identified 2026-08-25: a real, standalone layer that sits
ABOVE both the price-only regime classifier (regime_classifier_live.py /
regime_decision_engine.price_regime()) and the news leg (delisting_poc,
paused pending Gemini tokens as of this writing), and produces the ONE
final answer Model D actually acts on. Per the formal spec written
2026-08-23 ("Regime Classifier -- Job Specification"): the price classifier
"does not combine itself with the news leg's output... that priority
ordering happens in a layer above both systems, not inside this one."
This file IS that layer. Neither regime_classifier_live.py nor
regime_decision_engine.py's HALF 1 (price_regime()) is modified -- both are
reused verbatim, imported, not re-derived.

WHY A SEPARATE FILE, NOT A REVIVAL OF regime_decision_engine.py's HALF 2:
that file's apply_news_adjustment() was marked superseded for an
architectural reason, not a logic reason -- it lived INSIDE the same file
as the price regime computation, blurring "what does price say" with "what
does news say." The actual combination RULE in that old function was sound
(confirmed+relevant+clear-direction gate, CRASH override, softening not
auto-promotion) and is reused here, verbatim in spirit, but now correctly
isolated in its own layer that takes both systems' outputs as plain
arguments instead of computing one of them itself.

WHAT'S NEW HERE, beyond reviving the old override rule:
1. SYMBOL SCOPING. The finalized news-leg->bridge contract (model-d-
   bridge-2026-08-23.md) includes `symbols_affected: list | "systemwide"`.
   The old logic implicitly treated every news item as systemwide. That's
   wrong for a 10-coin watchlist -- a security incident naming one coin
   should not push the other 9 coins' regime call to CRASH. This layer
   only applies a regime override to symbols actually in scope.
2. CATEGORY-BASED ROUTING. Per the bridge doc's own decision #2 (5
   categories, not 4, specifically because delisting-risk and operational-
   change items need DIFFERENT target actions, not just a regime nudge):
   only "macro/systemic" news is treated as a regime-scalar override here.
   "exchange_delisting_halt" and "exchange_operational" route to their own
   named actions (PULL_FROM_WATCHLIST / UPDATE_COST_MODEL_OR_REVALIDATE)
   and do NOT touch the CRASH/CALM/BULL regime call at all -- matches the
   bridge doc's explicit statement that merging these would just push an
   unresolved distinction downstream.
3. FAIL-SAFE ON UNKNOWN CATEGORIES. Only 3 of the 5 category names are
   confirmed on this side of the bridge (macro/systemic,
   exchange_delisting_halt, exchange_operational -- see model-d-bridge-
   2026-08-23.md). The other 2 (referenced elsewhere in this project as
   "regulatory" and "security incident" tests, e.g. SEC v. Ripple, CoinEx
   hack) have never had their exact category-string and target-action
   pinned down against the news leg's own scope doc. Rather than GUESS at
   what an unrecognized category should do, this layer passes it through
   as informational-only, no automatic action -- fail-safe, same direction
   as every other classifier in this project (missing/uncertain
   information never triggers a risk-reducing OR risk-taking action on its
   own).

STATUS: reference implementation only. `delisting_poc` (the actual news
ingestion/classification pipeline) is not live yet -- paused by the user,
2026-08-25, pending (a) a final readiness check before wiring to the
bridge, and (b) Gemini API tokens (exhausted backtesting the news leg's
own filtration system, expected available later the same day). This file
is built and sanity-checked now so it is ready the moment real classified
news items start arriving -- nothing here executes real orders or pulls
anything from a real watchlist.

Run: python3 news_price_arbitration_layer.py
"""
import regime_decision_engine as RDE

# ---------------------------------------------------------------------------
# The adopted 10-coin watchlist (model_d_final_policy.py), reused verbatim --
# needed here only to validate symbols_affected entries are real watchlist
# members, not to re-derive tier assignment (this layer doesn't size trades).
# ---------------------------------------------------------------------------
WATCHLIST_SYMBOLS = (
    "COMPUSDT", "CRVUSDT", "ONTUSDT", "CELRUSDT", "ILVUSDT", "SYSUSDT", "XNOUSDT",  # tier-1
    "AUDIOUSDT", "LPTUSDT", "FLUXUSDT",  # tier-2
)

# Categories with a CONFIRMED target action, per model-d-bridge-2026-08-23.md
# decision #2. Anything else is passed through informational-only (fail-safe).
CATEGORY_REGIME_OVERRIDE = "macro/systemic"          # only this one touches the regime scalar
CATEGORY_DELISTING_HALT = "exchange_delisting_halt"  # -> PULL_FROM_WATCHLIST
CATEGORY_OPERATIONAL = "exchange_operational"        # -> UPDATE_COST_MODEL_OR_REVALIDATE
KNOWN_CATEGORIES = (CATEGORY_REGIME_OVERRIDE, CATEGORY_DELISTING_HALT, CATEGORY_OPERATIONAL)


def _symbol_in_scope(symbol, symbols_affected):
    """symbols_affected is either the literal string 'systemwide' or a list
    of ticker strings. Fails safe to NOT in scope if the field is missing or
    malformed -- an ambiguous scope should never accidentally widen a
    regime override to symbols it wasn't actually about."""
    if symbols_affected == "systemwide":
        return True
    if isinstance(symbols_affected, (list, tuple, set)):
        return symbol in symbols_affected
    return False


def _validate_news_assessment(news_assessment):
    """Returns a normalized dict with safe defaults for every field, and a
    list of validation notes (missing/malformed fields don't raise -- this
    layer fails safe by treating anything malformed as not-actionable, same
    discipline as every other classifier in this project)."""
    if news_assessment is None:
        return None, []

    notes = []
    out = {}
    out["confirmed"] = bool(news_assessment.get("confirmed", False))
    out["relevant"] = bool(news_assessment.get("relevant", False))
    direction = news_assessment.get("direction", "neutral")
    if direction not in ("up", "down", "neutral"):
        notes.append(f"direction={direction!r} not recognized -- treated as 'neutral'")
        direction = "neutral"
    out["direction"] = direction
    out["note"] = str(news_assessment.get("note", ""))
    out["source"] = str(news_assessment.get("source", "unknown"))
    out["category"] = news_assessment.get("category")
    symbols_affected = news_assessment.get("symbols_affected")
    if symbols_affected != "systemwide" and not isinstance(symbols_affected, (list, tuple, set)):
        notes.append(f"symbols_affected={symbols_affected!r} not recognized (expected 'systemwide' or a list) -- treated as affecting nothing")
        symbols_affected = []
    out["symbols_affected"] = symbols_affected
    out["timestamp"] = news_assessment.get("timestamp")
    return out, notes


def arbitrate(symbol, price_result, news_assessment=None):
    """THE single entry point. symbol: one of WATCHLIST_SYMBOLS (raises if
    not -- a caller bug, same convention as regime_classifier_live.py's
    classify_tier2_trade). price_result: output of
    regime_decision_engine.price_regime() (HALF 1, reused verbatim -- this
    layer never recomputes it). news_assessment: optional dict matching the
    finalized bridge contract, or None if no news leg is connected /
    delisting_poc is paused (exactly today's real situation).

    Returns a full, auditable dict: final_regime for THIS symbol,
    watchlist_action (None unless a delisting/operational item fired), and
    the complete reasoning trail. Never executes anything."""
    if symbol not in WATCHLIST_SYMBOLS:
        raise ValueError(f"{symbol} is not on the adopted 10-coin watchlist -- caller bug, not a data condition")

    result = {
        "symbol": symbol,
        "price_regime": price_result["regime"],
        "price_basis": price_result.get("basis"),
        "final_regime": price_result["regime"],
        "watchlist_action": None,
        "action_reason": None,
        "news_note": "no news assessment supplied -- price regime used as-is (delisting_poc not connected / no live news leg output)",
        "validation_notes": [],
    }

    news, notes = _validate_news_assessment(news_assessment)
    result["validation_notes"] = notes
    if news is None:
        return result

    category = news["category"]
    confirmed = news["confirmed"]
    relevant = news["relevant"]
    direction = news["direction"]
    note = news["note"]

    # --- Category routing: only macro/systemic touches the regime scalar ---
    if category == CATEGORY_REGIME_OVERRIDE:
        in_scope = _symbol_in_scope(symbol, news["symbols_affected"])
        if not (confirmed and relevant and direction in ("up", "down") and in_scope):
            result["news_note"] = (
                f"macro/systemic news present but did not meet override bar "
                f"(confirmed={confirmed}, relevant={relevant}, direction={direction}, "
                f"in_scope={in_scope}) -- price regime unchanged. {note}"
            )
            return result

        price_regime = price_result["regime"]
        if direction == "down" and price_regime != "CRASH-like":
            result["final_regime"] = "CRASH-like"
            result["news_note"] = f"OVERRIDDEN by confirmed, relevant, systemic down-direction news ahead of price catching up: {note}"
        elif direction == "up" and price_regime == "CRASH-like":
            result["final_regime"] = "CALM-like"
            result["news_note"] = f"news signals recovery, softened from CRASH-like to CALM-like (not straight to BULL-like -- that still requires the trend signal's own confirmation): {note}"
        elif direction == "up" and price_regime == "CALM-like":
            result["final_regime"] = price_regime
            result["news_note"] = f"news is bullish but trend basket hasn't confirmed BULL-like yet -- flagged EARLY, not auto-promoted: {note}"
        else:
            result["news_note"] = f"macro/systemic news noted but did not change the regime call: {note}"
        return result

    # --- Category routing: delisting/halt risk -> watchlist action, regime untouched ---
    if category == CATEGORY_DELISTING_HALT:
        in_scope = _symbol_in_scope(symbol, news["symbols_affected"])
        if confirmed and in_scope:
            result["watchlist_action"] = "PULL_FROM_WATCHLIST"
            result["action_reason"] = f"confirmed exchange delisting/trading-halt risk naming {symbol}: {note}"
        else:
            result["action_reason"] = f"delisting/halt category present but not actioned (confirmed={confirmed}, in_scope={in_scope}): {note}"
        result["news_note"] = "delisting/halt category does not affect the CRASH/CALM/BULL regime call by design -- see model-d-bridge-2026-08-23.md decision #2"
        return result

    # --- Category routing: exchange-operational change -> its own action, regime untouched ---
    if category == CATEGORY_OPERATIONAL:
        in_scope = _symbol_in_scope(symbol, news["symbols_affected"])
        if confirmed and in_scope:
            result["watchlist_action"] = "UPDATE_COST_MODEL_OR_REVALIDATE"
            result["action_reason"] = f"confirmed exchange-operational change affecting {symbol}: {note}"
        else:
            result["action_reason"] = f"operational category present but not actioned (confirmed={confirmed}, in_scope={in_scope}): {note}"
        result["news_note"] = "exchange-operational category does not affect the CRASH/CALM/BULL regime call by design -- see model-d-bridge-2026-08-23.md decision #2"
        return result

    # --- Fail-safe: unrecognized/unconfirmed category -> informational only ---
    result["news_note"] = (
        f"category={category!r} not one of the {len(KNOWN_CATEGORIES)} categories with a confirmed target "
        f"action on this side of the bridge ({', '.join(KNOWN_CATEGORIES)}) -- passed through informational-only, "
        f"no automatic action taken (fail-safe: an unmapped category never guesses a risk action). {note}"
    )
    return result


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks():
    print("=== SANITY CHECKS ===")

    p_crash = {"regime": "CRASH-like", "basis": "test"}
    p_calm = {"regime": "CALM-like", "basis": "test"}
    p_bull = {"regime": "BULL-like", "basis": "test"}

    # 1. No news -> price regime unchanged, no action
    r = arbitrate("COMPUSDT", p_calm, None)
    assert r["final_regime"] == "CALM-like" and r["watchlist_action"] is None
    print("  no-news -> price regime unchanged, no action -- OK")

    # 2. Invalid symbol -> raises (caller bug, not a data condition)
    try:
        arbitrate("NOTAWATCHLISTCOIN", p_calm, None)
        raise AssertionError("expected ValueError for non-watchlist symbol")
    except ValueError:
        pass
    print("  non-watchlist symbol -> raises (caller bug) -- OK")

    # 3. macro/systemic, confirmed+relevant+down+systemwide -> CRASH override, all symbols
    news_down = {"confirmed": True, "relevant": True, "direction": "down",
                 "category": "macro/systemic", "symbols_affected": "systemwide",
                 "note": "test systemic shock", "source": "test"}
    for sym in ("COMPUSDT", "AUDIOUSDT"):
        r = arbitrate(sym, p_calm, news_down)
        assert r["final_regime"] == "CRASH-like", r
    print("  macro/systemic confirmed+relevant+down+systemwide -> CRASH override on every symbol -- OK")

    # 4. macro/systemic but symbol NOT in scope -> unchanged for that symbol
    news_scoped = {"confirmed": True, "relevant": True, "direction": "down",
                   "category": "macro/systemic", "symbols_affected": ["COMPUSDT"],
                   "note": "test scoped shock", "source": "test"}
    r_in = arbitrate("COMPUSDT", p_calm, news_scoped)
    r_out = arbitrate("AUDIOUSDT", p_calm, news_scoped)
    assert r_in["final_regime"] == "CRASH-like"
    assert r_out["final_regime"] == "CALM-like"
    print("  macro/systemic scoped to one symbol -> only that symbol's regime overridden, others untouched -- OK")

    # 5. Unconfirmed/irrelevant/ambiguous -> no override (same false-positive discipline as USDC/SVB, CoinEx)
    for bad_news in [
        {"confirmed": False, "relevant": True, "direction": "down", "category": "macro/systemic", "symbols_affected": "systemwide"},
        {"confirmed": True, "relevant": False, "direction": "down", "category": "macro/systemic", "symbols_affected": "systemwide"},
        {"confirmed": True, "relevant": True, "direction": "neutral", "category": "macro/systemic", "symbols_affected": "systemwide"},
    ]:
        r = arbitrate("COMPUSDT", p_calm, bad_news)
        assert r["final_regime"] == "CALM-like", bad_news
    print("  unconfirmed / irrelevant / ambiguous-direction news -> no override, any one of the three fails the gate -- OK")

    # 6. up-direction during CRASH -> softens to CALM, not straight to BULL
    news_up = {"confirmed": True, "relevant": True, "direction": "up",
               "category": "macro/systemic", "symbols_affected": "systemwide", "note": "test recovery", "source": "test"}
    r = arbitrate("COMPUSDT", p_crash, news_up)
    assert r["final_regime"] == "CALM-like"
    print("  confirmed+relevant+up news during CRASH -> softens to CALM-like, not auto-promoted to BULL -- OK")

    # 7. up-direction during CALM -> flagged early, NOT promoted to BULL
    r = arbitrate("COMPUSDT", p_calm, news_up)
    assert r["final_regime"] == "CALM-like"
    print("  confirmed+relevant+up news during CALM -> regime unchanged (flagged early only, trend basket must confirm BULL itself) -- OK")

    # 8. exchange_delisting_halt, confirmed, in scope -> PULL_FROM_WATCHLIST, regime untouched
    news_delist = {"confirmed": True, "relevant": True, "direction": "down",
                   "category": "exchange_delisting_halt", "symbols_affected": ["COMPUSDT"],
                   "note": "test delisting notice", "source": "test"}
    r = arbitrate("COMPUSDT", p_calm, news_delist)
    assert r["watchlist_action"] == "PULL_FROM_WATCHLIST"
    assert r["final_regime"] == "CALM-like", "delisting news must not touch the regime scalar"
    r_other = arbitrate("AUDIOUSDT", p_calm, news_delist)
    assert r_other["watchlist_action"] is None, "delisting news scoped to COMPUSDT must not action AUDIOUSDT"
    print("  exchange_delisting_halt confirmed+in-scope -> PULL_FROM_WATCHLIST, regime scalar untouched, other symbols unaffected -- OK")

    # 9. exchange_operational, confirmed, in scope -> UPDATE_COST_MODEL_OR_REVALIDATE, regime untouched
    news_ops = {"confirmed": True, "relevant": True, "direction": "neutral",
                "category": "exchange_operational", "symbols_affected": ["LPTUSDT"],
                "note": "test fee schedule change", "source": "test"}
    r = arbitrate("LPTUSDT", p_calm, news_ops)
    assert r["watchlist_action"] == "UPDATE_COST_MODEL_OR_REVALIDATE"
    assert r["final_regime"] == "CALM-like"
    print("  exchange_operational confirmed+in-scope -> UPDATE_COST_MODEL_OR_REVALIDATE, regime scalar untouched -- OK")

    # 10. Unrecognized category -> informational only, no action, no regime change (fail-safe)
    news_unknown = {"confirmed": True, "relevant": True, "direction": "down",
                     "category": "regulatory", "symbols_affected": ["ONTUSDT"],
                     "note": "test unrecognized category", "source": "test"}
    r = arbitrate("ONTUSDT", p_calm, news_unknown)
    assert r["watchlist_action"] is None
    assert r["final_regime"] == "CALM-like"
    assert "not one of the" in r["news_note"]
    print("  unrecognized category (e.g. 'regulatory', not yet pinned down against the news leg's own scope doc) -> informational-only, no action -- OK")

    # 11. Malformed news_assessment (missing fields, bad direction/scope values) never raises
    r = arbitrate("COMPUSDT", p_calm, {"category": "macro/systemic"})
    assert r["final_regime"] == "CALM-like"
    r2 = arbitrate("COMPUSDT", p_calm, {"confirmed": True, "relevant": True, "direction": "sideways",
                                         "category": "macro/systemic", "symbols_affected": "COMPUSDT"})
    assert r2["final_regime"] == "CALM-like"
    assert len(r2["validation_notes"]) >= 1
    print("  malformed/incomplete news_assessment (missing fields, bad direction/scope types) never raises, fails safe -- OK")

    print("=== SANITY CHECKS PASSED ===\n")


def main():
    run_sanity_checks()

    print("=== DEMO: today's real price-only regime, arbitration layer with NO news connected (actual current state) ===")
    import regime_classifier_v2 as R
    import regime_classifier_final as RF

    windows = R.basket_rolling_features("cache", R.REFERENCE_SYMBOLS, R.WINDOW_DAYS, R.STEP_DAYS)
    latest_window = windows[-1]
    symbols = RF.default_trend_basket_symbols()
    trend_basket = RF.precompute_trend_basket(symbols, "cache")
    latest_t = max(d["ts"][-1] for d in trend_basket.values())

    # Use the true trailing-60-day return (latest-bar anchored), not the
    # stepped-window value -- see this session's own discrepancy check.
    tier1_60d_rets = []
    for sym in R.REFERENCE_SYMBOLS:
        bars45 = R.resample_45m(R.load_bars(sym, "cache"))
        win_bars = int(round(60 * R.BARS_PER_DAY_45M))
        anchor, last = bars45[-1 - win_bars], bars45[-1]
        tier1_60d_rets.append((last["c"] - anchor["c"]) / anchor["c"] * 100.0)
    tier1_60d = sum(tier1_60d_rets) / len(tier1_60d_rets)

    price_result = RDE.price_regime("cache", latest_t, tier1_60d, trend_basket)
    print(f"  Price-only regime (HALF 1, unchanged): {price_result['regime']} ({price_result['basis']})")

    for sym in ("COMPUSDT", "AUDIOUSDT"):
        r = arbitrate(sym, price_result, None)
        print(f"  {sym:10s} final_regime={r['final_regime']:12s} action={r['watchlist_action']}  -- {r['news_note']}")

    print("\nHonest note: delisting_poc (the real news classification pipeline) is paused as of 2026-08-25")
    print("(readiness check pending + Gemini tokens exhausted until later today) -- so every call above runs")
    print("with news_assessment=None, exactly today's real operating condition. The moment it produces real")
    print("classified output matching the bridge contract, that dict is the news_assessment argument here --")
    print("no other change needed on this side.")
    print("\nReference implementation only. Not wired into live trading. Does not modify regime_classifier_live.py,")
    print("regime_decision_engine.py, or model_d_final_policy.py -- purely additive, sits above all three.")


if __name__ == "__main__":
    main()
