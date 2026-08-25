"""
regime_decision_engine.py

*** SUPERSEDED 2026-08-23 -- DO NOT USE FOR THE TIER-2 SIZING DECISION. ***
*** See regime_classifier_live.py, which replaces this file. ***

Why: this file's apply_news_adjustment() blends news directly into the
price regime call. The formal WHAT spec written later the same session
("Regime Classifier -- Job Specification") explicitly rules that out:
the classifier "does not combine itself with the news leg's output, and
does not need to know the news leg exists... That priority ordering
happens in a layer above both systems, not inside this one." So the
architecture this file implements is now known to be wrong, not just
superseded by a cleaner rewrite -- news belongs in a separate arbitration
layer (still unbuilt, and explicitly not this leg's job -- see the
2026-08-23 chat log: confirmed real-world crash events should route to
the spot leg, which owns the watchlist and live execution, not here).

Kept in ~/trading for the record, same convention as every other
superseded script in this project (v1-v5 of the original classifier,
etc.) -- not deleted, just not authoritative. The 3-way functional
regime definition in HALF 1 below (CRASH-like/CALM-like/BULL-like) is
still accurate and still reused elsewhere; only the news-blending in
HALF 2 is the part that's out of spec.

--- Original docstring follows, for context ---

Confirmed design (user, 2026-08-23): the news leg's only job is fetching
news. THIS leg's job is deciding which regime we're most likely in, using
EVERYTHING available -- the existing price signal, news, anything else --
and that regime decision picks which spotting system (sizing behavior)
gets used. Not married to exactly 3 regimes; open to more if risk
management ever needs a finer split.

Two honest, separate halves, on purpose:

HALF 1 -- PRICE REGIME (fully automatable, works historically AND live):
  reuses this leg's own already-validated functional 3-way definition
  (regime_classifier_functional_3way.py): CRASH-like = tier-1 60-day
  return <= -15%, BULL-like = trend-basket participation >= 70%,
  CALM-like = neither (the residual). Deterministic, no judgment calls,
  runs on cached data the same way every other script in this project
  does.

HALF 2 -- NEWS ADJUSTMENT (NOT automatable as plain code -- stated
  honestly, not glossed over): classifying raw, ambiguous news requires
  judgment (an LLM or a human), not a fixed formula. There is no live
  news API wired into this project. So this half is deliberately built
  as a SEPARATE INPUT this function accepts -- a small, structured
  assessment -- rather than something the script fetches itself. Today,
  that assessment comes from a live chat turn (the assistant reading real
  search results, same as the short-squeeze example run this session).
  If this needs to run unattended, that assessment would come from a
  scheduled task instead -- still an LLM doing the reading, just on a
  timer instead of on-demand. The code below is honest about that
  boundary rather than pretending a Python function can read the news.

COMBINATION RULE, stated plainly: news only overrides the price regime
when it is CONFIRMED (not rumor) AND system-wide or watchlist-relevant
(not an unrelated token/exchange) AND directionally clear. A confirmed,
relevant, clear-direction news event can push CALM/BULL toward CRASH (or
vice versa) BEFORE the slower price signal would catch up on its own --
that's the entire point of adding it. Anything less than that (rumor,
irrelevant symbol, ambiguous direction) leaves the price regime
untouched -- same false-positive discipline already validated in the
news leg's own historical work (USDC/SVB, CoinEx: real stress, correctly
ignored because it didn't meet this bar).

Run: python3 regime_decision_engine.py
"""
import regime_classifier_v2 as R
import regime_classifier_v3_breadth as V3
import regime_classifier_final as RF

CRASH_RET_THRESHOLD = -15.0
TREND_FLAG = 0.70

REGIME_ACTIONS = {
    "CRASH-like": "FULL_SIZE",
    "CALM-like": "FULL_SIZE",
    "BULL-like": "HALF_SIZE",
}


# ---------------------------------------------------------------------------
# HALF 1: price-only regime read (deterministic, automatable, reused as-is
# from this leg's own already-validated functional 3-way definition)
# ---------------------------------------------------------------------------

def price_regime(suffix_or_symbols_data, t_ms, tier1_60d_return_pct, trend_basket):
    """tier1_60d_return_pct: the tier-1 reference basket's own trailing
    60-day return at t_ms (caller supplies it -- computed the same way as
    regime_classifier_v2.basket_rolling_features(), reused unchanged).
    trend_basket: precomputed via regime_classifier_final.precompute_trend_basket()."""
    if tier1_60d_return_pct is not None and tier1_60d_return_pct <= CRASH_RET_THRESHOLD:
        return {"regime": "CRASH-like", "basis": f"tier-1 60d return {tier1_60d_return_pct:.1f}% <= {CRASH_RET_THRESHOLD}%"}
    participation = RF.trend_participation_at(trend_basket, t_ms)
    if participation is not None and participation >= TREND_FLAG:
        return {"regime": "BULL-like", "basis": f"trend participation {participation:.1%} >= {TREND_FLAG:.0%}"}
    return {"regime": "CALM-like", "basis": "neither CRASH nor BULL condition met (residual)"}


# ---------------------------------------------------------------------------
# HALF 2: news adjustment -- a structured INPUT, not something this script
# fetches. See docstring above for why.
# ---------------------------------------------------------------------------

def apply_news_adjustment(price_result, news_assessment):
    """news_assessment: {"confirmed": bool, "relevant": bool, "direction":
    "down"/"up"/"neutral", "note": str} -- supplied by whoever/whatever did
    the actual reading (a chat turn today, a scheduled task later).
    Returns the FINAL regime call plus a full, auditable reason trail."""
    if news_assessment is None:
        return {**price_result, "final_regime": price_result["regime"],
                "news_note": "no news assessment supplied -- price regime used as-is"}

    confirmed = news_assessment.get("confirmed", False)
    relevant = news_assessment.get("relevant", False)
    direction = news_assessment.get("direction", "neutral")
    note = news_assessment.get("note", "")

    if not (confirmed and relevant and direction in ("up", "down")):
        return {**price_result, "final_regime": price_result["regime"],
                "news_note": f"news present but did not meet override bar (confirmed={confirmed}, relevant={relevant}, direction={direction}) -- price regime unchanged. {note}"}

    if direction == "down" and price_result["regime"] != "CRASH-like":
        return {**price_result, "final_regime": "CRASH-like",
                "news_note": f"OVERRIDDEN by confirmed, relevant, down-direction news ahead of price catching up: {note}"}
    if direction == "up" and price_result["regime"] == "CRASH-like":
        return {**price_result, "final_regime": "CALM-like",
                "news_note": f"news signals recovery, price regime softened from CRASH-like to CALM-like (not straight to BULL-like -- that still requires the trend signal itself to confirm): {note}"}
    if direction == "up" and price_result["regime"] == "CALM-like":
        return {**price_result, "final_regime": price_result["regime"],
                "news_note": f"news is bullish but the trend signal hasn't confirmed BULL-like yet -- flagged as EARLY, not overridden, since 'trending up' needs the trend basket's own confirmation, not just one news event: {note}"}
    return {**price_result, "final_regime": price_result["regime"],
            "news_note": f"news noted but did not change the regime call: {note}"}


def regime_to_action(final_regime):
    return REGIME_ACTIONS.get(final_regime, "FULL_SIZE (unknown regime, fail-safe)")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks():
    print("=== SANITY CHECKS ===")

    p_crash = {"regime": "CRASH-like", "basis": "test"}
    p_calm = {"regime": "CALM-like", "basis": "test"}
    p_bull = {"regime": "BULL-like", "basis": "test"}

    # No news supplied -> price regime unchanged
    r = apply_news_adjustment(p_calm, None)
    assert r["final_regime"] == "CALM-like", "no-news sanity failed"

    # Rumor / unconfirmed -> ignored, same false-positive discipline as USDC/SVB, CoinEx
    r = apply_news_adjustment(p_calm, {"confirmed": False, "relevant": True, "direction": "down"})
    assert r["final_regime"] == "CALM-like", "unconfirmed-news sanity failed"

    # Confirmed but irrelevant (not watchlist/system-wide) -> ignored
    r = apply_news_adjustment(p_calm, {"confirmed": True, "relevant": False, "direction": "down"})
    assert r["final_regime"] == "CALM-like", "irrelevant-news sanity failed"

    # Confirmed + relevant + down -> overrides CALM to CRASH
    r = apply_news_adjustment(p_calm, {"confirmed": True, "relevant": True, "direction": "down", "note": "test shock"})
    assert r["final_regime"] == "CRASH-like", "confirmed-down-override sanity failed"

    # Confirmed + relevant + up, while in CRASH -> softens to CALM, not straight to BULL
    r = apply_news_adjustment(p_crash, {"confirmed": True, "relevant": True, "direction": "up", "note": "test recovery"})
    assert r["final_regime"] == "CALM-like", "crash-recovery-softening sanity failed"

    # Confirmed + relevant + up, while already CALM -> flagged early, NOT bumped to BULL
    r = apply_news_adjustment(p_calm, {"confirmed": True, "relevant": True, "direction": "up", "note": "test rally"})
    assert r["final_regime"] == "CALM-like", "calm-to-bull-should-not-auto-promote sanity failed"

    print("  apply_news_adjustment(): all 6 branches (no-news, unconfirmed, irrelevant, confirmed-down-override,")
    print("  crash-recovery-softening, calm-should-not-auto-promote-to-bull) verified against hand-built inputs -- OK")

    assert regime_to_action("CRASH-like") == "FULL_SIZE"
    assert regime_to_action("CALM-like") == "FULL_SIZE"
    assert regime_to_action("BULL-like") == "HALF_SIZE"
    print("  regime_to_action(): matches Model D's already-adopted mapping exactly -- OK")
    print("=== SANITY CHECKS PASSED ===\n")


# ---------------------------------------------------------------------------
# Demo: today's real example (short-squeeze rally news vs. our stale
# trailing price signal), run live, plus the plain price-only read per
# cached regime file for reference.
# ---------------------------------------------------------------------------

def main():
    run_sanity_checks()

    symbols = V3.breadth_basket_symbols()
    print(f"=== Price-only regime read, latest cached point per regime file ({len(symbols)}-symbol trend basket) ===\n")
    for suffix, label in R.REGIMES:
        windows = R.basket_rolling_features(suffix, R.REFERENCE_SYMBOLS, R.WINDOW_DAYS, R.STEP_DAYS)
        if not windows:
            continue
        latest_window = windows[-1]
        trend_basket = RF.precompute_trend_basket(symbols, suffix)
        pr = price_regime(suffix, latest_window["t_end"], latest_window["ret_pct"], trend_basket)
        print(f"  {label:14s} price-only regime: {pr['regime']:12s} ({pr['basis']})")

    print("\n=== LIVE DEMO: today's real short-squeeze news applied on top of the CRASH_2025-26 file's latest point ===")
    print("(this is the actual example from this session: BTC +22% short-squeeze headline, today, vs. our trailing 15.1%% trend participation)\n")
    windows = R.basket_rolling_features("cache", R.REFERENCE_SYMBOLS, R.WINDOW_DAYS, R.STEP_DAYS)
    latest_window = windows[-1]
    trend_basket = RF.precompute_trend_basket(symbols, "cache")
    pr = price_regime("cache", latest_window["t_end"], latest_window["ret_pct"], trend_basket)
    print(f"  Price-only read: {pr['regime']} ({pr['basis']})")

    news_today = {
        "confirmed": True, "relevant": True, "direction": "up",
        "note": "BTC +22%% short-squeeze rally reported same-day; SEC 'Regulation Crypto Assets' framework + CLARITY Act push both broad/positive, not watchlist-specific",
    }
    final = apply_news_adjustment(pr, news_today)
    print(f"  Final regime: {final['final_regime']}")
    print(f"  Reasoning: {final['news_note']}")
    print(f"  Model D action: {regime_to_action(final['final_regime'])}")

    print("\nHONEST NOTE: this demo's news_assessment was hand-typed from a real search done earlier this session,")
    print("not fetched automatically -- that's HALF 2's documented boundary, not an oversight.")
    print("Reference implementation only. Not wired into live trading.")


if __name__ == "__main__":
    main()
