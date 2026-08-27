"""
spec_report.py

Translation layer between this pipeline's internal scoring (routed_items.json,
category/confidence_score/confirmed per news-ingest-classify-design-notes.md)
and the fixed report shape required by the news-event leg spec (2026-08-24,
consolidated doc from the regime-classifier/synthesis chat). Does NOT change
anything upstream -- layer1_filter.py, dedup_filter.py, layer2_classifier.py,
and layer2_route.py all keep working exactly as before, with their own
category rubric and confidence formula untouched. This script only reads
routed_items.json and reshapes CONFIRMED-worthy items into the spec's format.

Why a translation layer instead of rewriting the classifier's rubric: the
existing 6-category rubric (project_specific / exchange_operational /
exchange_delisting_risk / regulatory_action / macro_systemic / none) has been
tuned through 3 rubric versions against real misclassifications, and all 140
of today's items are cached against it -- replacing it wholesale would force
a full re-classification (re-billed to Groq/Gemini) for a taxonomy swap alone.
Decision made 2026-08-24 with the user: keep the tuned rubric, translate at
the reporting boundary instead.

CATEGORY MAPPING (decided 2026-08-24)
--------------------------------------
  regulatory_action      -> regulatory shock                  (clean 1:1)
  macro_systemic         -> macro/geopolitical shock           (clean 1:1)
  exchange_delisting_risk -> exchange/counterparty failure     (spec has no
                             delisting-specific category -- flagged back as
                             a gap in the spec itself; mapped here so
                             nothing gets silently dropped)
  project_specific        -> disambiguated via keyword match against
                             title+text+reasoning (see PROJECT_SPECIFIC_
                             SUBCATEGORY_KEYWORDS below), since this one
                             internal category spans 3 spec categories
                             (hack/exploit, exchange/counterparty failure,
                             stablecoin/peg failure). Defaults to
                             hack/exploit if no keyword matches, since
                             that's the dominant real-world case for this
                             category so far.
  exchange_operational    -> NOT reported. Routine exchange ops
                             (maintenance windows, fee changes, min order
                             size, listing/delisting POLICY -- as opposed
                             to an actual delisting event) are not
                             negative/shock events per the spec's own
                             scope ("never touch ... the calm/bull price
                             question, that's not yours" -- routine ops
                             are the news-domain equivalent).
  none                     -> NOT reported (matches existing `confirmed`
                             semantics -- nothing worth surfacing at all).

CONFIDENCE TIER MAPPING (decided 2026-08-24)
----------------------------------------------
  CONFIRMED  -- source_type == "first_party" AND confirmed == True.
                Decision: a first-party source (WEEX announcing its own
                action) IS the primary source the spec requires -- no
                separate corroboration needed.
  LIKELY     -- source_type == "third_party" AND final_confirmed == True
                (i.e. Groq and/or Gemini scored it confirmed, or both
                agreed). Strong corroboration, but a news outlet
                reporting on something is not itself a primary source.
  UNCONFIRMED -- everything else that still has a real category (not
                "none"/not exchange_operational) -- includes Groq/Gemini
                disagreements (providers_agree == False), since active
                model disagreement is itself a form of "not yet
                confirmed." Per the spec's own posture table, UNCONFIRMED
                always gets "No action -- log only, do not fire"
                regardless of what category it mapped to.

EVENT ID CROSS-TIME MATCHING (added 2026-08-24, second pass)
----------------------------------------------------------------
event_id now matches against a persistent registry (event_registry.json)
of open events: same mapped spec_category, AND (shared watchlist symbol
OR >= EVENT_MATCH_JACCARD_THRESHOLD title-word overlap), AND within
EVENT_MATCH_WINDOW_DAYS of that event's last-seen mention. On a match,
the existing event_id is reused and the registry's last-seen timestamp
rolls forward. This is mechanical (no LLM cost), same design philosophy
as dedup_filter.py -- NOT semantic understanding. Known failure modes,
by design tradeoff rather than oversight:
  - Can MISS a real follow-up if it's worded so differently it shares no
    watchlist symbol and under 34% of title words with the original.
  - Can OVER-MERGE two unrelated stories that happen to share enough
    entity words and land in the same category within the window (e.g.
    two different exchanges both having a "hack" story within 30 days
    could theoretically collide if their titles share enough words --
    unlikely in practice since exchange/project names are usually part
    of the matched word set, but not impossible).
  - `matched_existing_event` on every report says plainly whether this
    ID was reused or newly created, and event_registry.json is a flat,
    readable file -- spot-check it rather than trusting merges blindly.

CANONICAL SINGLE-TRIGGER TIMESTAMP (added 2026-08-25, third pass)
----------------------------------------------------------------
Per the spec's "single-trigger discipline": every report sharing an
event_id must report the SAME trigger_timestamp/trigger_source, not each
follow-up's own arrival time. The registry now tracks the event's
earliest-known confirming source (trigger_timestamp/trigger_source/
trigger_title) and updates it only if a newly-matched item's own
timestamp precedes what's currently stored (handles out-of-order
ingestion, e.g. a slower outlet's earlier-dated report arriving after a
faster outlet's later-dated one). build_report() now pulls
trigger_timestamp/trigger_source/trigger_source_title from the registry
entry, not from the individual item -- the item's own timestamp/url still
appear in evidence (that report's specific citation), and
is_canonical_source flags whether this particular report IS the event's
earliest source or a later follow-up riding the same event_id.

LIVE VS HISTORICAL FLAG (fixed 2026-08-25, Task #7)
----------------------------------------------------------------
live_or_historical is no longer hardcoded. determine_live_or_historical()
reads it straight off the item's own "provider" field -- real live-
pipeline items always carry whichever model actually classified them
(e.g. "groq", "gemini"), while historical_stress_test.py's hand-curated
items are explicitly tagged provider="hand_labeled_historical". Zero
inference, zero LLM cost -- the distinction was already in the data.

DELIBERATELY NOT BUILT (see 2026-08-24/25 discussion, deferred by the
user's own choice, not an oversight):
  - Lead-time-to-drawdown backtesting (60-day trailing / 24h fast-drawdown
    / 72h fast-drawdown metrics). This is the price/Model D leg's job, not
    this pipeline's -- it has no price data anywhere. lead_time_note below
    says so explicitly rather than fabricating a number.

Usage
-----
    python3 spec_report.py
"""

import json
import os
import re
from datetime import datetime, timezone

INPUT_FILE = "routed_items.json"
OUTPUT_FILE = "news_event_reports.json"
EVENT_REGISTRY_FILE = "event_registry.json"

# How many days a follow-up can trail the LAST time this event was seen
# (not the first) before it's treated as a new, unrelated event. Sagas
# like a regulatory lawsuit can run for months, but 30 days is a
# reasonable default for "is this still the same acute story" without
# ever-growing false merges from stale entries -- adjust here if real
# data shows it's too tight or too loose.
EVENT_MATCH_WINDOW_DAYS = 30

# Jaccard overlap (intersection/union) of significant title words needed
# to call two items the same event, when they don't already share a
# watchlist symbol. 0.34 means roughly 1-in-3 significant words in
# common -- looser than dedup_filter.py's 0.75 same-day threshold on
# purpose, since follow-up headlines days later are often reworded a lot
# more than same-day cross-outlet copies of one story.
EVENT_MATCH_JACCARD_THRESHOLD = 0.34

# --- category mapping -------------------------------------------------

DIRECT_CATEGORY_MAP = {
    "regulatory_action": "regulatory shock",
    "macro_systemic": "macro/geopolitical shock",
    "exchange_delisting_risk": "exchange/counterparty failure",
}
NOT_REPORTED_CATEGORIES = {"exchange_operational", "none"}

# Keyword sets used only to disambiguate project_specific -- checked in
# this order, first match wins. Matched against title+text+reasoning
# combined, case-insensitive.
PROJECT_SPECIFIC_SUBCATEGORY_KEYWORDS = [
    ("stablecoin/peg failure", re.compile(
        r"depeg|de-peg|\bpeg\b|stablecoin", re.IGNORECASE)),
    ("exchange/counterparty failure", re.compile(
        r"insolven|bankrupt|exit scam|\bfreeze\b|\bfroze\b|\bseiz|collapse|"
        r"default|shut down|halts? network|halted network|"
        # Added 2026-08-25 after historical_stress_test_classify.py's real
        # Groq run correctly categorized Celsius (project_specific) but the
        # subcategory regex missed it -- "pauses withdrawals"/"trapped ...
        # assets" is exactly this failure mode's real-world phrasing and
        # wasn't covered by the freeze/insolvent/collapse wordlist above.
        r"paus(e|es|ed|ing) (withdrawals|redemptions)|withdrawal (pause|suspension|halt)|"
        r"trapped .{0,20}(assets|funds)|seeks? creditor protection",
        re.IGNORECASE)),
    ("hack/exploit", re.compile(
        r"hack|exploit|breach|vulnerab|compromise|\battack|\bdrain|stolen",
        re.IGNORECASE)),
]
PROJECT_SPECIFIC_DEFAULT = "hack/exploit"


def map_category(item):
    """Returns the spec's category string, or None if this item shouldn't
    be reported at all (routine ops / no real category)."""
    cat = item.get("category", "none")
    if cat in NOT_REPORTED_CATEGORIES:
        return None
    if cat in DIRECT_CATEGORY_MAP:
        return DIRECT_CATEGORY_MAP[cat]
    if cat == "project_specific":
        haystack = f"{item.get('title', '')} {item.get('text', '')} {item.get('reasoning', '')}"
        for spec_cat, pattern in PROJECT_SPECIFIC_SUBCATEGORY_KEYWORDS:
            if pattern.search(haystack):
                return spec_cat
        return PROJECT_SPECIFIC_DEFAULT
    return None  # unknown category -- don't report rather than guess


# --- confidence tier mapping -------------------------------------------

def map_confidence_tier(item):
    if item.get("source_type") == "first_party" and item.get("confirmed"):
        return "CONFIRMED"
    if item.get("source_type") == "third_party" and item.get("final_confirmed"):
        return "LIKELY"
    return "UNCONFIRMED"


# --- live vs historical (Task #7, added 2026-08-25) --------------------
#
# Mechanical, zero-cost, no LLM involved -- the distinction is already
# present in the data itself. Every item that came through the real live
# pipeline (unified_ingest.py -> layer1_filter.py -> dedup_filter.py ->
# layer2_classifier.py/layer2_route.py) has "provider" set to whichever
# model actually classified it (e.g. "groq", "gemini"). Items from
# historical_stress_test.py are explicitly tagged
# provider="hand_labeled_historical" (see that file's docstring) since
# they were never scored by a live classifier at all. That tag is the
# single source of truth here -- nothing to infer or guess.

HISTORICAL_PROVIDER_TAG = "hand_labeled_historical"

# Checked FIRST, ahead of the provider tag: a stress-test item run through
# the REAL Groq/Gemini classifier (historical_stress_test_classify.py, added
# 2026-08-25 for Task #8) still carries a real provider value ("groq",
# "gemini_second_opinion") so it's auditable which engine actually scored
# it -- but it represents a 2014-2023 event, not something the live feed
# just saw, so it needs its own explicit historical marker rather than
# being inferred from provider alone.
STRESS_TEST_PIPELINE_TAG = "historical_stress_test"


def determine_live_or_historical(item):
    if item.get("_source_pipeline") == STRESS_TEST_PIPELINE_TAG:
        return "historical"
    if item.get("provider") == HISTORICAL_PROVIDER_TAG:
        return "historical"
    return "live"


# --- posture table (informational only -- does not fire anything;
# the live execution bridge that would act on this doesn't exist yet,
# per the spec's own "Where reports go" section) -----------------------

POSTURE_TABLE = {
    "exchange/counterparty failure": "Full defensive: flatten or pause new entries in affected symbols",
    "stablecoin/peg failure": "Full defensive: flatten or pause new entries market-wide",
    "regulatory shock": "Reduce size, pause new entries in affected symbols",
    "macro/geopolitical shock": "Reduce size market-wide, no forced exits",
    "hack/exploit": "Pause new entries in affected symbol(s) only",
}
UNCONFIRMED_POSTURE = "No action -- log only, do not fire"


def recommended_posture(spec_category, tier):
    if tier == "UNCONFIRMED":
        return UNCONFIRMED_POSTURE
    return POSTURE_TABLE.get(spec_category, UNCONFIRMED_POSTURE)


# --- event ID: cross-time matching against a persistent registry --------
#
# Mechanical, no LLM cost -- deliberately not semantic understanding, so
# it WILL miss a follow-up worded very differently from the original, and
# COULD over-merge two coincidentally similar but unrelated stories that
# happen to share entity words and category within the window. Every
# report says plainly whether its event_id was matched to an existing
# open event or newly created (see matched_existing_event below), and
# event_registry.json is a flat, readable file worth spot-checking for
# surprising merges before trusting this blindly.
#
# Matching rule: same mapped spec_category, AND (shares at least one
# watchlist symbol from symbols_affected, OR Jaccard word-overlap of the
# title's significant words >= EVENT_MATCH_JACCARD_THRESHOLD), AND the
# candidate event's last-seen timestamp is within EVENT_MATCH_WINDOW_DAYS
# of this item's timestamp. First matching open event wins; on a match,
# the registry's last_seen is bumped to this item's timestamp so a slow
# trickle of follow-ups keeps the window rolling forward instead of being
# anchored to the very first mention.

_STOPWORDS = {
    "a", "an", "the", "is", "in", "on", "at", "to", "of", "for", "and",
    "or", "just", "in:", "breaking:", "update:", "says", "say", "said",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def entity_words(title):
    return {w for w in _WORD_RE.findall((title or "").lower()) if w not in _STOPWORDS}


def generate_new_event_id(item, spec_category, words):
    slug = "_".join(list(words)[:4]) or "event"
    ts = item.get("timestamp_utc", "")
    date_part = ts[:10].replace("-", "") if ts else "unknown"
    cat_prefix = spec_category.split("/")[0].split(" ")[0]
    return f"{cat_prefix}_{slug}_{date_part}"


def load_registry():
    if not os.path.exists(EVENT_REGISTRY_FILE):
        return {}
    with open(EVENT_REGISTRY_FILE) as f:
        return json.load(f)


def save_registry(registry):
    with open(EVENT_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def _parse_ts(ts):
    try:
        return datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
    except Exception:
        return None


def resolve_event_id(item, spec_category, symbols_affected, registry):
    """Returns (event_id, matched_existing: bool). Mutates registry
    in-memory -- caller is responsible for save_registry() once per run,
    not once per item."""
    words = entity_words(item.get("title", ""))
    this_ts = _parse_ts(item.get("timestamp_utc"))
    watchlist_syms = set(symbols_affected) if isinstance(symbols_affected, list) else set()

    for event_id, entry in registry.items():
        if entry.get("spec_category") != spec_category:
            continue
        last_ts = _parse_ts(entry.get("last_seen_timestamp"))
        if this_ts is not None and last_ts is not None:
            if abs((this_ts - last_ts).days) > EVENT_MATCH_WINDOW_DAYS:
                continue
        entry_syms = set(entry.get("symbols_affected", []))
        shares_symbol = bool(watchlist_syms & entry_syms)
        entry_words = set(entry.get("entity_words", []))
        union = words | entry_words
        jaccard = len(words & entry_words) / len(union) if union else 0.0
        if shares_symbol or jaccard >= EVENT_MATCH_JACCARD_THRESHOLD:
            entry["last_seen_timestamp"] = item.get("timestamp_utc")
            entry["entity_words"] = list(set(entry_words) | words)
            entry["symbols_affected"] = list(entry_syms | watchlist_syms)
            entry.setdefault("follow_up_titles", []).append(item.get("title", "")[:100])
            # Single-trigger discipline: keep whichever source has the
            # EARLIEST actual timestamp, not whichever arrived first in
            # ingestion order -- a slower outlet can report an earlier-
            # dated event after a faster outlet already broke it.
            existing_trigger_ts = _parse_ts(entry.get("trigger_timestamp"))
            if this_ts is not None and (existing_trigger_ts is None or this_ts < existing_trigger_ts):
                entry["trigger_timestamp"] = item.get("timestamp_utc")
                entry["trigger_source"] = item.get("url")
                entry["trigger_title"] = item.get("title")
            return event_id, True

    event_id = generate_new_event_id(item, spec_category, words)
    registry[event_id] = {
        "spec_category": spec_category,
        "first_seen_timestamp": item.get("timestamp_utc"),
        "last_seen_timestamp": item.get("timestamp_utc"),
        "trigger_timestamp": item.get("timestamp_utc"),
        "trigger_source": item.get("url"),
        "trigger_title": item.get("title"),
        "entity_words": list(words),
        "symbols_affected": list(watchlist_syms),
        "follow_up_titles": [],
    }
    return event_id, False


# --- main ----------------------------------------------------------------

def build_report(item, registry):
    spec_category = map_category(item)
    if spec_category is None:
        return None

    tier = map_confidence_tier(item)
    providers = [item.get("provider")]
    if item.get("second_opinion"):
        providers.append(item["second_opinion"].get("provider"))

    symbols_affected = item.get("symbols_affected", [])
    event_id, matched_existing = resolve_event_id(item, spec_category, symbols_affected, registry)

    # Single-trigger discipline: every report under this event_id points
    # at the SAME canonical earliest-known confirming source (tracked in
    # the registry), not this individual item's own arrival time. This
    # item's own timestamp/url are still preserved below in evidence, as
    # the citation for this specific report.
    entry = registry[event_id]
    canonical_ts = entry.get("trigger_timestamp")
    canonical_source = entry.get("trigger_source")
    canonical_title = entry.get("trigger_title")
    is_canonical_source = item.get("timestamp_utc") == canonical_ts and item.get("url") == canonical_source

    return {
        "event_id": event_id,
        "matched_existing_event": matched_existing,
        "category": spec_category,
        "trigger_timestamp": canonical_ts,
        "trigger_source": canonical_source,
        "trigger_source_title": canonical_title,
        "is_canonical_source": is_canonical_source,
        "confidence_tier": tier,
        "recommended_posture": recommended_posture(spec_category, tier),
        "lead_time_note": "not backtested -- price-drawdown computation belongs to the price/Model D leg, not this one",
        "live_or_historical": determine_live_or_historical(item),
        "symbols_affected": symbols_affected,
        "evidence": {
            "title": item.get("title"),
            "source": item.get("source"),
            "source_type": item.get("source_type"),
            "url": item.get("url"),
            "reasoning": item.get("reasoning"),
            "providers_checked": providers,
            "providers_agree": item.get("providers_agree"),
            "internal_category": item.get("category"),
            "internal_confidence_score": item.get("confidence_score"),
        },
        "_event_id_caveat": (
            f"matched mechanically -- same category + (shared watchlist symbol OR "
            f">= {EVENT_MATCH_JACCARD_THRESHOLD} title-word overlap) within "
            f"{EVENT_MATCH_WINDOW_DAYS} days of the event's last-seen mention. Not a "
            f"semantic/LLM judgment -- can still miss a follow-up worded very "
            f"differently, or over-merge unrelated stories that share entity words. "
            f"Check event_registry.json if this ID's history looks surprising."
        ),
    }


def main():
    with open(INPUT_FILE) as f:
        items = json.load(f)

    registry = load_registry()

    reports = []
    skipped_no_category = 0
    for item in items:
        report = build_report(item, registry)
        if report is None:
            skipped_no_category += 1
            continue
        reports.append(report)

    save_registry(registry)

    reports.sort(key=lambda r: {"CONFIRMED": 0, "LIKELY": 1, "UNCONFIRMED": 2}[r["confidence_tier"]])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(reports, f, indent=2)

    tier_counts = {"CONFIRMED": 0, "LIKELY": 0, "UNCONFIRMED": 0}
    for r in reports:
        tier_counts[r["confidence_tier"]] += 1
    matched_count = sum(1 for r in reports if r["matched_existing_event"])

    print(f"spec_report.py -- {len(items)} routed items in, {len(reports)} reportable events out "
          f"({skipped_no_category} skipped: routine ops / no real category)\n")
    print(f"Tiers: {tier_counts['CONFIRMED']} CONFIRMED, {tier_counts['LIKELY']} LIKELY, "
          f"{tier_counts['UNCONFIRMED']} UNCONFIRMED -- saved to {OUTPUT_FILE}")
    print(f"Event matching: {matched_count} matched an existing open event, "
          f"{len(reports) - matched_count} started a new event_id -- registry saved to "
          f"{EVENT_REGISTRY_FILE} ({len(registry)} open events tracked)\n")

    for r in reports:
        print(f"  [{r['confidence_tier']:11}] ({r['category']}) {r['evidence']['title'][:70]}")
        print(f"      posture: {r['recommended_posture']}")

    print(f"\nNote: live_or_historical is hardcoded \"live\" (this pipeline has never ingested "
          f"historical news). event_id now matches against open events mechanically (category + "
          f"symbol/word overlap within {EVENT_MATCH_WINDOW_DAYS} days) -- not semantic, so spot-check "
          f"event_registry.json before trusting a merge or a split.")
    print("\nDONE. Paste this full output back into chat.")


if __name__ == "__main__":
    main()
