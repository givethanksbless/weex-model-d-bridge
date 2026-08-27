"""
layer1_filter.py

Layer 1 of the ingest -> classify pipeline: a cheap, deterministic pre-filter
that strips obvious junk before anything reaches the LLM classifier (layer 2).
Reads ingested_items.json (produced by unified_ingest.py) -- does not touch
Telegram or WEEX directly, so it can be re-run/tuned without re-fetching.

Design discipline (locked before building, not tuned after the fact): tuned
for HIGH RECALL, not precision. Only rejects patterns confirmed as junk from
real pulled data (see news-ingest-classify-design-notes.md, 2026-08-23
entries). Anything ambiguous defaults to PASS, not reject -- a false
negative here just costs a few cents of LLM call in layer 2; a false
positive (real signal dropped here) has no audit trail downstream. Every
item's verdict (pass or reject) and reason is logged, nothing is silently
discarded.

Junk pattern classes (from real Watcher Guru / Wu Blockchain data):
  1. Bare liquidation/volume stats with no named cause
  2. ETF flow stats
  3. Sentiment/opinion quotes ("Highlight Clip:", "X says...")
  4. Trivia / index stats ("FUN FACT:", Fear & Greed Index)
  5. Individual price/rally stats with no named event

Safety override: if an item contains a strong event keyword (exploit, hack,
delist, suspend, halt, regulat, lawsuit, congress, tariff, etc.) it ALWAYS
passes regardless of matching a junk pattern above -- named-cause events
must never be silently dropped by a blunt pattern match. Overrides target
topics/institutions, not named people -- a person's name gets quoted on
everything, signal and noise alike, so it's the wrong axis to filter on.

Usage
-----
    python3 layer1_filter.py
"""

import json
import re

INPUT_FILE = "ingested_items.json"
PASSED_FILE = "layer1_passed.json"
REJECTED_FILE = "layer1_rejected.json"

# Strong event keywords -- presence of ANY of these forces a PASS, overriding
# any junk pattern match below. Named-cause events must never be silently
# dropped by a blunt pattern match.
EVENT_OVERRIDE_KEYWORDS = [
    "exploit", "hack", "breach", "delist", "de-list", "suspend", "halt",
    "regulat", "investigat", "lawsuit", "sec ", "cftc", r"\bban(?:ned|ning|s)?\b",
    "seiz", "freeze", "insolven", "bankrupt", "depeg", "de-peg", "exit scam",
    "rug pull", "vulnerab", "compromise",
    # legislative-process language (regulatory-action sub-type keyword-only
    # coverage missed, per the CLARITY Act edge case, 2026-08-23)
    r"\bbill\b", "legislation", "congress", "senate", "committee", "hearing",
    "testimony",
    # macro/systemic shock language -- previously zero override coverage
    "tariff", "sanction", "rate hike", "interest rate", "recession",
]

# (pattern, reason) -- case-insensitive. Only fires if no event-override
# keyword is present anywhere in the item.
JUNK_PATTERNS = [
    (r"highlight clip:", "opinion/interview clip"),
    (r"\bfun fact\b", "trivia"),
    (r"fear\s*&\s*greed index", "sentiment index stat"),
    (r"\$[\d,.]+\s*(million|billion|trillion|m|b)?\s*(worth of)?\s*"
     r"(longs?|shorts?)?\s*liquidat", "bare liquidation stat, no named cause"),
    (r"\betfs?\b.*(inflow|volume|record)", "ETF flow stat"),
    (r"\brecords?\b.*\b(trillion|billion|million)\b.*\bvolume\b",
     "exchange volume stat"),
    (r"all-time high.*(burn|volume|value)", "token metric stat"),
    (r"\bup more than \d+%|(\brally\b.*\bup\b.*%)", "price rally stat, no named event"),
]

_JUNK_COMPILED = [(re.compile(p, re.IGNORECASE), reason) for p, reason in JUNK_PATTERNS]
_EVENT_COMPILED = re.compile("|".join(EVENT_OVERRIDE_KEYWORDS), re.IGNORECASE)


def classify_layer1(item):
    """Returns (verdict, reason) where verdict is 'pass' or 'reject'."""
    haystack = f"{item.get('title', '')} {item.get('text', '')}"

    if _EVENT_COMPILED.search(haystack):
        return "pass", "event-keyword override"

    for pattern, reason in _JUNK_COMPILED:
        if pattern.search(haystack):
            return "reject", reason

    return "pass", "no junk pattern matched (default pass)"


def main():
    with open(INPUT_FILE) as f:
        items = json.load(f)

    passed, rejected = [], []
    for item in items:
        verdict, reason = classify_layer1(item)
        item_with_verdict = dict(item, layer1_reason=reason)
        if verdict == "pass":
            passed.append(item_with_verdict)
        else:
            rejected.append(item_with_verdict)

    with open(PASSED_FILE, "w") as f:
        json.dump(passed, f, indent=2)
    with open(REJECTED_FILE, "w") as f:
        json.dump(rejected, f, indent=2)

    print(f"Layer 1 filter results -- {len(items)} total items\n")
    print(f"PASSED ({len(passed)}) -- saved to {PASSED_FILE}:")
    for item in passed:
        print(f"  [{item['source']}] ({item['layer1_reason']}) {item['title'][:90]}")

    print(f"\nREJECTED ({len(rejected)}) -- saved to {REJECTED_FILE}:")
    for item in rejected:
        print(f"  [{item['source']}] ({item['layer1_reason']}) {item['title'][:90]}")

    print("\nDONE. Paste this full output back into chat.")


if __name__ == "__main__":
    main()
