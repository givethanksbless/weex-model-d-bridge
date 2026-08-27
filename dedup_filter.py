"""
dedup_filter.py

Cross-source duplicate filter, sitting between layer1_filter.py and
layer2_route.py / layer2_classifier.py in the pipeline. Purely mechanical
(no LLM, no Groq, no Gemini) -- catches the same underlying news story
reported by multiple outlets (e.g. the same delisting or exploit posted
near-identically by Watcher Guru, Wu Blockchain, and Cointelegraph within
the same hour) and keeps only one copy before it reaches layer 2.

Why this exists: items are scored independently in layer 2 (no cross-item
awareness), so sending 3 near-identical copies of the same story through
Groq just triples the token cost for the same verdict, three times over --
no added signal, since authority_weight already encodes source trust and a
repeat-count confidence boost would double-count / fight that existing
formula (see news-ingest-classify-design-notes.md, 2026-08-24 discussion).
This is a cost/cleanliness filter only -- it does not change confidence
scoring in any way.

Design discipline, same spirit as layer1_filter.py: biased toward KEEPING,
not dropping. A missed duplicate just costs a bit of extra layer-2 budget;
a wrongly-dropped distinct story is real signal lost with no audit trail
downstream. So the similarity threshold here is deliberately conservative,
and only items within TIME_WINDOW_HOURS of each other are ever compared --
two coincidentally similar-sounding items posted days apart are never
treated as duplicates. Every drop is logged with which item it matched and
the similarity score, so the call is auditable, never silent.

Ordering note: input is expected newest-first (unified_ingest.py sorts
that way and layer1_filter.py preserves order), so within a duplicate
cluster this keeps the newest copy and drops the older repost(s). Does NOT
special-case first_party vs third_party when picking which copy to keep --
in practice a first-party item (e.g. WEEX) is unlikely to closely word-match
a third-party news outlet's coverage of the same event, so this hasn't
come up on real data yet. Worth revisiting if it ever does.

Usage
-----
    python3 dedup_filter.py
"""

import json
import re
from datetime import datetime
from difflib import SequenceMatcher

INPUT_FILE = "layer1_passed.json"
PASSED_FILE = "dedup_passed.json"
DROPPED_FILE = "dedup_dropped.json"

TIME_WINDOW_HOURS = 6
SIMILARITY_THRESHOLD = 0.75  # conservative -- only drop clear same-story matches

_MD_LINK = re.compile(r"\[\*\*(.+?)\*\*\]\([^)]*\)")
_EMOJI = re.compile(
    "["
    "\U0001F1E0-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF"
    "]+",
    flags=re.UNICODE,
)
_PREFIXES = re.compile(
    r"^(just in:|breaking:|update:|urgent:|highlight clip:)\s*", re.IGNORECASE
)

# Numbers (prices, dollar figures, percentages) and parenthetical ticker
# codes, e.g. "(SAD)" or "(RL)". Pulled from the RAW title, before
# normalize() strips punctuation, so "$73,000" is captured whole rather
# than as separate digit groups.
_NUMBERS = re.compile(r"[\d,]+\.?\d*")
_TICKERS = re.compile(r"\(([A-Z]{2,10})\)")


def distinguishing_tokens(raw_title):
    """Numbers and ticker codes found in a title -- the parts template-
    style headlines vary ON, so two items should never be treated as
    duplicates if these differ, no matter how similar the surrounding
    boilerplate text scores. Added 2026-08-24 after real data showed
    '$73,000 Bitcoin' vs '$74,000 Bitcoin' and two different WEEX token
    listings both scoring above SIMILARITY_THRESHOLD on boilerplate alone."""
    numbers = set(_NUMBERS.findall(raw_title))
    tickers = set(_TICKERS.findall(raw_title))
    return numbers, tickers


def normalize(text):
    """Strip markdown link wrapping, emoji, boilerplate prefixes, and
    punctuation/case differences so the same headline from two outlets
    compares cleanly even when their formatting habits differ."""
    text = _MD_LINK.sub(r"\1", text or "")
    text = _EMOJI.sub("", text)
    text = text.strip()
    text = _PREFIXES.sub("", text).strip()
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_ts(ts):
    try:
        return datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
    except Exception:
        return None


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def main():
    with open(INPUT_FILE) as f:
        items = json.load(f)

    normalized = [normalize(it.get("title", "")) for it in items]
    timestamps = [parse_ts(it.get("timestamp_utc", "")) for it in items]
    tokens = [distinguishing_tokens(it.get("title", "")) for it in items]

    kept_idx = []
    dropped = []

    for i, item in enumerate(items):
        match = None
        for j in kept_idx:
            ti, tj = timestamps[i], timestamps[j]
            if ti is not None and tj is not None:
                if abs((ti - tj).total_seconds()) > TIME_WINDOW_HOURS * 3600:
                    continue

            # Guard added 2026-08-24: never call it a duplicate if the
            # two titles carry different numbers or ticker codes, even at
            # high text similarity -- that's exactly what distinguishes
            # "$73,000 Bitcoin" from "$74,000 Bitcoin", or two different
            # WEEX listings sharing the same promo template. Only blocks
            # when BOTH sides actually have tokens to compare and they
            # disagree; blank-vs-blank or one-sided is left to the normal
            # similarity check below.
            nums_i, tickers_i = tokens[i]
            nums_j, tickers_j = tokens[j]
            if nums_i and nums_j and nums_i != nums_j:
                continue
            if tickers_i and tickers_j and tickers_i != tickers_j:
                continue

            score = similarity(normalized[i], normalized[j])
            if score >= SIMILARITY_THRESHOLD:
                match = (j, score)
                break
        if match is not None:
            j, score = match
            dropped.append(dict(
                item,
                dedup_reason=(
                    f"duplicate of [{items[j]['source']}] "
                    f"{items[j]['title'][:60]} (similarity={score:.2f})"
                ),
            ))
        else:
            kept_idx.append(i)

    passed = [items[i] for i in kept_idx]

    with open(PASSED_FILE, "w") as f:
        json.dump(passed, f, indent=2)
    with open(DROPPED_FILE, "w") as f:
        json.dump(dropped, f, indent=2)

    print(f"Dedup filter results -- {len(items)} total items\n")
    print(f"PASSED ({len(passed)}) -- saved to {PASSED_FILE}")
    print(f"\nDROPPED as duplicates ({len(dropped)}) -- saved to {DROPPED_FILE}:")
    for d in dropped:
        print(f"  [{d['source']}] {d['title'][:70]}")
        print(f"      -- {d['dedup_reason']}")

    print("\nDONE. Paste this full output back into chat.")


if __name__ == "__main__":
    main()
