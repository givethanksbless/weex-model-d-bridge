"""
layer2_classifier.py

Layer 2 of the ingest -> classify pipeline: LLM-based classification using
the confidence-score schema locked in task #26. Runs only on items that
survived layer1_filter.py AND dedup_filter.py -- reads dedup_passed.json
(deduped output, not the raw layer1_passed.json -- wired up 2026-08-24 so
duplicate cross-outlet copies of the same story don't get billed to the
LLM three times over), calls Gemini for each item, computes
confidence_score and confirmed mechanically per the locked formula, saves
the full classified output.

Pipeline order: unified_ingest.py -> layer1_filter.py -> dedup_filter.py
-> layer2_classifier.py / layer2_route.py.

Schema (see news-ingest-classify-design-notes.md, 2026-08-23):
  category           -- LLM-judged: one of 5 categories or "none"
  specificity        -- LLM-judged: vague / moderate / specific
  reasoning          -- LLM-judged: 1-2 sentence audit trail
  authority_weight   -- mechanical: 1.0 first-party, 0.6 third-party
  specificity_score  -- mechanical: vague=0.2, moderate=0.5, specific=0.9
  confidence_score   -- mechanical: 0 if category==none, else
                         specificity_score * authority_weight
  confirmed          -- mechanical: confidence_score >= 0.5 ("worth surfacing
                         for human review" only -- routing/action stays out of
                         scope for this chat)

PROVIDER HISTORY (all on 2026-08-23):
  Started on Groq (free, no card) with openai/gpt-oss-20b, two rounds of
  rubric fixes based on real misclassifications (see prior versions of
  this file / news-ingest-classify-design-notes.md for the full trail),
  then hit Groq's real per-model daily token cap (200,000 TPD) mid-run.
  Switched provider entirely to Gemini (google-genai SDK, Google AI
  Studio) since its free tier has no daily token cap, just 1,500
  requests/day and 1,000,000 tokens/minute -- a much better fit for a
  token-heavy, low-request-count workload like this one. Model:
  gemini-2.5-flash.

  2026-08-23, key-format fix: Google migrated Gemini API keys from the
  old "Standard" format (AIza...) to a new "Auth" format (AQ.Ab...) in
  2026 -- AI Studio now issues AQ. keys by default, and old AIza keys
  are being fully phased out (rejected starting Sept 2026 per Google's
  docs). The key-validation prompt below was written against the old
  format and wrongly rejected valid AQ. keys -- fixed to accept both.

Usage
-----
    python3 layer2_classifier.py
"""

import json
import os
import re
import time

from google import genai
from google.genai import types

INPUT_FILE = "dedup_passed.json"
OUTPUT_FILE = "classified_items.json"
CONFIG_FILE = "gemini_config.json"

CONFIRMED_THRESHOLD = 0.5

# Bump this whenever CATEGORY_RUBRIC OR the provider/model changes
# meaningfully. Cached results stamped with an older version are treated
# as stale and re-classified instead of silently reused.
RUBRIC_VERSION = 3

MODEL = "gemini-3.5-flash"

AUTHORITY_WEIGHTS = {"first_party": 1.0, "third_party": 0.6}
SPECIFICITY_SCORES = {"vague": 0.2, "moderate": 0.5, "specific": 0.9}

# Real per-key rate limit, confirmed via a live 429 payload 2026-08-24:
# gemini-3.5-flash's free tier allows only 5 requests/minute per
# project/key (GenerateRequestsPerMinutePerProjectPerModel-FreeTier) --
# much tighter than the 2.5-flash numbers this file's docstring above
# still references from before the model was swapped to 3.5. The old
# 0.2s courtesy delay was never checked against this model's real limit.
GEMINI_RPM_LIMIT = 5
GEMINI_RPM_SAFETY_MARGIN = 4  # pace against 4/min per key, not right up against 5
MAX_GEMINI_WAIT_S = 60  # same discipline as the Groq fix: past this, stop and
                         # save instead of retrying into a wall


class GeminiCooldownRequired(Exception):
    """Raised instead of retrying when Gemini's real retryDelay exceeds
    MAX_GEMINI_WAIT_S. Callers should stop the run cleanly on this --
    progress already saved so far is preserved, nothing lost."""
    def __init__(self, wait_s):
        self.wait_s = wait_s
        super().__init__(f"Gemini asked for a {wait_s:.0f}s wait -- over the "
                          f"{MAX_GEMINI_WAIT_S}s threshold, stopping instead of retrying.")


_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


def parse_retry_delay(e):
    """Pulls the real retryDelay Gemini reports on a 429 (e.g. "'retryDelay':
    '54s'") instead of guessing -- same idea as honoring Groq's Retry-After
    header. Returns None if the error doesn't carry one."""
    m = _RETRY_DELAY_RE.search(str(e))
    return float(m.group(1)) if m else None

CATEGORY_RUBRIC = """You are classifying a single crypto news/announcement item for a risk-monitoring
system. Categories:

- exchange_delisting_risk: a specific exchange delisting a symbol, halting trading,
  or suspending withdrawals for a specific asset.
- regulatory_action: a government body, regulator, or legislature taking or
  considering action affecting crypto -- lawsuits, investigations, bills, hearings,
  rulings, sanctions. The ACTOR must actually be a government body, regulator, or
  legislature -- a private company, exchange, or asset manager announcing its own
  business plans (even compliance-adjacent ones, like "integrating tokenized
  assets") is NOT regulatory_action, no matter how official-sounding the language.
  IMPORTANT: a person's OPINION, ENDORSEMENT, or STATEMENT about what should happen
  ("X says Congress should pass...", "CEO says the bill will prevent...", a
  "Highlight Clip" of someone's take on legislation or the regulatory climate in
  general) is NOT regulatory_action, even when the speaker is an official -- that
  belongs under "none". But when an official body is actually DOING something
  concrete right now -- directing staff to draft rules, opening an investigation,
  issuing a ruling, holding a hearing -- that IS regulatory_action even if it's
  reported as a quote. Don't let caution about "opinion vs action" cause you to
  miss real action just because it's phrased as someone saying/announcing it.
- project_specific: a hack, exploit, governance failure, insolvency, exit scam, or
  other fundamental negative event specific to one project/protocol. If the text
  contains hack/exploit/breach/freeze/suspend language tied to a named project or
  token, treat that as strong evidence for this category -- do not default to
  "none" without a clear reason to override that signal.
- macro_systemic: a market-wide shock not specific to one project/exchange --
  tariffs, rate changes, broad liquidation cascades with a named macro cause.
- exchange_operational: an exchange changing trading-relevant infrastructure or
  policy -- maintenance windows, API deprecations, fee schedule changes, minimum
  order size changes, listing/delisting POLICY changes. Does NOT include
  promotional marketing (staking APY campaigns, airdrop promotions, token launch
  marketing) -- that belongs under "none".
- none: doesn't fit any category above -- price commentary, trivia, ETF flow
  stats, celebrity opinions, generic market chatter, promotional/marketing
  content (staking campaigns, airdrop announcements, "launch" marketing), and
  opinions/endorsements about legislation that hasn't actually moved.
  IMPORTANT, two more exclusions caught in real misclassifications:
  (1) Weekly/periodic digest or roundup posts that bundle multiple separate
  stories into one post ("Asia's weekly TOP10...", "X Weekly", a
  "subscribe for the full report" teaser) are "none" even if the bundled
  items sound individually regulatory/event-like -- the roundup itself is
  a newsletter teaser, not a single verified action. Each underlying story
  would need to be reported on its own to be classified as anything else.
  (2) News about a company or entity that is NOT crypto/blockchain-related
  (general tech, semiconductor, general corporate, general geopolitical)
  is "none" even when it's reported by a crypto news outlet and even when
  it uses charged words like "probe" or "diversion" -- a dispute between
  two non-crypto companies picked up by a crypto feed has no crypto-
  specific risk angle just because of where it was published.

Also judge specificity:
- specific: names an asset, date, dollar figure, or named entity taking a
  concrete, verifiable action.
- moderate: has some concrete detail but is partial or not fully verifiable.
- vague: opinion, speculation, or a bare claim with no named cause or entity.

Examples of items this system previously misclassified -- use these as the
exact boundary to hold:

Item: "WEEX is about to Launch LINK Staking!"
Correct: {"category": "none", "specificity": "specific", "reasoning": "Promotional staking campaign -- explicitly excluded from exchange_operational regardless of how specific the wording is."}

Item: "JUST IN: Coinbase CEO Brian Armstrong says the crypto Clarity Act will prevent another FTX collapse"
Correct: {"category": "none", "specificity": "vague", "reasoning": "An executive's opinion/endorsement about a bill's expected effect -- no regulatory body has actually acted."}

Item: "Over 500M SAND Minted in Suspected The Sandbox Security Breach; Upbit Issues Warning, Bithumb Suspends Deposits"
Correct: {"category": "project_specific", "specificity": "specific", "reasoning": "Named project (Sandbox/SAND) suffered a suspected security breach, with exchanges responding by issuing warnings and suspending deposits."}

Item: "JUST IN: CFTC Chair Selig directs staff to create formal crypto market structure rules"
Correct: {"category": "regulatory_action", "specificity": "specific", "reasoning": "The CFTC chair is actively directing an actual regulatory body to take a concrete step -- this is real regulatory action in progress, not an opinion about a bill, even though it's reported as a quote."}

Item: "Franklin Templeton Plans to Integrate Tokenized Assets Into Traditional Funds"
Correct: {"category": "none", "specificity": "moderate", "reasoning": "Franklin Templeton is a private asset manager announcing its own business plan -- not a government body, so this cannot be regulatory_action regardless of how compliance-adjacent it sounds."}

Item: "Highlight Clip: U.S. SEC Chairman: The U.S. Is Entering a Historic Era of Crypto Regulatory Reform"
Correct: {"category": "none", "specificity": "vague", "reasoning": "A framing statement about the overall regulatory climate, not a specific concrete action the SEC took."}

Item: "Asia's weekly TOP10 crypto news\n\nSouth Korea Plans a Joint Virtual-Asset Crime Investigation Unit, KRX Prepares to Launch a New Securities Market, Nomura-Backed Laser Digital Secures Japan's First New Crypto Registration in Four Years, Pakistan Launches Its Virtual-Asset Licensing Regime. For the complete article and weekly curated reports, subscribe to our Substack..."
Correct: {"category": "none", "specificity": "vague", "reasoning": "A weekly roundup bundling four separate stories with a subscribe teaser -- not a single verified action, regardless of how regulatory-sounding the bundled items are."}

Item: "UPDATE: Super Micro says its probe found no evidence management knew of the alleged $2.5B Nvidia diversion to China"
Correct: {"category": "none", "specificity": "vague", "reasoning": "Super Micro and Nvidia are semiconductor/tech companies, not crypto projects or exchanges -- general corporate news with no crypto-specific angle, even though a crypto outlet reported it."}

Respond with ONLY a JSON object, no markdown fences, no extra text:
{"category": "...", "specificity": "...", "reasoning": "one to two sentences"}
"""


def load_or_prompt_api_keys():
    """Returns a list of Gemini API keys to rotate through. Each key should
    come from its own Google Cloud/AI Studio project (same account is
    fine) -- that's what gives each one an independent daily quota.
    Stored under "api_keys" in CONFIG_FILE; transparently migrates the old
    single-"api_key" schema. GEMINI_API_KEY/GOOGLE_API_KEY env var, if
    set, short-circuits everything and is used alone (old behavior)."""
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return [env_key]

    VALID_PREFIXES = ("AQ.", "AIza")
    keys = []

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
        if "api_keys" in saved:
            keys = [k for k in saved["api_keys"] if k.startswith(VALID_PREFIXES)]
        elif "api_key" in saved and saved["api_key"].startswith(VALID_PREFIXES):
            keys = [saved["api_key"]]  # migrate old single-key format

    print(f"Found {len(keys)} saved Gemini key(s) in {CONFIG_FILE}." if keys
          else "No saved Gemini keys found.")
    print("Get a free key from aistudio.google.com -> Get API key -> Create API key "
          "in a NEW project each time (same account is fine) -- that's what gives "
          "each key its own independent daily quota. No card needed.\n")

    while True:
        prompt = (f"Paste another Gemini API key to add, or press Enter to continue "
                  f"with the {len(keys)} key(s) you have: " if keys else
                  "Paste your Gemini API key, then press Enter: ")
        key = input(prompt).strip()
        if not key:
            if keys:
                break
            print("Need at least one key to continue.\n")
            continue
        if key.startswith(VALID_PREFIXES) and len(key) > 20:
            if key in keys:
                print("Already have that one.\n")
            else:
                keys.append(key)
                print(f"Added. {len(keys)} key(s) so far.\n")
        else:
            print(f"That doesn't look like a valid key (got {len(key)} chars, "
                  f"expected it to start with 'AQ.' or 'AIza'). Try again.\n")

    with open(CONFIG_FILE, "w") as f:
        json.dump({"api_keys": keys}, f)
    print(f"\nSaved {len(keys)} key(s) to {CONFIG_FILE} -- won't ask again "
          f"unless you want to add more.\n")
    return keys


def is_daily_quota_error(e):
    """A per-day quota cap (GenerateRequestsPerDayPerProjectPerModel) won't
    clear on retry within the same run -- it only resets at the daily
    boundary. Distinct from is_transient_error: those (429 per-minute,
    503 overload) are worth a short backoff; this is not."""
    msg = str(e).lower()
    return "generaterequestsperdayperprojectpermodel" in msg or "perday" in msg


def is_daily_quota_error(e):
    """A per-day quota cap (GenerateRequestsPerDayPerProjectPerModel) won't
    clear on retry within the same run -- it only resets at the daily
    boundary. Distinct from is_transient_error: those (429 per-minute,
    503 overload) are worth a short backoff; this is not."""
    msg = str(e).lower()
    return "generaterequestsperdayperprojectpermodel" in msg or "perday" in msg


def is_transient_error(e):
    """Errors worth retrying within the same run -- rate limits (429) and
    temporary server-side overload (503), both of which clear on their own
    within seconds to minutes. Anything else (bad request, auth failure,
    etc.) is NOT retried -- retrying won't fix those."""
    msg = str(e).lower()
    return (
        "429" in msg or "resource_exhausted" in msg or "rate limit" in msg
        or "quota" in msg or "503" in msg or "unavailable" in msg
        or "high demand" in msg
    )


def classify_item(client, item, max_retries=2):
    user_content = f"Title: {item['title']}\n\nFull text: {item['text']}"
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=CATEGORY_RUBRIC,
                    temperature=0,  # deterministic -- reduce run-to-run
                                    # flip-flopping on boundary cases
                    # Gemini 3.x models always "think" to some degree --
                    # thinking_budget (the old numeric control) doesn't
                    # apply to 3.x, and even at the lowest thinking_level
                    # the model still emits a thought_signature that eats
                    # into the token budget. First run at max_output_tokens
                    # =300 with no thinking_level set (defaults to medium)
                    # burned the whole budget on thinking and left nothing
                    # for the actual JSON answer -- resp.text came back
                    # empty, silently misread as a real "none" classification
                    # every time. Fixed with thinking_level="low" (fastest,
                    # for high-throughput simple tasks -- this is exactly
                    # that) plus a much larger token ceiling as a safety
                    # margin.
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )
            raw = (resp.text or "").strip()
            # Strip markdown fences if the model adds them despite
            # response_mime_type -- belt and suspenders.
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {
                    "category": "none",
                    "specificity": "vague",
                    "reasoning": f"PARSE ERROR, raw response: {raw[:200]}",
                }
        except Exception as e:
            last_err = e
            if is_daily_quota_error(e):
                raise
            if is_transient_error(e) and attempt < max_retries:
                wait_s = parse_retry_delay(e)
                if wait_s is None:
                    wait_s = 20 * (attempt + 1)
                else:
                    wait_s += 1  # small safety margin, same as the Groq fix
                if wait_s > MAX_GEMINI_WAIT_S:
                    raise GeminiCooldownRequired(wait_s)
                print(f"    transient error, retrying in {wait_s:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_s)
                continue
            raise last_err
    raise last_err


# Watchlist symbols this pipeline cares about for downstream routing --
# mirrors delisting_check_poc.py's WATCHLIST_LIVE + WATCHLIST_CANDIDATE
# (kept as a plain copy here, not imported, since that file also carries
# network-calling functions this pipeline has no reason to pull in).
# Update both places together if the watchlist ever changes.
# Each ticker maps to a short list of unambiguous full-name aliases,
# matched case-insensitively; bare tickers themselves are matched
# case-sensitive/uppercase-only (word boundary) so common English words
# ("near", "op") don't false-positive against lowercase prose.
WATCHLIST = {
    "SOL": ["Solana"],
    "XRP": ["Ripple"],
    "DOGE": ["Dogecoin"],
    "ADA": ["Cardano"],
    "NEAR": [],  # ticker only -- "near" the word is too collision-prone to match case-insensitively
    "FIL": ["Filecoin"],
    "OP": ["Optimism"],
    "INJ": ["Injective"],
    "SUI": ["Sui"],
}

_TICKER_WORD_RE = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in WATCHLIST}
_ALIAS_RE = {
    t: [re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE) for a in aliases]
    for t, aliases in WATCHLIST.items()
}

# Broad market-mover language -- when no watchlist ticker is named
# specifically but the item is clearly about the whole market rather than
# one coin, it's still relevant to log, just not to one symbol.
_SYSTEMWIDE_HINTS = re.compile(
    r"\bcrypto market\b|\bmarket cap\b|\btariff|\bfed\b|\binterest rate|"
    r"\brecession\b|\bregulat",
    re.IGNORECASE,
)


def extract_symbols_affected(item, category):
    """Mechanical (no LLM cost) extraction of which watchlist symbols a
    news item concerns, added 2026-08-24 so downstream routing can key off
    actual coin-level relevance instead of a bare `relevant` bool. Matches
    bare uppercase ticker codes and a small set of unambiguous full names
    against the item's raw title+text (not the normalized/stripped text
    dedup_filter.py uses, since case matters here).

    Returns: a list of matched watchlist tickers (e.g. ["SOL"]); the
    string "systemwide" if the item reads as broad-market/macro with no
    specific watchlist coin named; or [] if it's specific to something
    NOT on the watchlist (e.g. a MANTRA or Sandbox exploit)."""
    haystack = f"{item.get('title', '')} {item.get('text', '')}"

    matched = [
        t for t in WATCHLIST
        if _TICKER_WORD_RE[t].search(haystack)
        or any(rx.search(haystack) for rx in _ALIAS_RE[t])
    ]
    if matched:
        return matched

    if category == "macro_systemic" or _SYSTEMWIDE_HINTS.search(haystack):
        return "systemwide"

    return []


# Mechanical, zero-cost correction for a pattern class that's reliably
# identifiable by shape alone -- added 2026-08-24 after "Asia's weekly
# TOP10 crypto news" (a bundled digest, not a single event) got tagged
# regulatory_action by the LLM. Deliberately applied as a POST-hoc
# override rather than relying only on the CATEGORY_RUBRIC prompt text,
# so it (a) fixes items ALREADY sitting in the cache without spending any
# Groq/Gemini calls to re-classify them, and (b) doesn't depend on the
# LLM reliably following the added rubric guidance every time. The
# rubric text addition still helps too, for genuinely new items -- this
# is belt-and-suspenders, not a replacement for it.
_DIGEST_PATTERN = re.compile(
    r"weekly\s+top\s*\d+|\bweekly\s+(digest|roundup|recap)\b|"
    r"subscribe to (our|the) (substack|newsletter)",
    re.IGNORECASE,
)


def apply_mechanical_category_override(item, category, specificity, reasoning):
    """Returns (category, specificity, reasoning), overridden if a known,
    reliably-mechanical pattern fires. Only ever downgrades TOWARD "none"
    -- never invents a more severe category the LLM didn't already see."""
    if category == "none":
        return category, specificity, reasoning
    haystack = f"{item.get('title', '')} {item.get('text', '')}"
    if _DIGEST_PATTERN.search(haystack):
        return (
            "none",
            "vague",
            f"MECHANICAL OVERRIDE: weekly digest/roundup pattern detected -- "
            f"bundled newsletter content, not a single verified action "
            f"(original LLM call was: {category}).",
        )
    return category, specificity, reasoning


def score_item(item, llm_result):
    category = llm_result.get("category", "none")
    specificity = llm_result.get("specificity", "vague")
    reasoning = llm_result.get("reasoning", "")
    category, specificity, reasoning = apply_mechanical_category_override(
        item, category, specificity, reasoning
    )

    authority_weight = AUTHORITY_WEIGHTS.get(item["source_type"], 0.6)
    specificity_score = SPECIFICITY_SCORES.get(specificity, 0.2)
    confidence_score = (
        0.0 if category == "none" else round(specificity_score * authority_weight, 3)
    )
    confirmed = confidence_score >= CONFIRMED_THRESHOLD
    symbols_affected = extract_symbols_affected(item, category)

    return dict(
        item,
        category=category,
        specificity=specificity,
        reasoning=reasoning,
        authority_weight=authority_weight,
        specificity_score=specificity_score,
        confidence_score=confidence_score,
        confirmed=confirmed,
        symbols_affected=symbols_affected,
        rubric_version=RUBRIC_VERSION,
    )


def reapply_mechanical_corrections(item, scored):
    """Applied to CACHE-HIT items (dict(cached) reuse, no API call) so
    mechanical fixes -- symbols_affected backfill, category overrides --
    reach entries already sitting in the cache without spending a Groq/
    Gemini call to re-classify them. Mutates and returns `scored`. Safe
    for layer2_classifier.py's own flat classified_items.json cache;
    NOT used on layer2_route.py's routed_items.json (nested Groq +
    second_opinion structure, plus final_confirmed's OR-merge logic --
    correcting there needs to touch both sides consistently, done as a
    one-off direct patch instead when a specific bad entry is known)."""
    scored.setdefault(
        "symbols_affected", extract_symbols_affected(item, scored.get("category", "none"))
    )
    new_category, new_specificity, new_reasoning = apply_mechanical_category_override(
        item, scored.get("category", "none"), scored.get("specificity", "vague"), scored.get("reasoning", "")
    )
    if new_category != scored.get("category"):
        authority_weight = scored.get(
            "authority_weight", AUTHORITY_WEIGHTS.get(item.get("source_type"), 0.6)
        )
        specificity_score = SPECIFICITY_SCORES.get(new_specificity, 0.2)
        confidence_score = (
            0.0 if new_category == "none" else round(specificity_score * authority_weight, 3)
        )
        scored["category"] = new_category
        scored["specificity"] = new_specificity
        scored["reasoning"] = new_reasoning
        scored["specificity_score"] = specificity_score
        scored["confidence_score"] = confidence_score
        scored["confirmed"] = confidence_score >= CONFIRMED_THRESHOLD
    return scored


# How close to CONFIRMED_THRESHOLD a confidence_score has to land to count
# as a "close call" worth a second opinion, independent of the specificity
# check below. Confidence_score only ever takes one of a handful of exact
# values (specificity_score {0.2, 0.5, 0.9} x authority_weight {1.0, 0.6}),
# so 0.1 is wide enough to catch specific+third_party (0.54, barely
# CONFIRMED) without also catching vague+first_party (0.2, a clear miss).
BOUNDARY_MARGIN = 0.1


def needs_second_opinion(scored_item):
    """True if a classified item is a boundary/low-confidence case worth
    re-checking with a second model, per the Groq-primary / Gemini-secondary
    routing plan (task #32): Groq classifies the full daily batch, and only
    items flagged here get a second Gemini pass. Realistically ~5-15 items
    out of a ~75-item batch, not the whole thing.

    Flags on either signal:
      - specificity == "moderate": the LLM itself said the evidence was
        partial/not fully verifiable -- the categorical middle ground,
        regardless of which way confidence_score landed.
      - confidence_score within BOUNDARY_MARGIN of CONFIRMED_THRESHOLD:
        catches close calls that specificity alone won't (e.g. a "specific"
        item from a third_party source scores 0.54 -- barely CONFIRMED).

    Does NOT re-check items already errored (is_error_reasoning) -- those
    need a straight re-run on the same provider, not a second opinion from
    a different one.
    """
    if is_error_reasoning(scored_item.get("reasoning", "")):
        return False
    if scored_item.get("specificity") == "moderate":
        return True
    return abs(scored_item.get("confidence_score", 0.0) - CONFIRMED_THRESHOLD) <= BOUNDARY_MARGIN


def is_error_reasoning(reasoning):
    return reasoning.startswith("PARSE ERROR") or reasoning.startswith("API ERROR")


def item_key(item):
    return (item.get("source", ""), item.get("raw_id", ""))


def load_cache():
    """Load previously classified items, keyed by (source, raw_id). Only
    entries that are current-rubric-version and not error fallbacks are
    usable as cache hits -- everything else gets re-classified."""
    if not os.path.exists(OUTPUT_FILE):
        return {}
    with open(OUTPUT_FILE) as f:
        prior = json.load(f)
    cache = {}
    for r in prior:
        if r.get("rubric_version") != RUBRIC_VERSION:
            continue
        if is_error_reasoning(r.get("reasoning", "")):
            continue
        cache[item_key(r)] = r
    return cache


def main():
    api_keys = load_or_prompt_api_keys()
    clients = [genai.Client(api_key=k) for k in api_keys]
    print(f"Using {len(clients)} Gemini key(s) in rotation.\n")

    with open(INPUT_FILE) as f:
        items = json.load(f)

    cache = load_cache()
    print(f"Classifying {len(items)} items that survived layer 1 "
          f"({len(cache)} already have a good cached result -- will be reused, not re-billed)...\n")

    results = []
    reused_count = 0
    called_count = 0
    stopped_early = False
    # Round-robin ALL available keys every call (2026-08-24 fix) instead of
    # sticking to one key until it's fully exhausted. Each key is capped at
    # GEMINI_RPM_LIMIT (5/min) -- spreading calls across N keys means each
    # individual key is only hit once every N calls, letting the overall
    # pace scale with key count instead of bottlenecking on one key's cap.
    available = list(range(len(clients)))
    rr_pos = 0
    for idx, item in enumerate(items, 1):
        key = item_key(item)
        cached = cache.get(key)
        if cached is not None:
            scored = dict(cached)  # reuse -- no API call
            scored = reapply_mechanical_corrections(item, scored)
            reused_count += 1
            tag_suffix = " [cached]"
        else:
            llm_result = None
            last_err = None
            try:
                while available:
                    client_idx = available[rr_pos % len(available)]
                    try:
                        llm_result = classify_item(clients[client_idx], item)
                        rr_pos += 1
                        break
                    except GeminiCooldownRequired:
                        raise
                    except Exception as e:
                        last_err = e
                        if is_daily_quota_error(e):
                            print(f"    key #{client_idx + 1} hit its daily quota -- "
                                  f"removing from rotation ({len(available) - 1} key(s) left)...")
                            available.remove(client_idx)
                            continue  # retry same item on whichever key is next
                        break  # not a daily-quota error -- give up on this item
            except GeminiCooldownRequired as e:
                print(f"\n{'!' * 80}")
                print(f"STOPPING: {e}")
                print(f"Processed {idx - 1}/{len(items)} items this run ({reused_count} from "
                      f"cache). Everything gathered so far is saved to {OUTPUT_FILE} -- "
                      f"re-run later and caching will pick up right here, no lost work.")
                print(f"{'!' * 80}\n")
                stopped_early = True
                break
            if not available:
                print(f"  [{idx}/{len(items)}] all Gemini keys exhausted for today -- stopping.")
                stopped_early = True
                break
            if llm_result is None:
                print(f"  [{idx}/{len(items)}] FAILED: {last_err}")
                llm_result = {
                    "category": "none",
                    "specificity": "vague",
                    "reasoning": f"API ERROR: {last_err}",
                }
            scored = score_item(item, llm_result)
            called_count += 1
            tag_suffix = ""
            # Pace against GEMINI_RPM_SAFETY_MARGIN per key, spread across
            # however many keys are still available -- replaces the old
            # flat 0.2s delay, which was never checked against the real
            # 5 RPM cap and was the direct cause of the 429s seen live.
            time.sleep(60 / (GEMINI_RPM_SAFETY_MARGIN * max(1, len(available))))
        results.append(scored)
        # Save after every item, not just at the end of the loop -- an
        # interrupted run (Ctrl-C, crash) should only lose the one
        # in-flight item, not every real (quota-costing) result gathered
        # so far. Sorted/pretty-printed once more after the loop finishes.
        with open(OUTPUT_FILE, "w") as out_f:
            json.dump(results, out_f, indent=2)
        confirmed_tag = "CONFIRMED" if scored["confirmed"] else ""
        print(
            f"  [{idx}/{len(items)}] {scored['category']:<24} "
            f"{scored['specificity']:<9} conf={scored['confidence_score']:.2f} "
            f"{confirmed_tag:<9} {item['title'][:55]}{tag_suffix}"
        )

    results.sort(key=lambda x: x["confidence_score"], reverse=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    confirmed_count = sum(1 for r in results if r["confirmed"])
    error_count = sum(1 for r in results if is_error_reasoning(r.get("reasoning", "")))
    print(f"\n{'=' * 80}")
    stop_note = " -- STOPPED EARLY, re-run later to finish the rest" if stopped_early else ""
    print(
        f"TOTAL: {len(results)}/{len(items)} items written -- {reused_count} reused from "
        f"cache, {called_count} newly classified ({error_count} of those errored), "
        f"{confirmed_count} CONFIRMED (confidence >= {CONFIRMED_THRESHOLD}) "
        f"-- saved to {OUTPUT_FILE}{stop_note}"
    )
    print(f"{'=' * 80}\n")

    if error_count:
        print(f"NOTE: {error_count} item(s) errored this run (parse failure or a "
              f"persistent API issue after retries) -- their results are NOT real "
              f"classifications. Re-run to retry; only errored/new items get re-billed.\n")

    print("CONFIRMED items, highest confidence first:")
    for r in results:
        if r["confirmed"]:
            print(f"  [{r['confidence_score']:.2f}] ({r['category']}) {r['title'][:80]}")

    print("\nDONE. Paste this full output back into chat.")


if __name__ == "__main__":
    main()
