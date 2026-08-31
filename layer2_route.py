"""
layer2_route.py

Task #32: Groq-primary / Gemini-secondary routing.

Stage 1 (primary, full batch): classify every item that survived layer 1
with Groq (openai/gpt-oss-20b, swapping to the sibling openai/gpt-oss-120b
if the active model hits its per-model daily/token cap). Groq's quota
(1,000 RPD / 200,000 TPD per model, confirmed in a prior session) has
plenty of headroom for a ~75-item daily batch, which is why it's primary.

Stage 2 (secondary, boundary items only): re-run ONLY the items flagged by
needs_second_opinion() (layer2_classifier.py -- specificity=="moderate" or
confidence_score within BOUNDARY_MARGIN of the 0.5 CONFIRMED_THRESHOLD)
through Gemini's key-pool classifier. Realistically 5-15 items out of a
75-item batch, not the whole thing -- keeping the ask well inside Gemini's
tiny per-key daily quota (20 requests/day/model/project) so the key pool
isn't needed just to cover routine volume, only real boundary cases.

Both stages share CATEGORY_RUBRIC and score_item() from layer2_classifier.py,
so a Groq result and a Gemini result for the same item are directly
comparable -- same rubric, same mechanical scoring formula, only the LLM
judgment call (category/specificity) can differ between providers.

Merge policy when Groq and Gemini disagree on confirmed status: OR, not
AND. This is a "worth surfacing for human review" queue, not an action
queue (routing/action is explicitly out of scope for this chat) -- so a
false negative (missing a real boundary case) costs more than a false
positive (one extra item for a human to glance at and dismiss). See
providers_agree / final_confirmed on each routed item.

STATUS (2026-08-23): scaffold only, written from the same rubric/scoring
code already proven out on Gemini -- NOT yet run against live Groq.
classify_item_groq() has not been validated end-to-end against real data:
its JSON-parsing robustness, its rubric fit on Groq's models, and
is_groq_model_exhausted_error()'s message-matching are all unconfirmed.
Run this against a small batch and check real output before trusting it
for the daily pipeline. See news-ingest-classify-design-notes.md.

Usage
-----
    python3 layer2_route.py
"""

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from google import genai

from layer2_classifier import (
    CATEGORY_RUBRIC,
    GEMINI_RPM_SAFETY_MARGIN,
    GeminiCooldownRequired,
    INPUT_FILE,
    RUBRIC_VERSION,
    extract_symbols_affected,
    is_daily_quota_error,
    is_error_reasoning,
    is_transient_error,
    item_key,
    load_or_prompt_api_keys,
    needs_second_opinion,
    score_item,
)
from layer2_classifier import classify_item as classify_item_gemini

OUTPUT_FILE = "routed_items.json"
GROQ_CONFIG_FILE = "groq_config.json"

# Groq's OpenAI-compatible endpoint -- no extra SDK dependency, just requests
# (already used elsewhere in this project).
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Two sibling models, same rubric. If the active one hits its per-model
# daily/token cap, swap to the sibling instead of failing the item --
# unlike Gemini's per-project cap, Groq's per-model cap clears by
# switching models (the sibling-model-swap trick confirmed in a prior
# session).
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

# Real per-minute caps, confirmed via console.groq.com -> Limits page
# (2026-08-24) for both gpt-oss models: 30 RPM, 8,000 TPM. The escalating
# Retry-After penalties seen 2026-08-23 and again this morning were very
# likely a symptom of this: the old 1.5s static pacing allowed ~40
# calls/min, already over the 30 RPM cap -- and even paced correctly for
# RPM, a handful of calls a minute can still blow past 8K TPM once
# CATEGORY_RUBRIC's system-prompt token cost is counted on every single
# call. Fixed with (a) a static floor that respects RPM, plus (b) an
# adaptive pause driven by each response's real "usage" field, so pacing
# reacts to actual token cost instead of a guess.
RPM_LIMIT = 30
TPM_LIMIT = 8000
MIN_REQUEST_INTERVAL = 60 / RPM_LIMIT  # 2.0s floor, RPM alone
TPM_SAFETY_MARGIN = 0.85  # pause before hitting the cap, not after

# Added 2026-08-24: if Groq's own Retry-After ever asks for longer than
# this, stop the whole run instead of sleeping through it and retrying.
# Both 2026-08-23 and this morning showed the same shape -- every retry
# during an active penalty window (even ones that correctly honored
# Retry-After) made the NEXT wait longer, not shorter (258s -> 408s,
# 681s -> 898s). Retrying into a real penalty doesn't wait it out, it
# feeds it. Past this threshold the right move is to go quiet entirely
# and let the person decide when to resume, not keep knocking.
MAX_ACCEPTABLE_WAIT_S = 60


class GroqCooldownRequired(Exception):
    """Raised instead of retrying when Groq's Retry-After exceeds
    MAX_ACCEPTABLE_WAIT_S. Caught in main() to stop the run cleanly --
    progress already saved so far is preserved, nothing lost."""
    def __init__(self, wait_s):
        self.wait_s = wait_s
        super().__init__(f"Groq asked for a {wait_s:.0f}s wait -- over the "
                          f"{MAX_ACCEPTABLE_WAIT_S}s threshold, stopping instead of retrying.")


# --- cross-run cooldown memory (added 2026-08-29) -----------------------
#
# This pipeline runs as an hourly GitHub Actions cron job, NOT a daemon --
# each hour is a completely fresh process with no memory of the last run
# beyond what's in the committed files. Before this, a GroqCooldownRequired
# stop just ended that hour's run; the NEXT hourly invocation had no idea
# a cooldown had been requested and would immediately retry from scratch,
# potentially hitting the exact same still-active penalty window again.
# Per this codebase's own documented history (see classify_item_groq's
# docstring), retrying INTO an active penalty window doesn't clear it, it
# extends it (258s -> 408s -> 681s -> 898s in a past session). An hourly
# cadence is normally plenty of gap for an ordinary transient 429 to clear
# on its own -- but if the real cause is a longer quota wall, blindly
# retrying every hour regardless wastes a run and gives no diagnostic
# signal beyond "stopped early" repeated forever.
#
# COOLDOWN_STATE_FILE persists the real deadline Groq asked for (now +
# wait_s) across process boundaries. The NEXT run checks it BEFORE
# attempting any fresh Groq call -- if still within the window, it skips
# Stage 1 entirely (zero API calls, zero risk of extending the penalty)
# and says exactly how long until it's worth trying again, instead of
# silently failing at the same spot hour after hour.
COOLDOWN_STATE_FILE = "groq_cooldown_state.json"


def save_cooldown(wait_s, reason):
    deadline = datetime.now(timezone.utc) + timedelta(seconds=wait_s)
    with open(COOLDOWN_STATE_FILE, "w") as f:
        json.dump({
            "cooldown_until": deadline.isoformat(),
            "requested_wait_s": wait_s,
            "reason": reason,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    return deadline


def load_active_cooldown():
    """Returns the cooldown deadline (datetime) if one is still active,
    else None. Missing/unparseable state is treated as no cooldown --
    fails open, since a stale/corrupt state file shouldn't be able to
    block the pipeline forever."""
    if not os.path.exists(COOLDOWN_STATE_FILE):
        return None
    try:
        with open(COOLDOWN_STATE_FILE) as f:
            state = json.load(f)
        deadline = datetime.fromisoformat(state["cooldown_until"])
    except Exception:
        return None
    if datetime.now(timezone.utc) < deadline:
        return deadline
    return None


def load_or_prompt_groq_key():
    """Groq: single key, no key-pool needed -- its quota (1,000 RPD /
    200,000 TPD per model, two swappable sibling models) is generous
    enough for one account to cover a full daily batch on its own."""
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key

    if os.path.exists(GROQ_CONFIG_FILE):
        with open(GROQ_CONFIG_FILE) as f:
            saved = json.load(f)
        if saved.get("api_key", "").startswith("gsk_"):
            return saved["api_key"]

    print("Get a free key from console.groq.com -> API Keys -> Create API key.")
    print("No card needed. Groq keys start with 'gsk_'.\n")
    while True:
        key = input("Paste your Groq API key, then press Enter: ").strip()
        if key.startswith("gsk_") and len(key) > 20:
            break
        print(f"\nThat doesn't look like a valid key (got {len(key)} chars, "
              f"expected it to start with 'gsk_'). Try again.\n")

    with open(GROQ_CONFIG_FILE, "w") as f:
        json.dump({"api_key": key}, f)
    print(f"\nSaved to {GROQ_CONFIG_FILE} -- won't ask again.\n")
    return key


def is_groq_model_exhausted_error(e):
    """Groq's per-model daily-request or daily-token cap. NOT yet
    confirmed against a real exhaustion response -- message-matching here
    is a best guess based on Groq's documented error shape (OpenAI-style
    'rate_limit_exceeded' with a 'tokens per day' / 'requests per day'
    reason) and should be double-checked against a real 429 payload
    before this is trusted the way is_daily_quota_error() already is for
    Gemini.
    """
    msg = str(e).lower()
    return (
        "rate_limit_exceeded" in msg
        or "tokens per day" in msg or "tpd" in msg
        or "requests per day" in msg or "rpd" in msg
    )


def classify_item_groq(api_key, item, max_retries=5):
    """Takes an item, returns (llm_result, usage_tokens): llm_result is a
    raw {"category", "specificity", "reasoning"} dict (or an error-shaped
    dict on parse failure) -- same shape classify_item() returns, so
    score_item() and needs_second_opinion() work on it unchanged.
    usage_tokens is the real total-token cost Groq reports for the call,
    added 2026-08-24 so the caller can pace against the actual 8K TPM cap
    instead of a guessed average. Swaps model on
    is_groq_model_exhausted_error instead of failing the item; ordinary
    transient errors (429 per-minute, 5xx) get a backoff -- honoring
    Groq's own Retry-After header when present (2026-08-23: hit Groq's
    real per-minute request cap in live testing, something the earlier
    RPD/TPD-only quota research never surfaced -- the fixed 20s/40s
    backoff copied from the Gemini path wasn't long enough to clear it,
    so this now trusts the server's own retry-after time first and only
    falls back to a widened fixed schedule if the server doesn't say).
    """
    user_content = f"Title: {item['title']}\n\nFull text: {item['text']}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = None
    model_idx = 0

    for attempt in range(max_retries + 1):
        model = GROQ_MODELS[model_idx]
        payload = {
            "model": model,
            "temperature": 0,  # deterministic, same reasoning as the Gemini side
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": CATEGORY_RUBRIC},
                {"role": "user", "content": user_content},
            ],
        }
        resp = None
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Real token cost for this call, straight from Groq -- used by
            # the caller to pace against the 8K TPM cap instead of guessing.
            usage_tokens = data.get("usage", {}).get("total_tokens", 0)
            raw = data["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
            try:
                return json.loads(raw), usage_tokens
            except json.JSONDecodeError:
                return {
                    "category": "none",
                    "specificity": "vague",
                    "reasoning": f"PARSE ERROR, raw response: {raw[:200]}",
                }, usage_tokens
        except Exception as e:
            last_err = e
            if is_groq_model_exhausted_error(e) and model_idx < len(GROQ_MODELS) - 1:
                model_idx += 1
                print(f"    {model} exhausted -- swapping to {GROQ_MODELS[model_idx]}...")
                continue  # retry immediately on the sibling model, same attempt budget
            if is_transient_error(e) and attempt < max_retries:
                # Prefer the server's own Retry-After (seconds) when Groq
                # sends one on a 429 -- it knows its own per-minute window
                # better than any fixed guess. Fall back to a wider fixed
                # backoff (was 20/40s, not enough in practice) otherwise.
                retry_after = None
                if resp is not None:
                    retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_s = float(retry_after) + 1  # small safety margin
                    except ValueError:
                        wait_s = 30 * (attempt + 1)
                else:
                    wait_s = 30 * (attempt + 1)
                if wait_s > MAX_ACCEPTABLE_WAIT_S:
                    raise GroqCooldownRequired(wait_s)
                print(f"    transient error, retrying in {wait_s:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_s)
                continue
            raise last_err
    raise last_err


def load_route_cache():
    """Load previously routed items, keyed by (source, raw_id) via the same
    item_key() layer2_classifier.load_cache() uses. Only current-rubric-
    version, non-error entries are reused -- everything else (rubric
    changed, or the cached entry was itself an API-error fallback) gets
    re-classified. A cached hit carries its second_opinion /
    providers_agree / final_confirmed along with it if Stage 2 already ran
    on that item, so a rerun doesn't re-spend Gemini quota on items that
    already got a second opinion either -- see the Stage 2 filter below.

    Added 2026-08-23: layer2_route.py had NO caching at all before this --
    every rerun reclassified all 75 items via Groq from scratch, Stage 1
    AND Stage 2 both. Three full batches back to back in one session (smoke
    test, a real run, then a rerun meant to just top up 4 missing second
    opinions) is what triggered Groq's escalating Retry-After penalty.
    """
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


def save_merged_cache(cache, results):
    """Writes OUTPUT_FILE as the union of the previously-loaded cache and
    this run's results, keyed by item_key -- so a run that stops early
    (Groq cooldown, external timeout, or a thin upstream item list) can
    never destroy cache history for items it didn't touch this cycle.
    Added 2026-08-31 after a real incident: a run that stopped early
    wrote only 4 items to disk, wiping 143 previously-cached entries and
    forcing the next run to reclassify almost everything from scratch,
    which is what caused it to blow through even a doubled timeout.
    """
    merged = dict(cache)
    for r in results:
        merged[item_key(r)] = r
    with open(OUTPUT_FILE, "w") as out_f:
        json.dump(list(merged.values()), out_f, indent=2)


def main():
    groq_key = load_or_prompt_groq_key()
    gemini_keys = load_or_prompt_api_keys()
    gemini_clients = [genai.Client(api_key=k) for k in gemini_keys]
    # Round-robin ALL available keys (2026-08-24 fix, matches
    # layer2_classifier.py's main()) instead of sticking to one key until
    # fully exhausted. Each key caps at 5 RPM on gemini-3.5-flash's free
    # tier -- spreading calls across N keys lets overall pace scale with
    # key count instead of bottlenecking on one key's cap. This is also
    # what was missing here specifically: this loop had NO pacing at all
    # between calls before, which is what triggered the live 429 seen
    # today (Stage 1 had already been fixed, Stage 2 hadn't).
    gemini_available = list(range(len(gemini_clients)))
    gemini_rr_pos = 0

    with open(INPUT_FILE) as f:
        items = json.load(f)

    cache = load_route_cache()

    # ---- Stage 1: Groq, full batch ----
    print(f"STAGE 1 (Groq, primary): classifying {len(items)} items "
          f"({len(cache)} already have a good cached result -- will be reused, not re-billed)...\n")

    active_cooldown = load_active_cooldown()
    if active_cooldown is not None:
        remaining_s = (active_cooldown - datetime.now(timezone.utc)).total_seconds()
        print(f"NOTE: a Groq cooldown from a prior run is still active -- "
              f"{remaining_s:.0f}s remaining (until {active_cooldown.isoformat()}). "
              f"Skipping any fresh Groq calls this run; cached items still process "
              f"normally below since they cost nothing.\n")

    results = []
    reused_count = 0
    stopped_early = False
    minute_window_start = time.time()
    tokens_this_minute = 0
    for idx, item in enumerate(items, 1):
        key = item_key(item)
        cached = cache.get(key)
        if cached is not None:
            scored = dict(cached)  # reuse -- no Groq call, keeps any prior second_opinion too
            # Backfill for cache entries written before symbols_affected
            # existed -- mechanical, no need to force a re-classify.
            scored.setdefault(
                "symbols_affected", extract_symbols_affected(item, scored.get("category", "none"))
            )
            reused_count += 1
            results.append(scored)
            confirmed_tag = "CONFIRMED" if scored.get("final_confirmed") else ""
            print(
                f"  [{idx}/{len(items)}] {scored['category']:<24} {scored['specificity']:<9} "
                f"conf={scored['confidence_score']:.2f} {confirmed_tag:<9} {item['title'][:50]} [cached]"
            )
            continue

        if active_cooldown is not None:
            print(f"\n{'!' * 80}")
            print(f"SKIPPING: Groq cooldown from a prior run is still active "
                  f"({(active_cooldown - datetime.now(timezone.utc)).total_seconds():.0f}s "
                  f"remaining) -- not attempting any new Groq calls this run, to avoid "
                  f"retrying into the same window and extending it further (see "
                  f"classify_item_groq's docstring for why that backfires).")
            print(f"Processed {idx - 1}/{len(items)} items this run ({reused_count} from "
                  f"cache). Everything gathered so far is saved to {OUTPUT_FILE} -- "
                  f"the next scheduled run will retry once the cooldown clears "
                  f"({(len(items) - idx + 1)} item(s) still waiting).")
            print(f"{'!' * 80}\n")
            stopped_early = True
            break

        try:
            llm_result, call_tokens = classify_item_groq(groq_key, item)
        except GroqCooldownRequired as e:
            deadline = save_cooldown(e.wait_s, str(e))
            print(f"\n{'!' * 80}")
            print(f"STOPPING: {e}")
            print(f"Saved this cooldown to {COOLDOWN_STATE_FILE} (until "
                  f"{deadline.isoformat()}) -- the next scheduled run will check this "
                  f"first and skip straight past any fresh Groq calls instead of "
                  f"retrying blind into the same window.")
            print(f"Processed {idx - 1}/{len(items)} items this run ({reused_count} from "
                  f"cache). Everything gathered so far is saved to {OUTPUT_FILE} -- "
                  f"re-run later and caching will pick up right here, no lost work.")
            print(f"{'!' * 80}\n")
            stopped_early = True
            break
        except Exception as e:
            print(f"  [{idx}/{len(items)}] Groq FAILED: {e}")
            llm_result = {
                "category": "none",
                "specificity": "vague",
                "reasoning": f"API ERROR: {e}",
            }
            call_tokens = 0
        scored = score_item(item, llm_result)
        scored["provider"] = "groq"
        scored["final_confirmed"] = scored["confirmed"]
        results.append(scored)
        confirmed_tag = "CONFIRMED" if scored["confirmed"] else ""
        print(
            f"  [{idx}/{len(items)}] {scored['category']:<24} {scored['specificity']:<9} "
            f"conf={scored['confidence_score']:.2f} {confirmed_tag:<9} {item['title'][:50]}"
        )
        save_merged_cache(cache, results)

        # Pacing fix (2026-08-24): replaces the old flat 1.5s sleep, which
        # allowed ~40 calls/min -- already over the real 30 RPM cap, and
        # the root cause of the escalating Retry-After penalties. This
        # enforces a 2.0s RPM floor every call, PLUS an adaptive pause
        # using each call's real token usage so a run also can't blow the
        # 8K TPM cap, whatever the true per-call token cost turns out to be.
        now = time.time()
        if now - minute_window_start >= 60:
            minute_window_start = now
            tokens_this_minute = 0
        tokens_this_minute += call_tokens
        if tokens_this_minute >= TPM_LIMIT * TPM_SAFETY_MARGIN:
            wait_s = 60 - (now - minute_window_start)
            if wait_s > 0:
                print(f"    approaching {TPM_LIMIT} TPM cap ({tokens_this_minute} used this "
                      f"window) -- pausing {wait_s:.0f}s for it to clear...")
                time.sleep(wait_s)
            minute_window_start = time.time()
            tokens_this_minute = 0
        else:
            time.sleep(MIN_REQUEST_INTERVAL)

    # Only flag items that don't already have a second opinion from a
    # cached prior run -- otherwise a rerun would re-spend Gemini quota
    # re-confirming things Stage 2 already settled.
    flagged = [
        i for i, r in enumerate(results)
        if needs_second_opinion(r) and r.get("second_opinion") is None
    ]
    print(f"\nSTAGE 1 done. {reused_count}/{len(items)} reused from cache. "
          f"{len(flagged)} item(s) newly flagged for a Gemini second opinion.\n")

    # ---- Stage 2: Gemini, flagged items only ----
    print(f"STAGE 2 (Gemini, secondary): re-checking {len(flagged)} flagged item(s)...\n")
    for n, i in enumerate(flagged, 1):
        item = items[i]
        llm_result = None
        last_err = None
        try:
            while gemini_available:
                client_idx = gemini_available[gemini_rr_pos % len(gemini_available)]
                try:
                    llm_result = classify_item_gemini(gemini_clients[client_idx], item)
                    gemini_rr_pos += 1
                    break
                except GeminiCooldownRequired:
                    raise
                except Exception as e:
                    last_err = e
                    if is_daily_quota_error(e):
                        print(f"    key #{client_idx + 1} hit its daily quota -- "
                              f"removing from rotation ({len(gemini_available) - 1} key(s) left)...")
                        gemini_available.remove(client_idx)
                        continue  # retry same item on whichever key is next
                    break  # not a daily-quota error -- give up on this item
        except GeminiCooldownRequired as e:
            print(f"\n{'!' * 80}")
            print(f"STOPPING (Stage 2): {e}")
            print(f"Got a second opinion on {n - 1}/{len(flagged)} flagged item(s) this run. "
                  f"Everything gathered so far (including all of Stage 1) is saved to "
                  f"{OUTPUT_FILE} -- re-run later and caching will pick up right here.")
            print(f"{'!' * 80}\n")
            stopped_early = True
            break

        if not gemini_available:
            print(f"  [{n}/{len(flagged)}] all Gemini keys exhausted for today -- stopping Stage 2.")
            stopped_early = True
            break

        if llm_result is None:
            print(f"  [{n}/{len(flagged)}] Gemini second opinion FAILED: {last_err}")
            continue  # keep the Groq result as final -- second opinion unavailable

        second = score_item(item, llm_result)
        second["provider"] = "gemini_second_opinion"
        agree = second["confirmed"] == results[i]["confirmed"]

        results[i]["second_opinion"] = second
        results[i]["providers_agree"] = agree
        # OR, not AND -- see module docstring: this is a review queue, a
        # false negative costs more than a false positive here.
        results[i]["final_confirmed"] = results[i]["confirmed"] or second["confirmed"]

        tag = "AGREE" if agree else "DISAGREE -- needs a human look"
        print(
            f"  [{n}/{len(flagged)}] groq_confirmed={results[i]['confirmed']} "
            f"gemini_confirmed={second['confirmed']} {tag} -- {item['title'][:45]}"
        )
        save_merged_cache(cache, results)
        # Pacing (2026-08-24): this loop had no delay at all before --
        # paces against GEMINI_RPM_SAFETY_MARGIN per key, spread across
        # however many keys are still in rotation, same formula as
        # layer2_classifier.py's main().
        time.sleep(60 / (GEMINI_RPM_SAFETY_MARGIN * max(1, len(gemini_available))))

    save_merged_cache(cache, results)

    final_confirmed_count = sum(1 for r in results if r["final_confirmed"])
    disagree_count = sum(1 for r in results if r.get("providers_agree") is False)
    print(f"\n{'=' * 80}")
    if stopped_early:
        remaining_note = (
            f"{len(items) - len(results)} item(s) still need Stage 1"
            if len(results) < len(items) else
            "Stage 1 is complete; some flagged item(s) still need a Stage 2 second opinion"
        )
        print(
            f"STOPPED EARLY (cooldown): {len(results)}/{len(items)} items through Stage 1 -- "
            f"{len(flagged)} got a second opinion this run, {disagree_count} disagreement(s), "
            f"{final_confirmed_count} final CONFIRMED so far -- saved to {OUTPUT_FILE}. "
            f"Re-run later to finish: {remaining_note}."
        )
    else:
        print(
            f"DONE: {len(items)} items -- {len(flagged)} got a second opinion, "
            f"{disagree_count} disagreement(s), {final_confirmed_count} final CONFIRMED "
            f"-- saved to {OUTPUT_FILE}"
        )
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
