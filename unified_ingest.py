"""
unified_ingest.py

Combines all proven ingestion sources -- WEEX announcements (first-party),
Watcher Guru, Wu Blockchain, CoinDesk, and Cointelegraph (third-party, all via
Telegram) -- into one common structured item format, as prep for the
classifier. Does NOT classify anything; this only normalizes raw items into
one shape and saves/prints them.

Reuses the already-authenticated Telegram session (telegram_ingest.session) and
saved API credentials (telegram_config.json) from telegram_ingest_smoketest.py --
that script needs to have been run and logged in successfully first.

Common item schema
-------------------
{
    "source": "weex" | "watcherguru" | "wublockchainenglish",
    "source_type": "first_party" | "third_party",
    "title": str,
    "text": str,
    "timestamp_utc": str,   # ISO 8601
    "url": str,
    "raw_id": str,
}

Usage
-----
    python3 unified_ingest.py
"""

import asyncio
import json
import os
import re
import urllib.request
from datetime import timezone

from telethon import TelegramClient

# -- WEEX (first-party) ------------------------------------------------------

WEEX_ANNOUNCEMENTS_URL = (
    "https://weexsupport.zendesk.com/api/v2/help_center/en-us/categories/"
    "18540264809497/articles.json"
)


def _strip_html(raw_html):
    return re.sub(r"<[^>]+>", " ", raw_html or "").strip()


def fetch_weex_items(per_page=20):
    url = f"{WEEX_ANNOUNCEMENTS_URL}?per_page={per_page}&sort_by=created_at&sort_order=desc"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (unified-ingest)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[weex] FAILED: {e}")
        return []

    articles = data.get("articles", [])
    items = []
    for art in articles:
        items.append({
            "source": "weex",
            "source_type": "first_party",
            "title": art.get("title", "") or "",
            "text": _strip_html(art.get("body", "")),
            "timestamp_utc": art.get("created_at", ""),
            "url": art.get("html_url", ""),
            "raw_id": str(art.get("id", "")),
        })
    print(f"[weex] OK -- {len(items)} items")
    return items


# -- Telegram (third-party) ---------------------------------------------------

CONFIG_FILE = "telegram_config.json"
SESSION_NAME = "telegram_ingest"  # reuse the already-authenticated session
TELEGRAM_CHANNELS = [
    "watcherguru", "wublockchainenglish",
    # Added 2026-08-23: official publisher channels (not aggregators like the
    # two above) -- same third_party weight in the rubric for now, but higher
    # editorial quality. Handles confirmed live via web search, not yet
    # smoke-tested through this script -- watch the first real run for a
    # resolve failure (wrong handle / channel renamed / private).
    "CoinDeskGlobal", "cointelegraph",
]
MESSAGES_PER_CHANNEL = 40


def load_telegram_credentials():
    if not os.path.exists(CONFIG_FILE):
        print(f"[telegram] FAILED -- {CONFIG_FILE} not found. Run "
              f"telegram_ingest_smoketest.py first to log in.")
        return None, None
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    return cfg["api_id"], cfg["api_hash"]


async def fetch_telegram_items():
    api_id, api_hash = load_telegram_credentials()
    if not api_id:
        return []

    client = TelegramClient(SESSION_NAME, api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        print("[telegram] FAILED -- session not authorized. Run "
              "telegram_ingest_smoketest.py first to log in via QR.")
        await client.disconnect()
        return []

    items = []
    for channel in TELEGRAM_CHANNELS:
        try:
            entity = await client.get_entity(channel)
        except Exception as e:
            print(f"[telegram:{channel}] FAILED to resolve: {e}")
            continue

        count = 0
        async for msg in client.iter_messages(entity, limit=MESSAGES_PER_CHANNEL):
            if not msg.text:
                continue
            text = msg.text.strip()
            title = text.split("\n")[0][:100]
            items.append({
                "source": channel,
                "source_type": "third_party",
                "title": title,
                "text": text,
                "timestamp_utc": msg.date.astimezone(timezone.utc).isoformat(),
                "url": f"https://t.me/{channel}/{msg.id}",
                "raw_id": str(msg.id),
            })
            count += 1
        print(f"[telegram:{channel}] OK -- {count} items")

    await client.disconnect()
    return items


# -- combine + report ----------------------------------------------------------

OUTPUT_FILE = "ingested_items.json"


async def main():
    print("Unified ingest -- WEEX + Watcher Guru + Wu Blockchain + CoinDesk + Cointelegraph\n")

    weex_items = fetch_weex_items()
    telegram_items = await fetch_telegram_items()

    all_items = weex_items + telegram_items
    all_items.sort(key=lambda x: x["timestamp_utc"], reverse=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {len(all_items)} items ({len(weex_items)} weex, "
          f"{len(telegram_items)} telegram) -- saved to {OUTPUT_FILE}")
    print(f"{'=' * 80}\n")

    for item in all_items:
        print(f"[{item['timestamp_utc']}] ({item['source']}/{item['source_type']}) {item['title']}")

    print("\nDONE. Paste this full output back into chat.")


if __name__ == "__main__":
    asyncio.run(main())
