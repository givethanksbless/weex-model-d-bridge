#!/usr/bin/env python3
"""
write_telegram_config.py

Generates telegram_config.json and telegram_ingest.session fresh on the
GitHub Actions runner, from GitHub encrypted Secrets. Mirrors the
write_creds.py / write_gemini_config.py pattern already used in this repo:
written at the start of every run, never committed, deleted again before
any commit step runs.

CONFIRMED schema (read directly from telegram_ingest_smoketest.py and
unified_ingest.py on the user's Mac on 2026-08-27, not guessed):
  telegram_config.json: {"api_id": <int>, "api_hash": "<string>"}
  telegram_ingest.session: a Telethon SQLite session file (binary) --
    already authenticated via one-time QR login on the user's Mac.
    Stored as a GitHub Secret in base64 form since Secrets are text-only.

SECURITY NOTE: this session file is equivalent to a logged-in Telegram
device, not a scoped API key -- treat the GitHub Secret holding it with
the same care as an account password. Decided 2026-08-27 with the user
to proceed anyway for the trial/demo phase, accepting that risk
knowingly, until dedicated always-on hardware removes the need to run
this in someone else's cloud at all.

Setup required before this script works -- in the repo's Settings >
Secrets and variables > Actions, add:
  TELEGRAM_API_ID     -- the numeric api_id from telegram_config.json
  TELEGRAM_API_HASH   -- the api_hash string from telegram_config.json
  TELEGRAM_SESSION_B64 -- output of:
      base64 < telegram_ingest.session | tr -d '\\n'
    (run that on the Mac, copy the single-line output as the secret value)

Example workflow step:
  - name: Write Telegram credentials (generated fresh every run, never committed)
    env:
      TELEGRAM_API_ID: ${{ secrets.TELEGRAM_API_ID }}
      TELEGRAM_API_HASH: ${{ secrets.TELEGRAM_API_HASH }}
      TELEGRAM_SESSION_B64: ${{ secrets.TELEGRAM_SESSION_B64 }}
    run: python3 write_telegram_config.py
"""
import base64
import json
import os


def _require(name):
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"Missing required environment variable: {name}. "
            f"Check that the GitHub Secret is set and mapped in the workflow's "
            f"'env:' block for this step."
        )
    return val


def main():
    api_id = _require("TELEGRAM_API_ID")
    api_hash = _require("TELEGRAM_API_HASH")
    session_b64 = _require("TELEGRAM_SESSION_B64")

    with open("telegram_config.json", "w") as f:
        json.dump({"api_id": int(api_id), "api_hash": api_hash}, f)

    session_bytes = base64.b64decode(session_b64)
    with open("telegram_ingest.session", "wb") as f:
        f.write(session_bytes)

    print("telegram_config.json and telegram_ingest.session written.")


if __name__ == "__main__":
    main()
