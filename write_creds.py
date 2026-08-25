#!/usr/bin/env python3
"""
write_creds.py

Generates local_creds.py fresh on the GitHub Actions runner, from GitHub
encrypted Secrets injected as environment variables. This file is NEVER
committed to the repo -- it's written at the start of every run and
deleted again before anything gets committed back (see the workflow
YAML's "Remove generated credentials file" step).

local_trading_bridge.py imports 7 names from local_creds.py (confirmed
directly from its own import block: ALPACA_KEY_ID, ALPACA_SECRET_KEY,
OANDA_TOKEN, OANDA_ACCOUNT_ID, WEEX_ACCESS_KEY, WEEX_PASSPHRASE,
WEEX_SECRET). The Model D WEEX bridge only ever uses the three WEEX
names -- but the import will fail with an ImportError, crashing the
whole script before it even starts, if the other four aren't defined.
So this writes real values (from Secrets) for the three WEEX names and
harmless empty-string placeholders for the four Alpaca/OANDA names this
bridge never touches.

Only the three WEEX_* secrets are required for this workflow. If
they're missing, this fails loudly and immediately (SystemExit) rather
than writing a broken credentials file and letting the real error
surface confusingly later inside local_trading_bridge.py.
"""
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
    weex_access_key = _require("WEEX_ACCESS_KEY")
    weex_secret = _require("WEEX_SECRET")
    weex_passphrase = _require("WEEX_PASSPHRASE")

    with open("local_creds.py", "w") as f:
        f.write("# Generated at runtime by write_creds.py -- never commit this file.\n")
        f.write(f"ALPACA_KEY_ID = {''!r}\n")
        f.write(f"ALPACA_SECRET_KEY = {''!r}\n")
        f.write(f"OANDA_TOKEN = {''!r}\n")
        f.write(f"OANDA_ACCOUNT_ID = {''!r}\n")
        f.write(f"WEEX_ACCESS_KEY = {weex_access_key!r}\n")
        f.write(f"WEEX_PASSPHRASE = {weex_passphrase!r}\n")
        f.write(f"WEEX_SECRET = {weex_secret!r}\n")

    print("local_creds.py written (WEEX credentials populated, Alpaca/OANDA left blank -- unused by this bridge).")


if __name__ == "__main__":
    main()
