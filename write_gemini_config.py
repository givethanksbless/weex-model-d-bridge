#!/usr/bin/env python3
"""
write_gemini_config.py

Generates gemini_config.json fresh on the GitHub Actions runner, from a
GitHub encrypted Secret injected as an environment variable. Mirrors the
write_creds.py pattern already used for WEEX credentials in the
weex-model-d-bridge repo: written at the start of every run, never
committed, deleted again before any commit step runs.

CONFIRMED schema (read directly from gemini_config.json on the user's
Mac on 2026-08-27, not guessed):
  {"api_keys": ["<key1>", "<key2>", "<key3>"]}

layer2_route.py's load_or_prompt_api_keys() reads this file if it exists
and is non-empty, and only falls back to interactive input() prompts if
it's missing or empty. Pre-writing this file with real keys before that
function runs means it never hits the interactive path -- which would
otherwise hang forever on a non-interactive CI runner.

Setup required before this script works:
  1. In the repo's Settings > Secrets and variables > Actions, add ONE
     secret named GEMINI_API_KEYS containing all 3 keys separated by
     commas, no spaces, e.g.:
       key1,key2,key3
  2. Reference that secret as an env var in the workflow step that runs
     this script (see example workflow snippet in the docstring below).

Example workflow step:
  - name: Write Gemini API keys (generated fresh every run, never committed)
    env:
      GEMINI_API_KEYS: ${{ secrets.GEMINI_API_KEYS }}
    run: python3 write_gemini_config.py
"""
import json
import os


def main():
    raw = os.environ.get("GEMINI_API_KEYS")
    if not raw:
        raise SystemExit(
            "Missing required environment variable: GEMINI_API_KEYS. "
            "Check that the GitHub Secret is set (comma-separated keys, "
            "no spaces) and mapped in the workflow's 'env:' block for "
            "this step."
        )

    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise SystemExit(
            "GEMINI_API_KEYS was set but contained no usable keys after "
            "splitting on commas -- check the secret's value."
        )

    with open("gemini_config.json", "w") as f:
        json.dump({"api_keys": keys}, f, indent=2)

    print(f"gemini_config.json written with {len(keys)} key(s).")


if __name__ == "__main__":
    main()
