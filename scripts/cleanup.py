"""
cleanup.py — Monthly history reset
====================================
Deletes card images older than KEEP_DAYS and trims processed_flares.json
down to only IDs from the same window so the state file stays lean.

Run automatically on the 1st of each month via cleanup.yml,
or manually: python scripts/cleanup.py
"""

import json
import os
import glob
import re
from datetime import datetime, timezone, timedelta

CARDS_DIR = "cards"
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "processed_flares.json")

# How many days of cards + IDs to keep (everything older is wiped)
KEEP_DAYS = 7


def cleanup():
    cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
    print(f"Cleaning up anything older than {cutoff.strftime('%Y-%m-%d %H:%M UTC')} ({KEEP_DAYS} days)")

    # ── 1. Delete old card images ─────────────────────────────────────────
    cards = glob.glob(os.path.join(CARDS_DIR, "flare_*.png"))
    removed_cards = 0
    for card in cards:
        # Parse date from filename (flare_YYYYMMDD_HHMM_CLASS.png)
        # checkout resets mtime so file timestamps are unreliable
        m = re.match(r"flare_(\d{8})_(\d{4})_", os.path.basename(card))
        if not m:
            continue
        file_dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        if file_dt < cutoff:
            os.remove(card)
            print(f"  [card] Deleted: {card}")
            removed_cards += 1

    print(f"  Cards removed: {removed_cards} / {len(cards)}")

    # ── 2. Trim processed_flares.json to the same window ─────────────────
    if not os.path.exists(STATE_FILE):
        print("  [state] No state file found — nothing to trim.")
        return

    with open(STATE_FILE, "r", encoding="utf-8") as fh:
        try:
            ids = set(json.load(fh))
        except (json.JSONDecodeError, TypeError):
            ids = set()

    before = len(ids)

    # IDs are formatted as "YYYY-MM-DDTHH:MM:SSZ_..." — trim by date prefix
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    trimmed = {fid for fid in ids if fid[:10] >= cutoff_str}

    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(sorted(trimmed), fh, indent=2)

    print(f"  [state] IDs trimmed: {before} → {len(trimmed)}")
    print("Cleanup complete.")


if __name__ == "__main__":
    cleanup()
