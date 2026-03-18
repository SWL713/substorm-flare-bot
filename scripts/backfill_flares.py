"""
backfill_flares.py — Retroactive gap filler
============================================
Scans the full 7-day NOAA flare catalog (primary + secondary), finds any
flare whose card doesn't already exist in cards/, and generates it.

Usage:
  # Dry-run — list gaps without generating anything
  python scripts/backfill_flares.py --dry-run

  # Generate all missing cards (safe to run multiple times — idempotent)
  python scripts/backfill_flares.py

  # Only backfill flares of class M and above
  python scripts/backfill_flares.py --min-class M

GitHub Actions (manual trigger):
  workflow_dispatch on backfill.yml
"""

import argparse
import os
import sys

# Reuse everything from the main script
sys.path.insert(0, os.path.dirname(__file__))
from generate_flare_card import (
    ensure_dirs,
    fetch_all_flares,
    load_processed_ids,
    save_processed_ids,
    render_card,
    flare_id,
    prune_old_cards,
    parse_time,
    CARDS_DIR,
)

CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


def card_exists(flare: dict) -> bool:
    """Check whether a card PNG already exists for this flare."""
    from datetime import datetime
    peak_dt = parse_time(flare["max_time"])
    flare_class = flare["max_class"]
    filename = f"flare_{peak_dt.strftime('%Y%m%d_%H%M')}_{flare_class.replace('.', 'p')}.png"
    return os.path.exists(os.path.join(CARDS_DIR, filename))


def class_rank(flare: dict) -> int:
    letter = (flare.get("max_class") or "A")[0].upper()
    return CLASS_ORDER.get(letter, 0)


def main():
    parser = argparse.ArgumentParser(description="Backfill missing flare cards")
    parser.add_argument("--dry-run", action="store_true",
                        help="List missing cards without generating them")
    parser.add_argument("--min-class", default="A",
                        choices=["A", "B", "C", "M", "X"],
                        help="Only backfill flares at or above this class (default: A = all)")
    args = parser.parse_args()

    ensure_dirs()

    print("Fetching 7-day flare catalog from primary + secondary GOES feeds...")
    all_flares = fetch_all_flares()
    processed_ids = load_processed_ids()

    min_rank = CLASS_ORDER.get(args.min_class.upper(), 0)
    candidates = [
        f for f in all_flares
        if class_rank(f) >= min_rank and not card_exists(f)
    ]

    if not candidates:
        print("Nothing to backfill — all flares already have cards.")
        return

    print(f"\nFound {len(candidates)} flare(s) without cards:")
    for f in candidates:
        fid = flare_id(f)
        already = "(already in processed_ids)" if fid in processed_ids else "(NEW — was missed)"
        print(f"  {f['max_class']:>5}  peak={f['max_time']}  sat=GOES-{f.get('satellite','')}  {already}")

    if args.dry_run:
        print("\n[dry-run] No cards generated.")
        return

    print(f"\nGenerating {len(candidates)} missing card(s)...")
    generated = 0
    for flare in candidates:
        fid = flare_id(flare)
        try:
            output = render_card(flare)
            processed_ids.add(fid)
            generated += 1
            print(f"  v {output}")
        except Exception as exc:
            print(f"  x Failed for {fid}: {exc}")

    save_processed_ids(processed_ids)
    prune_old_cards(keep=50)  # Be generous during backfill
    print(f"\nBackfill complete. {generated}/{len(candidates)} card(s) generated.")


if __name__ == "__main__":
    main()
