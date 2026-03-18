"""
backfill_flares.py — Retroactive gap filler
Fetches all data ONCE then processes all missing cards.

Usage:
  python scripts/backfill_flares.py            # generate all missing
  python scripts/backfill_flares.py --dry-run  # list gaps only
  python scripts/backfill_flares.py --min-class M
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_flare_card import (
    ensure_dirs,
    fetch_all_flares,
    fetch_xray_series,
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
    peak_dt = parse_time(flare["max_time"])
    flare_class = flare["max_class"]
    filename = f"flare_{peak_dt.strftime('%Y%m%d_%H%M')}_{flare_class.replace('.', 'p')}.png"
    return os.path.exists(os.path.join(CARDS_DIR, filename))


def class_rank(flare: dict) -> int:
    letter = (flare.get("max_class") or "A")[0].upper()
    return CLASS_ORDER.get(letter, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-class", default="A", choices=["A", "B", "C", "M", "X"])
    args = parser.parse_args()

    ensure_dirs()

    print("Fetching 7-day flare catalog...")
    all_flares = fetch_all_flares()
    processed_ids = load_processed_ids()

    min_rank = CLASS_ORDER.get(args.min_class.upper(), 0)
    candidates = [f for f in all_flares if class_rank(f) >= min_rank and not card_exists(f)]

    if not candidates:
        print("Nothing to backfill.")
        return

    print(f"Found {len(candidates)} missing card(s):")
    for f in candidates:
        status = "(in state)" if flare_id(f) in processed_ids else "(MISSED)"
        print(f"  {f['max_class']:>5}  peak={f['max_time']}  {status}")

    if args.dry_run:
        print("\n[dry-run] No cards generated.")
        return

    # ── Fetch X-ray data ONCE for all cards ──────────────────────────────
    print("\nFetching X-ray series...")
    xray_times, xray_fluxes = fetch_xray_series()
    # ─────────────────────────────────────────────────────────────────────

    print(f"Generating {len(candidates)} card(s)...")
    generated = 0
    for flare in candidates:
        fid = flare_id(flare)
        try:
            output = render_card(flare, xray_times, xray_fluxes)
            processed_ids.add(fid)
            generated += 1
            print(f"  v {output}")
        except Exception as exc:
            print(f"  x Failed for {fid}: {exc}")

    save_processed_ids(processed_ids)
    prune_old_cards(keep=5)
    print(f"\nBackfill complete. {generated}/{len(candidates)} generated.")


if __name__ == "__main__":
    main()
