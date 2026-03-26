"""
Substorm Flare Bot — generate_flare_card.py
"""

import json
import os
import glob
from datetime import datetime, timedelta

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter


FLARE_URLS = [
    "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
    "https://services.swpc.noaa.gov/json/goes/secondary/xray-flares-7-day.json",
]
XRAY_URLS = [
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
    "https://services.swpc.noaa.gov/json/goes/secondary/xrays-6-hour.json",
]

TEMPLATE_PATH = "template.png"
CARDS_DIR = "cards"
CHARTS_DIR = "charts"
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "processed_flares.json")
LEGACY_STATE_FILE = os.path.join(DATA_DIR, "last_flare_id.txt")


def ensure_dirs():
    for d in (CARDS_DIR, CHARTS_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def flare_id(flare: dict) -> str:
    # Satellite intentionally excluded: both GOES-16 and GOES-18 can report the same
    # physical flare, and we only want one card per event.
    return (
        f"{flare.get('begin_time', '')}"
        f"_{flare.get('max_time', '')}"
        f"_{flare.get('max_class', '')}"
    )


def load_processed_ids() -> set:
    ids: set = set()
    if not os.path.exists(STATE_FILE) and os.path.exists(LEGACY_STATE_FILE):
        with open(LEGACY_STATE_FILE, "r", encoding="utf-8") as fh:
            legacy_id = fh.read().strip()
        if legacy_id:
            ids.add(legacy_id)
        save_processed_ids(ids)
        print(f"[migrate] Promoted legacy ID: {legacy_id}")
        return ids
    if not os.path.exists(STATE_FILE):
        return ids
    with open(STATE_FILE, "r", encoding="utf-8") as fh:
        try:
            ids = set(json.load(fh))
        except (json.JSONDecodeError, TypeError):
            pass
    return ids


def save_processed_ids(ids: set):
    capped = sorted(ids)[-500:]
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(capped, fh, indent=2)


def _fetch_json(url: str) -> list:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "substorm-flare-bot"})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[warn] Could not fetch {url}: {exc}")
        return []


def fetch_all_flares() -> list:
    seen_ids: set = set()
    merged: list = []
    for url in FLARE_URLS:
        for f in _fetch_json(url):
            if not (f.get("max_time") and f.get("max_class")):
                continue
            fid = flare_id(f)
            if fid not in seen_ids:
                seen_ids.add(fid)
                merged.append(f)
    if not merged:
        raise RuntimeError("No valid flare data from any source.")
    merged.sort(key=lambda f: f["max_time"])
    return merged


def fetch_xray_series():
    """Fetch X-ray series ONCE — result is passed to all cards in the run."""
    for url in XRAY_URLS:
        data = _fetch_json(url)
        times, fluxes = [], []
        for row in data:
            if str(row.get("energy", "")).strip() != "0.1-0.8nm":
                continue
            flux = row.get("flux")
            time_tag = row.get("time_tag")
            if flux is None or time_tag is None:
                continue
            try:
                t = parse_time(time_tag)
                f = float(flux)
            except Exception:
                continue
            if f > 0:
                times.append(t)
                fluxes.append(f)

        if not times:
            print(f"[warn] No X-ray series from {url}")
            continue

        clean_t, clean_f = [], []
        for i, f in enumerate(fluxes):
            if f < 1e-8:
                continue
            prev_f = fluxes[i - 1] if i > 0 else None
            next_f = fluxes[i + 1] if i < len(fluxes) - 1 else None
            if prev_f and next_f:
                ref = (prev_f + next_f) / 2.0
                if ref > 0 and (f < ref / 20 or f > ref * 20):
                    continue
            clean_t.append(times[i])
            clean_f.append(f)

        if not clean_t:
            continue

        arr = np.array(clean_f)
        if len(arr) >= 5:
            arr = np.convolve(arr, np.ones(5) / 5, mode="same")

        print(f"[xray] Loaded {len(clean_t)} points from {url}")
        return clean_t, arr.tolist()

    raise RuntimeError("X-ray series unavailable from all sources.")


def class_color(flare_class: str):
    letter = (flare_class or "C")[0].upper()
    return {"X": (255, 130, 40), "M": (255, 170, 60),
            "C": (255, 220, 120), "B": (180, 210, 255)}.get(letter, (230, 230, 230))


def draw_glow_text(base, position, text, font, fill_rgb):
    for blur, alpha in ((24, 120), (10, 170)):
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text(position, text, font=font, fill=fill_rgb + (alpha,))
        base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(base).text(position, text, font=font, fill=(255, 235, 170, 255))


def build_chart(flare: dict, output_path: str, xray_times: list, xray_fluxes: list):
    """Build chart using pre-fetched X-ray data — no HTTP call here."""
    peak_time = parse_time(flare["max_time"])
    flare_class = flare["max_class"]
    now_utc = datetime.now().astimezone(peak_time.tzinfo)

    window_start = peak_time - timedelta(minutes=30)
    window_end = min(now_utc, peak_time + timedelta(minutes=30))
    if window_end <= window_start:
        window_start = now_utc - timedelta(minutes=30)
        window_end = now_utc

    filtered = [(t, f) for t, f in zip(xray_times, xray_fluxes) if window_start <= t <= window_end]
    if len(filtered) < 10:
        filtered = list(zip(xray_times, xray_fluxes))

    plot_times = [t for t, _ in filtered]
    plot_fluxes = [f for _, f in filtered]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.plot(plot_times, plot_fluxes, linewidth=8, color="#ffe9a8", alpha=0.10, solid_capstyle="round")
    ax.plot(plot_times, plot_fluxes, linewidth=2.5, color="#ffe9a8", solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-4)

    for value, label, color in [
        (1e-8, "A", "#888888"), (1e-7, "B", "#9cc9ff"),
        (1e-6, "C", "#ffe57a"), (1e-5, "M", "#ffb347"), (1e-4, "X", "#ff7043"),
    ]:
        ax.axhline(value, color=color, linewidth=0.9, alpha=0.45)

    ax.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    ax.set_yticklabels(["A", "B", "C", "M", "X"], color="#f5dfb0", fontsize=11)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.06, color="#ffffff")
    ax.tick_params(axis="x", colors="#f5dfb0", labelsize=10)
    ax.tick_params(axis="y", colors="#f5dfb0", labelsize=11)

    if plot_times and plot_fluxes:
        # Mark the actual maximum flux in the window, not just nearest to peak_time
        max_idx = max(range(len(plot_fluxes)), key=lambda i: plot_fluxes[i])
        max_time = plot_times[max_idx]
        max_flux = plot_fluxes[max_idx]
        ax.axvline(max_time, color="#ffb347", linewidth=1.0, alpha=0.6)
        ax.scatter([max_time], [max_flux], s=50, color="#ffd27a", zorder=5)
        ax.text(max_time, max_flux * 1.35, flare_class, color="#ffd27a",
                fontsize=13, fontweight="bold", ha="left", va="bottom")

    for spine in ax.spines.values():
        spine.set_color("#c9b27a")
        spine.set_alpha(0.22)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)


def render_card(flare: dict, xray_times: list, xray_fluxes: list) -> str:
    """Render a card using pre-fetched xray data."""
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    draw = ImageDraw.Draw(template)

    flare_class = flare["max_class"]
    peak_dt = parse_time(flare["max_time"])
    start_dt = parse_time(flare["begin_time"]) if flare.get("begin_time") else peak_dt
    end_dt = parse_time(flare["end_time"]) if flare.get("end_time") else None

    start_str = start_dt.strftime("%H:%M UTC")
    peak_str = peak_dt.strftime("%H:%M UTC")
    end_str = end_dt.strftime("%H:%M UTC") if end_dt else "ONGOING"
    duration_str = (
        f"{int((end_dt - start_dt).total_seconds() // 60)} Minutes" if end_dt else "ONGOING"
    )
    satellite = f"GOES-{flare.get('satellite', '??')}"

    font_class = load_font(120, True)
    font_info = load_font(28)
    font_chart = load_font(22, True)
    font_footer = load_font(18)
    fill_color = class_color(flare_class)

    bbox = draw.textbbox((0, 0), flare_class, font=font_class)
    x_class = (template.width - (bbox[2] - bbox[0])) // 2
    draw_glow_text(template, (x_class, 520), flare_class, font_class, fill_color)

    line1 = f"Start : {start_str}      Peak : {peak_str}"
    line2 = f"End : {end_str}      Duration : {duration_str}"
    x1 = (template.width - draw.textbbox((0, 0), line1, font=font_info)[2]) // 2
    x2 = (template.width - draw.textbbox((0, 0), line2, font=font_info)[2]) // 2
    draw.text((x1, 690), line1, font=font_info, fill=(230, 230, 230, 255))
    draw.text((x2, 735), line2, font=font_info, fill=(230, 230, 230, 255))

    chart_title = f"{satellite} X-Ray Flux (-30m / +30m around flare)"
    x_chart = (template.width - draw.textbbox((0, 0), chart_title, font=font_chart)[2]) // 2
    draw.text((x_chart, 805), chart_title, font=font_chart, fill=(240, 220, 170, 255))

    chart_path = os.path.join(CHARTS_DIR, f"chart_{peak_dt.strftime('%Y%m%d_%H%M')}.png")
    build_chart(flare, chart_path, xray_times, xray_fluxes)

    chart_img = Image.open(chart_path).convert("RGBA")
    chart_img.thumbnail((870, 395))
    template.alpha_composite(
        chart_img,
        (105 + (870 - chart_img.width) // 2, 865 + (395 - chart_img.height) // 2),
    )

    footer = f"Data: NOAA SWPC • Satellite: {satellite}"
    x_footer = (template.width - draw.textbbox((0, 0), footer, font=font_footer)[2]) // 2
    draw.text((x_footer, template.height - 36), footer, font=font_footer, fill=(220, 220, 220, 255))

    filename = f"flare_{peak_dt.strftime('%Y%m%d_%H%M')}_{flare_class.replace('.', 'p')}.png"
    output_path = os.path.join(CARDS_DIR, filename)
    template.convert("RGB").save(output_path)
    return output_path


def prune_old_cards(keep: int = 5):
    # Sort by the date embedded in the filename (flare_YYYYMMDD_HHMM_CLASS.png)
    # so backfill regenerating old cards never bumps out genuinely recent ones
    files = sorted(glob.glob(os.path.join(CARDS_DIR, "flare_*.png")), reverse=True)
    for old_file in files[keep:]:
        os.remove(old_file)
        print(f"[prune] Removed: {old_file}")


def main():
    ensure_dirs()

    # ── Fetch everything ONCE upfront ────────────────────────────────────
    all_flares = fetch_all_flares()
    xray_times, xray_fluxes = fetch_xray_series()
    processed_ids = load_processed_ids()
    # ─────────────────────────────────────────────────────────────────────

    new_flares = [f for f in all_flares if flare_id(f) not in processed_ids]

    if not new_flares:
        print("No new flares. Nothing to do.")
        return

    print(f"Found {len(new_flares)} new flare(s).")
    generated = []
    for flare in new_flares:
        fid = flare_id(flare)
        try:
            output = render_card(flare, xray_times, xray_fluxes)
            processed_ids.add(fid)
            generated.append(output)
            print(f"  v Generated: {output}")
        except Exception as exc:
            print(f"  x Failed for {fid}: {exc}")

    save_processed_ids(processed_ids)
    prune_old_cards()
    print(f"\nDone. {len(generated)}/{len(new_flares)} card(s) generated.")


if __name__ == "__main__":
    main()
