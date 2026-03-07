import os
import glob
from datetime import datetime, timedelta

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter


FLARE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"

TEMPLATE_PATH = "template.png"
CARDS_DIR = "cards"
CHARTS_DIR = "charts"
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "last_flare_id.txt")


def ensure_dirs():
    os.makedirs(CARDS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def flare_id(flare: dict) -> str:
    return f"{flare.get('begin_time','')}_{flare.get('max_time','')}_{flare.get('max_class','')}_{flare.get('satellite','')}"


def fetch_latest_flare() -> dict:
    response = requests.get(FLARE_URL, timeout=20, headers={"User-Agent": "substorm-flare-bot"})
    response.raise_for_status()
    data = response.json()

    valid = [f for f in data if f.get("max_time") and f.get("max_class")]
    if not valid:
        raise RuntimeError("No valid flare events found.")

    valid.sort(key=lambda f: f["max_time"])
    return valid[-1]


def fetch_xray_series():
    response = requests.get(XRAY_URL, timeout=20, headers={"User-Agent": "substorm-flare-bot"})
    response.raise_for_status()
    data = response.json()

    times = []
    fluxes = []

    for row in data:
        energy = str(row.get("energy", "")).strip()
        flux = row.get("flux")
        time_tag = row.get("time_tag")

        if energy != "0.1-0.8nm":
            continue
        if flux is None or time_tag is None:
            continue

        try:
            t = parse_time(time_tag)
            f = float(flux)
        except Exception:
            continue

        # Reject broken values
        if f <= 0:
            continue

        times.append(t)
        fluxes.append(f)

    if not times:
        raise RuntimeError("No valid X-ray time series data found.")

    # Remove obvious instrument failures and impossible dips/spikes
    clean_times = []
    clean_fluxes = []

    for i, f in enumerate(fluxes):
        # reject values that are too low for realistic flare plotting
        if f < 1e-8:
            continue

        prev_f = fluxes[i - 1] if i > 0 else None
        next_f = fluxes[i + 1] if i < len(fluxes) - 1 else None

        bad_point = False

        if prev_f is not None and next_f is not None:
            local_ref = (prev_f + next_f) / 2.0

            # giant downward glitch
            if local_ref > 0 and f < local_ref / 20:
                bad_point = True

            # giant upward glitch
            if local_ref > 0 and f > local_ref * 20:
                bad_point = True

        if not bad_point:
            clean_times.append(times[i])
            clean_fluxes.append(f)

    if not clean_times:
        raise RuntimeError("All X-ray points were filtered out.")

    # Light smoothing
    flux_arr = np.array(clean_fluxes)
    if len(flux_arr) >= 5:
        flux_arr = np.convolve(flux_arr, np.ones(5) / 5, mode="same")

    return clean_times, flux_arr.tolist()


def read_last_flare_id() -> str:
    if not os.path.exists(STATE_FILE):
        return ""
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_last_flare_id(value: str):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        f.write(value)


def class_color(flare_class: str):
    letter = (flare_class or "C")[0].upper()
    if letter == "X":
        return (255, 130, 40)
    if letter == "M":
        return (255, 170, 60)
    if letter == "C":
        return (255, 220, 120)
    if letter == "B":
        return (180, 210, 255)
    return (230, 230, 230)


def draw_glow_text(base: Image.Image, position, text: str, font, fill_rgb):
    glow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.text(position, text, font=font, fill=fill_rgb + (120,))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(24))
    base.alpha_composite(glow_layer)

    glow_layer2 = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow_draw2 = ImageDraw.Draw(glow_layer2)
    glow_draw2.text(position, text, font=font, fill=(255, 240, 180, 170))
    glow_layer2 = glow_layer2.filter(ImageFilter.GaussianBlur(10))
    base.alpha_composite(glow_layer2)

    final_draw = ImageDraw.Draw(base)
    final_draw.text(position, text, font=font, fill=(255, 235, 170, 255))


def build_chart(flare: dict, output_path: str):
    times, fluxes = fetch_xray_series()
    peak_time = parse_time(flare["max_time"])
    flare_class = flare["max_class"]

    now_utc = datetime.now().astimezone(peak_time.tzinfo)

    # Show 3 hours before flare peak up to now,
    # but cap post-flare region at 90 minutes after peak
    window_start = peak_time - timedelta(hours=3)
    ideal_window_end = peak_time + timedelta(minutes=90)
    window_end = min(now_utc, ideal_window_end)

    if window_end <= window_start:
        window_end = max(times[-1], peak_time)

    filtered = [
        (t, f) for t, f in zip(times, fluxes)
        if window_start <= t <= window_end
    ]

    # fallback if filtering gets too narrow
    if len(filtered) < 10:
        filtered = list(zip(times, fluxes))

    plot_times = [t for t, _ in filtered]
    plot_fluxes = [f for _, f in filtered]

    plt.figure(figsize=(8.2, 4.4))
    ax = plt.gca()
    ax.set_facecolor((0.03, 0.03, 0.07))
    plt.gcf().patch.set_facecolor((0.03, 0.03, 0.07))

    # Glow line and main line
    ax.plot(plot_times, plot_fluxes, linewidth=6, color="#ffe9a8", alpha=0.12)
    ax.plot(plot_times, plot_fluxes, linewidth=2.5, color="#ffe9a8")

    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-4)

    thresholds = [
        (1e-8, "A", "#888888"),
        (1e-7, "B", "#9cc9ff"),
        (1e-6, "C", "#ffe57a"),
        (1e-5, "M", "#ffb347"),
        (1e-4, "X", "#ff7043"),
    ]

    for value, label, color in thresholds:
        ax.axhline(value, color=color, linewidth=0.8, alpha=0.55)

    ax.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    ax.set_yticklabels(["A", "B", "C", "M", "X"], color="#f5dfb0", fontsize=11)

    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.35, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.12, color="#ffffff")

    ax.tick_params(axis="x", colors="#f5dfb0", labelsize=10)
    ax.tick_params(axis="y", colors="#f5dfb0", labelsize=11)

    if plot_times[0] <= peak_time <= plot_times[-1]:
        nearest_idx = min(range(len(plot_times)), key=lambda i: abs((plot_times[i] - peak_time).total_seconds()))
        peak_flux = plot_fluxes[nearest_idx]

        ax.axvline(peak_time, color="#ffb347", linewidth=1.0, alpha=0.7)
        ax.scatter([peak_time], [peak_flux], s=50, color="#ffd27a", zorder=5)
        ax.text(
            peak_time,
            peak_flux * 1.35,
            flare_class,
            color="#ffd27a",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    for spine in ax.spines.values():
        spine.set_color("#c9b27a")
        spine.set_alpha(0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()


def prune_old_cards():
    files = sorted(glob.glob(os.path.join(CARDS_DIR, "*.png")), key=os.path.getmtime, reverse=True)
    for old_file in files[10:]:
        os.remove(old_file)


def render_card(flare: dict):
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    draw = ImageDraw.Draw(template)

    flare_class = flare["max_class"]
    start_dt = parse_time(flare["begin_time"])
    peak_dt = parse_time(flare["max_time"])
    end_dt = parse_time(flare["end_time"]) if flare.get("end_time") else None

    start_str = start_dt.strftime("%H:%M UTC")
    peak_str = peak_dt.strftime("%H:%M UTC")
    end_str = end_dt.strftime("%H:%M UTC") if end_dt else "ONGOING"

    if end_dt:
        duration_minutes = int((end_dt - start_dt).total_seconds() // 60)
        duration_str = f"{duration_minutes} Minutes"
    else:
        duration_str = "ONGOING"

    satellite = f"GOES-{flare.get('satellite', '??')}"

    font_class = load_font(120, bold=True)
    font_info = load_font(28, bold=False)
    font_chart = load_font(22, bold=True)
    font_footer = load_font(18, bold=False)

    fill_color = class_color(flare_class)

    # Flare class
    bbox = draw.textbbox((0, 0), flare_class, font=font_class)
    text_w = bbox[2] - bbox[0]
    x_class = (template.width - text_w) // 2
    y_class = 520
    draw_glow_text(template, (x_class, y_class), flare_class, font_class, fill_color)

    # Info lines
    line1 = f"Start : {start_str}      Peak : {peak_str}"
    line2 = f"End : {end_str}      Duration : {duration_str}"

    bbox1 = draw.textbbox((0, 0), line1, font=font_info)
    bbox2 = draw.textbbox((0, 0), line2, font=font_info)

    x1 = (template.width - (bbox1[2] - bbox1[0])) // 2
    x2 = (template.width - (bbox2[2] - bbox2[0])) // 2

    draw.text((x1, 690), line1, font=font_info, fill=(230, 230, 230, 255))
    draw.text((x2, 735), line2, font=font_info, fill=(230, 230, 230, 255))

    # Chart title
    chart_title = f"{satellite} X-Ray Flux (Last 6 Hours)"
    bbox_chart = draw.textbbox((0, 0), chart_title, font=font_chart)
    x_chart = (template.width - (bbox_chart[2] - bbox_chart[0])) // 2
    draw.text((x_chart, 805), chart_title, font=font_chart, fill=(240, 220, 170, 255))

    # Build and paste chart
    chart_path = os.path.join(CHARTS_DIR, "latest_chart.png")
    build_chart(flare, chart_path)

    chart_img = Image.open(chart_path).convert("RGBA")

    # Placement box tuned to your current template
    target_x = 105
    target_y = 865
    target_w = 870
    target_h = 395

    chart_img.thumbnail((target_w, target_h))
    paste_x = target_x + (target_w - chart_img.width) // 2
    paste_y = target_y + (target_h - chart_img.height) // 2

    template.alpha_composite(chart_img, (paste_x, paste_y))

    # Footer
    footer = f"Data: NOAA SWPC • Satellite: {satellite}"
    bbox_footer = draw.textbbox((0, 0), footer, font=font_footer)
    x_footer = (template.width - (bbox_footer[2] - bbox_footer[0])) // 2
    draw.text((x_footer, template.height - 36), footer, font=font_footer, fill=(220, 220, 220, 255))

    filename = f"flare_{peak_dt.strftime('%Y%m%d_%H%M')}_{flare_class.replace('.', 'p')}.png"
    output_path = os.path.join(CARDS_DIR, filename)
    template.convert("RGB").save(output_path)
    return output_path


def main():
    ensure_dirs()
    latest = fetch_latest_flare()
    current_id = flare_id(latest)
    previous_id = read_last_flare_id()

    if current_id == previous_id:
        print("No new flare detected. Nothing to do.")
        return

    output = render_card(latest)
    write_last_flare_id(current_id)
    prune_old_cards()
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
