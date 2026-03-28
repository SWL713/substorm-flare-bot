"""
Substorm Flare Bot — generate_flare_card.py

Two card modes:
  IN PROGRESS — fires immediately when live X-ray flux crosses M1.0 and stays
                there for at least 3 minutes. Only triggers for M and X class.
  COMPLETED   — fires once NOAA finalises the event in their 7-day flares feed.

In-progress cards are pruned once the matching completed card is generated.
"""

import json
import os
import glob
import re
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ── Sources ───────────────────────────────────────────────────────────────────
FLARE_URLS = [
    "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
    "https://services.swpc.noaa.gov/json/goes/secondary/xray-flares-7-day.json",
]
XRAY_URLS = [
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
    "https://services.swpc.noaa.gov/json/goes/secondary/xrays-6-hour.json",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
TEMPLATE_PATH      = "template.png"
CARDS_DIR          = "cards"
CHARTS_DIR         = "charts"
DATA_DIR           = "data"
STATE_FILE         = os.path.join(DATA_DIR, "processed_flares.json")
LEGACY_STATE_FILE  = os.path.join(DATA_DIR, "last_flare_id.txt")
ACTIVE_EVENTS_FILE = os.path.join(DATA_DIR, "active_events.json")

# ── In-progress settings (M and X only) ───────────────────────────────────────
INPROGRESS_MIN_FLUX      = 1e-5   # M1.0 — minimum to trigger a live card
INPROGRESS_SUSTAIN_MINS  = 3      # must stay above threshold this long
ACTIVE_EVENT_MAX_AGE_HRS = 12     # drop unresolved active events after this


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in (CARDS_DIR, CHARTS_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)


def load_font(size, bold=False):
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


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def flux_to_class(flux):
    if   flux >= 1e-4: return f"X{flux / 1e-4:.1f}"
    elif flux >= 1e-5: return f"M{flux / 1e-5:.1f}"
    elif flux >= 1e-6: return f"C{flux / 1e-6:.1f}"
    elif flux >= 1e-7: return f"B{flux / 1e-7:.1f}"
    else:              return f"A{flux / 1e-8:.1f}"


def class_color(flare_class):
    letter = (flare_class or "C")[0].upper()
    return {
        "X": (255, 100, 30),
        "M": (255, 170, 60),
        "C": (255, 220, 120),
        "B": (180, 210, 255),
    }.get(letter, (230, 230, 230))


def draw_glow_text(base, position, text, font, fill_rgb):
    for blur, alpha in ((24, 120), (10, 170)):
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text(position, text, font=font, fill=fill_rgb + (alpha,))
        base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(base).text(position, text, font=font, fill=(255, 235, 170, 255))


def draw_status_banner(template, text, bg_rgba, text_rgba, y=462, height=44):
    overlay = Image.new("RGBA", template.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([(0, y), (template.width, y + height)], fill=bg_rgba)
    template.alpha_composite(overlay)
    font  = load_font(22, bold=True)
    draw2 = ImageDraw.Draw(template)
    bbox  = draw2.textbbox((0, 0), text, font=font)
    x     = (template.width - (bbox[2] - bbox[0])) // 2
    draw2.text((x, y + (height - (bbox[3] - bbox[1])) // 2), text, font=font, fill=text_rgba)


# ── State ─────────────────────────────────────────────────────────────────────

def flare_id(flare):
    return (f"{flare.get('begin_time', '')}"
            f"_{flare.get('max_time', '')}"
            f"_{flare.get('max_class', '')}")


def _normalize_id(fid):
    return re.sub(r"_\d+$", "", fid)


def load_processed_ids():
    ids = set()
    if not os.path.exists(STATE_FILE) and os.path.exists(LEGACY_STATE_FILE):
        with open(LEGACY_STATE_FILE, "r", encoding="utf-8") as fh:
            legacy_id = fh.read().strip()
        if legacy_id:
            ids.add(_normalize_id(legacy_id))
        save_processed_ids(ids)
        print(f"[migrate] Promoted legacy ID: {legacy_id}")
        return ids
    if not os.path.exists(STATE_FILE):
        return ids
    with open(STATE_FILE, "r", encoding="utf-8") as fh:
        try:
            ids = set(_normalize_id(x) for x in json.load(fh))
        except (json.JSONDecodeError, TypeError):
            pass
    return ids


def save_processed_ids(ids):
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(sorted(ids)[-500:], fh, indent=2)


def load_active_events():
    if not os.path.exists(ACTIVE_EVENTS_FILE):
        return []
    with open(ACTIVE_EVENTS_FILE, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except (json.JSONDecodeError, TypeError):
            return []


def save_active_events(events):
    with open(ACTIVE_EVENTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(events, fh, indent=2)


# ── Network ───────────────────────────────────────────────────────────────────

def _fetch_json(url):
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "substorm-flare-bot"})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[warn] Could not fetch {url}: {exc}")
        return []


def fetch_all_flares():
    seen, merged = set(), []
    for url in FLARE_URLS:
        for f in _fetch_json(url):
            if not (f.get("max_time") and f.get("max_class")):
                continue
            fid = flare_id(f)
            if fid not in seen:
                seen.add(fid)
                merged.append(f)
    if not merged:
        raise RuntimeError("No valid flare data from any source.")
    merged.sort(key=lambda f: f["max_time"])
    return merged


def fetch_xray_series():
    for url in XRAY_URLS:
        data = _fetch_json(url)
        times, fluxes = [], []
        for row in data:
            if str(row.get("energy", "")).strip() != "0.1-0.8nm":
                continue
            flux, time_tag = row.get("flux"), row.get("time_tag")
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


# ── M/X detection — scans full 6-hour history ─────────────────────────────────

def find_mx_events_in_series(xray_times, xray_fluxes, lookback_hours=6):
    """
    Scans the entire X-ray time series (up to lookback_hours) for distinct M or X
    class events — whether currently active OR already peaked and declining.

    Returns a list of event dicts, one per distinct M/X burst found, most recent first.
    Each event has the same shape as the old detect_active_mx_flare() return value.

    Logic:
      - Walk the series looking for contiguous regions where flux >= M1.0.
      - Each such region that lasted >= INPROGRESS_SUSTAIN_MINS is one event.
      - Events separated by flux dropping below C-level are treated as distinct.
    """
    if not xray_times or not xray_fluxes:
        return []

    now     = xray_times[-1]
    cutoff  = now - timedelta(hours=lookback_hours)
    indices = [i for i, t in enumerate(xray_times) if t >= cutoff]
    if not indices:
        return []

    events = []
    in_burst     = False
    burst_start  = None
    burst_indices = []

    for i in indices:
        flux = xray_fluxes[i]
        if flux >= INPROGRESS_MIN_FLUX:
            if not in_burst:
                in_burst = True
                burst_start = i
                burst_indices = []
            burst_indices.append(i)
        else:
            if in_burst:
                # Close out this burst
                _maybe_add_event(events, xray_times, xray_fluxes, burst_indices, now)
                in_burst = False
                burst_indices = []

    # Handle burst still active at end of series
    if in_burst and burst_indices:
        _maybe_add_event(events, xray_times, xray_fluxes, burst_indices, now)

    events.sort(key=lambda e: e["peak_time"], reverse=True)
    return events


def _maybe_add_event(events, xray_times, xray_fluxes, burst_indices, now):
    """Build an event dict from a burst segment if it meets the minimum duration."""
    duration_mins = (
        xray_times[burst_indices[-1]] - xray_times[burst_indices[0]]
    ).total_seconds() / 60

    if duration_mins < INPROGRESS_SUSTAIN_MINS:
        return  # Too brief — noise, not a real flare

    peak_flux  = max(xray_fluxes[i] for i in burst_indices)
    peak_idx   = next(i for i in burst_indices if xray_fluxes[i] == peak_flux)
    peak_time  = xray_times[peak_idx]
    peak_class = flux_to_class(peak_flux)

    # Estimate event start: look back before the burst for the last sub-C reading
    event_start = xray_times[burst_indices[0]]
    for i in range(burst_indices[0], -1, -1):
        if xray_fluxes[i] < 1e-6:
            event_start = xray_times[min(i + 1, len(xray_times) - 1)]
            break

    current_flux  = xray_fluxes[burst_indices[-1]]
    still_active  = burst_indices[-1] == len(xray_fluxes) - 1

    status = "active" if still_active else "peaked"
    print(f"[scan] {peak_class} event found — peak {peak_time.strftime('%H:%M UTC')}, "
          f"duration {duration_mins:.0f} min, status: {status}")

    events.append({
        "event_start":    event_start.isoformat(),
        "peak_time":      peak_time.isoformat(),
        "peak_flux":      peak_flux,
        "peak_class":     peak_class,
        "current_flux":   current_flux,
        "current_class":  flux_to_class(current_flux),
        "last_updated":   now.isoformat(),
        "still_active":   still_active,
        "card_filename":  None,
    })


def active_event_already_carded(event_start_dt, active_events):
    for ae in active_events:
        ae_start = datetime.fromisoformat(ae["event_start"])
        if abs((ae_start - event_start_dt).total_seconds()) < 30 * 60:
            return ae
    return None


def find_active_event_for_flare(flare, active_events):
    if not flare.get("begin_time"):
        return None
    begin = parse_time(flare["begin_time"])
    for ae in active_events:
        ae_start = datetime.fromisoformat(ae["event_start"])
        if abs((ae_start - begin).total_seconds()) < 30 * 60:
            return ae
    return None


# ── Chart ─────────────────────────────────────────────────────────────────────

def _build_chart(peak_time, flare_class, chart_path, xray_times, xray_fluxes,
                  inprogress=False, event_start=None):
    now_utc = datetime.now().astimezone(peak_time.tzinfo)

    # ── Adaptive window ───────────────────────────────────────────────────────
    # Pre-peak  : 2× rise time (floor 15 min) — shows the full ramp
    # Post-peak : 1.5× decay time (floor 30 min) — shows the tail
    if event_start is not None:
        rise_mins = max(15, (peak_time - event_start).total_seconds() / 60 * 2)
    else:
        rise_mins = 30

    if inprogress:
        decay_mins = max(30, (now_utc - peak_time).total_seconds() / 60 * 1.5)
    else:
        decay_mins = 30
        peak_idx = next((i for i, t in enumerate(xray_times) if t >= peak_time), None)
        if peak_idx is not None:
            for i in range(peak_idx, len(xray_fluxes)):
                if xray_fluxes[i] < 1e-6:
                    decay_mins = max(30, (xray_times[i] - peak_time).total_seconds() / 60 * 1.5)
                    break
            else:
                decay_mins = max(30, (xray_times[-1] - peak_time).total_seconds() / 60 * 1.5)

    window_start = peak_time - timedelta(minutes=rise_mins)
    window_end   = min(now_utc, peak_time + timedelta(minutes=decay_mins))
    if window_end <= window_start:
        window_start = now_utc - timedelta(hours=1)
        window_end   = now_utc

    filtered = [(t, f) for t, f in zip(xray_times, xray_fluxes)
                if window_start <= t <= window_end]
    if len(filtered) < 10:
        filtered = list(zip(xray_times, xray_fluxes))

    plot_times  = [t for t, _ in filtered]
    plot_fluxes = [f for _, f in filtered]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    line_color = "#ff9933" if inprogress else "#ffe9a8"
    ax.plot(plot_times, plot_fluxes, linewidth=8,   color=line_color, alpha=0.10, solid_capstyle="round")
    ax.plot(plot_times, plot_fluxes, linewidth=2.5, color=line_color, solid_capstyle="round")
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
        max_idx  = max(range(len(plot_fluxes)), key=lambda i: plot_fluxes[i])
        max_time = plot_times[max_idx]
        max_flux = plot_fluxes[max_idx]
        ax.axvline(max_time, color="#ffb347", linewidth=1.0, alpha=0.6)
        ax.scatter([max_time], [max_flux], s=50, color="#ffd27a", zorder=5)
        ax.text(max_time, max_flux * 1.35, flare_class,
                color="#ffd27a", fontsize=13, fontweight="bold", ha="left", va="bottom")

    if inprogress and plot_times:
        ax.axvline(plot_times[-1], color="#ff4444", linewidth=1.4, linestyle="--", alpha=0.85)
        ax.text(plot_times[-1], 1.3e-5, " NOW", color="#ff6666",
                fontsize=10, fontweight="bold", ha="left", va="bottom")

    for spine in ax.spines.values():
        spine.set_color("#c9b27a")
        spine.set_alpha(0.22)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)


def _composite_chart(template, chart_path):
    chart_img = Image.open(chart_path).convert("RGBA")
    chart_img.thumbnail((870, 395))
    template.alpha_composite(
        chart_img,
        (105 + (870 - chart_img.width) // 2, 865 + (395 - chart_img.height) // 2),
    )


# ── Card renderers ────────────────────────────────────────────────────────────

def render_inprogress_card(active, xray_times, xray_fluxes):
    """Orange IN PROGRESS card for a live M or X flare."""
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    draw     = ImageDraw.Draw(template)

    peak_class    = active["peak_class"]
    peak_time     = parse_time(active["peak_time"])
    start_time    = parse_time(active["event_start"])
    current_class = active["current_class"]
    fill_color    = class_color(peak_class)

    still_active = active.get("still_active", True)
    banner_text  = ("  IN PROGRESS  -  UPDATING LIVE"
                    if still_active else
                    "  PEAKED  -  AWAITING NOAA CONFIRMATION")
    footer_text  = ("Data: NOAA SWPC  -  Status: IN PROGRESS  -  card will update"
                    if still_active else
                    "Data: NOAA SWPC  -  Status: PEAKED  -  final card coming soon")

    draw_status_banner(template,
                       text=banner_text,
                       bg_rgba=(160, 80, 0, 210),
                       text_rgba=(255, 210, 120, 255))

    font_class  = load_font(120, True)
    font_info   = load_font(28)
    font_chart  = load_font(22, True)
    font_footer = load_font(18)

    bbox    = draw.textbbox((0, 0), peak_class, font=font_class)
    x_class = (template.width - (bbox[2] - bbox[0])) // 2
    draw_glow_text(template, (x_class, 520), peak_class, font_class, fill_color)

    now_str   = parse_time(active["last_updated"]).strftime("%H:%M UTC")
    start_str = start_time.strftime("%H:%M UTC")

    line1 = f"Start : {start_str}      Current : {current_class}"
    line2 = f"Peak so far : {peak_class}      Last update : {now_str}"
    x1 = (template.width - draw.textbbox((0, 0), line1, font=font_info)[2]) // 2
    x2 = (template.width - draw.textbbox((0, 0), line2, font=font_info)[2]) // 2
    draw.text((x1, 690), line1, font=font_info, fill=(230, 230, 230, 255))
    draw.text((x2, 735), line2, font=font_info, fill=(230, 230, 230, 255))

    chart_title = "GOES X-Ray Flux  -  Live  (final card issued when event closes)"
    x_chart = (template.width - draw.textbbox((0, 0), chart_title, font=font_chart)[2]) // 2
    draw.text((x_chart, 805), chart_title, font=font_chart, fill=(255, 190, 80, 255))

    chart_path = os.path.join(CHARTS_DIR,
                               f"chart_live_{start_time.strftime('%Y%m%d_%H%M')}.png")
    _build_chart(peak_time, peak_class, chart_path, xray_times, xray_fluxes,
                  inprogress=True, event_start=start_time)
    _composite_chart(template, chart_path)

    x_footer = (template.width - draw.textbbox((0, 0), footer_text, font=font_footer)[2]) // 2
    draw.text((x_footer, template.height - 36), footer_text,
               font=font_footer, fill=(255, 180, 80, 255))

    filename    = (f"flare_{start_time.strftime('%Y%m%d_%H%M')}"
                   f"_{peak_class.replace('.', 'p')}_inprogress.png")
    output_path = os.path.join(CARDS_DIR, filename)
    template.convert("RGB").save(output_path)
    return output_path


def render_card(flare, xray_times, xray_fluxes):
    """Green CONFIRMED / FINAL DATA card from a finalised NOAA event."""
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    draw     = ImageDraw.Draw(template)

    flare_class = flare["max_class"]
    peak_dt     = parse_time(flare["max_time"])
    start_dt    = parse_time(flare["begin_time"]) if flare.get("begin_time") else peak_dt
    end_dt      = parse_time(flare["end_time"])   if flare.get("end_time")   else None
    satellite   = f"GOES-{flare.get('satellite', '??')}"
    fill_color  = class_color(flare_class)

    start_str    = start_dt.strftime("%H:%M UTC")
    peak_str     = peak_dt.strftime("%H:%M UTC")
    end_str      = end_dt.strftime("%H:%M UTC") if end_dt else "ONGOING"
    duration_str = (f"{int((end_dt - start_dt).total_seconds() // 60)} Minutes"
                    if end_dt else "ONGOING")

    draw_status_banner(template,
                       text="  CONFIRMED  -  FINAL DATA",
                       bg_rgba=(30, 120, 60, 210),
                       text_rgba=(200, 255, 210, 255))

    font_class  = load_font(120, True)
    font_info   = load_font(28)
    font_chart  = load_font(22, True)
    font_footer = load_font(18)

    bbox    = draw.textbbox((0, 0), flare_class, font=font_class)
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
    _build_chart(peak_dt, flare_class, chart_path, xray_times, xray_fluxes,
                  inprogress=False, event_start=start_dt)
    _composite_chart(template, chart_path)

    footer = f"Data: NOAA SWPC  -  Satellite: {satellite}"
    x_footer = (template.width - draw.textbbox((0, 0), footer, font=font_footer)[2]) // 2
    draw.text((x_footer, template.height - 36), footer,
               font=font_footer, fill=(220, 220, 220, 255))

    filename    = f"flare_{peak_dt.strftime('%Y%m%d_%H%M')}_{flare_class.replace('.', 'p')}.png"
    output_path = os.path.join(CARDS_DIR, filename)
    template.convert("RGB").save(output_path)
    return output_path


# ── Pruning ───────────────────────────────────────────────────────────────────

def prune_old_cards(keep=5):
    completed = sorted(
        [f for f in glob.glob(os.path.join(CARDS_DIR, "flare_*.png"))
         if "_inprogress" not in f],
        reverse=True,
    )
    for old in completed[keep:]:
        os.remove(old)
        print(f"[prune] Removed: {old}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    all_flares             = fetch_all_flares()
    xray_times, xray_fluxes = fetch_xray_series()
    processed_ids          = load_processed_ids()
    active_events          = load_active_events()
    now_utc                = datetime.now(timezone.utc)

    # ── Expire stale active events ───────────────────────────────────────────
    fresh_active = []
    for ae in active_events:
        ae_dt = datetime.fromisoformat(ae["event_start"])
        if ae_dt.tzinfo is None:
            ae_dt = ae_dt.replace(tzinfo=timezone.utc)
        age_hrs = (now_utc - ae_dt).total_seconds() / 3600
        if age_hrs <= ACTIVE_EVENT_MAX_AGE_HRS:
            fresh_active.append(ae)
        else:
            print(f"[expire] Dropping stale active event: {ae['event_start']}")
            if ae.get("card_filename"):
                stale = os.path.join(CARDS_DIR, ae["card_filename"])
                if os.path.exists(stale):
                    os.remove(stale)
                    print(f"[expire] Removed stale card: {ae['card_filename']}")
    active_events = fresh_active

    generated = []

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Completed cards from finalised NOAA events
    # ═══════════════════════════════════════════════════════════════════════
    new_flares = [f for f in all_flares if flare_id(f) not in processed_ids]
    if new_flares:
        print(f"Found {len(new_flares)} new completed flare(s).")

    for flare in new_flares:
        fid = flare_id(flare)
        try:
            output = render_card(flare, xray_times, xray_fluxes)
            processed_ids.add(fid)
            generated.append(output)
            print(f"  [done] Completed card: {output}")

            # Remove the matching in-progress card
            matched = find_active_event_for_flare(flare, active_events)
            if matched:
                if matched.get("card_filename"):
                    ip_path = os.path.join(CARDS_DIR, matched["card_filename"])
                    if os.path.exists(ip_path):
                        os.remove(ip_path)
                        print(f"  [swap] Removed in-progress card: {matched['card_filename']}")
                active_events = [ae for ae in active_events if ae is not matched]

        except Exception as exc:
            print(f"  [fail] {fid}: {exc}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — In-progress/retrospective cards from X-ray history (M/X only)
    # Scans the full 6-hour series — catches events whether currently active
    # or already peaked and awaiting NOAA confirmation.
    # ═══════════════════════════════════════════════════════════════════════
    mx_events = find_mx_events_in_series(xray_times, xray_fluxes)

    if not mx_events:
        print("[scan] No M/X events found in X-ray history.")

    for active_flare in mx_events:
        event_start_dt = datetime.fromisoformat(active_flare["event_start"])
        already        = active_event_already_carded(event_start_dt, active_events)

        if not already:
            try:
                output = render_inprogress_card(active_flare, xray_times, xray_fluxes)
                active_flare["card_filename"] = os.path.basename(output)
                active_events.append(active_flare)
                generated.append(output)
                status = "active" if active_flare.get("still_active") else "peaked — awaiting NOAA"
                print(f"  [card] In-progress card ({status}): {output}")
            except Exception as exc:
                print(f"  [fail] In-progress render: {exc}")
        else:
            print(f"[scan] Card already exists for event at {active_flare['event_start']} — skipping.")

    save_processed_ids(processed_ids)
    save_active_events(active_events)

    if not generated:
        print("Nothing new to commit.")
        return

    prune_old_cards()
    print(f"\nDone. {len(generated)} card(s) generated.")


if __name__ == "__main__":
    main()
