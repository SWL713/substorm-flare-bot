"""
Substorm Flare Bot — generate_flare_card.py

Per M/X flare, the bot sends at most TWO Telegram messages:

  1. RISING TEXT ALERT — short text-only line when live X-ray flux first
     crosses M1 and is still climbing. One per "alert window" (90-min
     cooldown), so a noisy multi-peak flare doesn't spam.
  2. CARD — full image card once NOAA finalises the event in the 7-day
     xray-flares feed (the canonical source of truth). One per unique
     flare ID forever, tracked in processed_flares.json.

The previous in-progress card path is gone — it duplicated NOAA's job
and the per-event tracker drifted between runs, causing 3-4 cards per
real flare. Cards now come exclusively from NOAA.
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
RISING_ALERT_FILE  = os.path.join(DATA_DIR, "rising_alert_state.json")
RISING_ALERT_COOLDOWN_MIN = 90  # minutes between rising-flare text alerts

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
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
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


def _enrich_from_donki(flares):
    """Enrich flares with location + active region from NASA DONKI."""
    try:
        start = (datetime.now(timezone.utc) - timedelta(days=8)).strftime('%Y-%m-%d')
        end = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        url = f'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate={start}&endDate={end}'
        donki = _fetch_json(url)
        if not donki:
            return
        for flare in flares:
            if flare.get('location') or not flare.get('max_time'):
                continue
            try:
                fp = datetime.fromisoformat(flare['max_time'].replace('Z', '+00:00'))
            except Exception:
                continue
            for df in donki:
                if not isinstance(df, dict):
                    continue
                try:
                    dp = datetime.fromisoformat(str(df.get('peakTime', '')).replace('Z', '+00:00'))
                    if abs((dp - fp).total_seconds()) < 600:
                        if df.get('sourceLocation'):
                            flare['location'] = df['sourceLocation']
                        if df.get('activeRegionNum'):
                            # NOAA SWPC public AR format is 4-digit. DONKI
                            # sometimes returns 5-digit with a leading '1'
                            # (cycle namespace) — strip it so 14220 → 4220.
                            try:
                                ar_n = int(df['activeRegionNum'])
                                ar_s = str(ar_n)
                                if len(ar_s) == 5 and ar_s[0] == '1':
                                    ar_n = int(ar_s[1:])
                                flare['active_region'] = ar_n
                            except (ValueError, TypeError):
                                flare['active_region'] = df['activeRegionNum']
                        print(f"  [donki] Enriched {flare['max_class']} with loc={flare.get('location')} AR={flare.get('active_region')}")
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"  [donki] Enrichment failed: {e}")


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
    _enrich_from_donki(merged)
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
            smoothed = np.convolve(arr, np.ones(5) / 5, mode="same")
            # Convolve in 'same' mode pads edges with zeros, causing artefacts
            # for the first and last 2 points — restore originals there
            smoothed[:2]  = arr[:2]
            smoothed[-2:] = arr[-2:]
            arr = smoothed

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
                  inprogress=False, event_start=None, sdo_mode=False):
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

    # When SDO is beside the chart, generate at squarer aspect ratio so text
    # stays the same size after thumbnail. Without SDO, original wide ratio.
    fig, ax = plt.subplots(figsize=(5.2, 4.4) if sdo_mode else (8.2, 4.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    line_color = "#ff9933" if inprogress else "#ffe9a8"
    ax.plot(plot_times, plot_fluxes, linewidth=8,   color=line_color, alpha=0.10, solid_capstyle="round")
    ax.plot(plot_times, plot_fluxes, linewidth=2.5, color=line_color, solid_capstyle="round")
    ax.set_yscale("log")

    # Y-axis: tight autoscale — 10% padding above max, 10% below min (log scale)
    if plot_fluxes:
        data_max = max(plot_fluxes)
        data_min = min(f for f in plot_fluxes if f > 0) if any(f > 0 for f in plot_fluxes) else 1e-8
        y_max = data_max * 2.0   # ~10% in log space above peak
        y_min = data_min / 2.0   # ~10% in log space below floor
    else:
        y_min, y_max = 1e-8, 1e-4
    ax.set_ylim(y_min, y_max)

    # Build tick labels — only show levels within visible range
    all_ticks   = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    all_labels  = ["A",  "B",  "C",  "M",  "X",  "X10"]
    all_colors  = ["#888888", "#9cc9ff", "#ffe57a", "#ffb347", "#ff7043", "#ff3300"]
    yticks, ylabels, tick_colors = [], [], []
    for val, lab, col in zip(all_ticks, all_labels, all_colors):
        if y_min <= val <= y_max:
            yticks.append(val); ylabels.append(lab); tick_colors.append(col)

    for value, color in zip(yticks, tick_colors):
        ax.axhline(value, color=color, linewidth=0.9, alpha=0.45)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color="#f5dfb0", fontsize=11)
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
        label_y = min(max_flux * 1.35, y_max * 0.75)
        ax.text(max_time, label_y, flare_class,
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


def _fetch_sdo_image(peak_time_str, location_str):
    """Fetch SDO/AIA 131A full sun image from Helioviewer and draw region box."""
    import math as _m
    try:
        peak_z = peak_time_str.replace('+00:00', 'Z').replace(' ', 'T')
        if not peak_z.endswith('Z'):
            peak_z += 'Z'
        url = (f'https://api.helioviewer.org/v2/takeScreenshot/'
               f'?date={peak_z}&imageScale=4.5&x0=0&y0=0&width=512&height=512'
               f'&layers=[SDO,AIA,AIA,131,1,100]&display=true&watermark=false')
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        sdo_img = Image.open(__import__('io').BytesIO(resp.content)).convert("RGBA")

        # Draw red bounding box at flare location
        if location_str and len(location_str) >= 4:
            try:
                lat_v = int(location_str[1:3]) * (1 if location_str[0] == 'N' else -1)
                lon_v = int(location_str[4:]) * (1 if location_str[3] == 'W' else -1)
                R = 960 / 4.5  # solar radius in pixels
                x = 256 + R * _m.sin(_m.radians(lon_v)) * _m.cos(_m.radians(lat_v))
                y = 256 - R * _m.sin(_m.radians(lat_v))
                box_size = 70
                draw = ImageDraw.Draw(sdo_img)
                draw.rectangle(
                    [x - box_size/2, y - box_size/2, x + box_size/2, y + box_size/2],
                    outline='#ff3333', width=3)
            except Exception as e:
                print(f"  [sdo] Could not draw region box: {e}")

        return sdo_img
    except Exception as e:
        print(f"  [sdo] Failed to fetch SDO image: {e}")
        return None


def _composite_chart(template, chart_path, sdo_img=None):
    chart_img = Image.open(chart_path).convert("RGBA")

    x_base, y_base = 105, 865
    panel_w, panel_h = 870, 395

    if sdo_img:
        # SDO is square, sized to panel height, with gap
        gap = 15
        sdo_size = panel_h
        chart_w = panel_w - sdo_size - gap
        chart_shift_left = 35
        chart_extra_h = 55  # extend chart taller

        # Chart: scale to fit left portion, taller than panel
        chart_img.thumbnail((chart_w, panel_h + chart_extra_h))
        template.alpha_composite(
            chart_img,
            (x_base - chart_shift_left + (chart_w - chart_img.width) // 2,
             y_base + (panel_h - chart_img.height) // 2))

        # SDO: exact square at panel height, flush right, feathered edges
        sdo_resized = sdo_img.resize((sdo_size, sdo_size), Image.LANCZOS)
        # Radial fade — circular vignette from center outward
        import math as _m
        mask = Image.new('L', (sdo_size, sdo_size), 255)
        cx, cy = sdo_size / 2, sdo_size / 2
        r_full = sdo_size * 0.38   # fully opaque inside this radius
        r_edge = sdo_size * 0.50   # fully transparent at this radius
        for y in range(sdo_size):
            for x in range(sdo_size):
                d = _m.sqrt((x - cx)**2 + (y - cy)**2)
                if d <= r_full:
                    a = 255
                elif d >= r_edge:
                    a = 0
                else:
                    a = int(255 * (1 - (d - r_full) / (r_edge - r_full)))
                mask.putpixel((x, y), a)
        sdo_resized.putalpha(mask)
        template.alpha_composite(
            sdo_resized,
            (x_base + chart_w + gap, y_base))
    else:
        # No SDO — chart fills full width (original behavior)
        chart_img.thumbnail((panel_w, panel_h))
        template.alpha_composite(
            chart_img,
            (x_base + (panel_w - chart_img.width) // 2,
             y_base + (panel_h - chart_img.height) // 2))


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

    loc_str = active.get("location", "")
    sdo_img = _fetch_sdo_image(active["peak_time"], loc_str) if loc_str else None

    chart_path = os.path.join(CHARTS_DIR,
                               f"chart_live_{start_time.strftime('%Y%m%d_%H%M')}.png")
    _build_chart(peak_time, peak_class, chart_path, xray_times, xray_fluxes,
                  inprogress=True, event_start=start_time, sdo_mode=bool(sdo_img))
    _composite_chart(template, chart_path, sdo_img)

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

    # Source line (active region + heliographic location) — only drawn when
    # DONKI gave us at least one. Limb flares often have neither, in which
    # case we skip the line entirely rather than print an empty placeholder.
    ar_num   = flare.get("active_region")
    loc_text = flare.get("location") or flare.get("source_location")
    src_bits = []
    if ar_num:   src_bits.append(f"Region : AR {ar_num}")
    if loc_text: src_bits.append(f"Location : {loc_text}")
    if src_bits:
        line3 = "      ".join(src_bits)
        x3 = (template.width - draw.textbbox((0, 0), line3, font=font_info)[2]) // 2
        draw.text((x3, 775), line3, font=font_info, fill=(200, 220, 240, 255))

    chart_title = f"{satellite} X-Ray Flux"
    x_chart = (template.width - draw.textbbox((0, 0), chart_title, font=font_chart)[2]) // 2
    draw.text((x_chart, 815), chart_title, font=font_chart, fill=(240, 220, 170, 255))

    loc_str = flare.get("location", flare.get("source_location", ""))
    sdo_img = _fetch_sdo_image(flare["max_time"], loc_str) if loc_str else None

    chart_path = os.path.join(CHARTS_DIR, f"chart_{peak_dt.strftime('%Y%m%d_%H%M')}.png")
    _build_chart(peak_dt, flare_class, chart_path, xray_times, xray_fluxes,
                  inprogress=False, event_start=start_dt, sdo_mode=bool(sdo_img))
    _composite_chart(template, chart_path, sdo_img)

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

def load_rising_alert_state():
    if not os.path.exists(RISING_ALERT_FILE):
        return {}
    try:
        with open(RISING_ALERT_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, TypeError):
        return {}


def save_rising_alert_state(state):
    with open(RISING_ALERT_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def main():
    ensure_dirs()

    all_flares             = fetch_all_flares()
    xray_times, xray_fluxes = fetch_xray_series()
    processed_ids          = load_processed_ids()
    now_utc                = datetime.now(timezone.utc)
    generated              = []

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Cards from NOAA's finalised 7-day flares feed.
    # processed_ids tracks which flare_ids have already been carded so each
    # NOAA flare is posted exactly once, ever.
    # ═══════════════════════════════════════════════════════════════════════
    new_flares = [f for f in all_flares
                  if flare_id(f) not in processed_ids
                  and (f.get("max_class", "")[0].upper() in ("M", "X"))]
    if new_flares:
        print(f"Found {len(new_flares)} new NOAA-confirmed flare(s).")

    for flare in new_flares:
        fid = flare_id(flare)
        try:
            output = render_card(flare, xray_times, xray_fluxes)
            processed_ids.add(fid)
            generated.append(output)
            print(f"  [done] Card: {output}")

            fc = flare.get("max_class", "?")
            peak_t = flare.get("max_time", "")[:16].replace("T", " ")
            ar  = flare.get("active_region")
            loc = flare.get("location") or flare.get("source_location")
            # Append source line only when DONKI gave us at least one of AR / loc.
            # Limb flares often have neither — keep the caption clean in that case.
            source_bits = []
            if ar:  source_bits.append(f"AR {ar}")
            if loc: source_bits.append(loc)
            source_line = (" · ".join(source_bits))
            caption = f"<b>Solar Flare — {fc}</b>\nPeak: {peak_t} UTC"
            if source_line:
                caption += f"\nSource: {source_line}"
            send_telegram_photo(output, caption)
        except Exception as exc:
            print(f"  [fail] {fid}: {exc}")

    save_processed_ids(processed_ids)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — Single rising-alert text message when live X-ray crosses M1
    # and is climbing. 90-minute cooldown so a noisy multi-peak flare doesn't
    # spam. No card here — NOAA Step 1 handles all imagery once finalised.
    # ═══════════════════════════════════════════════════════════════════════
    state = load_rising_alert_state()
    last_iso = state.get("last_alert_at")
    last_dt = None
    if last_iso:
        try:
            last_dt = datetime.fromisoformat(last_iso)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except Exception:
            last_dt = None

    cooldown_active = (
        last_dt is not None
        and (now_utc - last_dt).total_seconds() < RISING_ALERT_COOLDOWN_MIN * 60
    )

    if cooldown_active:
        mins_since = int((now_utc - last_dt).total_seconds() / 60)
        print(f"[rising] Cooldown active ({mins_since}/{RISING_ALERT_COOLDOWN_MIN} min) — skipping rising-alert check")
    elif xray_times and xray_fluxes and len(xray_fluxes) >= 5:
        # Look at the last 5 samples (~5 min). Rising = last sample is the
        # max AND is >= 1e-5 (M1). Avoids triggering on flat-but-elevated
        # decaying flares.
        recent = xray_fluxes[-5:]
        latest = recent[-1]
        if latest >= 1e-5 and latest == max(recent):
            fc = flux_to_class(latest)
            now_str = now_utc.strftime("%H:%M UTC")
            send_telegram_message(
                f"<b>🔺 Solar Flare in progress — {fc}</b>\n"
                f"Live X-ray rising past M1 at {now_str}.\n"
                f"Card will follow when NOAA finalises the event."
            )
            state["last_alert_at"] = now_utc.isoformat()
            save_rising_alert_state(state)
            print(f"  [rising] Alert sent: {fc} at {now_str}")

    if not generated:
        print("No new cards this run.")
        return

    prune_old_cards()
    print(f"\nDone. {len(generated)} card(s) generated.")


def send_telegram_message(text):
    """Send a plain text message to the Telegram group + channel."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids = [
        os.environ.get("TELEGRAM_CHAT_ID"),
        os.environ.get("TELEGRAM_CHANNEL_ID"),
    ]
    if not bot_token:
        return
    for chat_id in chat_ids:
        if not chat_id:
            continue
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=15,
            )
            result = resp.json()
            if result.get("ok"):
                print(f"  [telegram] Text sent to {chat_id}")
            else:
                print(f"  [telegram] Text send failed to {chat_id}: {result}")
        except Exception as e:
            print(f"  [telegram] Text error to {chat_id}: {e}")


def send_telegram_photo(image_path, caption):
    """Send a flare card image to the Telegram group + channel."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids = [
        os.environ.get("TELEGRAM_CHAT_ID"),
        os.environ.get("TELEGRAM_CHANNEL_ID"),
    ]
    if not bot_token:
        return
    for chat_id in chat_ids:
        if not chat_id:
            continue
        try:
            with open(image_path, "rb") as photo:
                resp = requests.post(
                    f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"photo": photo},
                    timeout=30,
                )
            result = resp.json()
            if result.get("ok"):
                print(f"  [telegram] Alert sent to {chat_id}: {os.path.basename(image_path)}")
            else:
                print(f"  [telegram] Send failed to {chat_id}: {result}")
        except Exception as e:
            print(f"  [telegram] Error sending to {chat_id}: {e}")


if __name__ == "__main__":
    main()
