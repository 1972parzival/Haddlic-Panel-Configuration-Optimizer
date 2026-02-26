"""
Haddlic Panel Configuration — Duck-Curve Optimized Solar Panel Visualization
----------------------------------------------
Run this script directly. You will be prompted for:
  - Your address (auto-geocoded via geopy if installed)
  - Total number of solar panels

Outputs a PNG saved to the same directory as this script.

Dependencies:
    pip install matplotlib numpy pandas geopy
    (geopy is optional — you can enter lat/lon manually if not installed)
"""

import sys
import os
import json
import urllib.request
import urllib.parse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Wedge, FancyBboxPatch, Polygon
import numpy as np

# ─────────────────────────────────────────────
# OUTPUT PATH — same directory as this script
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# USER INPUTS
# ─────────────────────────────────────────────
print("\n╔══════════════════════════════════════════════╗")
print("║   Haddlic Panel Configuration Optimizer      ║")
print("╚══════════════════════════════════════════════╝\n")

address = input("Enter your address (e.g. 7 Mrs Macquaries Rd, Sydney NSW 2000): ").strip()
if not address:
    address = "72 Mrs Macquaries Rd, Sydney NSW 2000, Australia"
    print(f"  → Using default: {address}")

while True:
    try:
        total_panels = int(input("Enter total number of solar panels (e.g. 200): ").strip())
        if total_panels <= 0:
            raise ValueError
        break
    except ValueError:
        print("  Please enter a positive whole number.")

# ─────────────────────────────────────────────
# GEOCODING — uses built-in urllib, no pip installs needed
# ─────────────────────────────────────────────
import urllib.request
import urllib.parse
import json

latitude_deg  = None
longitude_deg = None

try:
    print(f"\n  Geocoding '{address}'...")
    import ssl
    ctx     = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    query   = urllib.parse.urlencode({"q": address, "format": "json", "limit": 1})
    url     = f"https://nominatim.openstreetmap.org/search?{query}"
    req     = urllib.request.Request(url, headers={"User-Agent": "solar-duck-curve-optimizer/1.0"})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        results = json.loads(resp.read().decode())
    if results:
        latitude_deg  = float(results[0]["lat"])
        longitude_deg = float(results[0]["lon"])
        print(f"  ✓ Found: {latitude_deg:.4f}°, {longitude_deg:.4f}°")
    else:
        print("  ✗ Address not found — please enter coordinates manually.")
except Exception as e:
    print(f"  ✗ Geocoding failed ({e}) — please enter coordinates manually.")

if latitude_deg is None:
    print()
    while True:
        try:
            latitude_deg = float(input("  Enter latitude  (e.g.  45.5 for 45.5°N, -33.9 for S): ").strip())
            break
        except ValueError:
            print("  Please enter a valid decimal number.")
    while True:
        try:
            longitude_deg = float(input("  Enter longitude (e.g. -94.4): ").strip())
            break
        except ValueError:
            print("  Please enter a valid decimal number.")

print()

# ─────────────────────────────────────────────
# DERIVED VALUES
# ─────────────────────────────────────────────
latitude_factor     = min(1.0, abs(latitude_deg) / 45.0)
south_ratio         = 0.30 * (1 - latitude_factor)
east_ratio          = 0.20 + 0.05 * (1 - latitude_factor)
west_ratio          = 0.30 + 0.25 * latitude_factor
low_tilt_west_ratio = 1 - (south_ratio + east_ratio + west_ratio)

rows = [
    ("South-facing",  south_ratio),
    ("East-facing",   east_ratio),
    ("West-facing",   west_ratio),
    ("Low-tilt West", low_tilt_west_ratio),
]
df = pd.DataFrame(rows, columns=["Orientation", "Ratio"])
df["Percent"]     = df["Ratio"] * 100
df["Panel Count"] = df["Ratio"].apply(lambda r: int(round(r * total_panels)))

# Duck curve data
hours  = np.linspace(0, 24, 500)
demand = (0.6
          + 0.15 * np.exp(-0.5*(hours - 8)**2 / 6)
          + 0.35 * np.exp(-0.5*(hours - 19)**2 / 4))
S = np.exp(-0.5*(hours-12)**2/4); S /= S.max()
E = np.exp(-0.5*(hours- 8)**2/4); E /= E.max()
W = np.exp(-0.5*(hours-16)**2/4); W /= W.max()
L = np.exp(-0.5*(hours-17)**2/5); L /= L.max()
solar_total = south_ratio*S + east_ratio*E + west_ratio*W + low_tilt_west_ratio*L
net_load    = demand - solar_total

# ── Duck curve improvement stats ──
# Baseline: standard install = equator-facing at latitude tilt (south in N hemi, north in S hemi)
# Both face the equator so the curve shape is the same midday-peaked S curve;
# what changes is the label used in the UI.
baseline_facing = "south" if latitude_deg >= 0 else "north"
solar_baseline  = S * solar_total.max()
net_baseline    = demand - solar_baseline

# 1. RMSE of net load vs mean demand (lower = flatter = better duck-curve fit)
rmse_opt  = float(np.sqrt(np.mean((net_load     - demand.mean())**2)))
rmse_base = float(np.sqrt(np.mean((net_baseline - demand.mean())**2)))
pct_better = (rmse_base - rmse_opt) / rmse_base * 100

# 2. Evening ramp: hours between afternoon trough and evening peak
afternoon_mask = (hours >= 12) & (hours <= 17)
evening_mask   = (hours >= 17) & (hours <= 22)

def ramp_hours(net):
    trough_hr = hours[afternoon_mask][np.argmin(net[afternoon_mask])]
    peak_hr   = hours[evening_mask ][np.argmax(net[evening_mask  ])]
    return peak_hr - trough_hr

ramp_opt  = ramp_hours(net_load)
ramp_base = ramp_hours(net_baseline)

# 3. Revenue-weighted output
# The net load curve IS the price signal: when net load is high, grid is stressed
# and electricity is most valuable. Normalise net load to a [0,1] price proxy.
# Only daytime hours (6-21) where solar actually produces.
solar_hours_mask = (hours >= 6) & (hours <= 21)

# Price proxy = demand curve (higher demand = higher electricity price).
# Using raw demand is more stable than net_baseline which can be near-zero
# during hours the baseline barely produces, causing division instability.
price_proxy = demand / demand.max()   # normalise to 0-1

# Revenue = solar output * price at each hour (daytime only)
# Use simple dot product (equivalent to integration with uniform spacing)
dt = hours[1] - hours[0]
rev_opt        = float(np.sum(solar_total[solar_hours_mask]    * price_proxy[solar_hours_mask]) * dt)
rev_base       = float(np.sum(solar_baseline[solar_hours_mask] * price_proxy[solar_hours_mask]) * dt)
rev_pct_better = (rev_opt - rev_base) / rev_base * 100

# Solar geometry
declination    = 23.5
solar_elev     = min(90 - abs(latitude_deg) + declination, 89.0)
solar_elev_rad = np.radians(solar_elev)
panel_tilt_deg = abs(latitude_deg)
panel_tilt_rad = np.radians(panel_tilt_deg)

# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
fig.suptitle("Haddlic Panel Configuration — Duck Curve Optimization",
             fontsize=17, color="white", fontweight="bold", y=0.985)

gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.38,
                      left=0.27, right=0.97, top=0.94, bottom=0.05)

ax_duck  = fig.add_subplot(gs[0, 0])
ax_bar   = fig.add_subplot(gs[0, 1])
ax_earth = fig.add_subplot(gs[1, 0])
ax_field = fig.add_subplot(gs[1, 1])

BG  = "#161b22"
GRD = "#30363d"

for ax in [ax_duck, ax_bar, ax_earth, ax_field]:
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_edgecolor(GRD)

def style(ax, title, grid=True):
    ax.tick_params(colors="#c9d1d9", labelsize=8.5)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=9)
    if grid:
        ax.grid(True, linestyle=':', linewidth=0.6, color=GRD)

# ══════════════════════════════════════════════
# PLOT 1 — Duck Curve
# ══════════════════════════════════════════════
ax_duck.plot(hours, demand,      label="Demand",       color="#e6edf3", lw=2)
ax_duck.plot(hours, solar_total, label="Solar Output", color="#f0a500", lw=2)
ax_duck.plot(hours, net_load,    label="Net Load",     color="#f85149", lw=2, ls="--")
ax_duck.fill_between(hours, net_load, demand, where=(net_load < demand),
                     alpha=0.10, color="#f0a500")
ax_duck.set_xlabel("Hour of Day")
ax_duck.set_ylabel("Normalized Power")
ax_duck.legend(fontsize=8, facecolor="#21262d", labelcolor="#c9d1d9", edgecolor=GRD)
style(ax_duck, "Duck Curve — Haddlic Configuration")


# ══════════════════════════════════════════════
# PLOT 2 — Bar Chart
# ══════════════════════════════════════════════
colors_bar = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]
bars = ax_bar.bar(df["Orientation"], df["Percent"], color=colors_bar,
                  width=0.55, edgecolor="#0d1117", linewidth=1.2)
for bar, row in zip(bars, df.itertuples()):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{row.Percent:.1f}%\n({row._4}p)",
                ha="center", va="bottom", fontsize=8, color="#c9d1d9")
ax_bar.set_ylabel("Share of Total Panel Area (%)")
ax_bar.tick_params(axis='x', labelsize=7.5, rotation=12)
style(ax_bar, f"Haddlic Mix — Panel Distribution ({total_panels} Panels)")

# ══════════════════════════════════════════════
# PLOT 3 — Earth Cross-Section
# ══════════════════════════════════════════════
ax_earth.set_xlim(-3.0, 3.0)
ax_earth.set_ylim(-2.4, 2.4)
ax_earth.set_aspect("equal")
ax_earth.axis("off")
style(ax_earth, f"Earth Cross-Section — Latitude {latitude_deg:.1f}°{'N' if latitude_deg >= 0 else 'S'}", grid=False)

# Space background
grad = np.zeros((300, 400, 3))
grad[:,:,0] = 0.051; grad[:,:,1] = 0.067; grad[:,:,2] = 0.090
ax_earth.imshow(grad, extent=[-3, 3, -2.4, 2.4], aspect="auto", zorder=0)

# Stars
rng = np.random.default_rng(42)
ax_earth.scatter(rng.uniform(-3, 3, 60), rng.uniform(-2.4, 2.4, 60),
                 s=rng.uniform(0.3, 1.5, 60), color="white", alpha=0.6, zorder=1)

earth_r = 1.15

# Atmosphere glow + ocean
ax_earth.add_patch(plt.Circle((0, 0), earth_r+0.09, color="#1a3a5c", zorder=2, alpha=0.55))
ax_earth.add_patch(plt.Circle((0, 0), earth_r,      color="#1a5fa8", zorder=3))

# Night side (right half shaded)
night_theta = np.linspace(-np.pi/2, np.pi/2, 300)
night_xs = np.concatenate([[0], earth_r*np.cos(night_theta), [0]])
night_ys = np.concatenate([[earth_r], earth_r*np.sin(night_theta), [-earth_r]])
ax_earth.add_patch(plt.Polygon(np.column_stack([night_xs, night_ys]),
                               facecolor="#05101f", alpha=0.58, zorder=4, edgecolor="none"))

# Polar ice caps
for cy, sign in [(earth_r*0.93, 1), (-earth_r*0.93, -1)]:
    cap_r = earth_r * 0.26
    ct = np.linspace(np.pi, 0, 100)
    ax_earth.fill(cap_r*np.cos(ct), cy + sign*cap_r*0.32*np.sin(ct),
                  color="white", alpha=0.45, zorder=6)

# Equator line + right-side label
eq_theta = np.linspace(np.pi, 0, 300)
ax_earth.plot(earth_r*np.cos(eq_theta), np.zeros(300),
              color="white", lw=1.4, ls=(0,(6,4)), zorder=7, alpha=0.8)
ax_earth.text(earth_r + 0.12, 0.06, "Equator", color="white",
              fontsize=7.5, ha="left", va="bottom", alpha=0.8, zorder=8)

# Latitude arc on LEFT (dayside), angle label pulled to RIGHT
lat_rad = np.radians(latitude_deg)
pt_x = -np.cos(lat_rad) * earth_r
pt_y =  np.sin(lat_rad) * earth_r

arc_r_inner = 0.62
arc_theta   = np.linspace(np.pi, np.pi - lat_rad, 120)
arc_xs      = arc_r_inner * np.cos(arc_theta)
arc_ys      = arc_r_inner * np.sin(arc_theta)
ax_earth.plot(arc_xs, arc_ys, color="#f0a500", lw=2.8, zorder=9)
ax_earth.annotate("", xy=(arc_xs[-1], arc_ys[-1]), xytext=(arc_xs[-5], arc_ys[-5]),
                  arrowprops=dict(arrowstyle="-|>", color="#f0a500",
                                  lw=1.5, mutation_scale=11), zorder=10)
ax_earth.plot([0, -arc_r_inner], [0, 0], color="#ffffff",
              lw=1.2, ls="--", alpha=0.55, zorder=8)

mid_a     = np.pi - lat_rad/2
arc_mid_x = arc_r_inner * np.cos(mid_a)
arc_mid_y = arc_r_inner * np.sin(mid_a)
ax_earth.annotate(f"{abs(latitude_deg):.1f}°{'N' if latitude_deg >= 0 else 'S'}",
                  xy=(arc_mid_x, arc_mid_y), xytext=(0.85, arc_mid_y + 0.05),
                  color="#f0a500", fontsize=10, fontweight="bold",
                  ha="left", va="center", zorder=11,
                  arrowprops=dict(arrowstyle="-", color="#f0a500", lw=0.8, alpha=0.6))

# ── Procedural landmasses seeded from lat/lon ──
def make_blob(cx, cy, radius, n_pts, rng, roughness=0.35):
    """Random organic blob polygon clipped to Earth circle."""
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    angles += rng.uniform(0, 2*np.pi)          # random rotation
    r = radius * (1 + roughness * rng.uniform(-1, 1, n_pts))
    r = np.clip(r, radius*0.3, radius*1.5)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    # clip to inside earth circle
    dists = np.sqrt(xs**2 + ys**2)
    mask  = dists < earth_r * 0.97
    if mask.sum() < 3:
        return None, None
    return xs[mask], ys[mask]

land_rng = np.random.default_rng(int(abs(latitude_deg)*100 + abs(longitude_deg)*10) % 99999)

# Large continent cluster — offset based on longitude quadrant
lon_phase = (longitude_deg % 360) / 360   # 0–1
lat_phase  = latitude_deg / 90            # -1 to 1

# Primary big continent (roughly where the user is, but stylised)
# Map lat/lon to Earth-circle coords on the dayside (left half)
cont_angle  = np.pi + lat_phase * 0.7     # dayside: π ± offset
cont_cx     = earth_r * 0.55 * np.cos(cont_angle)
cont_cy     = earth_r * 0.55 * np.sin(cont_angle)
bxs, bys = make_blob(cont_cx, cont_cy, 0.38, 14, land_rng, roughness=0.45)
if bxs is not None:
    ax_earth.add_patch(plt.Polygon(np.column_stack([bxs, bys]),
                                   facecolor="#2d6a2d", alpha=0.88, zorder=5, edgecolor="none"))

# Secondary continent (opposite side, partially in night)
cont2_angle = cont_angle + np.pi * 0.6 + land_rng.uniform(-0.3, 0.3)
cont2_cx    = earth_r * 0.50 * np.cos(cont2_angle)
cont2_cy    = earth_r * 0.50 * np.sin(cont2_angle)
bxs2, bys2 = make_blob(cont2_cx, cont2_cy, 0.28, 11, land_rng, roughness=0.40)
if bxs2 is not None:
    ax_earth.add_patch(plt.Polygon(np.column_stack([bxs2, bys2]),
                                   facecolor="#2d6a2d", alpha=0.75, zorder=5, edgecolor="none"))

# Small island / peninsula guaranteed near the location point (pt_x, pt_y)
island_cx = pt_x * 0.72 + land_rng.uniform(-0.08, 0.08)
island_cy = pt_y * 0.72 + land_rng.uniform(-0.08, 0.08)
bxs3, bys3 = make_blob(island_cx, island_cy, 0.13, 9, land_rng, roughness=0.35)
if bxs3 is not None:
    ax_earth.add_patch(plt.Polygon(np.column_stack([bxs3, bys3]),
                                   facecolor="#3a7a2a", alpha=0.90, zorder=5, edgecolor="none"))

# Tiny scattered islands
for _ in range(3):
    ang  = land_rng.uniform(np.pi*0.5, np.pi*1.5)
    dist = land_rng.uniform(0.3, 0.9) * earth_r
    icx, icy = dist*np.cos(ang), dist*np.sin(ang)
    bxi, byi = make_blob(icx, icy, land_rng.uniform(0.06, 0.11), 7, land_rng, roughness=0.3)
    if bxi is not None:
        ax_earth.add_patch(plt.Polygon(np.column_stack([bxi, byi]),
                                       facecolor="#2d6a2d", alpha=0.70, zorder=5, edgecolor="none"))

# Latitude dashed circle
lat_half    = np.linspace(np.pi, 0, 200)
lat_circ_r  = np.cos(lat_rad) * earth_r
ax_earth.plot(lat_circ_r * np.cos(lat_half),
              np.sin(lat_rad) * earth_r * np.ones(200),
              color="#f0a500", lw=0.9, ls=":", zorder=7, alpha=0.55)

# Surface dot + location label (right side via leader)
ax_earth.plot(pt_x, pt_y, "o", color="#f0a500", ms=11, zorder=12,
              markeredgecolor="white", markeredgewidth=2.0)
ax_earth.annotate(address.split(",")[0],   # just street / city name
                  xy=(pt_x, pt_y), xytext=(pt_x + 0.90, pt_y + 0.10),
                  color="#f0a500", fontsize=8.5, fontweight="bold",
                  ha="left", va="center", zorder=13,
                  arrowprops=dict(arrowstyle="-", color="#f0a500", lw=0.8, alpha=0.6))

# Surface normal arrow + label (right side via leader)
norm_len    = 0.60
norm_tip_x  = pt_x + pt_x/earth_r * norm_len
norm_tip_y  = pt_y + pt_y/earth_r * norm_len
ax_earth.annotate("", xy=(norm_tip_x, norm_tip_y), xytext=(pt_x, pt_y),
                  arrowprops=dict(arrowstyle="-|>", color="#ff7f50",
                                  lw=2.2, mutation_scale=11), zorder=12)
ax_earth.annotate("Surface\nnormal", xy=(norm_tip_x, norm_tip_y),
                  xytext=(norm_tip_x + 1.10, norm_tip_y + 0.10),
                  color="#ff7f50", fontsize=7.5, ha="left", va="center", zorder=13,
                  arrowprops=dict(arrowstyle="-", color="#ff7f50", lw=0.8, alpha=0.6))

# Parallel sun rays from left
ray_color = "#FFE066"
for ry in np.linspace(-0.90, 0.90, 10):
    if abs(ry) < earth_r:
        hit_x = -np.sqrt(earth_r**2 - ry**2)
        alpha = 0.45 + 0.45*(1 - abs(ry)/earth_r)
        ax_earth.annotate("", xy=(hit_x, ry), xytext=(-2.88, ry),
                          arrowprops=dict(arrowstyle="-|>", color=ray_color,
                                          lw=1.4, mutation_scale=8, alpha=alpha), zorder=8)
    else:
        ax_earth.plot([-2.88, -earth_r-0.2], [ry, ry],
                      color=ray_color, lw=0.8, alpha=0.18, zorder=7)
ax_earth.text(-2.85, -1.10, "Parallel\nsun rays →", color=ray_color,
              fontsize=8, ha="left", va="top", zorder=10)
ax_earth.add_patch(plt.Circle((-2.98, 0), 0.55, color="#FFD700", zorder=1, clip_on=True))

# ══════════════════════════════════════════════
# PLOT 4 — Four Panel Types: side-elevation view
# ══════════════════════════════════════════════
ax_field.set_xlim(-0.5, 13.5)
ax_field.set_ylim(-1.3, 6.5)
ax_field.set_aspect("equal")
ax_field.axis("off")
style(ax_field, "Haddlic Panel Types — Side Profile & Peak Output Time", grid=False)
ax_field.set_facecolor("#0d1523")

# Ground + grass
ax_field.plot([-0.5, 13.5], [0, 0], color="#4a8a30", lw=2.0, zorder=3)
rng2 = np.random.default_rng(11)
for gx in np.arange(0.0, 13.0, 0.28):
    bh = rng2.uniform(0.08, 0.18)
    ax_field.plot([gx, gx+rng2.uniform(-0.05, 0.05)], [0, bh],
                  color="#3a7020", lw=1.2, alpha=0.65, zorder=2)

def draw_panel(ax, base_x, tilt_deg, face_right, color, label, peak_time,
               panel_len=1.8, post_h=0.35, thick=0.10):
    tilt = np.radians(tilt_deg)
    sign = 1 if face_right else -1

    bx = base_x - sign * panel_len/2 * np.cos(tilt)
    by = post_h  - panel_len/2 * np.sin(tilt)
    tx = base_x + sign * panel_len/2 * np.cos(tilt)
    ty = post_h  + panel_len/2 * np.sin(tilt)

    nx = -sign * np.sin(tilt)
    ny =  np.cos(tilt)

    # Panel face
    pts_face = np.array([
        [bx,          by],
        [tx,          ty],
        [tx+thick*nx, ty+thick*ny],
        [bx+thick*nx, by+thick*ny],
    ])
    ax.add_patch(plt.Polygon(pts_face, closed=True,
                             facecolor=color, edgecolor="#90CAF9",
                             lw=1.6, zorder=6, alpha=0.92))

    # Cell grid lines
    for frac in [0.25, 0.5, 0.75]:
        gx0 = bx + frac*(tx-bx); gy0 = by + frac*(ty-by)
        ax.plot([gx0, gx0+thick*nx], [gy0, gy0+thick*ny],
                color="#64B5F6", lw=0.9, zorder=7, alpha=0.7)

    # Sheen
    sheen_pts = np.array([
        [bx + 0.10*(tx-bx),              by + 0.10*(ty-by)],
        [bx + 0.30*(tx-bx),              by + 0.30*(ty-by)],
        [bx + 0.30*(tx-bx)+thick*nx*0.5, by + 0.30*(ty-by)+thick*ny*0.5],
        [bx + 0.10*(tx-bx)+thick*nx*0.5, by + 0.10*(ty-by)+thick*ny*0.5],
    ])
    ax.add_patch(plt.Polygon(sheen_pts, closed=True,
                             facecolor="#90CAF9", alpha=0.30, zorder=7))

    # Support post
    ax.plot([base_x, base_x], [0, post_h],
            color="#8b949e", lw=3.0, solid_capstyle="round", zorder=5)
    ax.plot([base_x-0.18, base_x+0.18], [0, 0],
            color="#8b949e", lw=3.5, zorder=5)

    # Tilt arc (untouched)
    arc_R = 0.55
    arc_t  = np.linspace(0, tilt, 60)
    arc_xa = base_x + sign * arc_R * np.cos(arc_t)
    arc_ya = post_h + arc_R * np.sin(arc_t)
    ax.plot(arc_xa, arc_ya, color="#ff7f50", lw=1.8, zorder=8)

    # Angle text above panel top edge
    ax.text(base_x, ty + thick*ny + 0.28,
            f"{tilt_deg:.0f}°", color="#ff7f50", fontsize=11,
            ha="center", va="bottom", fontweight="bold", zorder=9)

    return (bx, by, tx, ty, base_x)

panel_defs = [
    (1.5,  "East-facing",  20,               False, "#1976D2", "Morning\n7–10 AM"),
    (4.8,  "South-facing", int(panel_tilt_deg), True, "#1565C0", "Midday\n10 AM–2 PM"),
    (8.1,  "West-facing",  30,               True,  "#0D47A1", "Afternoon\n2–6 PM"),
    (11.4, "Low-tilt West", 8,               True,  "#1a3a6a", "Late Afternoon\n4–7 PM"),
]

for (cx, lbl, tilt_d, face_r, col, peak) in panel_defs:
    draw_panel(ax_field, cx, tilt_d, face_r, col, lbl, peak)

# Label table below ground
row_y_name = -0.45
row_y_peak = -0.85
ax_field.plot([-0.5, 13.5], [row_y_name+0.18]*2, color=GRD, lw=1.0, zorder=8)
for (cx, lbl, tilt_d, face_r, col, peak) in panel_defs:
    ax_field.plot(cx, row_y_name, "o", color=col, ms=7,
                  markeredgecolor="#90CAF9", markeredgewidth=0.8, zorder=10)
    ax_field.text(cx+0.22, row_y_name, lbl,
                  color="#e6edf3", fontsize=8.8, fontweight="bold",
                  ha="left", va="center", zorder=10)
    ax_field.text(cx+0.22, row_y_peak, peak.replace("\n", "  "),
                  color="#f0a500", fontsize=8.0, fontstyle="italic",
                  ha="left", va="center", zorder=10)

# Direction / info labels
ax_field.text(0.1,  5.8, "← EAST", color="#8b949e", fontsize=8, alpha=0.75, zorder=10)
ax_field.text(13.3, 5.8, "WEST →", color="#8b949e", fontsize=8, ha="right", alpha=0.75, zorder=10)
ax_field.text(6.5,  6.1, "Side profile view — cross-section looking North",
              color="#6e7681", fontsize=8, ha="center", zorder=10)
ax_field.text(6.5,  5.7,
              f"Lat {abs(latitude_deg):.1f}°{'N' if latitude_deg >= 0 else 'S'}"
              f"  |  South-facing optimal tilt ≈ {int(panel_tilt_deg)}°",
              color="#8b949e", fontsize=8.5, ha="center", zorder=10)

# ──────────────────────────────────────────────
# LEFT SIDEBAR TEXT BLOCK
# ──────────────────────────────────────────────
text_block = (
    f"Location:\n{address}\n\n"
    f"Latitude:   {latitude_deg:.4f}°\n"
    f"Longitude: {longitude_deg:.4f}°\n\n"
    f"Lat. Factor: {latitude_factor:.2f}\n\n"
    f"Solar elev.\n(summer noon):\n  ~{solar_elev:.0f}°\n\n"
    f"Panel tilt:\n  ~{panel_tilt_deg:.0f}°\n\n"
    f"─── Curve Fit ───\n"
    f"Curve fit: +{pct_better:.0f}%\n"
    f"Revenue:   +{rev_pct_better:.0f}%\n"
    f"vs standard all-{baseline_facing} install\n"
    f"(Haddlic config)\n\n"
    f"Eve. ramp span:\n"
    f"  Opt:  {ramp_opt:.1f} hrs\n"
    f"  Base: {ramp_base:.1f} hrs\n\n"
    "─── Panel Mix ───\n"
)
for _, row in df.iterrows():
    text_block += f"{row['Orientation']}:\n  {row['Percent']:.1f}%  ({row['Panel Count']}p)\n"

fig.text(
    0.005, 0.50, text_block,
    fontsize=9.5, verticalalignment="center",
    family="monospace", color="#c9d1d9",
    bbox=dict(boxstyle="round,pad=0.65", facecolor="#161b22",
              edgecolor="#30363d", linewidth=1.5)
)

# ──────────────────────────────────────────────
# SAVE — same directory as this script
# ──────────────────────────────────────────────
safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in address)
safe_name = safe_name.strip().replace(" ", "_")[:60]
out_path  = os.path.join(SCRIPT_DIR, f"solar_output_{safe_name}.png")

plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# Terminal summary table
W = 56

def trow(text):
    inner = "  " + str(text)
    return "||" + inner[:W].ljust(W) + "||"

def tdiv(ch="="):
    return "+" + ch * W + "+"

print()
print("\u2554" + "\u2550" * W + "\u2557")
print("\u2551" + ("  HADDLIC PANEL CONFIGURATION RESULTS").ljust(W) + "\u2551")
print("\u2560" + "\u2550" * W + "\u2563")
print("\u2551" + ("  Location   : " + address)[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Latitude   : " + f"{latitude_deg:+.4f}\u00b0   Longitude: {longitude_deg:+.4f}\u00b0")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Panels     : " + str(total_panels))[:W].ljust(W) + "\u2551")
print("\u2560" + "\u2550" * W + "\u2563")
print("\u2551" + "  PANEL DISTRIBUTION".ljust(W) + "\u2551")
print("\u2560" + "\u2500" * W + "\u2563")
for _, r in df.iterrows():
    bar  = "\u2588" * int(r["Percent"] / 2.5)
    line = "  " + f'{r["Orientation"]:<18}  {r["Percent"]:5.1f}%  ({r["Panel Count"]:3d}p)  {bar}'
    print("\u2551" + line[:W].ljust(W) + "\u2551")
print("\u2560" + "\u2550" * W + "\u2563")
print("\u2551" + "  HADDLIC PERFORMANCE VS STANDARD INSTALL".ljust(W) + "\u2551")
print("\u2560" + "\u2500" * W + "\u2563")
print("\u2551" + ("  Baseline        :  all-" + baseline_facing + " standard install")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Curve fit       :  +" + f"{pct_better:.1f}% better")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Revenue         :  +" + f"{rev_pct_better:.1f}% better")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Eve. ramp (opt) :  " + f"{ramp_opt:.1f} hrs")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Eve. ramp (base):  " + f"{ramp_base:.1f} hrs")[:W].ljust(W) + "\u2551")
print("\u2560" + "\u2550" * W + "\u2563")
print("\u2551" + "  SOLAR GEOMETRY".ljust(W) + "\u2551")
print("\u2560" + "\u2500" * W + "\u2563")
print("\u2551" + ("  Solar elev (summer noon) :  ~" + f"{solar_elev:.0f}\u00b0")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Optimal panel tilt       :  ~" + f"{panel_tilt_deg:.0f}\u00b0")[:W].ljust(W) + "\u2551")
print("\u2560" + "\u2550" * W + "\u2563")
print("\u2551" + "  FORMULA SHEET".ljust(W) + "\u2551")
print("\u2560" + "\u2500" * W + "\u2563")
print("\u2551" + "  -- Latitude Factor --".ljust(W) + "\u2551")
print("\u2551" + "  f = min(1.0, |lat| / 45.0)".ljust(W) + "\u2551")
print("\u2551" + ("  f = min(1.0, " + f"{abs(latitude_deg):.2f} / 45.0) = {latitude_factor:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Panel Allocation Ratios --".ljust(W) + "\u2551")
print("\u2551" + "  South  = 0.30 x (1 - f)".ljust(W) + "\u2551")
print("\u2551" + ("  South  = 0.30 x (1 - " + f"{latitude_factor:.2f}) = {south_ratio:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + "  East   = 0.20 + 0.05 x (1 - f)".ljust(W) + "\u2551")
print("\u2551" + ("  East   = 0.20 + 0.05 x (1 - " + f"{latitude_factor:.2f}) = {east_ratio:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + "  West   = 0.30 + 0.25 x f".ljust(W) + "\u2551")
print("\u2551" + ("  West   = 0.30 + 0.25 x " + f"{latitude_factor:.2f} = {west_ratio:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + "  LTW    = 1 - (South + East + West)".ljust(W) + "\u2551")
print("\u2551" + ("  LTW    = 1 - " + f"({south_ratio:.4f} + {east_ratio:.4f} + {west_ratio:.4f}) = {low_tilt_west_ratio:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Solar Geometry --".ljust(W) + "\u2551")
print("\u2551" + "  Solar elev = 90 - |lat| + declination".ljust(W) + "\u2551")
print("\u2551" + ("  Solar elev = 90 - " + f"{abs(latitude_deg):.1f} + 23.5 = {solar_elev:.1f}" + "\u00b0")[:W].ljust(W) + "\u2551")
print("\u2551" + "  Panel tilt = |lat|".ljust(W) + "\u2551")
print("\u2551" + ("  Panel tilt = " + f"{panel_tilt_deg:.1f}" + "\u00b0")[:W].ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Demand Curve D(t) --".ljust(W) + "\u2551")
print("\u2551" + "  D(t) = 0.60".ljust(W) + "\u2551")
print("\u2551" + "       + 0.15 x exp(-0.5(t-8)^2 / 6)   [morning]".ljust(W) + "\u2551")
print("\u2551" + "       + 0.35 x exp(-0.5(t-19)^2 / 4)  [evening]".ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Individual Panel Curves (normalised) --".ljust(W) + "\u2551")
print("\u2551" + "  S(t) = exp(-0.5(t-12)^2 / 4)  South  peak 12:00".ljust(W) + "\u2551")
print("\u2551" + "  E(t) = exp(-0.5(t- 8)^2 / 4)  East   peak 08:00".ljust(W) + "\u2551")
print("\u2551" + "  W(t) = exp(-0.5(t-16)^2 / 4)  West   peak 16:00".ljust(W) + "\u2551")
print("\u2551" + "  L(t) = exp(-0.5(t-17)^2 / 5)  LTW    peak 17:00".ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Haddlic Composite Solar Output --".ljust(W) + "\u2551")
print("\u2551" + "  H(t) = rS x S(t) + rE x E(t) + rW x W(t) + rL x L(t)".ljust(W) + "\u2551")
print("\u2551" + ("  H(t) = " + f"{south_ratio:.3f}S + {east_ratio:.3f}E + {west_ratio:.3f}W + {low_tilt_west_ratio:.3f}L")[:W].ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Baseline (standard equator-facing) --".ljust(W) + "\u2551")
print("\u2551" + ("  B(t) = S(t) x max(H(t))  [all-" + baseline_facing + ", peak-scaled]")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  B(t) = S(t) x " + f"{solar_total.max():.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Net Load --".ljust(W) + "\u2551")
print("\u2551" + "  Net(t) = D(t) - H(t)  [Haddlic]".ljust(W) + "\u2551")
print("\u2551" + "  Net(t) = D(t) - B(t)  [Baseline]".ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Curve Fit (RMSE vs flat demand) --".ljust(W) + "\u2551")
print("\u2551" + "  RMSE = sqrt( mean( (Net(t) - mean(D))^2 ) )".ljust(W) + "\u2551")
print("\u2551" + ("  RMSE Haddlic  = " + f"{rmse_opt:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  RMSE Baseline = " + f"{rmse_base:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Improvement   = (RMSE_b - RMSE_h) / RMSE_b = +" + f"{pct_better:.1f}%")[:W].ljust(W) + "\u2551")
print("\u2551" + "".ljust(W) + "\u2551")
print("\u2551" + "  -- Revenue (demand-weighted output) --".ljust(W) + "\u2551")
print("\u2551" + "  P(t) = D(t) / max(D)         [price proxy]".ljust(W) + "\u2551")
print("\u2551" + "  Rev  = SUM( Solar(t) x P(t) x dt )  t in [6,21]".ljust(W) + "\u2551")
print("\u2551" + ("  Rev Haddlic  = " + f"{rev_opt:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Rev Baseline = " + f"{rev_base:.4f}")[:W].ljust(W) + "\u2551")
print("\u2551" + ("  Improvement  = (Rev_h - Rev_b) / Rev_b = +" + f"{rev_pct_better:.1f}%")[:W].ljust(W) + "\u2551")

print("\u2560" + "\u2550" * W + "\u2563")
print("\u2551" + "  OUTPUT FILE".ljust(W) + "\u2551")
print("\u2560" + "\u2500" * W + "\u2563")
for chunk in [out_path[i:i+W-4] for i in range(0, len(out_path), W-4)]:
    print("\u2551" + ("  " + chunk)[:W].ljust(W) + "\u2551")
print("\u255a" + "\u2550" * W + "\u255d")
print()

# Open the saved image with the system default viewer
import subprocess, sys
if sys.platform == "darwin":
    subprocess.Popen(["open", out_path])
elif sys.platform.startswith("linux"):
    subprocess.Popen(["xdg-open", out_path],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
elif sys.platform == "win32":
    subprocess.Popen(["start", out_path], shell=True)