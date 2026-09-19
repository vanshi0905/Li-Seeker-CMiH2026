"""
Ultra-High-Resolution (300 DPI, 4K+) GeoTIFF Raster Snapshot Generator.

Generates publication-grade, full-extent cartographic snapshots for:
1. Mineral Exploration Indices (Scientific Colormaps):
   - Cardoso Lithium Pegmatite Index (LPI) [Colormap: magma]
   - Cardoso Lithium Mica Discrimination Ratio (LMDR) [Colormap: plasma]
   - Crosta PCA Al-OH Hydroxyl Alteration [Colormap: turbo]
   - REE Neodymium (Nd3+) 740nm Absorption [Colormap: cividis]
   - Ferric Iron Oxide Gossan Index [Colormap: YlOrRd]

2. False-Color / Natural Color RGB Composites (3-Band Blends):
   - True Color RGB: B04 (Red), B03 (Green), B02 (Blue)
   - SWIR Mineral Composite: B12 (Red), B8A (Green), B04 (Blue)
   - REE / Hydroxyl Alteration Composite: B11 (Red), B06 (Green), B02 (Blue)

3. Individual Sentinel-2 Optical/Infrared Bands:
   - B02, B03, B04, B06, B08, B11, B12, B8A [Colormap: bone]

Features:
- Dynamic range normalization (2nd to 98th percentile contrast stretch)
- Clean NoData (-9999.0 / NaN) masking
- Professional cartographic North Arrow and calibrated 10 km Scale Bar
- Coordinate grid in WGS 84 Decimal Degrees
- Rich scientific metadata, statistics, and exploration interpretation side panel
- Minimum 3840x2160 (4K+ UHD) resolution at 300 DPI
"""

import os
import sys
import re
import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "sentinel_bands")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "high_res_snapshots")
DEFAULT_BOUNDS = (22.25, 82.25, 22.75, 82.75)  # min_lat, min_lon, max_lat, max_lon
DPI = 300


# ─────────────────────────────────────────────────────────────────────────────
# METADATA DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────
MINERAL_INDICES_META = {
    "index_cardoso_LPI_lithium_pegmatite.tif": {
        "title": "Cardoso Lithium Pegmatite Index (LPI)",
        "subtitle": "Katghora District, Korba | Sentinel-2 MSI Multi-Spectral Ratio",
        "colormap": "magma",
        "cbar_label": "LPI Value: (B11 / B12) × (B2 / B4)",
        "layer_name": "Cardoso LPI",
        "formula": "(B11 / B12) × (B02 / B04)",
        "bands": "B11 (1610nm), B12 (2190nm), B02 (490nm), B04 (665nm)",
        "target": "Spodumene / Petalite Pegmatites\nFelsic Leucosomes vs Country Rock",
        "interpretation": "Couples the 2.20µm Al-OH mica\nabsorption dip (B11/B12) with\nfelsic quartz-feldspar albedo\n(B02/B04) to isolate pegmatites.",
    },
    "index_cardoso_LMDR_lithium_mica.tif": {
        "title": "Cardoso Lithium Mica Discrimination Ratio (LMDR)",
        "subtitle": "Katghora District, Korba | Sentinel-2 MSI Normalized Difference",
        "colormap": "plasma",
        "cbar_label": "LMDR Value: (B11 - B12) / (B8A + B2)",
        "layer_name": "Cardoso LMDR",
        "formula": "(B11 - B12) / (B8A + B02)",
        "bands": "B11 (1610nm), B12 (2190nm), B8A (865nm), B02 (490nm)",
        "target": "Lepidolite / Zinnwaldite / Muscovite\nLithium-Bearing Mica Halos",
        "interpretation": "Normalizes SWIR-2 Al-OH absorption\ndepth against NIR and visible\nreflectance to delineate lithium mica\nmetasomatic selvages.",
    },
    "index_crosta_PCA_AlOH_hydroxyl.tif": {
        "title": "Crosta Feature-Oriented PCA: Al-OH Hydroxyl Alteration",
        "subtitle": "Katghora District, Korba | Principal Component Analysis [B2, B4, B11, B12]",
        "colormap": "turbo",
        "cbar_label": "PC-4 Alteration Score (Al-OH Hydroxyl)",
        "layer_name": "Crosta PCA (Al-OH)",
        "formula": "Eigenvector PC-4 on [B2, B4, B11, B12]",
        "bands": "B02 (490nm), B04 (665nm), B11 (1610nm), B12 (2190nm)",
        "target": "Hydrothermal Greisen & Argillic Caps\nEndocontact Hydroxyl Alteration",
        "interpretation": "Selects the PC with opposite signs\nand highest loading magnitude\nbetween B11 and B12, suppressing\nvegetation and highlighting clays.",
    },
    "index_REE_neodymium_740nm.tif": {
        "title": "Neodymium (Nd3+) REE Spectral Absorption Index",
        "subtitle": "Katghora District, Korba | Sentinel-2 MSI 740nm Electronic Trough",
        "colormap": "cividis",
        "cbar_label": "Nd3+ Ratio: B8A (865nm) / B06 (740nm)",
        "layer_name": "REE Neodymium (Nd3+)",
        "formula": "B8A / B06",
        "bands": "B8A (865nm Narrow NIR) / B06 (740nm Red Edge 2)",
        "target": "Monazite, Bastnasite, Xenotime\nRare Earth Element Pegmatites",
        "interpretation": "Exploits the narrow electronic\nabsorption trough of trivalent\nNeodymium (Nd3+) at 740nm (B06)\nflanked by the 865nm continuum.",
    },
    "index_ferric_iron_oxide_gossan.tif": {
        "title": "Ferric Iron Oxide (Fe3+) Gossan Cap Index",
        "subtitle": "Katghora District, Korba | Sentinel-2 MSI Visible Spectral Ratio",
        "colormap": "YlOrRd",
        "cbar_label": "Fe3+ Gossan Index: B04 (Red) / B02 (Blue)",
        "layer_name": "Ferric Iron (Gossan)",
        "formula": "B04 / B02",
        "bands": "B04 (665nm Red) / B02 (490nm Blue)",
        "target": "Hematite, Goethite, Jarosite\nOxidized Pegmatite Gossan Caps",
        "interpretation": "Strong absorption of ferric iron\nin blue (B02) paired with high\nreflectance in red (B04) maps\nsupergene weathering caps.",
    },
}

SENTINEL_BANDS_META = {
    "sentinel2_B02_blue_490nm.tif": {
        "title": "Sentinel-2 MSI Band 02: Blue (490 nm)",
        "subtitle": "Katghora District | 10m Ground Resolution | BOA Surface Reflectance",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 02 (Blue)",
        "center_wl": "492.4 nm (Bandwidth: 66 nm)",
        "res": "10 meters",
        "target": "Atmospheric scattering correction,\nquartz-feldspar felsic albedo.",
    },
    "sentinel2_B03_green_560nm.tif": {
        "title": "Sentinel-2 MSI Band 03: Green (560 nm)",
        "subtitle": "Katghora District | 10m Ground Resolution | BOA Surface Reflectance",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 03 (Green)",
        "center_wl": "559.8 nm (Bandwidth: 36 nm)",
        "res": "10 meters",
        "target": "Vegetation green reflectance peak,\niron hydroxide discrimination.",
    },
    "sentinel2_B04_red_665nm.tif": {
        "title": "Sentinel-2 MSI Band 04: Red (665 nm)",
        "subtitle": "Katghora District | 10m Ground Resolution | BOA Surface Reflectance",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 04 (Red)",
        "center_wl": "664.6 nm (Bandwidth: 31 nm)",
        "res": "10 meters",
        "target": "Chlorophyll absorption boundary,\nferric iron oxide reflection.",
    },
    "sentinel2_B06_rededge_740nm_REE.tif": {
        "title": "Sentinel-2 MSI Band 06: Red Edge 2 / REE (740 nm)",
        "subtitle": "Katghora District | 20m Ground Resolution | Nd3+ Rare Earth Absorption",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 06 (Red Edge 2)",
        "center_wl": "740.5 nm (Bandwidth: 15 nm)",
        "res": "20 meters",
        "target": "Trivalent Neodymium (Nd3+)\nsharp electronic absorption dip.",
    },
    "sentinel2_B08_broad_nir_842nm.tif": {
        "title": "Sentinel-2 MSI Band 08: Broad NIR (842 nm)",
        "subtitle": "Katghora District | 10m Ground Resolution | BOA Surface Reflectance",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 08 (Broad NIR)",
        "center_wl": "832.8 nm (Bandwidth: 106 nm)",
        "res": "10 meters",
        "target": "Canopy biomass scattering plateaus,\nwater-body shoreline delineation.",
    },
    "sentinel2_B11_swir1_1610nm.tif": {
        "title": "Sentinel-2 MSI Band 11: SWIR-1 (1610 nm)",
        "subtitle": "Katghora District | 20m Ground Resolution | Felsic Silicate Shoulder",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 11 (SWIR-1)",
        "center_wl": "1613.7 nm (Bandwidth: 91 nm)",
        "res": "20 meters",
        "target": "High felsic pegmatite reflectance,\nhydrothermal alteration continuum.",
    },
    "sentinel2_B12_swir2_2190nm_AlOH.tif": {
        "title": "Sentinel-2 MSI Band 12: SWIR-2 / Al-OH (2190 nm)",
        "subtitle": "Katghora District | 20m Ground Resolution | Al-OH Vibrational Absorption",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 12 (SWIR-2)",
        "center_wl": "2202.4 nm (Bandwidth: 175 nm)",
        "res": "20 meters",
        "target": "Diagnostic 2.20µm Al-OH\nabsorption in lepidolite & clays.",
    },
    "sentinel2_B8A_narrow_nir_865nm.tif": {
        "title": "Sentinel-2 MSI Band 8A: Narrow NIR (865 nm)",
        "subtitle": "Katghora District | 20m Ground Resolution | Unabsorbed NIR Continuum",
        "colormap": "bone",
        "cbar_label": "Surface Reflectance (BOA)",
        "layer_name": "Band 8A (Narrow NIR)",
        "center_wl": "864.7 nm (Bandwidth: 21 nm)",
        "res": "20 meters",
        "target": "Atmospheric water-vapor immune\nNIR reference continuum.",
    },
}

RGB_COMPOSITES_META = {
    "composite_true_color_RGB_B04_B03_B02.png": {
        "title": "Sentinel-2 True Color Natural RGB Composite",
        "subtitle": "Katghora District, Korba | Bands: Red (B04), Green (B03), Blue (B02)",
        "r_band": "sentinel2_B04_red_665nm.tif",
        "g_band": "sentinel2_B03_green_560nm.tif",
        "b_band": "sentinel2_B02_blue_490nm.tif",
        "r_desc": "B04 (665 nm) - Red Visible Spectrum",
        "g_desc": "B03 (560 nm) - Green Visible Spectrum",
        "b_desc": "B02 (490 nm) - Blue Visible Spectrum",
        "composite_name": "True Color RGB (B04-B03-B02)",
        "interpretation": (
            "• Vegetation: Natural green tones\n"
            "• Bare Soils / Rocks: Tan & ochre hues\n"
            "• Water / Reservoirs: Deep blue-black\n"
            "• Urban / Cleared Land: Bright whitish-gray"
        ),
        "significance": "Provides baseline geomorphological context and true-to-eye landcover reference.",
    },
    "composite_swir_mineral_B12_B8A_B04.png": {
        "title": "Sentinel-2 SWIR Mineral & Pegmatite False-Color Composite",
        "subtitle": "Katghora District, Korba | Bands: Red (B12), Green (B8A), Blue (B04)",
        "r_band": "sentinel2_B12_swir2_2190nm_AlOH.tif",
        "g_band": "sentinel2_B8A_narrow_nir_865nm.tif",
        "b_band": "sentinel2_B04_red_665nm.tif",
        "r_desc": "B12 (2190 nm) - Al-OH / Hydroxyl Absorption",
        "g_desc": "B8A (865 nm) - Narrow NIR Biomass Reflection",
        "b_desc": "B04 (665 nm) - Visible Red Ferric Iron",
        "composite_name": "SWIR Mineral Composite (B12-B8A-B04)",
        "interpretation": (
            "• Pegmatite & Leucosomes: Violet / Magenta tones\n"
            "• Dense Forest Canopy: Intense vibrant green\n"
            "• Ferric Gossan Caps: Golden amber to red-orange\n"
            "• Host Gneiss / Country Rock: Muted bronze / brown\n"
            "• Hasdeo River / Water Bodies: Deep black"
        ),
        "significance": "Optimal false-color blend for geological mapping; separates hydroxyl-bearing pegmatites from background forest.",
    },
    "composite_ree_alteration_B11_B06_B02.png": {
        "title": "Sentinel-2 REE & Hydroxyl Alteration False-Color Composite",
        "subtitle": "Katghora District, Korba | Bands: Red (B11), Green (B06), Blue (B02)",
        "r_band": "sentinel2_B11_swir1_1610nm.tif",
        "g_band": "sentinel2_B06_rededge_740nm_REE.tif",
        "b_band": "sentinel2_B02_blue_490nm.tif",
        "r_desc": "B11 (1610 nm) - SWIR-1 Felsic Shoulder",
        "g_desc": "B06 (740 nm) - REE Nd3+ Electronic Trough",
        "b_desc": "B02 (490 nm) - Visible Blue Albedo",
        "composite_name": "REE Alteration Composite (B11-B06-B02)",
        "interpretation": (
            "• REE Alteration Zones: Bright cyan / yellow-gold\n"
            "• Greisen / Mica Selvages: Vivid reddish-orange\n"
            "• Unaltered Metasediments: Muted olive-gray\n"
            "• Agricultural Fields: Lime green hues\n"
            "• Water / Sludge: Pitch dark navy"
        ),
        "significance": "Directly illuminates Nd3+ REE electronic absorption zones coupled with hydrothermal SWIR alteration.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GEOSPATIAL HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def parse_bounds_from_tif(filepath: str) -> tuple[float, float, float, float]:
    """Extracts bounding box from GeoTIFF ImageDescription or falls back to default."""
    try:
        with tifffile.TiffFile(filepath) as tif:
            desc = tif.pages[0].tags.get("ImageDescription")
            if desc and desc.value:
                match = re.search(r"bounds=\(([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+)\)", desc.value)
                if match:
                    min_lat, min_lon, max_lat, max_lon = map(float, match.groups())
                    return min_lat, min_lon, max_lat, max_lon
    except Exception:
        pass
    return DEFAULT_BOUNDS


def normalize_band_percentile(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0, nodata: float = -9999.0):
    """
    Computes dynamic contrast stretching using the 2nd and 98th percentiles.
    Cleanly masks NoData/NaN/Inf values.
    Returns: (stretched_array, valid_mask, p2, p98, stats_dict)
    """
    valid = (arr != nodata) & ~np.isnan(arr) & ~np.isinf(arr) & (arr > -9000.0)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32), valid, 0.0, 1.0, {}

    valid_vals = arr[valid]
    p2 = float(np.percentile(valid_vals, p_low))
    p98 = float(np.percentile(valid_vals, p_high))

    if p98 <= p2:
        p98 = p2 + 1e-4

    stretched = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
    stretched[~valid] = 0.0

    stats = {
        "min": float(valid_vals.min()),
        "max": float(valid_vals.max()),
        "mean": float(valid_vals.mean()),
        "std": float(valid_vals.std()),
        "p2": p2,
        "p98": p98,
        "valid_count": int(valid.sum()),
        "total_count": int(arr.size),
        "nodata_count": int((~valid).sum()),
    }
    return stretched, valid, p2, p98, stats


def add_cartographic_elements(ax, bounds: tuple[float, float, float, float]):
    """
    Adds a subtle cartographic North Arrow, calibrated Scale Bar, and Geographic Grid.
    """
    min_lat, min_lon, max_lat, max_lon = bounds

    # Grid & Ticks
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect(1.0 / np.cos(np.radians((min_lat + max_lat) / 2.0)))

    # Formatting ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.2f}°E"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.2f}°N"))

    ax.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2, color="#334155")
    ax.tick_params(axis="both", which="minor", length=3, width=0.8, color="#64748b")
    ax.grid(True, which="major", linestyle="--", color="#64748b", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":", color="#94a3b8", alpha=0.20, linewidth=0.5)

    ax.set_xlabel("Longitude (WGS 84 / Geographic Grid)", fontsize=12, fontweight="bold", color="#1e293b", labelpad=8)
    ax.set_ylabel("Latitude (WGS 84 / Geographic Grid)", fontsize=12, fontweight="bold", color="#1e293b", labelpad=8)

    # 1. Subtle North Arrow in Top-Right
    # Compact, elegant card (5.5% x 9.5% of axes) with solid white background to avoid obscuring data
    na_w, na_h = 0.055, 0.095
    na_x, na_y = 0.925, 0.880
    na_card = patches.FancyBboxPatch(
        (na_x, na_y), na_w, na_h, boxstyle="round,pad=0.008",
        transform=ax.transAxes, facecolor="white", edgecolor="#64748b", alpha=0.98, lw=1.1, zorder=30
    )
    ax.add_patch(na_card)
    cx = na_x + na_w / 2.0
    ax.text(cx, na_y + na_h - 0.022, "N", transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#0f172a", zorder=32)
    p_l = patches.Polygon([[cx, na_y + na_h - 0.035], [cx - 0.012, na_y + 0.016], [cx, na_y + 0.026]],
                          transform=ax.transAxes, facecolor="#0f172a", edgecolor="#0f172a", zorder=31)
    p_r = patches.Polygon([[cx, na_y + na_h - 0.035], [cx + 0.012, na_y + 0.016], [cx, na_y + 0.026]],
                          transform=ax.transAxes, facecolor="#ffffff", edgecolor="#0f172a", lw=1.1, zorder=31)
    ax.add_patch(p_l)
    ax.add_patch(p_r)

    # 2. Dynamic Calibrated Scale Bar in Bottom-Left
    mid_lat = (min_lat + max_lat) / 2.0
    km_per_deg_lon = 111.320 * np.cos(np.radians(mid_lat))
    total_lon_deg = max_lon - min_lon
    total_width_km = total_lon_deg * km_per_deg_lon

    if total_width_km >= 40:
        bar_km = 10.0
    elif total_width_km >= 20:
        bar_km = 5.0
    elif total_width_km >= 8:
        bar_km = 2.0
    else:
        bar_km = 1.0

    deg_bar = bar_km / km_per_deg_lon
    seg_len_axes = (deg_bar / total_lon_deg) / 2.0  # half-bar segment in axes fraction
    half_km = bar_km / 2.0

    sb_w = 2 * seg_len_axes + 0.065
    sb_h = 0.078
    sb_x, sb_y = 0.025, 0.025
    sb_card = patches.FancyBboxPatch(
        (sb_x, sb_y), sb_w, sb_h, boxstyle="round,pad=0.008",
        transform=ax.transAxes, facecolor="white", edgecolor="#64748b", alpha=0.98, lw=1.1, zorder=30
    )
    ax.add_patch(sb_card)

    x_sb = sb_x + 0.018
    y_sb = sb_y + 0.022
    h_sb = 0.013
    seg1 = patches.Rectangle((x_sb, y_sb), seg_len_axes, h_sb,
                             transform=ax.transAxes, facecolor="#0f172a", edgecolor="#0f172a", zorder=31)
    seg2 = patches.Rectangle((x_sb + seg_len_axes, y_sb), seg_len_axes, h_sb,
                             transform=ax.transAxes, facecolor="#ffffff", edgecolor="#0f172a", lw=1.1, zorder=31)
    ax.add_patch(seg1)
    ax.add_patch(seg2)

    half_label = f"{half_km:g}"
    full_label = f"{bar_km:g} km"
    ax.text(x_sb, y_sb + 0.019, "0", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a", zorder=32)
    ax.text(x_sb + seg_len_axes, y_sb + 0.019, half_label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a", zorder=32)
    ax.text(x_sb + 2 * seg_len_axes, y_sb + 0.019, full_label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a", zorder=32)
    ax.text(x_sb + seg_len_axes, y_sb - 0.014, "Calibrated Scale (WGS 84)", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#475569", zorder=32)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE LAYER RENDERING ENGINE (Indices & Individual Bands)
# ─────────────────────────────────────────────────────────────────────────────
def render_single_layer(tif_filename: str, meta: dict, output_path: str, input_dir: str = INPUT_DIR):
    """
    Renders an ultra-high-resolution (300 DPI, 4K+) cartographic image for a single raster layer.
    """
    tif_path = os.path.join(input_dir, tif_filename)
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Input GeoTIFF not found: {tif_path}")

    arr = tifffile.imread(tif_path).astype(np.float32)
    bounds = parse_bounds_from_tif(tif_path)
    min_lat, min_lon, max_lat, max_lon = bounds
    extent = [min_lon, max_lon, min_lat, max_lat]

    stretched, valid, p2, p98, stats = normalize_band_percentile(arr, 2.0, 98.0)

    # Mask out NoData cleanly
    masked_data = np.ma.masked_where(~valid, arr)

    colormap_name = meta.get("colormap", "viridis")
    cmap = plt.get_cmap(colormap_name).copy()
    try:
        cmap = cmap.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    except AttributeError:
        cmap.set_bad(color="#ffffff", alpha=0.0)  # Pure white / transparent background for nodata

    norm = plt.Normalize(vmin=p2, vmax=p98, clip=True)

    # Canvas dimensions: 22 x 14.5 inches at 300 DPI -> ensures > 4800 x 3600 pixels
    fig, ax = plt.subplots(figsize=(22.0, 14.5), dpi=DPI, facecolor="#ffffff")

    im = ax.imshow(
        masked_data,
        extent=extent,
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest"
    )

    # Cartographic Elements (Grid, North Arrow, Scale Bar)
    add_cartographic_elements(ax, bounds)

    # Titles & Header
    ax.set_title(
        f"{meta['title']}\n{meta['subtitle']}",
        fontsize=16, fontweight="bold", pad=18, color="#0f172a", loc="center"
    )

    # Divider for Colorbar and Info Panel
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.30)
    cbar = fig.colorbar(im, cax=cax, extend="both")
    cbar.set_label(meta["cbar_label"], fontsize=11.5, fontweight="bold", color="#1e293b", labelpad=14)
    cbar.ax.tick_params(labelsize=10, color="#334155")

    # Side Info Panel - padded sufficiently (1.20) so the colorbar label is 100% visible and unclipped
    info_ax = divider.append_axes("right", size="16.0%", pad=1.20)
    info_ax.axis("off")

    formula_text = meta.get("formula", meta.get("center_wl", "N/A"))
    bands_text = meta.get("bands", f"Spatial Res: {meta.get('res', '10m')}")
    target_text = meta.get("target", "Lithium / Rare-Metal Exploration")
    interp_text = meta.get("interpretation", meta.get("significance", "Surface Reflectance layer."))

    lat_mid = (min_lat + max_lat) / 2.0
    km_lat = (max_lat - min_lat) * 111.0
    km_lon = (max_lon - min_lon) * 111.32 * np.cos(np.radians(lat_mid))
    area_km2 = int(round(km_lat * km_lon))

    info_str = (
        f"LAYER SPECIFICATIONS\n"
        f"──────────────────────────────\n"
        f"Layer:    {meta['layer_name']}\n"
        f"Colormap: {colormap_name.upper()}\n"
        f"Sensor:   Sentinel-2 MSI (L2A)\n"
        f"Format:   GeoTIFF Float32\n\n"
        f"SPECTRAL FORMULATION\n"
        f"──────────────────────────────\n"
        f"{formula_text}\n"
        f"{bands_text}\n\n"
        f"DYNAMIC STRETCH (2%-98%)\n"
        f"──────────────────────────────\n"
        f"• 2nd Percentile:  {p2:.4f}\n"
        f"• 98th Percentile: {p98:.4f}\n"
        f"• Mean Value:      {stats.get('mean', 0.0):.4f}\n"
        f"• Std Deviation:   {stats.get('std', 0.0):.4f}\n"
        f"• Valid Pixels:    {stats.get('valid_count', 0):,} ({100*stats.get('valid_count', 0)/stats.get('total_count', 1):.1f}%)\n"
        f"• Masked NoData:   {stats.get('nodata_count', 0):,}\n\n"
        f"EXPLORATION TARGET\n"
        f"──────────────────────────────\n"
        f"{target_text}\n\n"
        f"REMOTE SENSING VECTOR\n"
        f"──────────────────────────────\n"
        f"{interp_text}\n\n"
        f"CARTOGRAPHIC FRAME\n"
        f"──────────────────────────────\n"
        f"Extent: {min_lat:.2f}°N – {max_lat:.2f}°N\n"
        f"        {min_lon:.2f}°E – {max_lon:.2f}°E\n"
        f"Grid:   {arr.shape[0]} × {arr.shape[1]} pixels\n"
        f"Area:   ~{area_km2:,} km²\n"
        f"CRS:    WGS 84 (EPSG:4326)\n"
        f"DPI:    300 Ultra-HD (4K+)"
    )

    info_ax.text(
        0.02, 0.99, info_str, transform=info_ax.transAxes, fontsize=10.0,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.4)
    )

    # Export crisp PNG or high-quality JPG
    save_kwargs = {
        "dpi": DPI,
        "bbox_inches": "tight",
        "facecolor": "#ffffff",
        "edgecolor": "none",
    }
    if output_path.lower().endswith((".jpg", ".jpeg")):
        save_kwargs["pil_kwargs"] = {"quality": 95}

    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

    # Verify dimensions
    with Image.open(output_path) as img:
        width, height = img.size
        size_bytes = os.path.getsize(output_path)

    return width, height, size_bytes


# ─────────────────────────────────────────────────────────────────────────────
# RGB COMPOSITE RENDERING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def render_rgb_composite(meta: dict, output_path: str, input_dir: str = INPUT_DIR):
    """
    Renders an ultra-high-resolution (300 DPI, 4K+) 3-band RGB composite image.
    """
    r_path = os.path.join(input_dir, meta["r_band"])
    g_path = os.path.join(input_dir, meta["g_band"])
    b_path = os.path.join(input_dir, meta["b_band"])

    for p in [r_path, g_path, b_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input band GeoTIFF not found: {p}")

    r_arr = tifffile.imread(r_path).astype(np.float32)
    g_arr = tifffile.imread(g_path).astype(np.float32)
    b_arr = tifffile.imread(b_path).astype(np.float32)

    bounds = parse_bounds_from_tif(r_path)
    min_lat, min_lon, max_lat, max_lon = bounds
    extent = [min_lon, max_lon, min_lat, max_lat]

    # Percentile stretch each channel independently
    r_norm, r_val, r_p2, r_p98, r_stats = normalize_band_percentile(r_arr, 2.0, 98.0)
    g_norm, g_val, g_p2, g_p98, g_stats = normalize_band_percentile(g_arr, 2.0, 98.0)
    b_norm, b_val, b_p2, b_p98, b_stats = normalize_band_percentile(b_arr, 2.0, 98.0)

    overall_valid = r_val & g_val & b_val

    # Construct RGBA stack
    rgba = np.zeros((*r_norm.shape, 4), dtype=np.float32)
    rgba[..., 0] = r_norm
    rgba[..., 1] = g_norm
    rgba[..., 2] = b_norm
    rgba[..., 3] = np.where(overall_valid, 1.0, 0.0)

    # Canvas dimensions: 21 x 14.5 inches at 300 DPI
    fig, ax = plt.subplots(figsize=(21.0, 14.5), dpi=DPI, facecolor="#ffffff")

    ax.imshow(
        rgba,
        extent=extent,
        origin="upper",
        interpolation="nearest"
    )

    # Cartographic Elements (Grid, North Arrow, Scale Bar)
    add_cartographic_elements(ax, bounds)

    # Titles & Header
    ax.set_title(
        f"{meta['title']}\n{meta['subtitle']}",
        fontsize=16, fontweight="bold", pad=18, color="#0f172a", loc="center"
    )

    # Divider for Side Legend / Interpretation Panel
    divider = make_axes_locatable(ax)
    info_ax = divider.append_axes("right", size="19.5%", pad=0.45)
    info_ax.axis("off")

    info_str = (
        f"RGB COMPOSITE GUIDE\n"
        f"──────────────────────────────\n"
        f"Composite: {meta['composite_name']}\n"
        f"Sensor:    Sentinel-2 MSI (Level-2A)\n"
        f"Dynamic:   2%–98% Percentile Stretch\n\n"
        f"CHANNEL ALLOCATION\n"
        f"──────────────────────────────\n"
        f"• RED CHANNEL:\n"
        f"  {meta['r_desc']}\n"
        f"  Stretch: [{r_p2:.4f} – {r_p98:.4f}]\n\n"
        f"• GREEN CHANNEL:\n"
        f"  {meta['g_desc']}\n"
        f"  Stretch: [{g_p2:.4f} – {g_p98:.4f}]\n\n"
        f"• BLUE CHANNEL:\n"
        f"  {meta['b_desc']}\n"
        f"  Stretch: [{b_p2:.4f} – {b_p98:.4f}]\n\n"
        f"COLOR INTERPRETATION KEY\n"
        f"──────────────────────────────\n"
        f"{meta['interpretation']}\n\n"
        f"EXPLORATION SIGNIFICANCE\n"
        f"──────────────────────────────\n"
        f"{meta['significance']}\n\n"
        f"CARTOGRAPHIC METADATA\n"
        f"──────────────────────────────\n"
        f"Grid:  {r_arr.shape[0]} × {r_arr.shape[1]} pixels\n"
        f"Lat:   {min_lat:.2f}°N – {max_lat:.2f}°N\n"
        f"Lon:   {min_lon:.2f}°E – {max_lon:.2f}°E\n"
        f"Valid: {overall_valid.sum():,} pixels\n"
        f"CRS:   WGS 84 (EPSG:4326)\n"
        f"DPI:   300 Ultra-HD (4K+)"
    )

    info_ax.text(
        0.02, 0.99, info_str, transform=info_ax.transAxes, fontsize=10.0,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.4)
    )

    # Export crisp PNG or high-quality JPG
    save_kwargs = {
        "dpi": DPI,
        "bbox_inches": "tight",
        "facecolor": "#ffffff",
        "edgecolor": "none",
    }
    if output_path.lower().endswith((".jpg", ".jpeg")):
        save_kwargs["pil_kwargs"] = {"quality": 95}

    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

    with Image.open(output_path) as img:
        width, height = img.size
        size_bytes = os.path.getsize(output_path)

    return width, height, size_bytes


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def main(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR, export_format: str = "png"):
    print("=" * 80)
    print("ULTRA-HIGH-RESOLUTION (300 DPI, 4K+) CARTOGRAPHIC RASTER EXPORT ENGINE")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Export Format:    {export_format.upper()}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    generated_records = []
    ext = f".{export_format.lower().lstrip('.')}"

    # 1. RENDER MINERAL INDICES
    print("\n>>> 1. Processing Mineral Exploration Indices (Scientific Colormaps)...")
    for tif_name, meta in MINERAL_INDICES_META.items():
        tif_path = os.path.join(input_dir, tif_name)
        if not os.path.exists(tif_path):
            print(f"  [SKIPPED] Input index GeoTIFF not found: {tif_name}")
            continue
        out_name = os.path.splitext(tif_name)[0] + ext
        out_path = os.path.join(output_dir, out_name)
        print(f"  Rendering [{meta['colormap'].upper()}]: {tif_name} -> {out_name}...")
        w, h, sz = render_single_layer(tif_name, meta, out_path, input_dir=input_dir)
        is_4k = (w >= 3840 and h >= 2160) or (w * h >= 3840 * 2160)
        print(f"    -> Resolution: {w}x{h} px | Size: {sz:,} bytes | 4K+ Compliant: {is_4k}")
        generated_records.append({
            "category": "Mineral Index",
            "layer": meta["layer_name"],
            "filename": out_name,
            "path": out_path,
            "width": w,
            "height": h,
            "bytes": sz,
            "colormap": meta["colormap"],
            "compliant": is_4k
        })

    # 2. RENDER RGB COMPOSITES
    print("\n>>> 2. Synthesizing False-Color & Natural RGB Composites (3-Band Blends)...")
    for comp_base, meta in RGB_COMPOSITES_META.items():
        r_p = os.path.join(input_dir, meta["r_band"])
        g_p = os.path.join(input_dir, meta["g_band"])
        b_p = os.path.join(input_dir, meta["b_band"])
        if not (os.path.exists(r_p) and os.path.exists(g_p) and os.path.exists(b_p)):
            print(f"  [SKIPPED] Missing required bands for composite: {meta['composite_name']}")
            continue
        out_name = os.path.splitext(comp_base)[0] + ext
        out_path = os.path.join(output_dir, out_name)
        print(f"  Synthesizing [{meta['composite_name']}]: {out_name}...")
        w, h, sz = render_rgb_composite(meta, out_path, input_dir=input_dir)
        is_4k = (w >= 3840 and h >= 2160) or (w * h >= 3840 * 2160)
        print(f"    -> Resolution: {w}x{h} px | Size: {sz:,} bytes | 4K+ Compliant: {is_4k}")
        generated_records.append({
            "category": "RGB Composite",
            "layer": meta["composite_name"],
            "filename": out_name,
            "path": out_path,
            "width": w,
            "height": h,
            "bytes": sz,
            "colormap": "RGB Multi-Channel",
            "compliant": is_4k
        })

    # 3. RENDER INDIVIDUAL SENTINEL-2 BANDS
    print("\n>>> 3. Processing Individual Sentinel-2 Optical & Infrared Bands...")
    for tif_name, meta in SENTINEL_BANDS_META.items():
        tif_path = os.path.join(input_dir, tif_name)
        if not os.path.exists(tif_path):
            print(f"  [SKIPPED] Input band GeoTIFF not found: {tif_name}")
            continue
        out_name = os.path.splitext(tif_name)[0] + ext
        out_path = os.path.join(output_dir, out_name)
        print(f"  Rendering [{meta['colormap'].upper()}]: {tif_name} -> {out_name}...")
        w, h, sz = render_single_layer(tif_name, meta, out_path, input_dir=input_dir)
        is_4k = (w >= 3840 and h >= 2160) or (w * h >= 3840 * 2160)
        print(f"    -> Resolution: {w}x{h} px | Size: {sz:,} bytes | 4K+ Compliant: {is_4k}")
        generated_records.append({
            "category": "Sentinel-2 Band",
            "layer": meta["layer_name"],
            "filename": out_name,
            "path": out_path,
            "width": w,
            "height": h,
            "bytes": sz,
            "colormap": meta["colormap"],
            "compliant": is_4k
        })

    # 4. AUTO-DISCOVERY OF ANY ADDITIONAL GEOTIFFS
    known_tifs = set(MINERAL_INDICES_META.keys()) | set(SENTINEL_BANDS_META.keys())
    if os.path.exists(input_dir):
        all_tifs = {f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))}
        unprocessed = sorted(all_tifs - known_tifs)
        if unprocessed:
            print(f"\n>>> 4. Auto-Discovered Additional GeoTIFFs ({len(unprocessed)} files)...")
            for tif_name in unprocessed:
                stem = os.path.splitext(tif_name)[0]
                out_name = stem + ext
                out_path = os.path.join(output_dir, out_name)
                custom_meta = {
                    "title": f"GeoTIFF Raster Layer: {stem}",
                    "subtitle": f"High-Resolution Exploration Layer | {os.path.basename(input_dir)}",
                    "colormap": "turbo",
                    "cbar_label": f"Raster Value: {stem}",
                    "layer_name": stem,
                    "formula": "Single-Band Normalized Array",
                    "bands": "Auto-Discovered GeoTIFF",
                    "target": "Exploration Target Area",
                    "interpretation": "Spatial distribution of raster values stretched dynamically across 2nd–98th percentiles.",
                }
                print(f"  Rendering auto-discovered: {tif_name} -> {out_name}...")
                w, h, sz = render_single_layer(tif_name, custom_meta, out_path, input_dir=input_dir)
                is_4k = (w >= 3840 and h >= 2160) or (w * h >= 3840 * 2160)
                print(f"    -> Resolution: {w}x{h} px | Size: {sz:,} bytes | 4K+ Compliant: {is_4k}")
                generated_records.append({
                    "category": "Auto-Discovered Raster",
                    "layer": stem,
                    "filename": out_name,
                    "path": out_path,
                    "width": w,
                    "height": h,
                    "bytes": sz,
                    "colormap": "turbo",
                    "compliant": is_4k
                })

    print("\n" + "=" * 80)
    print("SUMMARY OF GENERATED HIGH-RESOLUTION ARTIFACTS:")
    print("=" * 80)
    print(f"{'Category':<16} | {'Layer / Product':<38} | {'Dimensions':<12} | {'File Size':<10} | {'Status'}")
    print("-" * 95)
    all_passed = True
    for r in generated_records:
        status = "PASS (4K+)" if r["compliant"] else "WARN (<4K)"
        if not r["compliant"]:
            all_passed = False
        print(f"{r['category']:<16} | {r['layer']:<38} | {r['width']}x{r['height']:<10} | {r['bytes']//1024:>6} KB | {status}")

    print("-" * 95)
    print(f"Total Snapshots Generated: {len(generated_records)} files")
    print(f"All 4K+ & 300 DPI Requirements Met: {all_passed}")
    print(f"Destination: {output_dir}")
    print("=" * 80)

    return generated_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-High-Resolution GeoTIFF Cartographic Snapshot Generator")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR, help="Path to GeoTIFF directory")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for snapshots")
    parser.add_argument("--format", type=str, choices=["png", "jpg", "jpeg"], default="png", help="Export image format")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, export_format=args.format)
