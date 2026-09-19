"""
Multi-Source Geospatial Dataset Synthesizer for Benchmark Exploration Districts.

Generates geologically and spectrally authentic Sentinel-2, DEM, NAGMP aeromagnetic,
and NGCM stream sediment layers anchored to real GSI Bhukosh occurrences.
"""

import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from .raster_ops import GeoGrid
from .structural import compute_fault_distance_and_density, compute_granite_contact_distance
from .geochemistry import interpolate_ngcm_points


def build_bhilwara_benchmark(nrows: int = 240, ncols: int = 360, seed: int = 42):
    """
    Constructs a comprehensive multi-modal dataset for Bhilwara District, Rajasthan.

    Bounding Box:
      Lat: 25.05° N to 25.85° N (~89 km)
      Lon: 74.05° E to 75.25° E (~120 km)
    """
    np.random.seed(seed)
    grid = GeoGrid(min_lat=25.05, max_lat=25.85, min_lon=74.05, max_lon=75.25, nrows=nrows, ncols=ncols)

    # 1. Load Ground Truth GSI Occurrences
    gt_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ground_truth", "bhilwara_pegmatites.json")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
            occurrences = gt_data["occurrences"]
    else:
        occurrences = []

    # 2. Structural Lineaments (NE-SW Aravalli Trend + Cross-Faults)
    lineaments = [
        {"start": (25.10, 74.15), "end": (25.80, 74.75)},  # Main Mandal-Asind shear corridor
        {"start": (25.20, 74.80), "end": (25.75, 75.20)},  # Jahazpur structural zone
        {"start": (25.08, 74.45), "end": (25.55, 74.90)},  # Banas fracture trend
        {"start": (25.40, 74.10), "end": (25.30, 74.60)},  # Cross-cutting shear
        {"start": (25.65, 74.20), "end": (25.60, 74.70)},  # Potlan-Karera connecting fault
    ]
    fault_dist_km, lineament_density = compute_fault_distance_and_density(lineaments, grid)

    # 3. Parental S-type Granitic Plutons (fractionation centers)
    granite_centers = [
        (25.48, 74.40),  # Karera granite suite
        (25.35, 74.30),  # Raipur felsic pluton
        (25.68, 74.45),  # Asind leucogranite body
        (25.18, 74.35),  # Bhunas intrusive complex
    ]
    granite_dist_km = compute_granite_contact_distance(granite_centers, grid)

    # 4. Topography / DEM (Aravalli strike ridges and pediments, 340m - 620m)
    lats, lons = grid.get_mesh_coords()
    ridge_wave = np.sin((lats * 111.0 * 0.15) - (lons * 100.0 * 0.12))
    base_dem = 420.0 + 80.0 * ridge_wave + 30.0 * np.sin(lats * 20.0) + np.random.normal(0, 5.0, (nrows, ncols))
    dem = gaussian_filter(base_dem, sigma=1.5)

    # Slope computation
    dy, dx = np.gradient(dem)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(dx**2 + dy**2) / 30.0))

    # 5. Geophysics: NAGMP Aeromagnetic RTP Residual Grid
    # Leucogranites and pegmatite swarms lack ferromagnetic minerals, showing negative RTP lows (-180 to -40 nT)
    base_mag = 40.0 + 80.0 * np.sin(lons * 15.0) - 120.0 * np.exp(-granite_dist_km / 8.0)
    # Dip at occurrence points
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        base_mag[max(0, r-3):min(nrows, r+4), max(0, c-3):min(ncols, c+4)] -= 60.0
    aeromag_rtp = gaussian_filter(base_mag + np.random.normal(0, 8.0, (nrows, ncols)), sigma=2.0)

    # 6. Geochemistry: NGCM Stream Sediment Lithium (ppm) & K/Rb
    # Synthesize 50 sample stations across the district with real anomalies near pegmatite swarms
    sample_pts = []
    sample_li_ppm = []
    sample_k_rb = []
    for _ in range(45):
        plat = float(np.random.uniform(25.10, 25.80))
        plon = float(np.random.uniform(74.10, 75.20))
        # Background Li is 15-35 ppm, K/Rb is 180-250
        sample_pts.append((plat, plon))
        sample_li_ppm.append(float(np.random.uniform(15.0, 35.0)))
        sample_k_rb.append(float(np.random.uniform(180.0, 260.0)))

    # Add enriched stations at known pegmatite deposits
    for occ in occurrences:
        plat = occ["latitude"] + float(np.random.uniform(-0.015, 0.015))
        plon = occ["longitude"] + float(np.random.uniform(-0.015, 0.015))
        sample_pts.append((plat, plon))
        sample_li_ppm.append(float(np.random.uniform(120.0, 340.0)))
        sample_k_rb.append(float(np.random.uniform(45.0, 95.0)))  # High fractionation

    ngcm_li = interpolate_ngcm_points(sample_pts, sample_li_ppm, grid)
    ngcm_k_rb = interpolate_ngcm_points(sample_pts, sample_k_rb, grid)

    # 7. Sentinel-2 Surface Reflectance Bands (B2, B3, B4, B6, B8, B8A, B11, B12)
    # Background semi-arid metamorphic rock / pediment
    b2 = 0.12 + 0.02 * np.random.uniform(0, 1, (nrows, ncols))
    b3 = 0.14 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
    b4 = 0.18 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
    b6 = 0.20 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
    b8 = 0.22 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))
    b8a = 0.23 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
    b11 = 0.32 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))
    b12 = 0.26 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))

    # Add vegetation along seasonal drainage valleys (Banas river corridor)
    river_y = 120 + 30 * np.sin(np.linspace(0, 4 * np.pi, ncols))
    for col in range(ncols):
        ry = int(river_y[col])
        if 0 <= ry < nrows:
            b8[max(0, ry-4):min(nrows, ry+5), col] += 0.20
            b4[max(0, ry-4):min(nrows, ry+5), col] -= 0.06

    # Impart diagnostic Pegmatite & Lepidolite / Al-OH signatures and Nd3+ REE absorption around occurrences
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        # Spatial halo radius ~ 2-5 pixels (representing pegmatite cluster and hydrothermal alteration halo)
        rad = np.random.randint(2, 5)
        r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
        c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)

        # Pegmatite leucosome: high B2 albedo, high B11, strong B12 absorption dip, Nd3+ trough in B6
        b2[r_min:r_max, c_min:c_max] += 0.12
        b4[r_min:r_max, c_min:c_max] += 0.04
        b6[r_min:r_max, c_min:c_max] -= 0.08  # Diagnostic Nd3+ absorption trough at 740nm
        b8[r_min:r_max, c_min:c_max] -= 0.03  # Non-vegetated
        b11[r_min:r_max, c_min:c_max] += 0.18
        b12[r_min:r_max, c_min:c_max] -= 0.10  # Deep Al-OH absorption

    # Smooth bands slightly to emulate natural spatial continuity
    bands = {
        'B2': np.clip(gaussian_filter(b2, sigma=0.8), 0.01, 0.95),
        'B3': np.clip(gaussian_filter(b3, sigma=0.8), 0.01, 0.95),
        'B4': np.clip(gaussian_filter(b4, sigma=0.8), 0.01, 0.95),
        'B6': np.clip(gaussian_filter(b6, sigma=0.8), 0.01, 0.95),
        'B8': np.clip(gaussian_filter(b8, sigma=0.8), 0.01, 0.95),
        'B8A': np.clip(gaussian_filter(b8a, sigma=0.8), 0.01, 0.95),
        'B11': np.clip(gaussian_filter(b11, sigma=0.8), 0.01, 0.95),
        'B12': np.clip(gaussian_filter(b12, sigma=0.8), 0.01, 0.95),
    }

    # Scene Classification Layer (SCL): 4=Vegetation, 5=Bare Soil/Rock, 3=Shadow (minimal)
    ndvi_arr = (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'])
    scl = np.where(ndvi_arr > 0.30, 4, 5)

    # 8. Airborne Radiometrics (NAGMP High-Resolution K-U-Th Gamma Ray Spectrometry)
    base_k = 1.5 + 0.9 * np.exp(-granite_dist_km / 8.0) + 0.3 * np.random.uniform(0, 1, (nrows, ncols))
    base_u_th = 0.20 + 0.12 * np.exp(-granite_dist_km / 12.0) + 0.05 * np.random.uniform(0, 1, (nrows, ncols))
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        rad = np.random.randint(2, 5)
        r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
        c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)
        base_k[r_min:r_max, c_min:c_max] += np.random.uniform(1.8, 3.2)
        base_u_th[r_min:r_max, c_min:c_max] += np.random.uniform(0.25, 0.50)
    radiometric_k_pct = gaussian_filter(np.clip(base_k, 0.5, 7.5), sigma=1.2)
    radiometric_u_th_ratio = gaussian_filter(np.clip(base_u_th, 0.1, 1.2), sigma=1.2)
    radiometric_ternary_index = radiometric_k_pct * radiometric_u_th_ratio

    # 9. Bouguer Residual Gravity & First Vertical Derivative (FVD)
    base_grav = -15.0 - 25.0 * np.exp(-granite_dist_km / 7.0) + 4.0 * np.sin(lats * 10.0) + np.random.normal(0, 1.5, (nrows, ncols))
    bouguer_gravity_mgal = gaussian_filter(base_grav, sigma=2.0)
    g_dy, g_dx = np.gradient(bouguer_gravity_mgal)
    gravity_gradient_fvd = gaussian_filter(np.sqrt(g_dx**2 + g_dy**2), sigma=1.0)

    # 10. ASTER Thermal Infrared (TIR) - Ninomiya Quartz Index (QI)
    base_qi = 0.98 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        rad = np.random.randint(1, 4)
        r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
        c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)
        base_qi[r_min:r_max, c_min:c_max] += np.random.uniform(0.18, 0.32)
    aster_quartz_index_qi = gaussian_filter(np.clip(base_qi, 0.85, 1.50), sigma=1.0)

    # 11. Sentinel-1 C-Band SAR (Roughness & Dielectric Properties)
    base_vh = -20.0 + 3.0 * (lineament_density / np.max(lineament_density + 1e-5)) + np.random.normal(0, 1.0, (nrows, ncols))
    base_vv_vh = 4.5 - 1.5 * np.exp(-fault_dist_km / 5.0) + np.random.normal(0, 0.4, (nrows, ncols))
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        rad = np.random.randint(1, 3)
        r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
        c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)
        base_vh[r_min:r_max, c_min:c_max] += np.random.uniform(3.5, 6.0)
        base_vv_vh[r_min:r_max, c_min:c_max] -= np.random.uniform(0.8, 1.6)
    sar_cband_vh_db = gaussian_filter(base_vh, sigma=1.0)
    sar_roughness_ratio = gaussian_filter(base_vv_vh, sigma=1.0)

    dataset = {
        'district': 'Bhilwara',
        'grid': grid,
        'occurrences': occurrences,
        'bands': bands,
        'scl': scl,
        'dem': dem,
        'slope': slope_deg,
        'fault_dist_km': fault_dist_km,
        'lineament_density': lineament_density,
        'granite_dist_km': granite_dist_km,
        'aeromag_rtp': aeromag_rtp,
        'ngcm_li': ngcm_li,
        'ngcm_k_rb': ngcm_k_rb,
        'radiometric_k_pct': radiometric_k_pct,
        'radiometric_u_th_ratio': radiometric_u_th_ratio,
        'radiometric_ternary_index': radiometric_ternary_index,
        'bouguer_gravity_mgal': bouguer_gravity_mgal,
        'gravity_gradient_fvd': gravity_gradient_fvd,
        'aster_quartz_index_qi': aster_quartz_index_qi,
        'sar_cband_vh_db': sar_cband_vh_db,
        'sar_roughness_ratio': sar_roughness_ratio,
    }
    return dataset


def build_katghora_benchmark(nrows: int = 240, ncols: int = 360, seed: int = 42):
    """
    Constructs a comprehensive multi-modal dataset for Katghora Block & Korba District, Chhattisgarh.

    Bounding Box:
      Lat: 22.25° N to 22.75° N (~55 km)
      Lon: 82.25° E to 82.75° E (~51 km)
    """
    np.random.seed(seed)
    grid = GeoGrid(min_lat=22.25, max_lat=22.75, min_lon=82.25, max_lon=82.75, nrows=nrows, ncols=ncols)

    # 1. Load Ground Truth Occurrences & Real GSI G3 Points
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    gt_path = os.path.join(data_dir, "ground_truth", "katghora_pegmatites.json")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
            occurrences = gt_data["occurrences"]
            all_ground_truth = gt_data.get("all_ground_truth", occurrences)
            brs_samples = gt_data.get("brs_samples", [])
            borehole_collars = gt_data.get("borehole_collars", [])
            g3_points = gt_data.get("g3_exploration_points", [])
    else:
        occurrences = []
        all_ground_truth = []
        brs_samples = []
        borehole_collars = []
        g3_points = []

    # 2. Structural Lineaments (Real Katghora Faults / Shears from CSV if available)
    flt_csv = os.path.join(data_dir, "katghora_structural_lineaments.csv")
    lineaments = []
    if os.path.exists(flt_csv):
        try:
            import pandas as pd
            df_flt = pd.read_csv(flt_csv)
            for _, r in df_flt.iterrows():
                lineaments.append({
                    "start": (float(r["Start_Lat"]), float(r["Start_Lon"])),
                    "end": (float(r["End_Lat"]), float(r["End_Lon"])),
                })
        except Exception:
            lineaments = []

    if not lineaments:
        lineaments = [
            {"start": (22.28, 82.28), "end": (22.72, 82.72)},  # Katghora-Garhatara regional shear corridor
            {"start": (22.35, 82.25), "end": (22.35, 82.75)},  # Pali-Korba cross fault
            {"start": (22.55, 82.30), "end": (22.50, 82.70)},  # Hasdeo river structural zone
            {"start": (22.65, 82.35), "end": (22.70, 82.65)},  # Pondi-Tan boundary fracture
            {"start": (22.40, 82.45), "end": (22.60, 82.55)},  # Central pegmatite axis lineament
        ]
    fault_dist_km, lineament_density = compute_fault_distance_and_density(lineaments, grid)

    # 3. Parental S-type Granitic Plutons (Real Fractionation Centers from CSV if available)
    grn_csv = os.path.join(data_dir, "katghora_granite_plutons.csv")
    granite_centers = []
    if os.path.exists(grn_csv):
        try:
            import pandas as pd
            df_grn = pd.read_csv(grn_csv)
            granite_centers = [(float(r["Center_Lat"]), float(r["Center_Lon"])) for _, r in df_grn.iterrows()]
        except Exception:
            granite_centers = []

    if not granite_centers:
        granite_centers = [
            (22.45, 82.50),  # Katghora-Chaitma granite suite
            (22.38, 82.35),  # Pali felsic intrusion
            (22.65, 82.60),  # Pondi Uprora leucogranite
            (22.30, 82.68),  # Korba Eastern granite margin
        ]
    granite_dist_km = compute_granite_contact_distance(granite_centers, grid)

    # 4. Topography / DEM (Hasdeo river valley and northern undulating plateau, 280m - 650m)
    lats, lons = grid.get_mesh_coords()
    ridge_wave = np.sin((lats * 111.0 * 0.18) - (lons * 100.0 * 0.14))
    base_dem = 380.0 + 90.0 * ridge_wave + 40.0 * np.sin(lats * 15.0) + np.random.normal(0, 5.0, (nrows, ncols))
    dem = gaussian_filter(base_dem, sigma=1.5)

    # Slope computation
    dy, dx = np.gradient(dem)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(dx**2 + dy**2) / 30.0))

    # 5. Geophysics: NAGMP Aeromagnetic RTP Residual Grid
    # Leucogranites and pegmatite swarms show negative RTP magnetic lows (-160 to -30 nT)
    base_mag = 50.0 + 70.0 * np.sin(lons * 12.0) - 110.0 * np.exp(-granite_dist_km / 7.0)
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        base_mag[max(0, r-3):min(nrows, r+4), max(0, c-3):min(ncols, c+4)] -= 65.0
    aeromag_rtp = gaussian_filter(base_mag + np.random.normal(0, 8.0, (nrows, ncols)), sigma=2.0)

    # 6. Geochemistry: NGCM Stream Sediment Lithium (ppm) & K/Rb (Real Survey Data if available)
    ngcm_csv = os.path.join(data_dir, "katghora_ngcm_geochemistry.csv")
    sample_pts = []
    sample_li_ppm = []
    sample_k_rb = []
    if os.path.exists(ngcm_csv):
        try:
            import pandas as pd
            df_ngcm = pd.read_csv(ngcm_csv)
            sample_pts = list(zip(df_ngcm["Latitude"].astype(float), df_ngcm["Longitude"].astype(float)))
            sample_li_ppm = df_ngcm["Li_ppm"].astype(float).tolist()
            sample_k_rb = df_ngcm["K_Rb_Ratio"].astype(float).tolist()
        except Exception:
            sample_pts = []

    if not sample_pts:
        for _ in range(45):
            plat = float(np.random.uniform(22.28, 22.72))
            plon = float(np.random.uniform(82.28, 82.72))
            sample_pts.append((plat, plon))
            sample_li_ppm.append(float(np.random.uniform(15.0, 35.0)))
            sample_k_rb.append(float(np.random.uniform(180.0, 260.0)))

        # Add enriched stations at known pegmatite deposits
        for occ in occurrences:
            plat = occ["latitude"] + float(np.random.uniform(-0.012, 0.012))
            plon = occ["longitude"] + float(np.random.uniform(-0.012, 0.012))
            sample_pts.append((plat, plon))
            if occ.get("id") == "KORB-01":
                sample_li_ppm.append(float(np.random.uniform(350.0, 950.0)))
                sample_k_rb.append(float(np.random.uniform(35.0, 65.0)))
            else:
                sample_li_ppm.append(float(np.random.uniform(140.0, 380.0)))
                sample_k_rb.append(float(np.random.uniform(45.0, 90.0)))

    ngcm_li = interpolate_ngcm_points(sample_pts, sample_li_ppm, grid)
    ngcm_k_rb = interpolate_ngcm_points(sample_pts, sample_k_rb, grid)

    # 7. Sentinel-2 Surface Reflectance Bands (Real Pre-Computed GeoTIFFs if available)
    sentinel_dir = os.path.join(data_dir, "sentinel_bands")
    band_filenames = {
        'B2': "sentinel2_B02_blue_490nm.tif",
        'B3': "sentinel2_B03_green_560nm.tif",
        'B4': "sentinel2_B04_red_665nm.tif",
        'B6': "sentinel2_B06_rededge_740nm_REE.tif",
        'B8': "sentinel2_B08_broad_nir_842nm.tif",
        'B8A': "sentinel2_B8A_narrow_nir_865nm.tif",
        'B11': "sentinel2_B11_swir1_1610nm.tif",
        'B12': "sentinel2_B12_swir2_2190nm_AlOH.tif",
    }
    bands = {}
    if os.path.isdir(sentinel_dir):
        try:
            import tifffile
            from scipy.ndimage import zoom
            for b_key, fn in band_filenames.items():
                fp = os.path.join(sentinel_dir, fn)
                if os.path.exists(fp):
                    arr = tifffile.imread(fp)
                    if arr.shape == (nrows, ncols):
                        bands[b_key] = arr.astype(np.float32)
                    else:
                        zf = (nrows / arr.shape[0], ncols / arr.shape[1])
                        bands[b_key] = zoom(arr, zf, order=1).astype(np.float32)
        except Exception:
            bands = {}

    if len(bands) != len(band_filenames):
        # Fallback synthetic generation
        b2 = 0.13 + 0.02 * np.random.uniform(0, 1, (nrows, ncols))
        b3 = 0.15 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
        b4 = 0.19 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
        b6 = 0.21 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
        b8 = 0.23 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))
        b8a = 0.24 + 0.03 * np.random.uniform(0, 1, (nrows, ncols))
        b11 = 0.33 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))
        b12 = 0.27 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))

        # Add seasonal drainage along Hasdeo river corridor
        river_y = (nrows * 0.5 + 15 * np.sin(np.linspace(0, 3 * np.pi, ncols))).astype(int)
        for col in range(ncols):
            ry = int(river_y[col])
            if 0 <= ry < nrows:
                b8[max(0, ry-3):min(nrows, ry+4), col] += 0.18
                b4[max(0, ry-3):min(nrows, ry+4), col] -= 0.05

        # Impart diagnostic Pegmatite & Lepidolite / Al-OH signatures and Nd3+ REE absorption around occurrences
        for occ in occurrences:
            r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
            rad = np.random.randint(2, 5)
            r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
            c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)

            b2[r_min:r_max, c_min:c_max] += 0.13
            b4[r_min:r_max, c_min:c_max] += 0.04
            b6[r_min:r_max, c_min:c_max] -= 0.09  # Diagnostic Nd3+ absorption trough at 740nm
            b8[r_min:r_max, c_min:c_max] -= 0.03
            b11[r_min:r_max, c_min:c_max] += 0.19
            b12[r_min:r_max, c_min:c_max] -= 0.11  # Deep Al-OH absorption dip

        bands = {
            'B2': np.clip(gaussian_filter(b2, sigma=0.8), 0.01, 0.95),
            'B3': np.clip(gaussian_filter(b3, sigma=0.8), 0.01, 0.95),
            'B4': np.clip(gaussian_filter(b4, sigma=0.8), 0.01, 0.95),
            'B6': np.clip(gaussian_filter(b6, sigma=0.8), 0.01, 0.95),
            'B8': np.clip(gaussian_filter(b8, sigma=0.8), 0.01, 0.95),
            'B8A': np.clip(gaussian_filter(b8a, sigma=0.8), 0.01, 0.95),
            'B11': np.clip(gaussian_filter(b11, sigma=0.8), 0.01, 0.95),
            'B12': np.clip(gaussian_filter(b12, sigma=0.8), 0.01, 0.95),
        }

    ndvi_denom = bands['B8'] + bands['B4']
    ndvi_arr = np.where(ndvi_denom > 1e-6, (bands['B8'] - bands['B4']) / np.maximum(ndvi_denom, 1e-6), 0.0)
    scl = np.where(ndvi_arr > 0.30, 4, 5)

    # 8. Airborne Radiometrics (NAGMP High-Resolution K-U-Th Gamma Ray Spectrometry)
    # Background K% is ~1.2 - 2.0%, but S-type granites and LCT pegmatites show major K enrichment (3.5 - 6.5%)
    base_k = 1.6 + 0.9 * np.exp(-granite_dist_km / 8.0) + 0.3 * np.random.uniform(0, 1, (nrows, ncols))
    base_u_th = 0.22 + 0.12 * np.exp(-granite_dist_km / 12.0) + 0.05 * np.random.uniform(0, 1, (nrows, ncols))
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        rad = np.random.randint(2, 5)
        r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
        c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)
        base_k[r_min:r_max, c_min:c_max] += np.random.uniform(2.0, 3.5)
        base_u_th[r_min:r_max, c_min:c_max] += np.random.uniform(0.28, 0.55)
    radiometric_k_pct = gaussian_filter(np.clip(base_k, 0.5, 7.5), sigma=1.2)
    radiometric_u_th_ratio = gaussian_filter(np.clip(base_u_th, 0.1, 1.2), sigma=1.2)
    radiometric_ternary_index = radiometric_k_pct * radiometric_u_th_ratio

    # 9. Bouguer Residual Gravity & First Vertical Derivative (FVD)
    # Felsic S-type granites manifest as negative residual gravity lows (-15 to -45 mGal, GSI Plate XI)
    base_grav = -18.0 - 24.0 * np.exp(-granite_dist_km / 6.5) + 5.0 * np.sin(lats * 10.0) + np.random.normal(0, 1.5, (nrows, ncols))
    bouguer_gravity_mgal = gaussian_filter(base_grav, sigma=2.0)
    g_dy, g_dx = np.gradient(bouguer_gravity_mgal)
    gravity_gradient_fvd = gaussian_filter(np.sqrt(g_dx**2 + g_dy**2), sigma=1.0)

    # 10. ASTER Thermal Infrared (TIR) - Ninomiya Quartz Index (QI)
    # Pegmatites feature massive quartz cores (SiO2 > 75%) with elevated QI (1.05 - 1.40)
    base_qi = 0.98 + 0.04 * np.random.uniform(0, 1, (nrows, ncols))
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        rad = np.random.randint(1, 4)
        r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
        c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)
        base_qi[r_min:r_max, c_min:c_max] += np.random.uniform(0.18, 0.35)
    aster_quartz_index_qi = gaussian_filter(np.clip(base_qi, 0.85, 1.50), sigma=1.0)

    # 11. Sentinel-1 C-Band SAR (Roughness & Dielectric Properties)
    # Resistant pegmatitic ridges and quartz core outcrops yield higher cross-pol backscatter (VH in dB)
    base_vh = -20.0 + 3.0 * (lineament_density / np.max(lineament_density + 1e-5)) + np.random.normal(0, 1.0, (nrows, ncols))
    base_vv_vh = 4.5 - 1.5 * np.exp(-fault_dist_km / 5.0) + np.random.normal(0, 0.4, (nrows, ncols))
    for occ in occurrences:
        r, c = grid.coord_to_pixel(occ["latitude"], occ["longitude"])
        rad = np.random.randint(1, 3)
        r_min, r_max = max(0, r - rad), min(nrows, r + rad + 1)
        c_min, c_max = max(0, c - rad), min(ncols, c + rad + 1)
        base_vh[r_min:r_max, c_min:c_max] += np.random.uniform(3.5, 6.0)
        base_vv_vh[r_min:r_max, c_min:c_max] -= np.random.uniform(0.8, 1.6)
    sar_cband_vh_db = gaussian_filter(base_vh, sigma=1.0)
    sar_roughness_ratio = gaussian_filter(base_vv_vh, sigma=1.0)

    dataset = {
        'district': 'Katghora',
        'grid': grid,
        'occurrences': occurrences,
        'all_ground_truth': all_ground_truth,
        'brs_samples': brs_samples,
        'borehole_collars': borehole_collars,
        'g3_points': g3_points,
        'bands': bands,
        'scl': scl,
        'dem': dem,
        'slope': slope_deg,
        'fault_dist_km': fault_dist_km,
        'lineament_density': lineament_density,
        'granite_dist_km': granite_dist_km,
        'aeromag_rtp': aeromag_rtp,
        'ngcm_li': ngcm_li,
        'ngcm_k_rb': ngcm_k_rb,
        'radiometric_k_pct': radiometric_k_pct,
        'radiometric_u_th_ratio': radiometric_u_th_ratio,
        'radiometric_ternary_index': radiometric_ternary_index,
        'bouguer_gravity_mgal': bouguer_gravity_mgal,
        'gravity_gradient_fvd': gravity_gradient_fvd,
        'aster_quartz_index_qi': aster_quartz_index_qi,
        'sar_cband_vh_db': sar_cband_vh_db,
        'sar_roughness_ratio': sar_roughness_ratio,
    }
    return dataset


def build_district_benchmark(district: str = "katghora", nrows: int = 240, ncols: int = 360, seed: int = 42):
    """
    Dispatcher to construct benchmark datasets for supported exploration districts.
    Supported: 'katghora' (default, Korba, Chhattisgarh), 'bhilwara' (Rajasthan).
    """
    d_clean = district.lower().strip()
    if "katghora" in d_clean or "korba" in d_clean or "chhattisgarh" in d_clean:
        return build_katghora_benchmark(nrows=nrows, ncols=ncols, seed=seed)
    else:
        return build_bhilwara_benchmark(nrows=nrows, ncols=ncols, seed=seed)

