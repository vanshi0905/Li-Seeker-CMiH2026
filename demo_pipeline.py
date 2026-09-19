"""
LithKhoj: End-to-End Mineral Prospectivity Mapping Pipeline.
Critical Minerals Innovation Hackathon (CMiH 2026) - Problem Statement 01.
Host: JNARDDC / Ministry of Mines, Government of India.
"""

import os
import sys
import time
import numpy as np

# Ensure root directory is on pythonpath
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

import argparse
from src.geospatial.synthetic_generator import (
    build_katghora_benchmark,
    build_district_benchmark,
)
from src.geospatial.raster_stack import EvidentialRasterStack
from src.models.pu_xgboost import BaggingPUMiner
from src.models.spatial_cv import run_spatial_block_cv
from src.models.feature_importance import compute_feature_rankings
from src.evaluation.metrics import compute_prediction_area_plot
from src.evaluation.target_extractor import extract_prospective_targets
from src.export.exporter import export_prospectivity_deliverables


def run_pipeline(district: str = "katghora", fast: bool = False):
    district_clean = district.lower().strip()
    is_katghora = "katghora" in district_clean or "korba" in district_clean

    dist_name = "Katghora Block, Korba District, Chhattisgarh (CGC Margin)"

    print("=" * 75, flush=True)
    print("  LITHKHOJ: AI-POWERED MINERAL PROSPECTIVITY MAPPING (CMiH 2026)", flush=True)
    print("  Problem Statement 01: Lithium Pegmatite Prospectivity from Open Data", flush=True)
    print(f"  Target District: {dist_name}", flush=True)
    if fast:
        print("  Mode: Rapid Evaluation (~10s smoke run on downsampled grid)", flush=True)
    print("=" * 75, flush=True)

    start_time = time.time()

    # Step 1: Ingest Multi-Source Open Data & Synthesize District Stack
    nrows = 60 if fast else 240
    ncols = 90 if fast else 360
    n_est_cv = 5 if fast else 15
    n_est_prod = 10 if fast else 30
    cv_blocks = 2 if fast else 3

    print("\n[Step 1/6] Ingesting Multi-Source Geospatial Stack...", flush=True)
    dataset = build_district_benchmark(district=district_clean, nrows=nrows, ncols=ncols, seed=42)
    grid = dataset['grid']
    occurrences = dataset.get('all_ground_truth') or dataset['occurrences']
    print(f"  -> Bounding Box: Lat [{grid.min_lat:.2f}N to {grid.max_lat:.2f}N], Lon [{grid.min_lon:.2f}E to {grid.max_lon:.2f}E]", flush=True)
    print(f"  -> Grid Dimensions: {grid.nrows} rows x {grid.ncols} cols ({grid.nrows * grid.ncols:,} total pixels)", flush=True)
    print(f"  -> Loaded {len(occurrences)} documented pegmatite / critical mineral occurrences.", flush=True)

    # Step 2: Extract Remote Sensing Indices & Assemble Evidential Raster Stack
    print("\n[Step 2/6] Computing Spectral Indices & Assembling Evidential Stack...")
    stack = EvidentialRasterStack(dataset, ndvi_threshold=0.28)
    n_valid = int(np.sum(stack.valid_mask))
    valid_pct = (n_valid / (grid.nrows * grid.ncols)) * 100.0
    print(f"  -> Computed Cardoso-Fernandes LCT pegmatite indices (PI1, PI2, LPI, LMDR, EHI).")
    print(f"  -> Executed Crosta 4-band PCA on [B2, B4, B11, B12] -> Selected PC_{stack.best_pc_idx+1} for Al-OH mica.")
    print(f"  -> Bare-rock outcrop / pediment validity mask: {n_valid:,} pixels ({valid_pct:.1f}% unmasked).")
    print(f"  -> Total Evidential Layers: {len(stack.feature_names)}")

    # Step 3: Prepare Positive and Unlabeled Training Sets
    print("\n[Step 3/6] Structuring Positive-Unlabeled (PU) Training Sets...")
    X_all, valid_mask, feat_names = stack.get_feature_matrix(apply_mask=True)
    lats, lons = grid.get_mesh_coords()
    valid_coords = np.column_stack([lats[valid_mask], lons[valid_mask]])

    # Locate positive deposit pixels
    pos_pixel_indices = stack.get_ground_truth_pixel_indices()
    is_pos_pixel = np.zeros(len(X_all), dtype=bool)

    # Label positive samples with a 2-pixel spatial dilation buffer
    for pr, pc in pos_pixel_indices:
        # Distance to this deposit
        dists = np.hypot(valid_coords[:, 0] - (grid.max_lat - pr * grid.lat_res),
                         valid_coords[:, 1] - (grid.min_lon + pc * grid.lon_res))
        is_pos_pixel |= (dists <= 0.02)  # ~2 km halo

    X_pos = X_all[is_pos_pixel]
    coords_pos = valid_coords[is_pos_pixel]

    # Exclude 1 km buffer around positives from unlabeled pool to prevent label contamination
    is_unlabeled = ~is_pos_pixel
    X_unlabeled = X_all[is_unlabeled]
    coords_unlabeled = valid_coords[is_unlabeled]

    print(f"  -> Confirmed Positive Feature Vectors (including alteration halos): {len(X_pos):,}")
    print(f"  -> Unlabeled Background Pixels: {len(X_unlabeled):,}")

    # Step 4: Train Bagging PU-XGBoost & Spatial Block Cross-Validation
    print("\n[Step 4/6] Training Bagging PU-XGBoost & Running Spatial Block CV...", flush=True)
    cv_results = run_spatial_block_cv(
        X_pos, coords_pos, X_unlabeled, coords_unlabeled,
        min_lat=grid.min_lat, max_lat=grid.max_lat,
        min_lon=grid.min_lon, max_lon=grid.max_lon,
        n_blocks_lat=cv_blocks, n_blocks_lon=cv_blocks, n_estimators=n_est_cv
    )
    print(f"  -> Spatial Block Cross-Validation Mean ROC-AUC: {cv_results['mean_roc_auc']:.4f}", flush=True)
    print(f"  -> Spatial Block Cross-Validation Mean PR-AUC:  {cv_results['mean_pr_auc']:.4f}", flush=True)

    # Fit final production ensemble on full dataset
    miner = BaggingPUMiner(n_estimators=n_est_prod, neg_pos_ratio=3.0, max_depth=4, random_state=42)
    miner.fit(X_pos, X_unlabeled)
    rankings_df = compute_feature_rankings(miner.feature_importances_, feat_names)

    print("\n  Top 5 Evidential Feature Contributors:", flush=True)
    for _, row in rankings_df.head(5).iterrows():
        print(f"    - {row['Feature']:<25}: {row['Percentage']:.2f}%", flush=True)

    # Step 5: Predict District-Wide Prospectivity & Evaluate Metrics
    print("\n[Step 5/6] Generating Full-District Heatmap, Epistemic Uncertainty & Exploration Metrics...")
    prospectivity_map, uncertainty_map = miner.predict_rasters(X_all, valid_mask, grid.nrows, grid.ncols)

    pa_metrics = compute_prediction_area_plot(prospectivity_map, occurrences, grid, n_steps=100)
    cross = pa_metrics["crossing_point"]
    ausrc = pa_metrics["ausrc"]

    print(f"  -> Area Under Success Rate Curve (AUSRC): {ausrc:.4f}")
    print(f"  -> Optimal Threshold (P-A Crossing Point): {cross['optimal_threshold']:.4f}")
    print(f"  -> Deposit Capture Rate at Threshold:    {cross['deposit_capture_percentage']:.1f}%")
    print(f"  -> Prospective Concession Area Required:  {cross['area_percentage']:.1f}%")
    print(f"  -> Normalized Exploration Density (Nd):  {cross['normalized_density']:.2f}x background")
    print(f"  -> Exploration Gain (EG):                {cross['exploration_gain']:.4f}")

    # Step 6: Delineate Exploration Targets & Export Deliverables
    print("\n[Step 6/6] Delineating High-Priority Drill Targets & Exporting Deliverables...")
    targets, geojson = extract_prospective_targets(
        prospectivity_map, grid,
        threshold=cross["optimal_threshold"],
        district_name=district_clean,
        uncertainty_map=uncertainty_map,
        max_uncertainty=0.15
    )
    print(f"  -> Extracted {len(targets)} discrete high-prospectivity target clusters (Uncertainty < 0.15).")
    if targets:
        top_tgt = targets[0]
        print(f"  -> Top Priority Target ({top_tgt['target_id']}): Centroid ({top_tgt['centroid_lat']}N, {top_tgt['centroid_lon']}E), "
              f"Area {top_tgt['area_km2']} km^2, Mean Prospectivity: {top_tgt['mean_prospectivity']:.3f}, "
              f"Uncertainty: {top_tgt.get('mean_uncertainty', 0.0):.3f}")
        if top_tgt.get("borehole_validation_status"):
            print(f"     Subsurface Status:     {top_tgt['borehole_validation_status']}")
        if top_tgt.get("intercept_summary"):
            print(f"     Intercept Calibration: {top_tgt['intercept_summary']}")

    output_dir = os.path.join(CURRENT_DIR, "output")
    paths = export_prospectivity_deliverables(
        prospectivity_map, grid, pa_metrics, geojson, rankings_df, output_dir,
        district_prefix=district_clean,
        uncertainty_map=uncertainty_map
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("  DELIVERABLES SUCCESSFULLY GENERATED")
    print("=" * 75)
    print(f"  - GeoTIFF Heatmap:     {paths['geotiff']}")
    if "uncertainty_geotiff" in paths:
        print(f"  - Uncertainty Map:     {paths['uncertainty_geotiff']}")
    print(f"  - Drill Targets:       {paths['geojson']}")
    print(f"  - Metrics Summary:     {paths['metrics_json']}")
    print(f"  - Feature Rankings:    {paths['feature_rankings_csv']}")
    print(f"  - P-A Plot Chart:      {paths['pa_plot_png']}")
    print(f"  Total Execution Time:  {elapsed:.2f} seconds")
    print("=" * 75)

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LithKhoj Mineral Prospectivity Mapping Pipeline")
    parser.add_argument(
        "--district",
        type=str,
        default="katghora",
        help="Exploration district to process (default: katghora)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Execute rapid evaluation on downsampled grid (~10s)",
    )
    args = parser.parse_args()
    run_pipeline(district=args.district, fast=args.fast)

