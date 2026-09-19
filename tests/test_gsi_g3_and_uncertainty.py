"""
Tests for GSI G3 real exploration datasets, Predictive Uncertainty Quantification,
and 3D Borehole Intercept Calibration.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest

from src.geospatial.raster_ops import GeoGrid
from src.evaluation.target_extractor import extract_prospective_targets
from src.models.pu_xgboost import BaggingPUMiner


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def test_gsi_g3_brs_samples_integrity():
    csv_path = os.path.join(DATA_DIR, "katghora_gsi_brs_samples.csv")
    assert os.path.exists(csv_path), "BRS samples CSV must exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 80, f"Expected 80 BRS bedrock samples, got {len(df)}"

    required_cols = [
        "sr_no", "sample_id", "lithology", "latitude", "longitude",
        "li_ppm", "cs_ppm", "be_ppm", "rb_ppm", "ta_ppm", "nb_ppm"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required column {c} in BRS samples"

    # Coordinates validation within Katghora block
    assert (df["latitude"] >= 22.40).all() and (df["latitude"] <= 22.65).all()
    assert (df["longitude"] >= 82.40).all() and (df["longitude"] <= 82.70).all()

    # Li assay range check (50 - 500 ppm in surface pegmatite/leucogranite)
    assert (df["li_ppm"] > 0).all()
    assert df["li_ppm"].max() >= 400.0


def test_gsi_g3_borehole_collars_integrity():
    csv_path = os.path.join(DATA_DIR, "katghora_borehole_collars.csv")
    assert os.path.exists(csv_path), "Borehole collars CSV must exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 15, f"Expected 15 borehole collars, got {len(df)}"

    expected_bhs = [f"KRKC-{i:02d}" for i in range(1, 16)]
    assert sorted(df["borehole_id"].tolist()) == sorted(expected_bhs)

    # Validate coordinate conversion from UTM 44N
    assert (df["latitude"] >= 22.50).all() and (df["latitude"] <= 22.55).all()
    assert (df["longitude"] >= 82.54).all() and (df["longitude"] <= 82.58).all()

    # Check 400m grid tiers in UTM (lines separated by ~400m)
    north_tiers = sorted(np.round(df["northing_utm44n"] / 200.0).unique())
    assert len(north_tiers) == 3, f"Expected 3 drilling lines, got {len(north_tiers)}"
    east_tiers = sorted(np.round(df["easting_utm44n"] / 200.0).unique())
    assert len(east_tiers) == 5, f"Expected 5 borehole columns, got {len(east_tiers)}"


def test_gsi_g3_drill_core_assays_integrity():
    csv_path = os.path.join(DATA_DIR, "katghora_drill_core_assays.csv")
    assert os.path.exists(csv_path), "Drill core assays CSV must exist"
    df = pd.read_csv(csv_path)
    assert len(df) >= 450, f"Expected >= 450 core assays, got {len(df)}"

    assert "borehole_id" in df.columns
    assert "sample_id" in df.columns
    assert "li_ppm" in df.columns
    assert "from_m" in df.columns
    assert "to_m" in df.columns

    # Subsurface high-grade lithium checks
    assert df["li_ppm"].max() >= 1200.0, "Subsurface core assays must detect high-grade Li (>1200 ppm)"
    bhs = df["borehole_id"].unique()
    assert len(bhs) == 15, "All 15 boreholes must be represented in assays"


def test_ground_truth_json_95_g3_points():
    json_path = os.path.join(DATA_DIR, "ground_truth", "katghora_pegmatites.json")
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    # 10 regional occurrences preserved for backward compatibility
    assert len(data["occurrences"]) == 10
    # 80 BRS bedrock samples
    assert len(data["brs_samples"]) == 80
    # 15 Borehole collars
    assert len(data["borehole_collars"]) == 15
    # Combined G3 field validated points = 95
    assert len(data["g3_exploration_points"]) == 95
    # Total ground truth = 105
    assert data["total_ground_truth_count"] == 105


def test_pu_miner_uncertainty_quantification():
    np.random.seed(42)
    X_pos = np.random.normal(loc=2.5, scale=0.4, size=(10, 4))
    X_unlabeled = np.random.normal(loc=0.0, scale=1.0, size=(100, 4))

    miner = BaggingPUMiner(n_estimators=6, max_depth=3, random_state=42)
    miner.fit(X_pos, X_unlabeled)

    # Multi-model architecture pool verification
    assert len(miner.models) == 6
    assert len(miner.model_types) == 6

    # Test predictive uncertainty
    X_test = np.random.normal(loc=1.0, scale=0.5, size=(20, 4))
    mean_prob, uncertainty = miner.predict_with_uncertainty(X_test)

    assert len(mean_prob) == 20
    assert len(uncertainty) == 20
    assert np.all((mean_prob >= 0.0) & (mean_prob <= 1.0))
    assert np.all((uncertainty >= 0.0) & (uncertainty <= 0.5))

    # Test 2D rasters
    valid_mask = np.array([True, True, False, True], dtype=bool)
    X_grid = np.random.normal(size=(3, 4))
    pmap, umap = miner.predict_rasters(X_grid, valid_mask, 2, 2)
    assert pmap.shape == (2, 2)
    assert umap.shape == (2, 2)
    assert np.isnan(pmap[1, 0])  # Masked pixel should be NaN
    assert np.isnan(umap[1, 0])


def test_target_extractor_uncertainty_and_borehole_calibration():
    grid = GeoGrid(min_lat=22.25, max_lat=22.75, min_lon=82.25, max_lon=82.75, nrows=100, ncols=100)
    prospectivity = np.zeros((100, 100), dtype=float)
    uncertainty = np.full((100, 100), 0.04, dtype=float)

    # Place anomaly right over GSI drilling grid (22.522N, 82.555E)
    r, c = grid.coord_to_pixel(22.522, 82.555)
    prospectivity[r-2:r+3, c-2:c+3] = 0.92

    targets, geojson = extract_prospective_targets(
        prospectivity, grid,
        threshold=0.60,
        district_name="katghora",
        uncertainty_map=uncertainty,
        max_uncertainty=0.15
    )

    assert len(targets) >= 1
    t0 = targets[0]
    assert t0["mean_uncertainty"] == pytest.approx(0.04, 0.01)
    assert "calibrated_boreholes" in t0
    assert len(t0["calibrated_boreholes"]) > 0
    assert t0["borehole_validation_status"] == "Field Validated - High-Grade Subsurface Intercept"
    assert t0["peak_downhole_li_ppm"] is not None
    assert t0["peak_downhole_li_ppm"] >= 200.0

    # GeoJSON properties check
    feat_props = geojson["features"][0]["properties"]
    assert "mean_uncertainty" in feat_props
    assert "borehole_validation_status" in feat_props
    assert "calibrated_boreholes" in feat_props


def test_pu_miner_predict_rasters_full_grid_unmasked():
    """Verify predict_rasters works when X_all has nrows*ncols rows (unmasked full grid)."""
    np.random.seed(42)
    miner = BaggingPUMiner(n_estimators=3, max_depth=2, random_state=42)
    miner.fit(np.ones((6, 4)), np.zeros((12, 4)))

    nrows, ncols = 8, 10
    X_full = np.random.normal(size=(nrows * ncols, 4))
    mask = np.ones((nrows, ncols), dtype=bool)
    mask[0, 0] = False
    mask[2, 3] = False

    pmap, umap = miner.predict_rasters(X_full, mask, nrows, ncols)
    assert pmap.shape == (nrows, ncols)
    assert umap.shape == (nrows, ncols)
    assert np.isnan(pmap[0, 0])
    assert np.isnan(umap[0, 0])
    assert np.isnan(pmap[2, 3])
    assert not np.isnan(pmap[1, 1])
    assert not np.isnan(umap[1, 1])


def test_target_extractor_sequential_target_id_renumbering():
    """Verify targets are sequentially numbered TGT-01, TGT-02 strictly by rank after sorting."""
    grid = GeoGrid(min_lat=22.25, max_lat=22.75, min_lon=82.25, max_lon=82.75, nrows=50, ncols=50)
    prospectivity = np.zeros((50, 50), dtype=float)

    # Cluster A (lower prospectivity placed first in array scan order)
    prospectivity[5:10, 5:10] = 0.75
    # Cluster B (higher prospectivity placed later in array scan order)
    prospectivity[35:40, 35:40] = 0.95

    targets, geojson = extract_prospective_targets(
        prospectivity, grid, threshold=0.70, district_name="katghora"
    )
    assert len(targets) == 2
    assert targets[0]["target_id"] == "TGT-01"
    assert targets[0]["rank"] == 1
    assert targets[0]["mean_prospectivity"] == pytest.approx(0.95, 0.01)
    assert targets[1]["target_id"] == "TGT-02"
    assert targets[1]["rank"] == 2
    assert targets[1]["mean_prospectivity"] == pytest.approx(0.75, 0.01)


def test_katghora_benchmark_real_data_ingestion():
    """Verify build_katghora_benchmark ingests real GSI ground truth and Sentinel bands."""
    from src.geospatial.synthetic_generator import build_katghora_benchmark
    ds = build_katghora_benchmark(nrows=240, ncols=360, seed=42)

    # 10 legacy occurrences preserved for backward compatibility
    assert len(ds["occurrences"]) == 10
    # 105 total ground truth points available
    assert "all_ground_truth" in ds
    assert len(ds["all_ground_truth"]) == 105
    assert len(ds["brs_samples"]) == 80
    assert len(ds["borehole_collars"]) == 15

    # All 8 Sentinel bands present and non-empty
    for b in ['B2', 'B3', 'B4', 'B6', 'B8', 'B8A', 'B11', 'B12']:
        assert b in ds['bands']
        assert ds['bands'][b].shape == (240, 360)
        assert not np.isnan(ds['bands'][b]).any()


def test_target_extractor_blind_district_fallback():
    """Verify target extractor handles districts with no borehole collars gracefully."""
    grid = GeoGrid(min_lat=25.05, max_lat=25.85, min_lon=74.05, max_lon=75.25, nrows=50, ncols=50)
    prospectivity = np.zeros((50, 50), dtype=float)
    prospectivity[20:25, 20:25] = 0.88

    targets, geojson = extract_prospective_targets(
        prospectivity, grid, threshold=0.70, district_name="bhilwara"
    )
    assert len(targets) >= 1
    t0 = targets[0]
    assert t0["target_id"] == "TGT-01"
    assert t0["calibrated_boreholes"] == []
    assert "Blind Discovery Target" in t0["borehole_validation_status"]

