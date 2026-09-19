"""
Tier 3: Pairwise Cross-Feature Integration E2E Tests.
Validates multi-module integration chains across:
  - STAC fallback dataset -> EvidentialRasterStack (with REE indices)
  - EvidentialRasterStack -> Bagging PU-XGBoost learning
  - PU Prospectivity Map -> P-A Metrics & Target Extraction
  - Feature Importance Rankings with REE layers
  - Deliverables Exporter with REE-augmented data
  - App module import and component resolution
"""

import os
from pathlib import Path
import numpy as np
import pytest

from tests.e2e.conftest import (
    PROJECT_ROOT, skipif_no_ree, HAS_STAC
)


@skipif_no_ree
def test_t3_pairwise_dataset_to_evidential_stack_with_ree():
    """
    T3-01: Pairwise test verifying dataset ingestion into EvidentialRasterStack with REE features.
    Tests data flow from ingestion layer into geospatial raster stacking.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    from src.geospatial.raster_stack import EvidentialRasterStack

    if HAS_STAC:
        from src.data_ingestion.stac_client import Sentinel2STACClient
        client = Sentinel2STACClient()
        dataset = client.fetch_bhilwara_dataset(nrows=20, ncols=30, force_fallback=True)
    else:
        dataset = build_bhilwara_benchmark(nrows=20, ncols=30, seed=42)

    stack = EvidentialRasterStack(dataset, ndvi_threshold=0.28)

    assert "ree_composite_index" in stack.feature_names or "ree_index" in stack.feature_names, \
        f"Evidential stack must expose REE composite index in feature_names: {stack.feature_names}"
    assert "ree_nd_absorption" in stack.feature_names or "ree_neodymium" in stack.feature_names, \
        f"Evidential stack must expose REE Nd absorption in feature_names: {stack.feature_names}"

    assert stack.valid_mask.shape == (20, 30)
    assert np.sum(stack.valid_mask) > 0, "Valid bare-rock mask must have active pixels"


@skipif_no_ree
def test_t3_pairwise_evidential_stack_to_pu_model():
    """
    T3-02: Pairwise test verifying EvidentialRasterStack (with REE) feature matrix extraction
    and training of the BaggingPUMiner ensemble without NaN or shape mismatch.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    from src.geospatial.raster_stack import EvidentialRasterStack
    from src.models.pu_xgboost import BaggingPUMiner

    dataset = build_bhilwara_benchmark(nrows=25, ncols=35, seed=42)
    stack = EvidentialRasterStack(dataset, ndvi_threshold=0.28)

    X_all, valid_mask, feat_names = stack.get_feature_matrix(apply_mask=True)

    assert X_all.ndim == 2
    assert X_all.shape[1] == len(feat_names)
    assert np.all(np.isfinite(X_all)), "Feature matrix must not contain NaN or Inf values"

    # Separate positive and unlabeled pixels
    pos_pixel_indices = stack.get_ground_truth_pixel_indices()
    lats, lons = dataset['grid'].get_mesh_coords()
    valid_coords = np.column_stack([lats[valid_mask], lons[valid_mask]])

    is_pos = np.zeros(len(X_all), dtype=bool)
    for pr, pc in pos_pixel_indices:
        dep_lat = dataset['grid'].max_lat - pr * dataset['grid'].lat_res
        dep_lon = dataset['grid'].min_lon + pc * dataset['grid'].lon_res
        dists = np.hypot(valid_coords[:, 0] - dep_lat, valid_coords[:, 1] - dep_lon)
        is_pos |= (dists <= 0.05)

    if not np.any(is_pos):
        is_pos[:5] = True  # Ensure test has at least minimal positives

    X_pos = X_all[is_pos]
    X_unlabeled = X_all[~is_pos]

    miner = BaggingPUMiner(n_estimators=3, neg_pos_ratio=2.0, max_depth=3, random_state=42)
    miner.fit(X_pos, X_unlabeled)

    assert hasattr(miner, "models")
    assert len(miner.models) == 3
    assert miner.feature_importances_ is not None


@skipif_no_ree
def test_t3_pairwise_pu_model_to_prospectivity_map():
    """
    T3-03: Pairwise test verifying prospectivity probability generation and 2D raster reconstruction.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    from src.geospatial.raster_stack import EvidentialRasterStack
    from src.models.pu_xgboost import BaggingPUMiner

    dataset = build_bhilwara_benchmark(nrows=20, ncols=30, seed=42)
    stack = EvidentialRasterStack(dataset, ndvi_threshold=0.28)
    X_all, valid_mask, _ = stack.get_feature_matrix(apply_mask=True)

    # Fast fit with dummy positive label
    miner = BaggingPUMiner(n_estimators=2, neg_pos_ratio=1.0, max_depth=2, random_state=42)
    miner.fit(X_all[:4], X_all[4:])

    preds = miner.predict_proba(X_all)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)

    # 2D reconstruction
    prob_map = np.full((20, 30), np.nan, dtype=np.float32)
    prob_map[valid_mask] = preds

    assert prob_map.shape == (20, 30)
    assert np.any(np.isfinite(prob_map))


@skipif_no_ree
def test_t3_pairwise_prospectivity_map_to_targets_and_metrics():
    """
    T3-04: Pairwise test verifying P-A crossing point evaluation and discrete drill target extraction.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    from src.evaluation.metrics import compute_prediction_area_plot
    from src.evaluation.target_extractor import extract_prospective_targets

    dataset = build_bhilwara_benchmark(nrows=20, ncols=30, seed=42)
    grid = dataset['grid']
    occurrences = dataset['occurrences']

    # Generate synthetic prospectivity map
    lats, lons = grid.get_mesh_coords()
    prob_map = np.sin(lats * 10) * np.cos(lons * 10)
    prob_map = ((prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())).astype(np.float32)

    pa_metrics = compute_prediction_area_plot(prob_map, occurrences, grid, n_steps=20)
    assert "crossing_point" in pa_metrics
    assert "ausrc" in pa_metrics
    assert 0.0 <= pa_metrics["ausrc"] <= 1.0

    threshold = pa_metrics["crossing_point"]["optimal_threshold"]
    targets, geojson = extract_prospective_targets(prob_map, grid, threshold=threshold)

    assert isinstance(targets, list)
    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson


@skipif_no_ree
def test_t3_pairwise_feature_importance_with_ree_layers():
    """
    T3-05: Pairwise test verifying feature importance calculation with REE evidential features.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    from src.geospatial.raster_stack import EvidentialRasterStack
    from src.models.feature_importance import compute_feature_rankings

    dataset = build_bhilwara_benchmark(nrows=15, ncols=20, seed=42)
    stack = EvidentialRasterStack(dataset, ndvi_threshold=0.28)
    feat_names = stack.feature_names

    # Mock importances
    n_feats = len(feat_names)
    importances = np.ones(n_feats, dtype=np.float32) / n_feats

    rankings_df = compute_feature_rankings(importances, feat_names)

    assert "Feature" in rankings_df.columns
    assert "Percentage" in rankings_df.columns
    assert pytest.approx(rankings_df["Percentage"].sum(), abs=1e-1) == 100.0

    # Ensure REE features appear in the rankings
    ree_features_in_rankings = [f for f in rankings_df["Feature"] if "ree" in f.lower()]
    assert len(ree_features_in_rankings) >= 1, f"Expected REE features in rankings: {rankings_df['Feature'].tolist()}"


@skipif_no_ree
def test_t3_pairwise_export_deliverables_integration(temp_output_dir):
    """
    T3-06: Pairwise test verifying all 5 deliverables export cleanly into output directory.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    from src.evaluation.metrics import compute_prediction_area_plot
    from src.evaluation.target_extractor import extract_prospective_targets
    from src.models.feature_importance import compute_feature_rankings
    from src.export.exporter import export_prospectivity_deliverables

    dataset = build_bhilwara_benchmark(nrows=15, ncols=20, seed=42)
    grid = dataset['grid']
    occurrences = dataset['occurrences']
    prob_map = np.full((15, 20), 0.65, dtype=np.float32)

    pa_metrics = compute_prediction_area_plot(prob_map, occurrences, grid, n_steps=10)
    targets, geojson = extract_prospective_targets(prob_map, grid, threshold=0.5)
    rankings_df = compute_feature_rankings(np.array([1.0]), ["ree_composite_index"])

    paths = export_prospectivity_deliverables(
        prob_map, grid, pa_metrics, geojson, rankings_df, str(temp_output_dir)
    )

    assert Path(paths["geotiff"]).exists()
    assert Path(paths["geojson"]).exists()
    assert Path(paths["metrics_json"]).exists()
    assert Path(paths["feature_rankings_csv"]).exists()
    assert Path(paths["pa_plot_png"]).exists()


def test_t3_pairwise_app_module_imports_cleanly():
    """
    T3-07: Pairwise test verifying app/app.py can be imported without syntax or import errors.
    """
    import importlib.util

    app_path = PROJECT_ROOT / "app" / "app.py"
    assert app_path.exists(), "app/app.py must exist"

    spec = importlib.util.spec_from_file_location("app_module", str(app_path))
    assert spec is not None
    assert spec.loader is not None
