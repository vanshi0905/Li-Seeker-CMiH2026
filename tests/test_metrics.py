"""
Unit tests for mineral prospectivity evaluation metrics.
"""

import numpy as np
import pytest
from src.geospatial.raster_ops import GeoGrid
from src.evaluation.metrics import compute_prediction_area_plot
from src.evaluation.target_extractor import extract_prospective_targets


def test_prediction_area_plot_metrics():
    grid = GeoGrid(min_lat=25.0, max_lat=26.0, min_lon=74.0, max_lon=75.0, nrows=50, ncols=50)
    prospectivity = np.random.uniform(0.1, 0.4, (50, 50))

    # Place 3 high-confidence deposits
    deposits = [
        {"latitude": 25.5, "longitude": 74.5},
        {"latitude": 25.8, "longitude": 74.8},
        {"latitude": 25.2, "longitude": 74.2}
    ]
    for d in deposits:
        r, c = grid.coord_to_pixel(d["latitude"], d["longitude"])
        prospectivity[r, c] = 0.95

    res = compute_prediction_area_plot(prospectivity, deposits, grid, n_steps=20)

    assert "ausrc" in res
    assert 0.0 <= res["ausrc"] <= 1.0
    assert "crossing_point" in res
    cross = res["crossing_point"]
    assert 0.0 <= cross["optimal_threshold"] <= 1.0
    assert cross["normalized_density"] > 0.0


def test_target_extractor_delineates_polygons():
    grid = GeoGrid(min_lat=25.0, max_lat=26.0, min_lon=74.0, max_lon=75.0, nrows=50, ncols=50)
    prospectivity = np.zeros((50, 50), dtype=float)

    # Create a 4x4 block of high prospectivity
    prospectivity[10:14, 10:14] = 0.88

    targets, geojson = extract_prospective_targets(prospectivity, grid, threshold=0.70, min_pixels=4)
    assert len(targets) == 1
    assert targets[0]["target_id"] == "TGT-01"
    assert targets[0]["pixel_count"] == 16
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
