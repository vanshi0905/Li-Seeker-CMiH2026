"""
Tests for High-Resolution GeoTIFF Cartographic Snapshot Generator.
"""

import os
import tempfile
import numpy as np
import pytest
import tifffile
from PIL import Image
from generate_high_res_snapshots import (
    normalize_band_percentile,
    parse_bounds_from_tif,
    render_single_layer,
    add_cartographic_elements,
    main,
    INPUT_DIR,
    OUTPUT_DIR,
    MINERAL_INDICES_META,
    RGB_COMPOSITES_META,
    SENTINEL_BANDS_META,
)
import matplotlib.pyplot as plt


def test_normalize_band_percentile_stretching():
    # Array with outlier, nodata, and normal values
    data = np.array([
        [-9999.0, 10.0, 20.0],
        [30.0, np.nan, 50.0],
        [60.0, 70.0, 1000.0]  # 1000 is an outlier
    ], dtype=np.float32)

    stretched, valid, p2, p98, stats = normalize_band_percentile(data, 2.0, 98.0, nodata=-9999.0)

    # Check mask
    assert not valid[0, 0]
    assert not valid[1, 1]
    assert valid[0, 1]
    assert valid[2, 2]

    # Stretched values should be in [0, 1]
    assert np.all(stretched[valid] >= 0.0)
    assert np.all(stretched[valid] <= 1.0)
    assert stretched[~valid].sum() == 0.0

    # Percentiles correctly calculated
    assert p2 < p98
    assert stats["valid_count"] == 7
    assert stats["nodata_count"] == 2


def test_normalize_band_all_nodata():
    data = np.full((10, 10), -9999.0, dtype=np.float32)
    stretched, valid, p2, p98, stats = normalize_band_percentile(data)
    assert not np.any(valid)
    assert np.all(stretched == 0.0)


def test_normalize_band_constant_value():
    data = np.full((5, 5), 42.0, dtype=np.float32)
    stretched, valid, p2, p98, stats = normalize_band_percentile(data)
    assert np.all(valid)
    assert p98 > p2
    assert np.all(stretched >= 0.0)
    assert np.all(stretched <= 1.0)


def test_normalize_band_inf_and_nan():
    data = np.array([
        [np.inf, -np.inf, np.nan],
        [1.0, 2.0, 3.0],
        [-9999.0, 4.0, 5.0]
    ], dtype=np.float32)
    stretched, valid, p2, p98, stats = normalize_band_percentile(data)
    assert not valid[0, 0]
    assert not valid[0, 1]
    assert not valid[0, 2]
    assert not valid[2, 0]
    assert valid[1, 0]
    assert valid[2, 2]
    assert stats["valid_count"] == 5


def test_parse_bounds():
    tif_sample = os.path.join(INPUT_DIR, "index_cardoso_LPI_lithium_pegmatite.tif")
    if os.path.exists(tif_sample):
        bounds = parse_bounds_from_tif(tif_sample)
        assert len(bounds) == 4
        min_lat, min_lon, max_lat, max_lon = bounds
        assert min_lat < max_lat
        assert min_lon < max_lon
        assert min_lat == pytest.approx(22.25)
        assert max_lat == pytest.approx(22.75)


def test_all_16_snapshots_exist_and_meet_4k_specs():
    """Verify that all 16 snapshots are generated, valid, non-empty, and meet 4K resolution."""
    assert os.path.exists(OUTPUT_DIR), f"Output directory does not exist: {OUTPUT_DIR}"

    expected_files = (
        [os.path.splitext(k)[0] + ".png" for k in MINERAL_INDICES_META.keys()]
        + list(RGB_COMPOSITES_META.keys())
        + [os.path.splitext(k)[0] + ".png" for k in SENTINEL_BANDS_META.keys()]
    )

    assert len(expected_files) == 16

    for fname in expected_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(fpath), f"Missing expected snapshot: {fpath}"
        assert os.path.getsize(fpath) > 100_000, f"Snapshot too small or empty: {fpath}"

        with Image.open(fpath) as img:
            w, h = img.size
            # 4K UHD standard: 3840x2160
            assert w >= 3840, f"Width {w} < 3840 for {fname}"
            assert h >= 2160, f"Height {h} < 2160 for {fname}"
            assert img.mode in ("RGB", "RGBA")


def test_render_single_layer_custom_and_jpg():
    """Test single layer rendering to JPG with custom metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock geotiff
        tif_path = os.path.join(tmpdir, "custom_layer.tif")
        mock_data = np.random.uniform(0.1, 0.9, size=(50, 50)).astype(np.float32)
        mock_data[0, 0] = -9999.0
        tifffile.imwrite(tif_path, mock_data, description="District Prospectivity Grid: bounds=(20.0,80.0,21.0,81.0)")

        meta = {
            "title": "Custom Test Layer",
            "subtitle": "Test Prospectivity Region",
            "colormap": "inferno",
            "cbar_label": "Custom Metric [0-1]",
            "layer_name": "Test Layer",
            "formula": "B01 / B02",
            "bands": "Band Test",
            "target": "Testing Target",
            "interpretation": "Test interpretation",
        }

        out_jpg = os.path.join(tmpdir, "custom_layer.jpg")
        w, h, sz = render_single_layer("custom_layer.tif", meta, out_jpg, input_dir=tmpdir)

        assert os.path.exists(out_jpg)
        assert sz > 50_000
        assert w >= 3840
        assert h >= 2160


def test_cartographic_elements_dynamic_scaling():
    """Verify that cartographic elements adjust properly across different bounding extents."""
    fig, ax = plt.subplots(figsize=(10, 8))
    # Small extent (< 8 km)
    add_cartographic_elements(ax, (22.25, 82.25, 22.28, 82.28))
    # Large extent (> 50 km)
    add_cartographic_elements(ax, (20.0, 80.0, 22.0, 82.0))
    plt.close(fig)


def test_auto_discovery_in_main():
    """Test that main() auto-discovers and renders unlisted GeoTIFF files."""
    with tempfile.TemporaryDirectory() as in_tmp, tempfile.TemporaryDirectory() as out_tmp:
        # Place an unlisted GeoTIFF
        mock_data = np.random.uniform(1.0, 5.0, size=(40, 40)).astype(np.float32)
        mock_tif = os.path.join(in_tmp, "unlisted_prospectivity_map.tif")
        tifffile.imwrite(mock_tif, mock_data, description="District Prospectivity Grid: bounds=(22.0,82.0,23.0,83.0)")

        records = main(input_dir=in_tmp, output_dir=out_tmp, export_format="png")
        assert len(records) >= 1
        assert any(r["filename"] == "unlisted_prospectivity_map.png" for r in records)
        out_png = os.path.join(out_tmp, "unlisted_prospectivity_map.png")
        assert os.path.exists(out_png)
        with Image.open(out_png) as img:
            assert img.size[0] >= 3840
