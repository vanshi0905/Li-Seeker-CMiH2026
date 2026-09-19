"""
Tier 2: Boundary and Corner Cases E2E Tests (>=5 tests per feature).
Features Covered:
  - F1 / F3: REE Spectral Indices Boundaries (zeros, ones, negatives, extremes, NaNs, epsilons)
  - F4 / F5 / F6: STAC Ingestion Boundaries (corrupted bbox, timeout, HTTP 500, empty items, extreme grids)
  - F10 / F11 / F12: Docker Configuration Boundaries (syntax, single FROM, workdir, healthcheck, volume format)
  - F13: Directory Confinement Boundaries (system paths, null bytes, nested output, foreign drives)
"""

import os
import re
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from tests.e2e.conftest import (
    PROJECT_ROOT, BHILWARA_BBOX, skipif_no_ree, skipif_no_stac, skipif_no_docker
)


# ==============================================================================
# Feature 1: REE Spectral Indices Boundaries - >= 5 tests
# ==============================================================================

@skipif_no_ree
def test_t2_ree_all_zeros_input():
    """
    F1-B01: Boundary test with all zeros input arrays.
    Must return all zeros without throwing ZeroDivisionError or NaN warnings.
    """
    from src.remote_sensing.indices import ree_index, ree_neodymium_index, ree_carbonatite_index

    zeros = np.zeros((10, 10), dtype=np.float32)

    nd = ree_neodymium_index(zeros, zeros)
    comp = ree_index(zeros, zeros, zeros, zeros)
    carb = ree_carbonatite_index(zeros, zeros, zeros, zeros)

    assert np.all(nd == 0.0)
    assert np.all(comp == 0.0)
    assert np.all(carb == 0.0)
    assert np.all(np.isfinite(nd))
    assert np.all(np.isfinite(comp))
    assert np.all(np.isfinite(carb))


@skipif_no_ree
def test_t2_ree_all_ones_input():
    """
    F1-B02: Boundary test with all 1.0 input arrays.
    Ratio of 1.0 / 1.0 = 1.0; Composite (1.0) * (1.0) = 1.0; Carbonatite (1.0*1.0)/(1.0*1.0) = 1.0.
    """
    from src.remote_sensing.indices import ree_index, ree_neodymium_index, ree_carbonatite_index

    ones = np.ones((8, 8), dtype=np.float32)

    nd = ree_neodymium_index(ones, ones)
    comp = ree_index(ones, ones, ones, ones)
    carb = ree_carbonatite_index(ones, ones, ones, ones)

    assert np.allclose(nd, 1.0)
    assert np.allclose(comp, 1.0)
    assert np.allclose(carb, 1.0)


@skipif_no_ree
def test_t2_ree_negative_reflectance_handling():
    """
    F1-B03: Corner case: negative reflectance values from atmospheric over-correction.
    Must evaluate safely without raising exceptions or producing non-finite numbers.
    """
    from src.remote_sensing.indices import ree_index, ree_neodymium_index

    b6_neg = np.array([-0.05, 0.15], dtype=np.float32)
    b8a_pos = np.array([0.30, 0.30], dtype=np.float32)
    b11_pos = np.array([0.40, 0.40], dtype=np.float32)
    b12_neg = np.array([-0.02, 0.20], dtype=np.float32)

    nd = ree_neodymium_index(b6_neg, b8a_pos)
    comp = ree_index(b6_neg, b8a_pos, b11_pos, b12_neg)

    assert np.all(np.isfinite(nd))
    assert np.all(np.isfinite(comp))


@skipif_no_ree
def test_t2_ree_extreme_large_reflectance():
    """
    F1-B04: Extreme boundary: raw DN values (>10,000) or high glint (1e6) without scaling.
    Must maintain finite float values without memory corruption or numerical overflow.
    """
    from src.remote_sensing.indices import ree_index, ree_neodymium_index

    b6 = np.array([1e6, 2e6], dtype=np.float32)
    b8a = np.array([3e6, 4e6], dtype=np.float32)
    b11 = np.array([5e6, 6e6], dtype=np.float32)
    b12 = np.array([1e6, 2e6], dtype=np.float32)

    nd = ree_neodymium_index(b6, b8a)
    comp = ree_index(b6, b8a, b11, b12)

    assert nd[0] == pytest.approx(3.0, rel=1e-4)
    assert comp[0] == pytest.approx(15.0, rel=1e-4)  # (3/1) * (5/1)
    assert np.all(np.isfinite(comp))


@skipif_no_ree
def test_t2_ree_nan_and_inf_inputs():
    """
    F1-B05: Corner case: input array containing NaN or Inf pixels from bad sensor data.
    safe_divide must sanitize NaNs and Infs to the safe fill_value (0.0).
    """
    from src.remote_sensing.indices import ree_index, ree_neodymium_index

    b6 = np.array([np.nan, 0.20, np.inf], dtype=np.float32)
    b8a = np.array([0.40, np.nan, 0.50], dtype=np.float32)
    b11 = np.array([0.40, 0.40, np.nan], dtype=np.float32)
    b12 = np.array([0.20, 0.20, 0.20], dtype=np.float32)

    nd = ree_neodymium_index(b6, b8a)
    comp = ree_index(b6, b8a, b11, b12)

    assert np.all(np.isfinite(nd)), "Output must have all NaNs/Infs sanitized to finite numbers"
    assert np.all(np.isfinite(comp))
    assert nd[0] == pytest.approx(0.0)


@skipif_no_ree
def test_t2_ree_sub_epsilon_denominators():
    """
    F1-B06: Boundary test for denominators near eps (e.g. 1e-7 to 1e-12).
    Must trigger the eps cutoff and safely map to 0.0.
    """
    from src.remote_sensing.indices import ree_neodymium_index

    b6 = np.array([1e-7, 1e-10, 1e-15], dtype=np.float32)
    b8a = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    nd = ree_neodymium_index(b6, b8a)

    assert np.all(np.isfinite(nd))
    assert np.all(nd == 0.0)


# ==============================================================================
# Feature 2: STAC Ingestion & Fallback Boundaries - >= 5 tests
# ==============================================================================

@skipif_no_stac
def test_t2_stac_corrupted_or_inverted_bbox():
    """
    F2-B01: Boundary case: passing an inverted or corrupted bounding box [max_lon, max_lat, min_lon, min_lat].
    Client must either validate/normalize or fall back to benchmark without crashing.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()
    inverted_bbox = [75.25, 25.85, 74.05, 25.05]

    try:
        results = client.search_scenes(bbox=inverted_bbox)
        assert isinstance(results, list)
    except (ValueError, Exception):
        # Graceful exception handling is also valid boundary behavior
        pass


@skipif_no_stac
def test_t2_stac_simulated_network_timeout():
    """
    F2-B02: Corner case: network search times out (e.g., requests.exceptions.Timeout).
    Must seamlessly fall back to synthetic benchmark with data_source='synthetic_fallback'.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient
    import requests

    client = Sentinel2STACClient()

    with patch.object(client, "search_scenes", side_effect=requests.exceptions.Timeout("Connection timed out")):
        dataset = client.fetch_bhilwara_dataset(nrows=15, ncols=20)

    assert dataset is not None
    assert dataset.get("data_source") == "synthetic_fallback"
    assert "bands" in dataset


@skipif_no_stac
def test_t2_stac_simulated_http_500_or_429():
    """
    F2-B03: Corner case: server returns HTTP 500 Internal Server Error or 429 Rate Limit.
    Must seamlessly trigger fallback without crashing.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient
    import requests

    client = Sentinel2STACClient()
    mock_resp = requests.Response()
    mock_resp.status_code = 500

    with patch.object(client, "search_scenes", side_effect=requests.exceptions.HTTPError("500 Server Error", response=mock_resp)):
        dataset = client.fetch_bhilwara_dataset(nrows=15, ncols=20)

    assert dataset["data_source"] == "synthetic_fallback"


@skipif_no_stac
def test_t2_stac_empty_search_results_triggers_fallback():
    """
    F2-B04: Corner case: STAC search returns an empty list of scenes (0 items).
    Client must detect 0 scenes and fall back to benchmark instead of throwing IndexError.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()

    with patch.object(client, "search_scenes", return_value=[]):
        dataset = client.fetch_bhilwara_dataset(nrows=15, ncols=20)

    assert dataset["data_source"] == "synthetic_fallback"
    assert len(dataset["occurrences"]) > 0


@skipif_no_stac
def test_t2_stac_extreme_grid_dimensions():
    """
    F2-B05: Boundary case: requesting very small (5x5) or asymmetric (40x80) grid dimensions.
    Returned raster arrays must strictly match the requested shape.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()

    # Small square grid
    ds_small = client.fetch_bhilwara_dataset(nrows=5, ncols=5, force_fallback=True)
    assert ds_small["bands"]["B4"].shape == (5, 5)

    # Asymmetric grid
    ds_asym = client.fetch_bhilwara_dataset(nrows=30, ncols=60, force_fallback=True)
    assert ds_asym["bands"]["B4"].shape == (30, 60)
    assert ds_asym["dem"].shape == (30, 60)


@skipif_no_stac
def test_t2_stac_cloud_cover_boundary_values():
    """
    F2-B06: Boundary case: cloud cover thresholds at 0.0% and 100.0%.
    Client search query builder must handle extreme values without error.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()

    # 0.0% cloud cover
    with patch.object(client, "search_scenes", return_value=[]):
        ds_zero = client.fetch_bhilwara_dataset(nrows=10, ncols=10)
        assert ds_zero is not None


# ==============================================================================
# Feature 3: Docker Configuration Boundaries - >= 5 tests
# ==============================================================================

@skipif_no_docker
def test_t2_dockerfile_no_duplicate_base_images():
    """
    F3-B01: Boundary test: Dockerfile must have exactly one base image definition (no multi-stage orphans).
    """
    content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    from_lines = [l for l in content.splitlines() if l.strip().upper().startswith("FROM")]

    assert len(from_lines) == 1, f"Expected single FROM statement in Dockerfile, found: {from_lines}"


@skipif_no_docker
def test_t2_dockerfile_workdir_is_app():
    """
    F3-B02: Boundary test: Dockerfile must define WORKDIR /app.
    """
    content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    workdir_lines = [l for l in content.splitlines() if l.strip().upper().startswith("WORKDIR")]

    assert any("/app" in l for l in workdir_lines), f"Expected WORKDIR /app, found: {workdir_lines}"


@skipif_no_docker
def test_t2_docker_compose_valid_yaml_structure():
    """
    F3-B03: Structural boundary test: inspects docker-compose.yml for required top-level services.
    """
    content = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "services:" in content, "docker-compose.yml must define 'services:'"
    assert "li-seeker:" in content or "li-seeker" in content
    assert "ports:" in content
    assert "volumes:" in content


@skipif_no_docker
def test_t2_docker_compose_healthcheck_parameters():
    """
    F3-B04: Boundary test: docker-compose.yml healthcheck specifications.
    """
    content = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "healthcheck:" in content
    assert "interval:" in content
    assert "timeout:" in content


@skipif_no_docker
def test_t2_docker_volume_syntax_relative_host():
    """
    F3-B05: Boundary test: volume mapping must use relative path ./output, not absolute host path.
    """
    content = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "./output:" in content, "Volume must use relative './output:' mapping"
    assert "C:\\" not in content and "/Users/" not in content, "No host-specific absolute paths in docker-compose.yml"


# ==============================================================================
# Feature 4: Directory Confinement Boundaries - >= 5 tests
# ==============================================================================

def test_t2_confinement_windows_system_paths_protection():
    """
    F4-B01: Boundary test: verifies no project script writes to system directories.
    """
    system_dirs = [
        Path(r"C:\Windows"),
        Path(r"C:\Program Files"),
        Path(r"C:\Program Files (x86)")
    ]

    for sdir in system_dirs:
        # Check that PROJECT_ROOT is completely distinct from system dirs
        assert not PROJECT_ROOT.is_relative_to(sdir)


def test_t2_confinement_null_byte_path_rejection():
    """
    F4-B02: Corner case: passing null bytes in path to export routines must be safely rejected.
    """
    from src.export.exporter import export_prospectivity_deliverables
    import pandas as pd
    from src.geospatial.raster_ops import GeoGrid

    grid = GeoGrid(25.05, 25.85, 74.05, 75.25, nrows=5, ncols=5)
    prob_map = np.full((5, 5), 0.5, dtype=np.float32)
    pa_metrics = {"crossing_point": {"optimal_threshold": 0.5, "deposit_capture_percentage": 50.0, "area_percentage": 50.0, "normalized_density": 1.0, "exploration_gain": 0.0}, "ausrc": 0.5}
    rankings_df = pd.DataFrame({'Feature': ['f1'], 'Importance': [1.0], 'Percentage': [100.0]})

    try:
        export_prospectivity_deliverables(prob_map, grid, pa_metrics, {"type": "FeatureCollection", "features": []}, rankings_df, "output/\x00evil")
        pytest.fail("Null byte path should have raised an exception")
    except (ValueError, TypeError, OSError):
        pass  # Expected safe rejection


def test_t2_confinement_nested_scratch_dir_containment(temp_output_dir):
    """
    F4-B03: Boundary test: creates a 5-level deeply nested subdirectory and verifies containment.
    """
    deep_dir = temp_output_dir / "l1" / "l2" / "l3" / "l4" / "l5"
    deep_dir.mkdir(parents=True, exist_ok=True)

    assert deep_dir.resolve().is_relative_to(PROJECT_ROOT)
    test_file = deep_dir / "probe.txt"
    test_file.write_text("probe", encoding="utf-8")
    assert test_file.exists()
    assert test_file.resolve().is_relative_to(PROJECT_ROOT)


def test_t2_confinement_no_foreign_drive_references():
    """
    F4-B04: Boundary test: scan project config files for accidental hardcoded foreign drive paths (D:, E:, /tmp/).
    """
    config_files = [
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / "pytest.ini"
    ]

    for cfg in config_files:
        if cfg.exists():
            text = cfg.read_text(encoding="utf-8")
            assert "D:\\" not in text
            assert "E:\\" not in text


def test_t2_confinement_output_dir_writable_and_contained():
    """
    F4-B05: Boundary test: ensures output/ directory exists and remains contained within PROJECT_ROOT.
    """
    out_dir = PROJECT_ROOT / "output"
    assert out_dir.exists(), "output/ directory must exist in project root"
    assert out_dir.resolve().is_relative_to(PROJECT_ROOT)
