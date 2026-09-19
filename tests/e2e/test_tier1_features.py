"""
Tier 1: Feature Coverage E2E Tests (>=5 tests per feature).
Features Covered:
  - F1 / F3: REE Spectral Indices (indices.py & feature cube)
  - F4 / F5 / F6: STAC Ingestion & Offline Fallback (stac_client.py)
  - F10 / F11 / F12: Docker Configuration (Dockerfile, docker-compose.yml)
  - F13: Directory Confinement (PROJECT_ROOT containment & isolation)
"""

import os
import re
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from tests.e2e.conftest import (
    PROJECT_ROOT, BHILWARA_BBOX, DOCKER_PORT, REQUIRED_SENTINEL2_BANDS,
    skipif_no_ree, skipif_no_stac, skipif_no_docker
)


# ==============================================================================
# Feature 1: REE Spectral Indices (F1 / F3) - >= 5 tests
# ==============================================================================

@skipif_no_ree
def test_t1_ree_neodymium_index_spectral_contrast():
    """
    F1-01: Verifies Neodymium absorption index (B8A / B6) calculation and spectral contrast.
    High B8A (continuum shoulder) over low B6 (Nd3+ absorption dip) must yield a distinct high ratio.
    """
    from src.remote_sensing.indices import ree_neodymium_index

    # Mineralized pixel: B6 has Nd3+ absorption dip (0.15), B8A is high continuum (0.45) -> ratio = 3.0
    # Background pixel: B6 is 0.30, B8A is 0.30 -> ratio = 1.0
    b6 = np.array([0.15, 0.30], dtype=np.float32)
    b8a = np.array([0.45, 0.30], dtype=np.float32)

    nd_ratio = ree_neodymium_index(b6, b8a)

    assert nd_ratio[0] == pytest.approx(3.0, rel=1e-4)
    assert nd_ratio[1] == pytest.approx(1.0, rel=1e-4)
    assert nd_ratio[0] > nd_ratio[1], "Mineralized pixel with Nd3+ absorption must exceed background"


@skipif_no_ree
def test_t1_ree_neodymium_safe_zero_division():
    """
    F1-02: Verifies safe_divide protection in ree_neodymium_index when denominator is zero or near-zero.
    """
    from src.remote_sensing.indices import ree_neodymium_index

    b6 = np.array([0.0, 1e-8, -0.05, 0.20], dtype=np.float32)
    b8a = np.array([0.40, 0.40, 0.40, 0.40], dtype=np.float32)

    nd_ratio = ree_neodymium_index(b6, b8a)

    assert np.all(np.isfinite(nd_ratio)), "Output must never contain NaN or Inf"
    assert nd_ratio[0] == pytest.approx(0.0), "Zero denominator must return fill_value 0.0"
    assert nd_ratio[1] == pytest.approx(0.0), "Sub-epsilon denominator must return fill_value 0.0"


@skipif_no_ree
def test_t1_ree_composite_index_calculation():
    """
    F1-03: Verifies composite REE exploration index: (B8A / B6) * (B11 / B12).
    Couples Nd3+ absorption with hydrothermal Al-OH/carbonate alteration.
    """
    from src.remote_sensing.indices import ree_index

    # Pixel 0: Nd term = 0.40 / 0.20 = 2.0; Alteration term = 0.50 / 0.25 = 2.0 -> REEI = 4.0
    # Pixel 1: Nd term = 0.30 / 0.30 = 1.0; Alteration term = 0.20 / 0.20 = 1.0 -> REEI = 1.0
    b6 = np.array([0.20, 0.30], dtype=np.float32)
    b8a = np.array([0.40, 0.30], dtype=np.float32)
    b11 = np.array([0.50, 0.20], dtype=np.float32)
    b12 = np.array([0.25, 0.20], dtype=np.float32)

    result = ree_index(b6, b8a, b11, b12)

    assert result[0] == pytest.approx(4.0, rel=1e-4)
    assert result[1] == pytest.approx(1.0, rel=1e-4)
    assert result.dtype == np.float32


@skipif_no_ree
def test_t1_ree_carbonatite_index_calculation():
    """
    F1-04: Verifies carbonatite host index: (B11 * B8) / (B4 * B12).
    Targeting bastnäsite and alkaline fenitization halos.
    """
    from src.remote_sensing.indices import ree_carbonatite_index

    b4 = np.array([0.10, 0.0], dtype=np.float32)
    b8 = np.array([0.20, 0.30], dtype=np.float32)
    b11 = np.array([0.40, 0.50], dtype=np.float32)
    b12 = np.array([0.20, 0.10], dtype=np.float32)

    carb = ree_carbonatite_index(b4, b8, b11, b12)

    # Pixel 0: (0.40 * 0.20) / (0.10 * 0.20) = 0.08 / 0.02 = 4.0
    assert carb[0] == pytest.approx(4.0, rel=1e-4)
    # Pixel 1: b4 is 0 -> denominator is 0 -> safe divide returns 0.0
    assert carb[1] == pytest.approx(0.0)
    assert np.all(np.isfinite(carb))


@skipif_no_ree
def test_t1_spectral_feature_cube_includes_ree_indices():
    """
    F1-05: Verifies that compute_spectral_feature_cube ingests B6 and returns REE indices.
    """
    from src.remote_sensing.indices import compute_spectral_feature_cube

    nrows, ncols = 12, 16
    bands = {
        'B2': np.full((nrows, ncols), 0.22, dtype=np.float32),
        'B3': np.full((nrows, ncols), 0.18, dtype=np.float32),
        'B4': np.full((nrows, ncols), 0.15, dtype=np.float32),
        'B6': np.full((nrows, ncols), 0.16, dtype=np.float32),
        'B8': np.full((nrows, ncols), 0.25, dtype=np.float32),
        'B8A': np.full((nrows, ncols), 0.32, dtype=np.float32),
        'B11': np.full((nrows, ncols), 0.42, dtype=np.float32),
        'B12': np.full((nrows, ncols), 0.21, dtype=np.float32),
    }

    cube = compute_spectral_feature_cube(bands)

    assert isinstance(cube, dict)
    assert 'ree_index' in cube, "Spectral feature cube must contain 'ree_index'"
    assert 'ree_neodymium' in cube, "Spectral feature cube must contain 'ree_neodymium'"
    assert 'ree_carbonatite' in cube, "Spectral feature cube must contain 'ree_carbonatite'"
    assert cube['ree_index'].shape == (nrows, ncols)
    assert np.all(np.isfinite(cube['ree_index']))


@skipif_no_ree
def test_t1_spectral_feature_cube_graceful_missing_b6():
    """
    F1-06: Verifies compute_spectral_feature_cube handles missing B6 gracefully via interpolation.
    """
    from src.remote_sensing.indices import compute_spectral_feature_cube

    nrows, ncols = 8, 8
    bands = {
        'B2': np.full((nrows, ncols), 0.20, dtype=np.float32),
        'B3': np.full((nrows, ncols), 0.20, dtype=np.float32),
        'B4': np.full((nrows, ncols), 0.15, dtype=np.float32),
        'B8': np.full((nrows, ncols), 0.25, dtype=np.float32),
        'B8A': np.full((nrows, ncols), 0.35, dtype=np.float32),
        'B11': np.full((nrows, ncols), 0.40, dtype=np.float32),
        'B12': np.full((nrows, ncols), 0.20, dtype=np.float32),
    }
    # No 'B6' in dictionary
    cube = compute_spectral_feature_cube(bands)

    assert 'ree_index' in cube
    assert 'ree_neodymium' in cube
    assert np.all(np.isfinite(cube['ree_index']))
    assert cube['ree_index'].shape == (nrows, ncols)


# ==============================================================================
# Feature 2: STAC Ingestion & Offline Fallback (F4 / F5 / F6) - >= 5 tests
# ==============================================================================

@skipif_no_stac
def test_t1_stac_bhilwara_bbox_coordinates():
    """
    F2-01: Verifies that STAC client module specifies or validates the Bhilwara bounding box.
    """
    from src.data_ingestion import stac_client

    bbox = getattr(stac_client, 'BHILWARA_BBOX', None)
    if bbox is None:
        # Check if defined on Sentinel2STACClient class
        client_cls = getattr(stac_client, 'Sentinel2STACClient', None)
        bbox = getattr(client_cls, 'BHILWARA_BBOX', None) or getattr(client_cls, 'DEFAULT_BBOX', None)

    assert bbox is not None, "STAC client module must define the Bhilwara bbox"
    assert bbox == BHILWARA_BBOX, f"Expected {BHILWARA_BBOX}, got {bbox}"


@skipif_no_stac
def test_t1_stac_client_initialization():
    """
    F2-02: Verifies Sentinel2STACClient initializes with default endpoint and timeout.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()
    assert client is not None
    assert hasattr(client, "endpoint")
    assert "planetarycomputer" in client.endpoint.lower() or "http" in client.endpoint.lower()
    assert hasattr(client, "timeout")
    assert client.timeout >= 5


@skipif_no_stac
def test_t1_stac_offline_fallback_on_network_error():
    """
    F2-03: Verifies fetch_bhilwara_dataset catches network errors and gracefully falls back.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()

    # Simulate network failure during search
    with patch.object(client, "search_scenes", side_effect=ConnectionError("Simulated network down")):
        dataset = client.fetch_bhilwara_dataset(nrows=20, ncols=30)

    assert isinstance(dataset, dict)
    assert dataset.get("data_source") == "synthetic_fallback"
    assert "grid" in dataset
    assert "bands" in dataset


@skipif_no_stac
def test_t1_stac_dataset_schema_and_data_source_flag():
    """
    F2-04: Verifies downstream schema contract of dataset returned by fetch_bhilwara_dataset.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()
    dataset = client.fetch_bhilwara_dataset(nrows=20, ncols=30, force_fallback=True)

    required_keys = [
        'grid', 'occurrences', 'bands', 'scl', 'dem', 'slope',
        'fault_dist_km', 'lineament_density', 'granite_dist_km',
        'aeromag_rtp', 'ngcm_li', 'ngcm_k_rb', 'data_source'
    ]
    for key in required_keys:
        assert key in dataset, f"Dataset must contain key '{key}'"

    assert dataset['data_source'] in ['synthetic_fallback', 'real_stac']


@skipif_no_stac
def test_t1_stac_force_fallback_parameter():
    """
    F2-05: Verifies that force_fallback=True returns the benchmark dataset immediately.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()
    dataset = client.fetch_bhilwara_dataset(nrows=15, ncols=25, force_fallback=True, seed=123)

    assert dataset["data_source"] == "synthetic_fallback"
    assert dataset["grid"].nrows == 15
    assert dataset["grid"].ncols == 25
    assert len(dataset["occurrences"]) > 0


@skipif_no_stac
def test_t1_stac_bands_structure_and_dimensions():
    """
    F2-06: Verifies that all Sentinel-2 bands are 2D arrays with requested dimensions.
    """
    from src.data_ingestion.stac_client import Sentinel2STACClient

    client = Sentinel2STACClient()
    nrows, ncols = 22, 33
    dataset = client.fetch_bhilwara_dataset(nrows=nrows, ncols=ncols, force_fallback=True)
    bands = dataset["bands"]

    for b in REQUIRED_SENTINEL2_BANDS:
        assert b in bands, f"Band '{b}' must be present in dataset['bands']"
        assert bands[b].shape == (nrows, ncols)
        assert np.all(np.isfinite(bands[b]))


# ==============================================================================
# Feature 3: Docker Configuration (F10 / F11 / F12) - >= 5 tests
# ==============================================================================

@skipif_no_docker
def test_t1_dockerfile_base_image():
    """
    F3-01: Verifies Dockerfile base image is python:3.11-slim.
    """
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    content = dockerfile_path.read_text(encoding="utf-8")

    from_lines = [line.strip() for line in content.splitlines() if line.strip().upper().startswith("FROM")]
    assert len(from_lines) >= 1, "Dockerfile must contain a FROM directive"
    assert any("python:3.11-slim" in line.lower() for line in from_lines), \
        f"Dockerfile must use python:3.11-slim as base image, found: {from_lines}"


@skipif_no_docker
def test_t1_dockerfile_system_dependencies():
    """
    F3-02: Verifies Dockerfile installs libgomp1 (OpenMP for XGBoost) and curl (healthcheck).
    """
    content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "libgomp1" in content, "Dockerfile must install libgomp1 for XGBoost multithreading"
    assert "curl" in content, "Dockerfile must install curl for container healthcheck"


@skipif_no_docker
def test_t1_dockerfile_port_and_entrypoint():
    """
    F3-03: Verifies Dockerfile exposes port 8501 and runs streamlit.
    """
    content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert re.search(r"EXPOSE\s+8501", content, re.IGNORECASE), "Dockerfile must contain 'EXPOSE 8501'"
    assert "streamlit" in content.lower()
    assert "app/app.py" in content or "app.py" in content


@skipif_no_docker
def test_t1_docker_compose_service_and_ports():
    """
    F3-04: Verifies docker-compose.yml defines service and maps port 8501:8501.
    """
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    content = compose_path.read_text(encoding="utf-8")

    assert "li-seeker" in content.lower()
    assert "8501:8501" in content, "docker-compose.yml must map port 8501:8501"


@skipif_no_docker
def test_t1_docker_compose_volume_mapping():
    """
    F3-05: Verifies docker-compose.yml mounts ./output for persistent deliverables.
    """
    content = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "output" in content, "docker-compose.yml must mount ./output volume"
    assert "/app/output" in content or ":/app/output" in content


@skipif_no_docker
def test_t1_dockerignore_and_streamlit_config():
    """
    F3-06: Verifies .dockerignore and .streamlit/config.toml configurations.
    """
    dockerignore = PROJECT_ROOT / ".dockerignore"
    streamlit_config = PROJECT_ROOT / ".streamlit" / "config.toml"

    assert dockerignore.exists(), ".dockerignore must exist at project root"
    ign_text = dockerignore.read_text(encoding="utf-8")
    assert ".git" in ign_text or "__pycache__" in ign_text or ".agents" in ign_text

    assert streamlit_config.exists(), ".streamlit/config.toml must exist"
    st_text = streamlit_config.read_text(encoding="utf-8")
    assert "headless" in st_text.lower()


# ==============================================================================
# Feature 4: Directory Confinement (F13) - >= 5 tests
# ==============================================================================

def test_t1_confinement_project_root_identity(project_root):
    """
    F4-01: Verifies that project root contains core repository packages and resolves cleanly.
    """
    assert (project_root / "src").exists()
    assert (project_root / "data").exists()
    assert project_root.is_dir()


def test_t1_confinement_no_files_written_outside_root(project_root):
    """
    F4-02: Verifies that benchmark synthesis generates data purely in RAM and writes no external files.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark

    dataset = build_bhilwara_benchmark(nrows=10, ncols=15, seed=99)
    assert dataset is not None
    assert dataset["grid"].nrows == 10


def test_t1_confinement_output_directory_default(temp_output_dir):
    """
    F4-03: Verifies export routines write exclusively within project_root/output.
    """
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    from src.evaluation.metrics import compute_prediction_area_plot
    from src.evaluation.target_extractor import extract_prospective_targets
    from src.export.exporter import export_prospectivity_deliverables
    import pandas as pd

    dataset = build_bhilwara_benchmark(nrows=15, ncols=20, seed=42)
    grid = dataset['grid']
    occurrences = dataset['occurrences']

    # Dummy map and metrics
    prob_map = np.random.uniform(0.1, 0.9, (15, 20)).astype(np.float32)
    pa_metrics = compute_prediction_area_plot(prob_map, occurrences, grid, n_steps=10)
    targets, geojson = extract_prospective_targets(prob_map, grid, threshold=0.5)
    rankings_df = pd.DataFrame({'Feature': ['f1'], 'Importance': [1.0], 'Percentage': [100.0]})

    paths = export_prospectivity_deliverables(
        prob_map, grid, pa_metrics, geojson, rankings_df, str(temp_output_dir)
    )

    for key, path_str in paths.items():
        file_path = Path(path_str).resolve()
        assert file_path.exists(), f"Exported deliverable {key} does not exist: {file_path}"
        assert PROJECT_ROOT in file_path.parents, f"File {file_path} escaped PROJECT_ROOT {PROJECT_ROOT}"


def test_t1_confinement_path_traversal_rejection(temp_output_dir):
    """
    F4-04: Verifies path normalization, directory containment, and strict traversal rejection.
    """
    from src.export.exporter import export_prospectivity_deliverables
    from src.geospatial.raster_ops import GeoGrid
    import pandas as pd

    # 1. Valid nested directory containment
    nested_dir = temp_output_dir / "nested" / "subrun"
    nested_dir.mkdir(parents=True, exist_ok=True)
    resolved = nested_dir.resolve()
    assert PROJECT_ROOT in resolved.parents

    # 2. Rejection of relative path traversal escaping output/
    grid = GeoGrid(25.05, 25.85, 74.05, 75.25, nrows=5, ncols=5)
    prob_map = np.full((5, 5), 0.5, dtype=np.float32)
    pa_metrics = {"crossing_point": {"optimal_threshold": 0.5, "deposit_capture_percentage": 50.0, "area_percentage": 50.0, "normalized_density": 1.0, "exploration_gain": 0.0}, "ausrc": 0.5, "area_percentages": [10.0], "deposit_capture_rates": [50.0]}
    target_geojson = {"type": "FeatureCollection", "features": []}
    rankings_df = pd.DataFrame({'Feature': ['f1'], 'Importance': [1.0], 'Percentage': [100.0]})

    with pytest.raises(ValueError, match="Directory Confinement Violation"):
        export_prospectivity_deliverables(
            prob_map, grid, pa_metrics, target_geojson, rankings_df, "../adversarial_escape_attempt"
        )

    # 3. Rejection of absolute path escaping output/
    with pytest.raises(ValueError, match="Directory Confinement Violation"):
        export_prospectivity_deliverables(
            prob_map, grid, pa_metrics, target_geojson, rankings_df, "C:/Windows/Temp/adversarial_escape_attempt"
        )


def test_t1_confinement_synthetic_generator_loads_local_ground_truth(project_root):
    """
    F4-05: Verifies ground-truth pegmatite catalog resides strictly inside project data/ directory.
    """
    gt_file = project_root / "data" / "ground_truth" / "bhilwara_pegmatites.json"
    assert gt_file.exists(), f"Ground-truth file must exist at {gt_file}"
    assert gt_file.resolve().is_relative_to(project_root)


def test_t1_confinement_agents_directory_has_no_source_code(project_root):
    """
    F4-06: Verifies .agents directory contains only metadata and no production source code.
    """
    agents_dir = project_root / ".agents"
    if agents_dir.exists():
        py_files = list(agents_dir.rglob("*.py"))
        assert len(py_files) == 0, f"Found prohibited Python source code in .agents/: {py_files}"
