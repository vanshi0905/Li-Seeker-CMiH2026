"""
E2E Test Configuration, Shared Fixtures, and Environment Validators.
Project: Li-Seeker MVP Improvements (CMiH 2026 - Problem Statement 01)
"""

import os
import sys
import shutil
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is at top of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Authoritative Constants from ORIGINAL_REQUEST.md & PROJECT.md
BHILWARA_BBOX = [74.05, 25.05, 75.25, 25.85]  # [min_lon, min_lat, max_lon, max_lat]
DOCKER_PORT = 8501
REQUIRED_SENTINEL2_BANDS = ['B2', 'B3', 'B4', 'B6', 'B8', 'B8A', 'B11', 'B12']

# Dynamic Feature Availability Checkers for Progressive Testability
try:
    from src.remote_sensing.indices import (
        ree_index, ree_neodymium_index, ree_carbonatite_index
    )
    HAS_REE = True
except ImportError:
    HAS_REE = False

try:
    from src.data_ingestion.stac_client import Sentinel2STACClient
    HAS_STAC = True
except (ImportError, ModuleNotFoundError):
    HAS_STAC = False

HAS_DOCKER = (
    (PROJECT_ROOT / "Dockerfile").exists() and
    (PROJECT_ROOT / "docker-compose.yml").exists()
)

skipif_no_ree = pytest.mark.skipif(
    not HAS_REE,
    reason="Milestone 1 (REE Spectral Indices) not yet implemented in src/remote_sensing/indices.py"
)

skipif_no_stac = pytest.mark.skipif(
    not HAS_STAC,
    reason="Milestone 2 (STAC Client) not yet implemented in src/data_ingestion/stac_client.py"
)

skipif_no_docker = pytest.mark.skipif(
    not HAS_DOCKER,
    reason="Milestone 3 (Docker files) not yet created at project root"
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the absolute Path to project root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def bhilwara_bbox() -> list[float]:
    """Returns the authoritative Bhilwara exploration bbox [min_lon, min_lat, max_lon, max_lat]."""
    return list(BHILWARA_BBOX)


@pytest.fixture(scope="module")
def small_benchmark_dataset():
    """Generates a fast 20x30 benchmark dataset for rapid test execution."""
    from src.geospatial.synthetic_generator import build_bhilwara_benchmark
    return build_bhilwara_benchmark(nrows=20, ncols=30, seed=42)


@pytest.fixture
def temp_output_dir(request):
    """Creates and yields an isolated scratch directory strictly within PROJECT_ROOT/output, cleaning up afterwards."""
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', request.node.name)
    scratch = PROJECT_ROOT / "output" / f"_e2e_{safe_name}"
    scratch.mkdir(parents=True, exist_ok=True)
    yield scratch
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
