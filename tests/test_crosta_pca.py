"""
Unit tests for Crosta Feature-Oriented PCA.
"""

import numpy as np
import pytest
from src.remote_sensing.crosta_pca import run_crosta_al_oh_pca


def test_crosta_pca_detects_al_oh_contrast():
    np.random.seed(42)
    shape = (20, 20)

    b2 = np.random.uniform(0.1, 0.3, shape).astype(np.float32)
    b4 = np.random.uniform(0.1, 0.3, shape).astype(np.float32)
    b11 = np.random.uniform(0.2, 0.4, shape).astype(np.float32)
    b12 = np.random.uniform(0.2, 0.4, shape).astype(np.float32)

    # Inject strong Al-OH signature (high B11, very low B12) at pixel (5, 5)
    b11[5, 5] = 0.85
    b12[5, 5] = 0.05

    pc_raster, best_idx, loadings, var_ratio = run_crosta_al_oh_pca(b2, b4, b11, b12)

    assert pc_raster.shape == shape
    assert 0 <= best_idx < 4
    assert loadings.shape == (4, 4)
    assert len(var_ratio) == 4

    # The anomalous pixel (5, 5) should have an elevated anomaly value
    assert pc_raster[5, 5] > np.mean(pc_raster)
