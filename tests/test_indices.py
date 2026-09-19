"""
Unit tests for Sentinel-2 spectral indices and band mathematics.
"""

import numpy as np
import pytest
from src.remote_sensing.indices import (
    safe_divide, al_oh_ratio, ndci,
    pegmatite_index_1, pegmatite_index_2, lithium_pegmatite_index,
    lithium_mica_ratio, exomorphic_halo_index, ferric_iron_ratio,
    ferrous_iron_ratio, gossan_index, ndvi, compute_spectral_feature_cube,
    ree_index, ree_neodymium_index, ree_carbonatite_index
)


def test_safe_divide_avoids_zero_division():
    a = np.array([10.0, 5.0, 0.0])
    b = np.array([2.0, 0.0, 1.0])
    res = safe_divide(a, b, fill_value=0.0)
    assert res[0] == pytest.approx(5.0)
    assert res[1] == pytest.approx(0.0)
    assert res[2] == pytest.approx(0.0)
    assert np.all(np.isfinite(res))


def test_al_oh_and_ndci_ranges():
    b11 = np.array([[0.40, 0.50], [0.20, 0.35]])
    b12 = np.array([[0.20, 0.25], [0.25, 0.15]])

    ratio = al_oh_ratio(b11, b12)
    assert ratio[0, 0] == pytest.approx(2.0)
    assert ratio[0, 1] == pytest.approx(2.0)

    ndci_val = ndci(b11, b12)
    assert np.all((ndci_val >= -1.0) & (ndci_val <= 1.0))
    # For b11=0.4, b12=0.2 -> (0.4-0.2)/(0.4+0.2) = 0.2/0.6 = 0.333
    assert ndci_val[0, 0] == pytest.approx(1.0 / 3.0)


def test_cardoso_pegmatite_indices():
    b2 = np.array([0.25])
    b4 = np.array([0.15])
    b8 = np.array([0.18])
    b11 = np.array([0.45])
    b12 = np.array([0.20])

    pi1 = pegmatite_index_1(b2, b4, b8, b11)
    # (0.25 + 0.45) / (0.15 + 0.18) = 0.70 / 0.33
    assert pi1[0] == pytest.approx(0.70 / 0.33, rel=1e-3)

    pi2 = pegmatite_index_2(b2, b4, b8, b11)
    # (0.25 * 0.45) / (0.15 * 0.18) = 0.1125 / 0.027 = 4.1667
    assert pi2[0] == pytest.approx(4.1667, rel=1e-3)

    lpi = lithium_pegmatite_index(b2, b4, b11, b12)
    # (0.45 / 0.20) * (0.25 / 0.15) = 2.25 * 1.6667 = 3.75
    assert lpi[0] == pytest.approx(3.75, rel=1e-3)


def test_ndvi_and_feature_cube():
    b4 = np.full((10, 10), 0.10)
    b8 = np.full((10, 10), 0.50)
    res_ndvi = ndvi(b4, b8)
    assert np.allclose(res_ndvi, (0.5 - 0.1) / (0.5 + 0.1))

    bands = {
        'B2': np.full((10, 10), 0.2),
        'B3': np.full((10, 10), 0.2),
        'B4': np.full((10, 10), 0.2),
        'B8': np.full((10, 10), 0.2),
        'B8A': np.full((10, 10), 0.2),
        'B11': np.full((10, 10), 0.4),
        'B12': np.full((10, 10), 0.2),
    }
    cube = compute_spectral_feature_cube(bands)
    assert 'lpi' in cube
    assert 'al_oh_ratio' in cube
    assert 'pi_1' in cube
    assert 'ree_index' in cube
    assert 'ree_neodymium' in cube
    assert 'ree_carbonatite' in cube
    assert cube['lpi'].shape == (10, 10)
    assert cube['ree_index'].shape == (10, 10)


def test_ree_spectral_indices():
    """
    Validates Rare Earth Element (REE) spectral indices:
    - Neodymium (Nd3+) absorption index: B8A / B6
    - Composite REE exploration index: (B8A / B6) * (B11 / B12)
    - Carbonatite alteration index: (B11 * B8) / (B4 * B12)
    - Safe division behavior with zero denominators and non-finite inputs
    - 2D grid shape handling
    - compute_spectral_feature_cube integration with B6, B06, and fallback
    """
    # 1. Neodymium index (B8A / B6)
    b6 = np.array([0.15, 0.20, 0.0, 0.10])
    b8a = np.array([0.30, 0.40, 0.50, 0.0])
    nd_idx = ree_neodymium_index(b6, b8a)
    assert nd_idx[0] == pytest.approx(0.30 / 0.15)  # 2.0
    assert nd_idx[1] == pytest.approx(0.40 / 0.20)  # 2.0
    assert nd_idx[2] == pytest.approx(0.0)           # safe divide on zero denominator
    assert nd_idx[3] == pytest.approx(0.0)           # zero numerator
    assert np.all(np.isfinite(nd_idx))

    # 2. Composite REE exploration index (B8A / B6) * (B11 / B12)
    b11 = np.array([0.40, 0.50, 0.30, 0.25])
    b12 = np.array([0.20, 0.25, 0.0, 0.10])
    ree_comp = ree_index(b6, b8a, b11, b12)
    # Pixel 0: (0.30 / 0.15) * (0.40 / 0.20) = 2.0 * 2.0 = 4.0
    assert ree_comp[0] == pytest.approx(4.0)
    # Pixel 1: (0.40 / 0.20) * (0.50 / 0.25) = 2.0 * 2.0 = 4.0
    assert ree_comp[1] == pytest.approx(4.0)
    # Pixel 2: zero b6 and zero b12 handled safely
    assert ree_comp[2] == pytest.approx(0.0)
    assert np.all(np.isfinite(ree_comp))

    # 3. Carbonatite index: (B11 * B8) / (B4 * B12)
    b4 = np.array([0.15, 0.10, 0.0])
    b8 = np.array([0.30, 0.20, 0.25])
    b11_c = np.array([0.40, 0.30, 0.20])
    b12_c = np.array([0.20, 0.15, 0.10])
    carb_idx = ree_carbonatite_index(b4, b8, b11_c, b12_c)
    # Pixel 0: (0.40 * 0.30) / (0.15 * 0.20) = 0.12 / 0.03 = 4.0
    assert carb_idx[0] == pytest.approx(4.0)
    # Pixel 1: (0.30 * 0.20) / (0.10 * 0.15) = 0.06 / 0.015 = 4.0
    assert carb_idx[1] == pytest.approx(4.0)
    # Pixel 2: b4=0 in denominator -> safe divide fill_value 0.0
    assert carb_idx[2] == pytest.approx(0.0)
    assert np.all(np.isfinite(carb_idx))

    # 4. 2D array shape and multidimensional evaluation
    shape = (15, 25)
    b6_2d = np.full(shape, 0.18, dtype=np.float32)
    b8a_2d = np.full(shape, 0.36, dtype=np.float32)
    b11_2d = np.full(shape, 0.45, dtype=np.float32)
    b12_2d = np.full(shape, 0.15, dtype=np.float32)
    res_2d = ree_index(b6_2d, b8a_2d, b11_2d, b12_2d)
    assert res_2d.shape == shape
    # (0.36 / 0.18) * (0.45 / 0.15) = 2.0 * 3.0 = 6.0
    assert np.allclose(res_2d, 6.0)

    # 5. Feature cube with explicit 'B6'
    bands_with_b6 = {
        'B2': np.full((8, 8), 0.2),
        'B3': np.full((8, 8), 0.2),
        'B4': np.full((8, 8), 0.2),
        'B6': np.full((8, 8), 0.15),
        'B8': np.full((8, 8), 0.3),
        'B8A': np.full((8, 8), 0.3),
        'B11': np.full((8, 8), 0.4),
        'B12': np.full((8, 8), 0.2),
    }
    cube_b6 = compute_spectral_feature_cube(bands_with_b6)
    assert 'ree_index' in cube_b6
    assert 'ree_neodymium' in cube_b6
    assert 'ree_carbonatite' in cube_b6
    assert cube_b6['ree_index'].shape == (8, 8)
    assert np.allclose(cube_b6['ree_neodymium'], 0.3 / 0.15)
    assert np.allclose(cube_b6['ree_index'], (0.3 / 0.15) * (0.4 / 0.2))

    # 6. Feature cube with alternative 'B06' key
    bands_with_b06 = {
        'B2': np.full((8, 8), 0.2),
        'B3': np.full((8, 8), 0.2),
        'B4': np.full((8, 8), 0.2),
        'B06': np.full((8, 8), 0.15),
        'B8': np.full((8, 8), 0.3),
        'B8A': np.full((8, 8), 0.3),
        'B11': np.full((8, 8), 0.4),
        'B12': np.full((8, 8), 0.2),
    }
    cube_b06 = compute_spectral_feature_cube(bands_with_b06)
    assert 'ree_index' in cube_b06
    assert np.allclose(cube_b06['ree_neodymium'], 0.3 / 0.15)

    # 7. Backward-compatible fallback when B6 is completely omitted
    bands_no_b6 = {k: v for k, v in bands_with_b6.items() if k != 'B6'}
    cube_fallback = compute_spectral_feature_cube(bands_no_b6)
    assert 'ree_index' in cube_fallback
    assert 'ree_neodymium' in cube_fallback
    assert 'ree_carbonatite' in cube_fallback
    assert np.all(np.isfinite(cube_fallback['ree_index']))
    assert np.all(np.isfinite(cube_fallback['ree_neodymium']))
    assert np.all(np.isfinite(cube_fallback['ree_carbonatite']))
    # Fallback B6 = (B4 + B8A) / 2 = (0.2 + 0.3) / 2 = 0.25
    expected_nd_fallback = 0.3 / 0.25
    assert np.allclose(cube_fallback['ree_neodymium'], expected_nd_fallback)

