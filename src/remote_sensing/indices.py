"""
Sentinel-2 Remote Sensing Spectral Indices for Lithium Pegmatite Exploration.

Formulations derived from:
1. Cardoso-Fernandes et al. (2019, 2020, 2021, 2022) - Remote Sensing & Ore Geology Reviews
2. Crosta et al. (2003) - Feature Oriented Principal Component Selection
3. Sabins (1999) - Remote Sensing: Principles and Interpretation
"""

import numpy as np


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0, eps: float = 1e-6) -> np.ndarray:
    """Safe division preventing ZeroDivisionError and NaN/Inf generation."""
    num = np.asarray(numerator, dtype=np.float32)
    den = np.asarray(denominator, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(den) > eps, num / den, fill_value)
    return np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)


def al_oh_ratio(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Muscovite / Lepidolite / Al-OH Hydrothermal Alteration Index.
    Ratio: B11 (SWIR-1, 1610nm) / B12 (SWIR-2, 2190nm).
    Highlights 2.20 micron Al-OH vibrational absorption dip in B12.
    """
    return safe_divide(b11, b12)


def ndci(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Clay / Hydroxyl Index.
    Ratio: (B11 - B12) / (B11 + B12).
    Range: [-1.0, 1.0]. Robust against terrain illumination shading.
    """
    return safe_divide(b11 - b12, b11 + b12)


def pegmatite_index_1(b2: np.ndarray, b4: np.ndarray, b8: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """
    Cardoso-Fernandes Pegmatite Index 1 (PI_1 / Additive Pegmatite Contrast).
    Formula: (B2 + B11) / (B4 + B8)
    Leverages high pegmatite albedo in Blue (B2) and SWIR-1 (B11).
    """
    return safe_divide(b2 + b11, b4 + b8)


def pegmatite_index_2(b2: np.ndarray, b4: np.ndarray, b8: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """
    Cardoso-Fernandes Pegmatite Index 2 (PI_2 / Multiplicative Contrast).
    Formula: (B2 * B11) / (B4 * B8)
    Strongly discriminates pegmatite leucosomes from dark metamorphic country rock.
    """
    return safe_divide(b2 * b11, b4 * b8)


def lithium_pegmatite_index(b2: np.ndarray, b4: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Cardoso-Fernandes Lithium Pegmatite Index (LPI).
    Formula: (B11 / B12) * (B2 / B4)
    Couples the Al-OH mica absorption (B11/B12) with felsic quartz-feldspar albedo (B2/B4).
    """
    mica = safe_divide(b11, b12)
    albedo = safe_divide(b2, b4)
    return mica * albedo


def lithium_mica_ratio(b2: np.ndarray, b8a: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Lithium Mica Discrimination Ratio (LMDR).
    Formula: (B11 - B12) / (B8a + B2)
    """
    return safe_divide(b11 - b12, b8a + b2)


def exomorphic_halo_index(b3: np.ndarray, b4: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Exomorphic Halo Index (EHI) targeting metasomatic tourmaline/biotite halos.
    Formula: (B4 / B3) * (B12 / B11)
    """
    return safe_divide(b4, b3) * safe_divide(b12, b11)


def ferric_iron_ratio(b2: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """
    Ferric Iron (Fe3+) Index (Hematite / Goethite gossan cap).
    Ratio: B4 (Red) / B2 (Blue).
    """
    return safe_divide(b4, b2)


def ferrous_iron_ratio(b3: np.ndarray, b4: np.ndarray, b8a: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Ferrous Iron (Fe2+) Index.
    Ratio: (B12 / B8a) + (B3 / B4).
    """
    return safe_divide(b12, b8a) + safe_divide(b3, b4)


def gossan_index(b4: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """
    Gossan Alteration Ratio.
    Ratio: B11 (SWIR-1) / B4 (Red).
    """
    return safe_divide(b11, b4)


def ndvi(b4: np.ndarray, b8: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.
    Ratio: (B8 - B4) / (B8 + B4).
    """
    return safe_divide(b8 - b4, b8 + b4)


def ree_neodymium_index(b6: np.ndarray, b8a: np.ndarray) -> np.ndarray:
    """
    Neodymium (Nd3+) REE Absorption Index.
    Ratio: B8A (Narrow NIR, 865nm) / B6 (Red Edge 2, 740nm).
    Exploits the sharp 740-745nm electronic absorption trough of trivalent Neodymium (Nd3+)
    bracketed by the unabsorbed 865nm continuum shoulder.
    Diagnostic for monazite, bastnasite, and xenotime.
    """
    return safe_divide(b8a, b6)


def ree_index(b6: np.ndarray, b8a: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Composite Rare Earth Element (REE) Exploration Index.
    Formula: (B8A / B6) * (B11 / B12)
    Couples diagnostic Nd3+ electronic absorption at 740nm (B8A / B6) with the hydrothermal
    hydroxyl / carbonate alteration signature (B11 / B12) characteristic of REE-bearing
    carbonatites and fractionated rare-element pegmatites.
    """
    nd_term = safe_divide(b8a, b6)
    alteration_term = safe_divide(b11, b12)
    return nd_term * alteration_term


def ree_carbonatite_index(b4: np.ndarray, b8: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Carbonatite / Alkaline Rock REE Alteration Index (Mars & Rowan, 2010).
    Formula: (B11 * B8) / (B4 * B12)
    Highlights carbonatite intrusions and fenitization halos hosting REE minerals.
    """
    return safe_divide(b11 * b8, b4 * b12)


def compute_spectral_feature_cube(bands: dict) -> dict:
    """
    Computes all standard LCT pegmatite, alteration, and REE indices from a Sentinel-2 band dictionary.
    
    Expected keys: 'B2', 'B3', 'B4', 'B8', 'B8A' (or 'B8a'), 'B11', 'B12', and optional 'B6' (or 'B06').
    Returns a dictionary of 2D numpy arrays.
    """
    b2 = bands['B2']
    b3 = bands['B3']
    b4 = bands['B4']
    b8 = bands['B8']
    b8a = bands.get('B8A', bands.get('B8a', bands['B8']))
    b11 = bands['B11']
    b12 = bands['B12']

    # Band 6 with backward-compatible virtual red-edge fallback if absent
    b6 = bands.get('B6', bands.get('B06', None))
    if b6 is None:
        b6 = safe_divide(b4 + b8a, 2.0)

    features = {
        'ndvi': ndvi(b4, b8),
        'al_oh_ratio': al_oh_ratio(b11, b12),
        'ndci': ndci(b11, b12),
        'pi_1': pegmatite_index_1(b2, b4, b8, b11),
        'pi_2': pegmatite_index_2(b2, b4, b8, b11),
        'lpi': lithium_pegmatite_index(b2, b4, b11, b12),
        'lmdr': lithium_mica_ratio(b2, b8a, b11, b12),
        'ehi': exomorphic_halo_index(b3, b4, b11, b12),
        'fe3_ratio': ferric_iron_ratio(b2, b4),
        'fe2_ratio': ferrous_iron_ratio(b3, b4, b8a, b12),
        'gossan': gossan_index(b4, b11),
        # REE Spectral Indices
        'ree_index': ree_index(b6, b8a, b11, b12),
        'ree_neodymium': ree_neodymium_index(b6, b8a),
        'ree_carbonatite': ree_carbonatite_index(b4, b8, b11, b12),
    }
    return features

