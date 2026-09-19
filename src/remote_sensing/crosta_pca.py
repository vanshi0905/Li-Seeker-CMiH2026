"""
Crosta Technique: Feature-Oriented Principal Component Selection (FOPCS).

Targeting diagnostic Al-OH vibrational absorptions (muscovite, lepidolite, sericite)
using Sentinel-2 bands [B2, B4, B11, B12].
Reference: Crosta et al. (2003); Cardoso-Fernandes et al. (2020)
"""

import numpy as np
from sklearn.decomposition import PCA


def run_crosta_al_oh_pca(b2: np.ndarray, b4: np.ndarray, b11: np.ndarray, b12: np.ndarray, mask: np.ndarray = None):
    """
    Executes Crosta FOPCS for hydroxyl (Al-OH / Lepidolite / Muscovite) mapping.

    Parameters:
    -----------
    b2, b4, b11, b12: 2D numpy arrays of identical shape (Sentinel-2 surface reflectance)
    mask: optional 2D boolean array (True for valid pixels, False for masked/clouds/veg)

    Returns:
    --------
    pc_al_oh: 2D numpy array with normalized anomaly scores
    best_pc_idx: index (0-3) of the selected principal component
    loadings: 4x4 matrix of eigenvector loadings
    explained_variance_ratio: variance explained by each PC
    """
    shape = b2.shape
    if mask is None:
        mask = np.ones(shape, dtype=bool)

    valid_idx = mask & np.isfinite(b2) & np.isfinite(b4) & np.isfinite(b11) & np.isfinite(b12)
    if np.sum(valid_idx) < 100:
        raise ValueError("Insufficient valid unmasked pixels for PCA decomposition (< 100).")

    # Stack 4 diagnostic bands: B2, B4, B11, B12
    stacked = np.column_stack([
        b2[valid_idx].flatten(),
        b4[valid_idx].flatten(),
        b11[valid_idx].flatten(),
        b12[valid_idx].flatten()
    ])

    pca = PCA(n_components=4)
    pcs = pca.fit_transform(stacked)

    # Loadings shape: (n_components, n_features) -> features are [B2, B4, B11, B12]
    # Indices: 0=B2, 1=B4, 2=B11, 3=B12
    loadings = pca.components_

    # Find the PC where B11 and B12 have opposite signs and maximum absolute contrast
    contrast_scores = []
    sign_multipliers = []

    for i in range(4):
        loading_b11 = loadings[i, 2]
        loading_b12 = loadings[i, 3]

        # Check for opposite signs
        opposite_sign = (loading_b11 * loading_b12) < 0
        contrast = abs(loading_b11 - loading_b12) if opposite_sign else 0.0
        contrast_scores.append(contrast)

        # We want Al-OH (high B11, low B12) to be positive:
        # If loading_b11 > 0 and loading_b12 < 0, score = +1
        # If loading_b11 < 0 and loading_b12 > 0, score = -1 (invert to make positive)
        if loading_b11 > loading_b12:
            sign_multipliers.append(1.0)
        else:
            sign_multipliers.append(-1.0)

    best_pc_idx = int(np.argmax(contrast_scores))
    sign = sign_multipliers[best_pc_idx]

    # Reconstruct 2D spatial raster
    pc_raster = np.full(shape, np.nan, dtype=np.float32)
    best_component_values = pcs[:, best_pc_idx] * sign

    # Standardize to zero mean, unit variance for comparability
    std = np.std(best_component_values)
    if std > 1e-6:
        best_component_values = (best_component_values - np.mean(best_component_values)) / std

    pc_raster[valid_idx] = best_component_values

    return pc_raster, best_pc_idx, loadings, pca.explained_variance_ratio_
