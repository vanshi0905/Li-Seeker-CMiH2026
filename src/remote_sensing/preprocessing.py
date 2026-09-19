"""
Sentinel-2 Preprocessing: Quality Masking, Bare-Rock Extraction, and Topographic Correction.
"""

import numpy as np


def apply_scl_mask(scl: np.ndarray, invalid_classes=None) -> np.ndarray:
    """
    Creates a boolean validity mask from Sentinel-2 Scene Classification Layer (SCL).
    
    Default invalid classes to mask out:
      3: Cloud shadows
      7: Cloud low probability / unclassified
      8: Cloud medium probability
      9: Cloud high probability
      10: Thin cirrus
      11: Snow / ice
    """
    if invalid_classes is None:
        invalid_classes = [3, 7, 8, 9, 10, 11]
    valid_mask = ~np.isin(scl, invalid_classes)
    return valid_mask


def filter_bare_rock_pediment(ndvi: np.ndarray, ndvi_threshold: float = 0.25) -> np.ndarray:
    """
    Isolates exposed bedrock outcrops, arid pediments, and bare ground by filtering out
    canopy and agricultural crops where NDVI > ndvi_threshold.
    """
    return (ndvi <= ndvi_threshold) & (ndvi >= -0.1)


def c_correction(band: np.ndarray, dem_slope_deg: np.ndarray, dem_aspect_deg: np.ndarray,
                 solar_zenith_deg: float, solar_azimuth_deg: float, mask: np.ndarray = None) -> np.ndarray:
    """
    Applies empirical C-correction (Teillet et al., 1982) to normalize topographic illumination.
    
    IL = cos(solar_zenith)*cos(slope) + sin(solar_zenith)*sin(slope)*cos(solar_azimuth - aspect)
    L_H = L_T * (cos(solar_zenith) + c) / (IL + c)
    where c = b / m from regression L_T = m * IL + b.
    """
    if mask is None:
        mask = np.ones(band.shape, dtype=bool)

    # Convert angles to radians
    sz_rad = np.deg2rad(solar_zenith_deg)
    sa_rad = np.deg2rad(solar_azimuth_deg)
    sl_rad = np.deg2rad(dem_slope_deg)
    as_rad = np.deg2rad(dem_aspect_deg)

    # Local illumination angle (cos i)
    cos_i = np.cos(sz_rad) * np.cos(sl_rad) + np.sin(sz_rad) * np.sin(sl_rad) * np.cos(sa_rad - as_rad)
    cos_sz = np.cos(sz_rad)

    # Fit linear regression on valid pixels
    valid = mask & np.isfinite(band) & (cos_i > 0.1)
    if np.sum(valid) < 500:
        # If insufficient points, return uncorrected
        return band

    x = cos_i[valid].flatten()
    y = band[valid].flatten()

    # m, b = np.polyfit(x, y, 1)
    poly = np.polyfit(x, y, 1)
    m, b = poly[0], poly[1]

    c = b / m if abs(m) > 1e-5 else 0.0

    # Apply C-correction
    corrected = np.copy(band)
    denominator = cos_i + c
    safe_den = np.where(denominator > 0.01, denominator, 0.01)
    factor = (cos_sz + c) / safe_den
    # Clamp factor to prevent overcorrection in deep shadows
    factor = np.clip(factor, 0.2, 3.0)

    corrected = band * factor
    return corrected
