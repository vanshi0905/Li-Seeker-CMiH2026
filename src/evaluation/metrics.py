"""
Mineral Prospectivity Evaluation Metrics Engine.

Implements domain-specific exploration metrics:
1. Prediction-Area (P-A) Plot & Crossing-Point Threshold (Chung & Fabbri, 2003; Yousefi & Carranza, 2015)
2. Success Rate Curve (SRC) & Area Under Success Rate Curve (AUSRC)
3. Normalized Exploration Density (Nd) & Exploration Gain (EG)
"""

import numpy as np


def compute_prediction_area_plot(prospectivity_map: np.ndarray, deposit_coords: list, grid, n_steps: int = 100):
    """
    Computes the Prediction-Area (P-A) plot curves.

    Parameters:
    -----------
    prospectivity_map: 2D numpy array of predicted prospectivity scores [0.0, 1.0]
    deposit_coords: list of dicts or tuples with lat, lon of known deposits
    grid: GeoGrid instance
    n_steps: number of threshold evaluation intervals

    Returns:
    --------
    dict with:
      - area_percentages: array of cumulative prospective area % [0, 100]
      - deposit_capture_rates: array of deposit prediction rate % [0, 100]
      - thresholds: corresponding prospectivity score cutoffs
      - crossing_point: dict with optimal threshold, area %, and capture rate %
      - ausrc: Area Under Success Rate Curve
    """
    valid_scores = prospectivity_map[np.isfinite(prospectivity_map)].flatten()
    total_pixels = len(valid_scores)
    if total_pixels == 0:
        raise ValueError("Prospectivity map has zero valid pixels.")

    # Extract prospectivity scores at known deposit locations
    deposit_scores = []
    for dep in deposit_coords:
        if isinstance(dep, dict):
            lat, lon = dep["latitude"], dep["longitude"]
        else:
            lat, lon = dep[0], dep[1]
        r, c = grid.coord_to_pixel(lat, lon)
        score = prospectivity_map[r, c]
        if np.isfinite(score):
            deposit_scores.append(score)

    deposit_scores = np.array(deposit_scores)
    total_deposits = len(deposit_scores)

    thresholds = np.linspace(1.0, 0.0, n_steps)
    area_percentages = []
    deposit_capture_rates = []

    for th in thresholds:
        cum_area_pct = (np.sum(valid_scores >= th) / total_pixels) * 100.0
        cum_dep_pct = (np.sum(deposit_scores >= th) / total_deposits) * 100.0 if total_deposits > 0 else 0.0
        area_percentages.append(cum_area_pct)
        deposit_capture_rates.append(cum_dep_pct)

    area_percentages = np.array(area_percentages)
    deposit_capture_rates = np.array(deposit_capture_rates)

    # Compute Success Rate Curve (fractional coordinates: 0.0 to 1.0)
    frac_area = area_percentages / 100.0
    frac_dep = deposit_capture_rates / 100.0
    # Sort by increasing area fraction for trapezoidal integration
    sort_idx = np.argsort(frac_area)
    ausrc = float(np.trapezoid(frac_dep[sort_idx], frac_area[sort_idx]))

    # Find P-A Crossing Point: where deposit_capture_rate = 100 - area_percentage
    diff = np.abs(deposit_capture_rates - (100.0 - area_percentages))
    cross_idx = int(np.argmin(diff))

    optimal_threshold = float(thresholds[cross_idx])
    optimal_area_pct = float(area_percentages[cross_idx])
    optimal_capture_pct = float(deposit_capture_rates[cross_idx])

    # Normalized Density (Nd) at optimal threshold
    a_frac = max(optimal_area_pct / 100.0, 1e-4)
    d_frac = optimal_capture_pct / 100.0
    normalized_density = float(d_frac / a_frac)
    exploration_gain = float(d_frac - a_frac)

    return {
        "thresholds": thresholds.tolist(),
        "area_percentages": area_percentages.tolist(),
        "deposit_capture_rates": deposit_capture_rates.tolist(),
        "ausrc": round(ausrc, 4),
        "crossing_point": {
            "optimal_threshold": round(optimal_threshold, 4),
            "area_percentage": round(optimal_area_pct, 2),
            "deposit_capture_percentage": round(optimal_capture_pct, 2),
            "normalized_density": round(normalized_density, 2),
            "exploration_gain": round(exploration_gain, 4),
        }
    }
