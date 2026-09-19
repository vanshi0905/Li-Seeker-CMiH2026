"""
National Geochemical Mapping (NGCM) Stream Sediment Pathfinder Layer Engine.
"""

import numpy as np
from scipy.spatial import cKDTree
from .raster_ops import GeoGrid


def interpolate_ngcm_points(points: list, values: list, grid: GeoGrid, power: float = 2.0, max_neighbors: int = 12):
    """
    Interpolates sparse geochemical stream sediment assays across the district grid
    using Inverse Distance Weighting (IDW) with KD-Tree acceleration.

    Parameters:
    -----------
    points: list of (lat, lon) coordinates
    values: list of float values (e.g. Li ppm, K/Rb ratio)
    grid: GeoGrid instance
    power: IDW distance weighting exponent
    max_neighbors: number of nearest sample stations to interpolate from

    Returns:
    --------
    2D numpy array of interpolated values across grid
    """
    pts = np.array(points, dtype=np.float32)
    vals = np.array(values, dtype=np.float32)

    lats, lons = grid.get_mesh_coords()
    grid_pts = np.column_stack([lats.flatten(), lons.flatten()])

    tree = cKDTree(pts)
    k = min(len(pts), max_neighbors)
    distances, indices = tree.query(grid_pts, k=k)

    # Avoid division by zero at sample locations
    eps = 1e-5
    distances = np.maximum(distances, eps)
    weights = 1.0 / (distances ** power)
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    norm_weights = weights / sum_weights

    interpolated = np.sum(vals[indices] * norm_weights, axis=1)
    return interpolated.reshape((grid.nrows, grid.ncols))
