"""
Structural Geology & Geological Controls Evidential Layer Engine.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .raster_ops import GeoGrid


def compute_fault_distance_and_density(lineaments: list, grid: GeoGrid, density_sigma: float = 3.0):
    """
    Computes Euclidean distance to mapped fault/shear lineaments and lineament density.

    Parameters:
    -----------
    lineaments: list of dicts with 'start': (lat, lon), 'end': (lat, lon)
    grid: GeoGrid instance
    density_sigma: Gaussian kernel sigma for density field smoothing

    Returns:
    --------
    dist_km: 2D numpy array of distance to nearest lineament in km
    density_norm: 2D numpy array of normalized lineament density [0, 1]
    """
    raster = np.zeros((grid.nrows, grid.ncols), dtype=np.uint8)

    # Rasterize line segments with Bresenham-like sampling
    for line in lineaments:
        r0, c0 = grid.coord_to_pixel(line['start'][0], line['start'][1])
        r1, c1 = grid.coord_to_pixel(line['end'][0], line['end'][1])

        num_pts = max(abs(r1 - r0), abs(c1 - c0), 2) * 2
        rows = np.linspace(r0, r1, num_pts).round().astype(int)
        cols = np.linspace(c0, c1, num_pts).round().astype(int)

        valid = (rows >= 0) & (rows < grid.nrows) & (cols >= 0) & (cols < grid.ncols)
        raster[rows[valid], cols[valid]] = 1

    # Distance transform (in pixel units)
    dist_pixels = distance_transform_edt(raster == 0)

    # Approximate km conversion: 1 deg lat ~= 111 km
    deg_to_km = 111.0
    pixel_size_km = ((grid.lat_res + grid.lon_res) / 2.0) * deg_to_km
    dist_km = dist_pixels * pixel_size_km

    # Density computation via Gaussian smoothing of lineament seed pixels
    density = gaussian_filter(raster.astype(float), sigma=density_sigma)
    max_d = np.max(density)
    density_norm = density / max_d if max_d > 0 else density

    return dist_km, density_norm


def compute_granite_contact_distance(granite_centers: list, grid: GeoGrid):
    """
    Computes distance in km to parental S-type granitic plutons.
    LCT pegmatites typically concentrate in the 1-5 km halo around fertile parental plutons.
    """
    raster = np.zeros((grid.nrows, grid.ncols), dtype=np.uint8)
    for center in granite_centers:
        r, c = grid.coord_to_pixel(center[0], center[1])
        raster[r, c] = 1

    dist_pixels = distance_transform_edt(raster == 0)
    deg_to_km = 111.0
    pixel_size_km = ((grid.lat_res + grid.lon_res) / 2.0) * deg_to_km
    dist_km = dist_pixels * pixel_size_km
    return dist_km
