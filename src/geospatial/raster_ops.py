"""
Geospatial Raster Operations & Coordinate Management.
"""

import numpy as np
import tifffile


class GeoGrid:
    """
    Defines a regular latitude/longitude bounding box and pixel grid for district prospectivity.
    """
    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, nrows: int, ncols: int):
        self.min_lat = float(min_lat)
        self.max_lat = float(max_lat)
        self.min_lon = float(min_lon)
        self.max_lon = float(max_lon)
        self.nrows = int(nrows)
        self.ncols = int(ncols)

        self.lat_res = (self.max_lat - self.min_lat) / self.nrows
        self.lon_res = (self.max_lon - self.min_lon) / self.ncols

        self.lat_coords = np.linspace(self.max_lat - self.lat_res/2, self.min_lat + self.lat_res/2, self.nrows)
        self.lon_coords = np.linspace(self.min_lon + self.lon_res/2, self.max_lon - self.lon_res/2, self.ncols)

    def coord_to_pixel(self, lat: float, lon: float):
        """Converts lat/lon to (row, col) indices, clamped to bounds."""
        row = int(np.round((self.max_lat - lat) / self.lat_res))
        col = int(np.round((lon - self.min_lon) / self.lon_res))
        row = np.clip(row, 0, self.nrows - 1)
        col = np.clip(col, 0, self.ncols - 1)
        return row, col

    def pixel_to_coord(self, row: int, col: int):
        """Converts (row, col) to (lat, lon)."""
        lat = self.max_lat - (row + 0.5) * self.lat_res
        lon = self.min_lon + (col + 0.5) * self.lon_res
        return float(lat), float(lon)

    def get_mesh_coords(self):
        """Returns 2D meshes of (lat, lon) for every pixel."""
        lons, lats = np.meshgrid(self.lon_coords, self.lat_coords)
        return lats, lons


def export_geotiff(raster: np.ndarray, filepath: str, grid: GeoGrid, nodata: float = -9999.0):
    """
    Exports a 2D float32 array as a standard GeoTIFF with coordinate tags.
    """
    out_raster = np.where(np.isnan(raster), nodata, raster).astype(np.float32)
    # Write GeoTIFF using tifffile
    tifffile.imwrite(
        filepath,
        out_raster,
        description=f"District Prospectivity Grid: bounds=({grid.min_lat},{grid.min_lon},{grid.max_lat},{grid.max_lon})"
    )
