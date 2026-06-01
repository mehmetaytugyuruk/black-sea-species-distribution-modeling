"""Build regular prediction grids over the Black Sea bounding box."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RegionConfig


def build_prediction_grid(region: RegionConfig, resolution_degrees: float, month: int = 7) -> pd.DataFrame:
    """Create a regular lon/lat grid covering the configured region."""

    if resolution_degrees <= 0:
        raise ValueError("Grid resolution must be positive")

    lons = np.arange(region.min_lon, region.max_lon + resolution_degrees / 2, resolution_degrees)
    lats = np.arange(region.min_lat, region.max_lat + resolution_degrees / 2, resolution_degrees)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid = pd.DataFrame(
        {
            "grid_id": np.arange(lon_grid.size),
            "longitude": lon_grid.ravel().round(6),
            "latitude": lat_grid.ravel().round(6),
            "month": int(month),
        }
    )
    grid["decimalLongitude"] = grid["longitude"]
    grid["decimalLatitude"] = grid["latitude"]
    return grid

