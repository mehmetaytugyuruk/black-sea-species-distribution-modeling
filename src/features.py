"""Feature engineering for occurrences, background points, and grids."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectSettings, RegionConfig


def engineer_features(
    data: pd.DataFrame,
    settings: ProjectSettings,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add configured model feature columns to a data frame."""

    log = logger or logging.getLogger(__name__)
    df = data.copy()

    df["latitude"] = _coordinate_series(df, "latitude", "decimalLatitude")
    df["longitude"] = _coordinate_series(df, "longitude", "decimalLongitude")
    df["month"] = pd.to_numeric(df.get("month"), errors="coerce").fillna(6).clip(1, 12).round().astype(int)

    feature_columns: list[str] = []
    if settings.feature_enabled("use_lat_lon"):
        feature_columns.extend(["latitude", "longitude"])
    if settings.feature_enabled("use_month"):
        feature_columns.append("month")
    if settings.feature_enabled("use_distance_to_coast"):
        df["distance_to_coast_km"] = approximate_distance_to_region_edge_km(
            df["longitude"].to_numpy(dtype=float),
            df["latitude"].to_numpy(dtype=float),
            settings.region,
        )
        df["distance_feature_source"] = "region_edge_approximation"
        feature_columns.append("distance_to_coast_km")

    if settings.feature_enabled("use_depth"):
        depth = sample_optional_bathymetry(df, settings, logger=log)
        if depth is not None:
            df["depth_m"] = depth
            feature_columns.append("depth_m")
        else:
            log.warning("Depth feature requested but no usable local GEBCO raster was available; continuing without depth")

    return df, feature_columns


def _coordinate_series(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in df.columns:
        return pd.to_numeric(df[preferred], errors="coerce")
    return pd.to_numeric(df[fallback], errors="coerce")


def approximate_distance_to_region_edge_km(
    longitude: np.ndarray,
    latitude: np.ndarray,
    region: RegionConfig,
) -> np.ndarray:
    """Approximate distance to the bounding-box edge in kilometers.

    This is a conservative v1 fallback when coastline vector data are not
    available. It should not be interpreted as an exact coastline distance.
    """

    latitude_rad = np.deg2rad(latitude)
    km_per_lon = 111.320 * np.cos(latitude_rad)
    km_per_lat = 110.574

    west = np.maximum(longitude - region.min_lon, 0) * km_per_lon
    east = np.maximum(region.max_lon - longitude, 0) * km_per_lon
    south = np.maximum(latitude - region.min_lat, 0) * km_per_lat
    north = np.maximum(region.max_lat - latitude, 0) * km_per_lat
    return np.minimum.reduce([west, east, south, north])


def sample_optional_bathymetry(
    df: pd.DataFrame,
    settings: ProjectSettings,
    logger: logging.Logger | None = None,
) -> pd.Series | None:
    """Sample a local GEBCO raster when configured and available."""

    log = logger or logging.getLogger(__name__)
    if not settings.source_enabled("gebco"):
        log.warning("Depth feature requested, but GEBCO is disabled in config")
        return None

    raster_path = settings.gebco_raster_path
    if not raster_path.exists():
        log.warning("Configured GEBCO raster not found at %s", raster_path)
        return None

    return sample_raster_values(raster_path, df["longitude"], df["latitude"], logger=log)


def sample_raster_values(
    raster_path: Path,
    longitude: pd.Series,
    latitude: pd.Series,
    logger: logging.Logger | None = None,
) -> pd.Series | None:
    """Sample raster values without making rasterio a hard dependency."""

    log = logger or logging.getLogger(__name__)
    try:
        import rasterio
    except ImportError:
        log.warning("rasterio is not installed; skipping optional bathymetry sampling")
        return None

    coords = list(zip(longitude.astype(float), latitude.astype(float)))
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_string() not in {"EPSG:4326", "OGC:CRS84"}:
            log.warning(
                "GEBCO raster CRS is %s; v1 expects lon/lat coordinates and will sample without reprojection",
                src.crs,
            )
        values = [float(sample[0]) if len(sample) else np.nan for sample in src.sample(coords)]
    return pd.Series(values, index=longitude.index)

