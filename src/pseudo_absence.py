"""Pseudo-absence/background point generation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .clean_occurrences import month_to_season
from .config import RegionConfig


def generate_pseudo_absences(
    presences: pd.DataFrame,
    region: RegionConfig,
    ratio: float,
    random_seed: int,
    minimum_per_species: int = 20,
) -> pd.DataFrame:
    """Generate random background points inside the configured bounding box."""

    if presences.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    rounded_presence_coords = {
        (row.species, round(float(row.decimalLatitude), 5), round(float(row.decimalLongitude), 5))
        for row in presences.itertuples(index=False)
    }

    for species, group in presences.groupby("species"):
        requested_count = max(minimum_per_species, int(math.ceil(len(group) * ratio)))
        generated = 0
        attempts = 0
        while generated < requested_count and attempts < requested_count * 50:
            attempts += 1
            lat = float(rng.uniform(region.min_lat, region.max_lat))
            lon = float(rng.uniform(region.min_lon, region.max_lon))
            coord_key = (species, round(lat, 5), round(lon, 5))
            if coord_key in rounded_presence_coords:
                continue
            month = int(rng.integers(1, 13))
            rows.append(
                {
                    "species": species,
                    "scientificName": species,
                    "acceptedScientificName": species,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "eventDate": pd.NA,
                    "year": pd.NA,
                    "month": month,
                    "season": month_to_season(month),
                    "basisOfRecord": "PSEUDO_ABSENCE",
                    "country": pd.NA,
                    "datasetKey": pd.NA,
                    "occurrenceKey": f"pseudo-{species}-{generated}",
                    "source": "pseudo_absence",
                    "presence": 0,
                }
            )
            generated += 1

    return pd.DataFrame(rows)

