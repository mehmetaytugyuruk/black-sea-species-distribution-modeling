"""GBIF occurrence download helpers."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from .config import RegionConfig

GBIF_OCCURRENCE_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_PAGE_SIZE = 300


class GBIFDownloadError(RuntimeError):
    """Raised when GBIF occurrence fetching fails."""


def fetch_gbif_occurrences(
    species_names: list[str],
    max_records_per_species: int,
    region: RegionConfig | None = None,
    timeout_seconds: int = 30,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fetch coordinate-bearing occurrence records from GBIF."""

    log = logger or logging.getLogger(__name__)
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for species in species_names:
            log.info("Fetching GBIF occurrences for %s", species)
            species_df = fetch_species_occurrences(
                species,
                max_records=max_records_per_species,
                region=region,
                session=session,
                timeout_seconds=timeout_seconds,
            )
            log.info("Fetched %s raw GBIF records for %s", len(species_df), species)
            frames.append(species_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_species_occurrences(
    scientific_name: str,
    max_records: int,
    region: RegionConfig | None = None,
    session: requests.Session | None = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Fetch occurrence pages for one species from GBIF."""

    if max_records <= 0:
        return pd.DataFrame()

    close_session = session is None
    request_session = session or requests.Session()
    rows: list[dict[str, Any]] = []
    offset = 0

    try:
        while len(rows) < max_records:
            limit = min(GBIF_PAGE_SIZE, max_records - len(rows))
            params = {
                "scientificName": scientific_name,
                "hasCoordinate": "true",
                "limit": limit,
                "offset": offset,
            }
            if region is not None:
                params["geometry"] = _bbox_polygon_wkt(region)
            response = request_session.get(
                GBIF_OCCURRENCE_SEARCH_URL,
                params=params,
                timeout=timeout_seconds,
            )
            if response.status_code == 429:
                raise GBIFDownloadError("GBIF rate limit reached while fetching records")
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise GBIFDownloadError(f"GBIF request failed: {exc}") from exc

            payload = response.json()
            results = payload.get("results", [])
            if not results:
                break

            for record in results:
                rows.append(_normalise_record(record, scientific_name))

            if payload.get("endOfRecords") or len(results) < limit:
                break
            offset += limit
    except requests.RequestException as exc:
        raise GBIFDownloadError(f"Could not connect to GBIF: {exc}") from exc
    finally:
        if close_session:
            request_session.close()

    return pd.DataFrame(rows)


def _bbox_polygon_wkt(region: RegionConfig) -> str:
    """Return a WKT polygon for the configured lon/lat bounding box."""

    return (
        "POLYGON(("
        f"{region.min_lon} {region.min_lat},"
        f"{region.max_lon} {region.min_lat},"
        f"{region.max_lon} {region.max_lat},"
        f"{region.min_lon} {region.max_lat},"
        f"{region.min_lon} {region.min_lat}"
        "))"
    )


def _normalise_record(record: dict[str, Any], configured_species: str) -> dict[str, Any]:
    """Keep only fields used by the pipeline."""

    return {
        "species": configured_species,
        "scientificName": record.get("scientificName"),
        "acceptedScientificName": record.get("acceptedScientificName"),
        "decimalLatitude": record.get("decimalLatitude"),
        "decimalLongitude": record.get("decimalLongitude"),
        "eventDate": record.get("eventDate"),
        "year": record.get("year"),
        "month": record.get("month"),
        "day": record.get("day"),
        "basisOfRecord": record.get("basisOfRecord"),
        "country": record.get("country"),
        "datasetKey": record.get("datasetKey"),
        "occurrenceKey": record.get("key"),
        "source": "GBIF",
    }
