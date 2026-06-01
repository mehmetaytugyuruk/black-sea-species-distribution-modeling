"""Occurrence cleaning, coordinate filtering, and summaries."""

from __future__ import annotations

import pandas as pd

from .config import RegionConfig

OUTPUT_COLUMNS = [
    "species",
    "scientificName",
    "acceptedScientificName",
    "decimalLatitude",
    "decimalLongitude",
    "eventDate",
    "year",
    "month",
    "season",
    "basisOfRecord",
    "country",
    "datasetKey",
    "occurrenceKey",
    "source",
]


def clean_occurrences(raw: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    """Clean occurrence coordinates and filter to the configured region."""

    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = raw.copy()
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df["decimalLatitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    df = df.dropna(subset=["species", "decimalLatitude", "decimalLongitude"])

    valid_coordinate_mask = (
        df["decimalLatitude"].between(-90, 90)
        & df["decimalLongitude"].between(-180, 180)
        & ~((df["decimalLatitude"] == 0) & (df["decimalLongitude"] == 0))
    )
    df = df.loc[valid_coordinate_mask].copy()

    black_sea_mask = (
        df["decimalLongitude"].between(region.min_lon, region.max_lon)
        & df["decimalLatitude"].between(region.min_lat, region.max_lat)
    )
    df = df.loc[black_sea_mask].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["month"] = _extract_month(df)
    df["season"] = df["month"].apply(month_to_season)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df["_lat_key"] = df["decimalLatitude"].round(5)
    df["_lon_key"] = df["decimalLongitude"].round(5)
    df = df.drop_duplicates(
        subset=["species", "_lat_key", "_lon_key", "eventDate"],
        keep="first",
    )
    df = df.drop(columns=["_lat_key", "_lon_key"])
    return df[OUTPUT_COLUMNS].sort_values(["species", "decimalLatitude", "decimalLongitude"]).reset_index(drop=True)


def _extract_month(df: pd.DataFrame) -> pd.Series:
    """Extract a numeric month from a GBIF month column or eventDate."""

    month = pd.to_numeric(df.get("month"), errors="coerce")
    parsed_dates = pd.to_datetime(df.get("eventDate"), errors="coerce", utc=True)
    date_month = parsed_dates.dt.month
    month = month.fillna(date_month).fillna(6)
    month = month.clip(lower=1, upper=12)
    return month.round().astype(int)


def month_to_season(month: int | float) -> str:
    """Map month number to a broad climatological season."""

    try:
        month_int = int(month)
    except (TypeError, ValueError):
        return "unknown"
    if month_int in (12, 1, 2):
        return "winter"
    if month_int in (3, 4, 5):
        return "spring"
    if month_int in (6, 7, 8):
        return "summer"
    if month_int in (9, 10, 11):
        return "autumn"
    return "unknown"


def build_species_summary(clean: pd.DataFrame, configured_species: list[str]) -> pd.DataFrame:
    """Build a per-species occurrence summary."""

    if clean.empty:
        return pd.DataFrame(
            {
                "species": configured_species,
                "clean_occurrences": [0 for _ in configured_species],
                "unique_months": [0 for _ in configured_species],
                "first_year": [pd.NA for _ in configured_species],
                "last_year": [pd.NA for _ in configured_species],
                "sources": ["" for _ in configured_species],
            }
        )

    grouped = clean.groupby("species", dropna=False)
    summary = grouped.agg(
        clean_occurrences=("species", "size"),
        unique_months=("month", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        sources=("source", lambda values: ";".join(sorted({str(value) for value in values.dropna()}))),
    ).reset_index()

    configured = pd.DataFrame({"species": configured_species})
    summary = configured.merge(summary, on="species", how="left")
    summary["clean_occurrences"] = summary["clean_occurrences"].fillna(0).astype(int)
    summary["unique_months"] = summary["unique_months"].fillna(0).astype(int)
    summary["sources"] = summary["sources"].fillna("")
    return summary

