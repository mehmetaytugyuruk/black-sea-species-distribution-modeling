"""Command-line entry point for the Black Sea SDM pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .build_black_sea_grid import build_prediction_grid
from .clean_occurrences import build_species_summary, clean_occurrences
from .config import ProjectSettings, load_settings
from .evaluate_model import evaluate_trained_models, extract_feature_importance
from .features import engineer_features
from .fetch_gbif import GBIFDownloadError, fetch_gbif_occurrences
from .fetch_obis import fetch_obis_occurrences
from .make_maps import (
    export_habitat_suitability_map,
    export_prediction_geojson,
    select_best_model_predictions,
)
from .pseudo_absence import generate_pseudo_absences
from .train_model import TrainedModelResult, train_species_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Black Sea species distribution modeling pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to project config.yaml")
    parser.add_argument("--sample", action="store_true", help="Run with deterministic synthetic sample records")
    args = parser.parse_args()

    configure_logging()
    run_pipeline(Path(args.config), sample=args.sample)


def run_pipeline(config_path: Path, sample: bool = False) -> None:
    """Run the complete SDM workflow."""

    log = logging.getLogger(__name__)
    settings = load_settings(config_path)
    settings.ensure_directories()

    log.info("Starting %s pipeline%s", settings.project_name, " in sample mode" if sample else "")
    raw_occurrences = load_occurrences(settings, sample=sample, logger=log)
    raw_output = settings.raw_data_dir / ("sample_occurrences.csv" if sample else "gbif_occurrences.csv")
    raw_occurrences.to_csv(raw_output, index=False)

    clean = clean_occurrences(raw_occurrences, settings.region)
    if clean.empty:
        raise RuntimeError("No clean occurrences remained after filtering. Try --sample or adjust config.yaml.")

    write_if_enabled(clean, settings.output_dir / "clean_occurrences.csv", settings.output_enabled("save_clean_occurrences"))
    (settings.processed_data_dir / "clean_occurrences.csv").parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(settings.processed_data_dir / "clean_occurrences.csv", index=False)

    species_summary = build_species_summary(clean, settings.species)
    write_if_enabled(species_summary, settings.output_dir / "species_summary.csv", settings.output_enabled("save_species_summary"))

    training = build_training_dataset(clean, settings, logger=log)
    write_if_enabled(training, settings.output_dir / "training_dataset.csv", settings.output_enabled("save_training_dataset"))
    training.to_csv(settings.processed_data_dir / "training_dataset.csv", index=False)

    feature_columns = infer_feature_columns(training, settings)
    if not feature_columns:
        raise RuntimeError("No model features were enabled in config.yaml")

    results = train_species_models(
        training_data=training,
        feature_columns=feature_columns,
        target_column=settings.target_column,
        model_names=settings.model_names,
        test_size=settings.test_size,
        random_seed=int(settings.modeling.get("random_seed", settings.random_seed)),
        logger=log,
    )
    if not results:
        raise RuntimeError("No models were trained. Check occurrence counts and pseudo-absence settings.")

    metrics = evaluate_trained_models(results)
    write_if_enabled(metrics, settings.output_dir / "model_metrics.csv", settings.output_enabled("save_metrics"))

    feature_importance = extract_feature_importance(results)
    write_if_enabled(
        feature_importance,
        settings.output_dir / "feature_importance.csv",
        settings.output_enabled("save_feature_importance"),
    )

    predictions = predict_grid(clean, results, settings, logger=log)
    if settings.output_enabled("save_prediction_grid"):
        export_prediction_geojson(predictions, settings.output_dir / "prediction_grid.geojson")

    map_predictions = select_best_model_predictions(predictions, metrics)
    if settings.output_enabled("save_folium_map"):
        export_habitat_suitability_map(
            map_predictions,
            settings.output_dir / "habitat_suitability_map.html",
            settings.region,
            logger=log,
        )

    log.info("Pipeline complete. Outputs written to %s", settings.output_dir)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_occurrences(settings: ProjectSettings, sample: bool, logger: logging.Logger) -> pd.DataFrame:
    """Load sample records or fetch occurrence records from configured sources."""

    if sample:
        logger.info("Using deterministic synthetic sample records")
        return generate_sample_occurrences(settings)

    frames: list[pd.DataFrame] = []
    if settings.source_enabled("gbif"):
        try:
            frames.append(
                fetch_gbif_occurrences(
                    species_names=settings.species,
                    max_records_per_species=settings.gbif_max_records_per_species,
                    region=settings.region,
                    logger=logger,
                )
            )
        except GBIFDownloadError as exc:
            raise RuntimeError(f"GBIF fetching failed: {exc}. Re-run with --sample to validate the pipeline.") from exc
    else:
        logger.warning("GBIF disabled in config; no GBIF records will be fetched")

    obis = fetch_obis_occurrences(
        species_names=settings.species,
        enabled=settings.source_enabled("obis"),
        logger=logger,
    )
    if not obis.empty:
        frames.append(obis)

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No occurrence records were fetched. Re-run with --sample to validate the pipeline.")
    return pd.concat(frames, ignore_index=True)


def build_training_dataset(
    clean: pd.DataFrame,
    settings: ProjectSettings,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Combine presences, pseudo-absences, and engineered features."""

    presences = clean.copy()
    presences["presence"] = 1
    absences = generate_pseudo_absences(
        presences=presences,
        region=settings.region,
        ratio=settings.pseudo_absence_ratio,
        random_seed=settings.random_seed,
    )
    if absences.empty:
        raise RuntimeError("Pseudo-absence generation produced no records")

    frames = [presences.dropna(axis=1, how="all"), absences.dropna(axis=1, how="all")]
    training = pd.concat(frames, ignore_index=True, sort=False)
    training, feature_columns = engineer_features(training, settings, logger=logger)
    training = training.dropna(subset=feature_columns + [settings.target_column]).copy()
    training[settings.target_column] = training[settings.target_column].astype(int)
    return training


def infer_feature_columns(training: pd.DataFrame, settings: ProjectSettings) -> list[str]:
    """Infer the same feature column order used by feature engineering."""

    feature_columns: list[str] = []
    if settings.feature_enabled("use_lat_lon"):
        feature_columns.extend(["latitude", "longitude"])
    if settings.feature_enabled("use_month"):
        feature_columns.append("month")
    if settings.feature_enabled("use_distance_to_coast"):
        feature_columns.append("distance_to_coast_km")
    if settings.feature_enabled("use_depth") and "depth_m" in training.columns and not training["depth_m"].isna().all():
        feature_columns.append("depth_m")
    return [column for column in feature_columns if column in training.columns]


def predict_grid(
    clean: pd.DataFrame,
    results: list[TrainedModelResult],
    settings: ProjectSettings,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Predict habitat suitability across the configured grid."""

    prediction_month = representative_prediction_month(clean)
    logger.info("Building prediction grid at %.3f degree resolution for month %s", settings.grid_resolution, prediction_month)
    base_grid = build_prediction_grid(settings.region, settings.grid_resolution, month=prediction_month)
    base_grid, _ = engineer_features(base_grid, settings, logger=logger)

    frames: list[pd.DataFrame] = []
    for result in results:
        X_grid = base_grid[result.feature_columns]
        suitability = result.final_model.predict_proba(X_grid)[:, 1]
        species_grid = base_grid.copy()
        species_grid["species"] = result.species
        species_grid["model"] = result.model_name
        species_grid["suitability"] = suitability
        keep_columns = [
            "grid_id",
            "species",
            "model",
            "latitude",
            "longitude",
            "month",
            "suitability",
        ]
        if "distance_to_coast_km" in species_grid.columns:
            keep_columns.insert(-1, "distance_to_coast_km")
        if "depth_m" in species_grid.columns:
            keep_columns.append("depth_m")
        frames.append(species_grid[keep_columns])

    return pd.concat(frames, ignore_index=True)


def representative_prediction_month(clean: pd.DataFrame) -> int:
    """Choose a representative month for grid predictions."""

    if clean.empty or "month" not in clean.columns:
        return 7
    month = int(round(float(clean["month"].median())))
    return max(1, min(12, month))


def write_if_enabled(data: pd.DataFrame, output_path: Path, enabled: bool) -> None:
    if not enabled:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def generate_sample_occurrences(settings: ProjectSettings, records_per_species: int = 80) -> pd.DataFrame:
    """Create deterministic synthetic records for offline testing."""

    rng = np.random.default_rng(settings.random_seed)
    profile_centers = {
        "Engraulis encrasicolus": (42.2, 34.2, 7),
        "Trachurus mediterraneus": (42.6, 36.2, 8),
        "Sprattus sprattus": (43.3, 32.0, 5),
        "Merlangius merlangus": (44.0, 34.8, 4),
        "Mullus barbatus": (41.8, 30.3, 9),
    }
    rows: list[dict[str, object]] = []
    for index, species in enumerate(settings.species):
        center_lat, center_lon, peak_month = profile_centers.get(
            species,
            (
                (settings.region.min_lat + settings.region.max_lat) / 2,
                (settings.region.min_lon + settings.region.max_lon) / 2,
                7,
            ),
        )
        for record_number in range(records_per_species):
            lat = float(np.clip(rng.normal(center_lat, 0.75), settings.region.min_lat + 0.05, settings.region.max_lat - 0.05))
            lon = float(np.clip(rng.normal(center_lon, 1.25), settings.region.min_lon + 0.05, settings.region.max_lon - 0.05))
            month = int(((round(rng.normal(peak_month, 2.0)) - 1) % 12) + 1)
            year = int(rng.integers(2017, 2025))
            day = int(rng.integers(1, 28))
            rows.append(
                {
                    "species": species,
                    "scientificName": species,
                    "acceptedScientificName": species,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "eventDate": f"{year}-{month:02d}-{day:02d}",
                    "year": year,
                    "month": month,
                    "day": day,
                    "basisOfRecord": "HUMAN_OBSERVATION",
                    "country": "Sample",
                    "datasetKey": "synthetic-sample",
                    "occurrenceKey": f"sample-{index}-{record_number}",
                    "source": "sample",
                }
            )

    if rows:
        duplicate = dict(rows[0])
        duplicate["occurrenceKey"] = "sample-duplicate"
        rows.append(duplicate)
        invalid = dict(rows[0])
        invalid["decimalLatitude"] = 95.0
        invalid["occurrenceKey"] = "sample-invalid-coordinate"
        rows.append(invalid)
        outside = dict(rows[0])
        outside["decimalLatitude"] = settings.region.max_lat + 5
        outside["decimalLongitude"] = settings.region.max_lon + 5
        outside["occurrenceKey"] = "sample-outside-region"
        rows.append(outside)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
