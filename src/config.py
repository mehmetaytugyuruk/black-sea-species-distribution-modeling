"""Project configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise RuntimeError("PyYAML is required. Install dependencies from requirements.txt.") from exc


@dataclass(frozen=True)
class RegionConfig:
    """Spatial settings for the modeling region."""

    name: str
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    crs: str = "EPSG:4326"
    projected_crs: str = "EPSG:32636"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return bounds as min_lon, min_lat, max_lon, max_lat."""

        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def center(self) -> tuple[float, float]:
        """Return center as latitude, longitude."""

        return ((self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2)

    def validate(self) -> None:
        """Validate coordinate bounds."""

        if self.min_lon >= self.max_lon:
            raise ValueError("region.min_lon must be smaller than region.max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("region.min_lat must be smaller than region.max_lat")
        if not (-180 <= self.min_lon <= 180 and -180 <= self.max_lon <= 180):
            raise ValueError("region longitudes must be valid WGS84 coordinates")
        if not (-90 <= self.min_lat <= 90 and -90 <= self.max_lat <= 90):
            raise ValueError("region latitudes must be valid WGS84 coordinates")


@dataclass(frozen=True)
class ProjectSettings:
    """Typed wrapper around config.yaml settings."""

    config_path: Path
    root_dir: Path
    project_name: str
    output_dir: Path
    region: RegionConfig
    species: list[str]
    data_sources: dict[str, Any]
    features: dict[str, Any]
    sampling: dict[str, Any]
    modeling: dict[str, Any]
    outputs: dict[str, Any]

    @property
    def raw_data_dir(self) -> Path:
        return self.root_dir / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def external_data_dir(self) -> Path:
        return self.root_dir / "data" / "external"

    @property
    def random_seed(self) -> int:
        return int(self.sampling.get("random_seed", self.modeling.get("random_seed", 42)))

    @property
    def grid_resolution(self) -> float:
        return float(self.sampling.get("grid_resolution_degrees", 0.25))

    @property
    def pseudo_absence_ratio(self) -> float:
        return float(self.sampling.get("pseudo_absence_ratio", 1.0))

    @property
    def target_column(self) -> str:
        return str(self.modeling.get("target_column", "presence"))

    @property
    def test_size(self) -> float:
        return float(self.modeling.get("test_size", 0.25))

    @property
    def model_names(self) -> list[str]:
        names = self.modeling.get("models", ["random_forest", "gradient_boosting"])
        return [str(name) for name in names]

    @property
    def gbif_max_records_per_species(self) -> int:
        gbif = self.data_sources.get("gbif", {})
        return int(gbif.get("max_records_per_species", 2000))

    @property
    def gebco_raster_path(self) -> Path:
        gebco = self.data_sources.get("gebco", {})
        configured_path = Path(str(gebco.get("local_raster_path", "data/external/gebco_black_sea.tif")))
        if configured_path.is_absolute():
            return configured_path
        return self.root_dir / configured_path

    def source_enabled(self, name: str) -> bool:
        source = self.data_sources.get(name, {})
        return bool(source.get("enabled", False))

    def feature_enabled(self, name: str) -> bool:
        return bool(self.features.get(name, False))

    def output_enabled(self, name: str) -> bool:
        return bool(self.outputs.get(name, True))

    def ensure_directories(self) -> None:
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.output_dir,
            self.root_dir / "notebooks",
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path) -> ProjectSettings:
    """Load config.yaml into a ProjectSettings object."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    root_dir = path.parent
    project = data.get("project", {})
    output_dir = Path(str(project.get("output_dir", "output")))
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    region_data = data.get("region", {})
    region = RegionConfig(
        name=str(region_data.get("name", "Black Sea")),
        min_lon=float(region_data["min_lon"]),
        max_lon=float(region_data["max_lon"]),
        min_lat=float(region_data["min_lat"]),
        max_lat=float(region_data["max_lat"]),
        crs=str(region_data.get("crs", "EPSG:4326")),
        projected_crs=str(region_data.get("projected_crs", "EPSG:32636")),
    )
    region.validate()

    species = [str(item).strip() for item in data.get("species", []) if str(item).strip()]
    if not species:
        raise ValueError("config.yaml must define at least one species")

    settings = ProjectSettings(
        config_path=path,
        root_dir=root_dir,
        project_name=str(project.get("name", root_dir.name)),
        output_dir=output_dir,
        region=region,
        species=species,
        data_sources=dict(data.get("data_sources", {})),
        features=dict(data.get("features", {})),
        sampling=dict(data.get("sampling", {})),
        modeling=dict(data.get("modeling", {})),
        outputs=dict(data.get("outputs", {})),
    )
    if settings.grid_resolution <= 0:
        raise ValueError("sampling.grid_resolution_degrees must be positive")
    if settings.pseudo_absence_ratio <= 0:
        raise ValueError("sampling.pseudo_absence_ratio must be positive")
    if not 0 < settings.test_size < 1:
        raise ValueError("modeling.test_size must be between 0 and 1")
    return settings

