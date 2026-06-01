"""GeoJSON and interactive map exports."""

from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RegionConfig


def export_prediction_geojson(predictions: pd.DataFrame, output_path: Path) -> None:
    """Write prediction points as a GeoJSON FeatureCollection."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        lon = float(getattr(row, "longitude"))
        lat = float(getattr(row, "latitude"))
        properties = {
            column: _json_value(getattr(row, column))
            for column in predictions.columns
            if column not in {"longitude", "latitude", "decimalLongitude", "decimalLatitude"}
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def export_habitat_suitability_map(
    predictions: pd.DataFrame,
    output_path: Path,
    region: RegionConfig,
    logger: logging.Logger | None = None,
) -> None:
    """Create an interactive habitat suitability map."""

    log = logger or logging.getLogger(__name__)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import folium
    except ImportError:
        log.warning("folium is not installed; writing a lightweight Leaflet fallback map")
        _export_leaflet_fallback(predictions, output_path, region)
        return

    center_lat, center_lon = region.center
    habitat_map = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    first_layer = True
    for (species, model_name), group in predictions.groupby(["species", "model"]):
        feature_group = folium.FeatureGroup(name=f"{species} - {model_name}", show=first_layer)
        first_layer = False
        for row in group.itertuples(index=False):
            suitability = float(getattr(row, "suitability"))
            popup = (
                f"<b>{escape(str(species))}</b><br>"
                f"Model: {escape(str(model_name))}<br>"
                f"Suitability: {suitability:.3f}<br>"
                f"Month: {int(getattr(row, 'month'))}"
            )
            folium.CircleMarker(
                location=[float(getattr(row, "latitude")), float(getattr(row, "longitude"))],
                radius=3,
                color=_suitability_color(suitability),
                fill=True,
                fill_color=_suitability_color(suitability),
                fill_opacity=0.75,
                weight=0,
                popup=folium.Popup(popup, max_width=260),
            ).add_to(feature_group)
        feature_group.add_to(habitat_map)

    folium.LayerControl(collapsed=False).add_to(habitat_map)
    habitat_map.save(str(output_path))


def select_best_model_predictions(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep the best evaluated model per species for a cleaner map."""

    if predictions.empty or metrics.empty:
        return predictions

    ranking = metrics.copy()
    ranking["rank_score"] = ranking["roc_auc"].fillna(ranking["f1_score"])
    ranking = ranking.sort_values(
        ["species", "rank_score", "f1_score", "precision"],
        ascending=[True, False, False, False],
    )
    best = ranking.drop_duplicates("species")[["species", "model"]]
    return predictions.merge(best, on=["species", "model"], how="inner")


def _export_leaflet_fallback(predictions: pd.DataFrame, output_path: Path, region: RegionConfig) -> None:
    records = [
        {
            "lat": float(row.latitude),
            "lon": float(row.longitude),
            "species": str(row.species),
            "model": str(row.model),
            "suitability": float(row.suitability),
            "month": int(row.month),
        }
        for row in predictions.itertuples(index=False)
    ]
    species_options = sorted({record["species"] for record in records})
    center_lat, center_lon = region.center
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Black Sea Habitat Suitability Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .control {{
      position: absolute;
      z-index: 1000;
      top: 12px;
      left: 12px;
      background: white;
      padding: 10px;
      border-radius: 6px;
      box-shadow: 0 1px 8px rgba(0,0,0,.25);
      font-family: Arial, sans-serif;
      font-size: 14px;
    }}
    select {{ max-width: 260px; }}
  </style>
</head>
<body>
  <div class="control">
    <label for="species">Species</label>
    <select id="species">
      <option value="all">All species</option>
      {''.join(f'<option value="{escape(species)}">{escape(species)}</option>' for species in species_options)}
    </select>
  </div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const records = {json.dumps(records)};
    const map = L.map('map').setView([{center_lat:.6f}, {center_lon:.6f}], 6);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 10,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    let layer = L.layerGroup().addTo(map);
    function color(value) {{
      if (value >= 0.8) return '#14532d';
      if (value >= 0.6) return '#22c55e';
      if (value >= 0.4) return '#facc15';
      if (value >= 0.2) return '#fb923c';
      return '#b91c1c';
    }}
    function render() {{
      const selected = document.getElementById('species').value;
      layer.clearLayers();
      records
        .filter(row => selected === 'all' || row.species === selected)
        .forEach(row => {{
          const marker = L.circleMarker([row.lat, row.lon], {{
            radius: 3,
            stroke: false,
            fillColor: color(row.suitability),
            fillOpacity: 0.75
          }});
          marker.bindPopup(`<b>${{row.species}}</b><br>Model: ${{row.model}}<br>Suitability: ${{row.suitability.toFixed(3)}}<br>Month: ${{row.month}}`);
          marker.addTo(layer);
        }});
    }}
    document.getElementById('species').addEventListener('change', render);
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _suitability_color(value: float) -> str:
    if value >= 0.8:
        return "#14532d"
    if value >= 0.6:
        return "#22c55e"
    if value >= 0.4:
        return "#facc15"
    if value >= 0.2:
        return "#fb923c"
    return "#b91c1c"


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value

