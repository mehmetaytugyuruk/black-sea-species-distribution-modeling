AGENTS.md

Project

Repository name: black-sea-species-distribution-modeling

This repository implements a reproducible Python project for Black Sea marine species distribution modeling using GIS and machine learning.

The goal is to estimate habitat suitability for selected Black Sea marine species by combining open biodiversity occurrence records with geospatial and environmental features.

Target CV description

The repository should support the following project description:

Developed a geospatial machine learning pipeline to estimate habitat suitability for selected Black Sea marine species using open biodiversity occurrence records and environmental oceanographic variables. Species records from GBIF and OBIS were spatially filtered, cleaned, and combined with features such as bathymetry, sea-surface temperature, salinity, seasonality, and distance to coast. A presence/background learning setup was built using pseudo-absence sampling, and Random Forest and gradient boosting models were trained to predict species-level suitability across a regular spatial grid. Model performance was evaluated using ROC-AUC, precision, recall, F1-score, and feature-importance analysis. Final outputs included interactive Folium suitability maps, GeoJSON prediction layers, summary tables, and reproducible Python modules for data processing, modelling, and GIS visualisation.

Important implementation rule

Build a robust minimum viable version first.

The first working version must not depend on paid APIs, private credentials, Copernicus login, or large external raster downloads.

If optional environmental sources such as GEBCO, Copernicus Marine, OBIS, SST, salinity, or chlorophyll are not available, implement the project so it still runs with the available features and clearly document those sources as optional extensions.

Do not fake data source integration. If a source is optional or not implemented in v1, document it honestly as optional or future work.

Required v1 functionality

Implement the following in the first working version:

* Read project settings from config.yaml
* Fetch species occurrence records from GBIF for the configured species
* Optionally support OBIS through a clean placeholder or module, but do not make OBIS required
* Filter records to the Black Sea bounding box
* Clean invalid coordinates
* Remove duplicate observations
* Extract month or seasonality features from occurrence dates when available
* Build a regular spatial prediction grid over the Black Sea region
* Generate pseudo-absence/background points inside the configured Black Sea bounding box
* Engineer features:
    * latitude
    * longitude
    * month
    * distance to coast or approximate distance-to-region-edge if coastline data is unavailable
    * optional bathymetry if a local GEBCO raster exists
* Train at least two machine learning models:
    * Random Forest classifier
    * Gradient Boosting classifier
* Evaluate models using:
    * ROC-AUC
    * precision
    * recall
    * F1-score
* Export feature importance when the model supports it
* Predict habitat suitability over the regular spatial grid
* Export a GeoJSON prediction layer
* Export an interactive Folium habitat suitability map
* Export summary CSV files
* Provide a clear README with installation, usage, methodology, outputs, and limitations

Expected repository structure

Create or maintain this structure:

black-sea-species-distribution-modeling/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── fetch_gbif.py
│   ├── fetch_obis.py
│   ├── clean_occurrences.py
│   ├── build_black_sea_grid.py
│   ├── features.py
│   ├── pseudo_absence.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── make_maps.py
│   └── run_pipeline.py
├── output/
├── README.md
├── requirements.txt
├── config.yaml
├── AGENTS.md
└── .gitignore

Required output files

The pipeline should produce these files when it runs successfully:

output/clean_occurrences.csv
output/training_dataset.csv
output/model_metrics.csv
output/feature_importance.csv
output/prediction_grid.geojson
output/habitat_suitability_map.html
output/species_summary.csv

Command-line usage

The main pipeline must be runnable with:

python -m src.run_pipeline --config config.yaml

Also provide a sample or test mode if external API access fails:

python -m src.run_pipeline --config config.yaml --sample

Code quality requirements

* Use Python 3.10 or newer
* Use clear functions with type hints
* Use logging instead of print statements where appropriate
* Avoid hard-coded paths
* Keep the runtime reasonable for a student project
* Handle missing optional files gracefully
* Do not commit large raw data files
* Do not commit model binaries or large generated outputs
* Keep modules small and readable
* Make README commands accurate

Python libraries

Use these libraries where appropriate:

pandas
numpy
geopandas
shapely
requests
scikit-learn
folium
matplotlib
pyyaml
joblib
rasterio

rasterio should only be required for optional bathymetry support. The pipeline should still run without a local bathymetry raster.

Data rules

Use these folders:

data/raw
data/processed
data/external
output

Do not commit large downloaded datasets.

Generated files should be reproducible from scripts.

README expectations

The README must include:

* Project summary
* Methodology
* Data sources
* Installation
* Usage
* Output files
* Model evaluation
* Limitations
* Optional extensions for GEBCO, Copernicus Marine, OBIS, SST, salinity, and chlorophyll

Validation

After implementation:

1. Run the pipeline.
2. Fix syntax errors.
3. Fix runtime errors.
4. Verify that output files are created.
5. Ensure README instructions match the actual commands.
6. Summarize what was implemented and what remains optional.