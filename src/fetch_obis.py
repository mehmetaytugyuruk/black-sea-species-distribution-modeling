"""OBIS placeholder module for a future optional occurrence source."""

from __future__ import annotations

import logging

import pandas as pd


def fetch_obis_occurrences(
    species_names: list[str],
    enabled: bool,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Return an empty frame unless OBIS support is explicitly implemented later.

    OBIS is intentionally not required for v1. This function exists so the
    pipeline has a clean extension point without pretending that OBIS data are
    being downloaded.
    """

    log = logger or logging.getLogger(__name__)
    if enabled:
        log.warning(
            "OBIS is enabled in config, but OBIS downloading is not implemented in v1. "
            "Continuing with other available sources for %s configured species.",
            len(species_names),
        )
    else:
        log.info("OBIS disabled in config; skipping optional OBIS source")
    return pd.DataFrame()

