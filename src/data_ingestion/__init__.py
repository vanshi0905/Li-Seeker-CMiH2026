"""
Data Ingestion Module for Li-Seeker.
Provides Sentinel-2 STAC querying and robust offline fallback datasets for Bhilwara Pegmatite Belt.
"""

from .stac_client import (
    Sentinel2STACClient,
    fetch_bhilwara_dataset,
    fetch_bhilwara_sentinel2_dataset,
    BHILWARA_BBOX,
    PLANETARY_COMPUTER_STAC_URL,
    EARTH_SEARCH_STAC_URL,
)

__all__ = [
    "Sentinel2STACClient",
    "fetch_bhilwara_dataset",
    "fetch_bhilwara_sentinel2_dataset",
    "BHILWARA_BBOX",
    "PLANETARY_COMPUTER_STAC_URL",
    "EARTH_SEARCH_STAC_URL",
]
