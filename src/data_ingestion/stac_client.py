"""
Sentinel-2 STAC Client & Resilient Data Ingestion Engine for Li-Seeker.
Critical Minerals Innovation Hackathon (CMiH 2026) - Problem Statement 01.

Provides automated querying of Sentinel-2 L2A BOA surface reflectance data
targeting the Bhilwara Pegmatite Belt (Rajasthan, India) from Microsoft
Planetary Computer and Element 84 Earth Search STAC endpoints, with guaranteed
zero-crash offline fallback to the synthetic benchmark generator.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from scipy.ndimage import zoom

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import pystac_client
except ImportError:
    pystac_client = None

try:
    import planetary_computer as pc
except ImportError:
    pc = None

from ..geospatial.synthetic_generator import (
    build_bhilwara_benchmark,
    build_katghora_benchmark,
    build_district_benchmark,
)

logger = logging.getLogger(__name__)

# Bhilwara District Pegmatite Belt Bounding Box [min_lon, min_lat, max_lon, max_lat] (EPSG:4326)
# Lat: 25.05° N to 25.85° N (~89 km span)
# Lon: 74.05° E to 75.25° E (~120 km span)
BHILWARA_BBOX: list[float] = [74.05, 25.05, 75.25, 25.85]

# Katghora Lithium-REE Block & Korba District Bounding Box (EPSG:4326)
# Lat: 22.25° N to 22.75° N (~55 km span)
# Lon: 82.25° E to 82.75° E (~51 km span)
KATGHORA_BBOX: list[float] = [82.25, 22.25, 82.75, 22.75]


# Default STAC Catalog Endpoints
PLANETARY_COMPUTER_STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
EARTH_SEARCH_STAC_URL: str = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION: str = "sentinel-2-l2a"
DEFAULT_DATETIME: str = "2024-01-01/2024-05-31"
DEFAULT_MAX_CLOUD: float = 20.0

# Required optical bands for Li-Seeker exploration & REE spectral indices
REQUIRED_BANDS: list[str] = ["B2", "B3", "B4", "B6", "B8", "B8A", "B11", "B12"]

# Asset key aliases across STAC providers
PC_BAND_MAP: dict[str, str] = {
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B6": "B06",
    "B8": "B08",
    "B8A": "B8A",
    "B11": "B11",
    "B12": "B12",
    "SCL": "SCL",
}

EARTH_SEARCH_BAND_MAP: dict[str, str] = {
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B6": "rededge2",
    "B8": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}


def _build_fallback_dataset(nrows: int, ncols: int, seed: int, district: str = "bhilwara") -> dict[str, Any]:
    """
    Constructs a full benchmark dataset via synthetic_generator with data_source tag.
    Ensures Band 6 is present (interpolated between B4 and B8 if not already provided).
    """
    dataset = build_district_benchmark(district=district, nrows=nrows, ncols=ncols, seed=seed)
    dataset["data_source"] = "synthetic_fallback"
    if "B6" not in dataset["bands"]:
        # Synthesize Band 6 (RedEdge 740nm) between B4 (Red) and B8 (NIR)
        dataset["bands"]["B6"] = (
            0.5 * (dataset["bands"]["B4"] + dataset["bands"]["B8"])
        ).astype(np.float32)
    return dataset


class Sentinel2STACClient:
    """
    Client for searching and retrieving Sentinel-2 L2A surface reflectance data
    from STAC APIs with automatic endpoint failover and offline fallback.
    """

    def __init__(
        self,
        endpoint: str = PLANETARY_COMPUTER_STAC_URL,
        backup_endpoint: Optional[str] = EARTH_SEARCH_STAC_URL,
        timeout: int = 15,
    ) -> None:
        self.endpoint = endpoint
        self.backup_endpoint = backup_endpoint
        self.timeout = timeout

    def build_search_query(
        self,
        bbox: list[float] = BHILWARA_BBOX,
        date_range: str = DEFAULT_DATETIME,
        max_cloud: float = DEFAULT_MAX_CLOUD,
        collection: str = DEFAULT_COLLECTION,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Constructs standardized search query parameters for the STAC API."""
        if len(bbox) != 4:
            raise ValueError(f"bbox must contain 4 coordinates [min_lon, min_lat, max_lon, max_lat], got {bbox}")
        min_lon, min_lat, max_lon, max_lat = bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(f"Invalid bbox coordinates: {bbox}")

        return {
            "collections": [collection],
            "bbox": [float(c) for c in bbox],
            "datetime": date_range,
            "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
            "limit": limit,
        }

    def _open_client(self, url: str) -> Any:
        """Opens a pystac_client Client instance with optional provider modifier."""
        if pystac_client is None:
            raise ImportError("pystac-client is not installed. Please install pystac-client>=0.8.0")

        modifier = None
        if "planetarycomputer" in url.lower() and pc is not None:
            modifier = pc.sign_inplace

        return pystac_client.Client.open(url, modifier=modifier, timeout=self.timeout)

    def search_scenes(
        self,
        bbox: list[float] = BHILWARA_BBOX,
        date_range: str = DEFAULT_DATETIME,
        max_cloud: float = DEFAULT_MAX_CLOUD,
        limit: int = 5,
        collection: str = DEFAULT_COLLECTION,
    ) -> list[dict[str, Any]]:
        """
        Queries the STAC API for Sentinel-2 L2A scenes covering the bounding box.
        Tries primary endpoint first; falls back to backup endpoint on failure.
        """
        query_params = self.build_search_query(
            bbox=bbox,
            date_range=date_range,
            max_cloud=max_cloud,
            collection=collection,
            limit=limit,
        )

        endpoints_to_try = [self.endpoint]
        if self.backup_endpoint and self.backup_endpoint != self.endpoint:
            endpoints_to_try.append(self.backup_endpoint)

        last_error: Optional[Exception] = None

        for endpoint_url in endpoints_to_try:
            try:
                client = self._open_client(endpoint_url)
                search = client.search(
                    collections=query_params["collections"],
                    bbox=query_params["bbox"],
                    datetime=query_params["datetime"],
                    query=query_params["query"],
                    max_items=limit,
                )
                items = list(search.items())
                if items:
                    logger.info(f"Successfully retrieved {len(items)} scenes from STAC endpoint: {endpoint_url}")
                    return [self._format_scene_dict(item, endpoint_url) for item in items]
                else:
                    logger.warning(f"No scenes matched query on {endpoint_url}")
            except Exception as exc:
                last_error = exc
                logger.warning(f"STAC search failed on {endpoint_url}: {type(exc).__name__} ({exc})")

        if last_error is not None:
            raise last_error

        return []

    def _format_scene_dict(self, item: Any, endpoint_url: str) -> dict[str, Any]:
        """Standardizes a pystac Item or raw dictionary into a Li-Seeker scene dict."""
        if hasattr(item, "to_dict"):
            item_dict = item.to_dict()
        elif isinstance(item, dict):
            item_dict = item
        else:
            item_dict = {}

        properties = item_dict.get("properties", {})
        dt = (
            properties.get("datetime")
            or getattr(item, "datetime", None)
            or ""
        )
        if hasattr(dt, "isoformat"):
            dt = dt.isoformat()

        cloud = properties.get("eo:cloud_cover", 0.0)
        item_id = item_dict.get("id") or getattr(item, "id", "unknown")
        bbox = item_dict.get("bbox") or getattr(item, "bbox", [])

        # Build asset dictionary
        assets = {}
        raw_assets = getattr(item, "assets", {}) or item_dict.get("assets", {})
        if isinstance(raw_assets, dict):
            for k, v in raw_assets.items():
                if hasattr(v, "href"):
                    assets[k] = v.href
                elif isinstance(v, dict) and "href" in v:
                    assets[k] = v["href"]
                elif isinstance(v, str):
                    assets[k] = v

        return {
            "id": item_id,
            "datetime": str(dt),
            "cloud_cover": float(cloud) if cloud is not None else 0.0,
            "bbox": bbox,
            "assets": assets,
            "stac_endpoint": endpoint_url,
            "_raw_item": item,
        }

    def fetch_scene_metadata(self, scene_id: str, collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        """Retrieves asset metadata and properties for a specific Sentinel-2 scene ID."""
        client = self._open_client(self.endpoint)
        search = client.search(collections=[collection], ids=[scene_id], max_items=1)
        items = list(search.items())
        if not items and self.backup_endpoint and self.backup_endpoint != self.endpoint:
            client_backup = self._open_client(self.backup_endpoint)
            search = client_backup.search(collections=[collection], ids=[scene_id], max_items=1)
            items = list(search.items())

        if not items:
            raise KeyError(f"Scene ID {scene_id} not found on STAC endpoints.")
        return self._format_scene_dict(items[0], self.endpoint)

    def decode_cog_overview(
        self,
        href_or_data: Union[str, bytes, np.ndarray],
        nrows: int,
        ncols: int,
        timeout: float = 10.0,
        is_scl: bool = False,
    ) -> np.ndarray:
        """
        Reads a band overview from a Cloud-Optimized GeoTIFF URL or in-memory array.
        Uses HTTP Range requests to read only the small overview header/segment (<1.5 MB),
        resampling to (nrows, ncols) and scaling to reflectance [0.0, 1.0].
        """
        # If already a numpy array (e.g. from mock testing)
        if isinstance(href_or_data, np.ndarray):
            if href_or_data.shape != (nrows, ncols):
                factors = (nrows / href_or_data.shape[0], ncols / href_or_data.shape[1])
                order = 0 if is_scl else 1
                arr = zoom(href_or_data, factors, order=order)
            else:
                arr = href_or_data.copy()

            if is_scl:
                return np.round(arr).astype(np.uint8)
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 10000.0
            return np.clip(arr, 0.0, 1.0)

        # Retrieve bytes
        if isinstance(href_or_data, str) and (
            href_or_data.startswith("http://") or href_or_data.startswith("https://")
        ):
            # Fetch the first 1.5MB containing COG overview IFD and data
            headers = {"Range": "bytes=0-1572863"}
            resp = requests.get(href_or_data, headers=headers, timeout=timeout)
            resp.raise_for_status()
            content = resp.content
        elif isinstance(href_or_data, str) and os.path.exists(href_or_data):
            with open(href_or_data, "rb") as f:
                content = f.read(1572864)
        elif isinstance(href_or_data, (bytes, bytearray)):
            content = bytes(href_or_data)
        else:
            raise ValueError(f"Invalid input type for decode_cog_overview: {type(href_or_data)}")

        if tifffile is None:
            raise ImportError("tifffile package is required for COG decoding.")

        with tifffile.TiffFile(io.BytesIO(content)) as tf:
            pages = tf.pages
            if not pages:
                raise ValueError("No TIFF pages found in COG header.")
            # Select overview page closest to target dimensions
            best_page = pages[-1]
            for p in reversed(pages):
                if p.shape[0] >= nrows and p.shape[1] >= ncols:
                    best_page = p
                    break

            raw_arr = best_page.asarray()
            factors = (nrows / raw_arr.shape[0], ncols / raw_arr.shape[1])
            order = 0 if is_scl else 1
            arr = zoom(raw_arr, factors, order=order)

            if is_scl:
                return np.round(arr).astype(np.uint8)

            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 10000.0
            return np.clip(arr, 0.0, 1.0)

    def fetch_bands(
        self,
        scene: dict[str, Any],
        nrows: int = 160,
        ncols: int = 240,
        timeout: float = 10.0,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Resolves asset links from the scene and decodes real 2D reflectance band arrays.
        Returns (bands_dict, scl_array).
        """
        assets = scene.get("assets", {})
        endpoint = scene.get("stac_endpoint", self.endpoint).lower()
        band_map = PC_BAND_MAP if "planetarycomputer" in endpoint else EARTH_SEARCH_BAND_MAP

        # Allow explicit mock bands injected into scene dict for fast testing
        if "mock_bands" in scene and isinstance(scene["mock_bands"], dict):
            mock_b = scene["mock_bands"]
            bands_res = {}
            for b in REQUIRED_BANDS:
                if b in mock_b:
                    bands_res[b] = self.decode_cog_overview(mock_b[b], nrows, ncols, is_scl=False)
                else:
                    bands_res[b] = np.full((nrows, ncols), 0.25, dtype=np.float32)
            scl_res = self.decode_cog_overview(scene.get("mock_scl", np.full((nrows, ncols), 4, dtype=np.uint8)), nrows, ncols, is_scl=True)
            return bands_res, scl_res

        bands_out: dict[str, np.ndarray] = {}

        for band_key in REQUIRED_BANDS:
            target_asset = band_map.get(band_key, band_key)
            if target_asset not in assets:
                # Try fallback names
                alt_asset = (EARTH_SEARCH_BAND_MAP if "planetarycomputer" in endpoint else PC_BAND_MAP).get(band_key)
                if alt_asset and alt_asset in assets:
                    target_asset = alt_asset
                elif band_key in assets:
                    target_asset = band_key
                else:
                    raise KeyError(f"Required band '{band_key}' (asset '{target_asset}') not found in scene assets.")

            href = assets[target_asset]
            bands_out[band_key] = self.decode_cog_overview(href, nrows, ncols, timeout=timeout, is_scl=False)

        # SCL decoding
        scl_asset_name = band_map.get("SCL", "SCL")
        if scl_asset_name in assets:
            scl_out = self.decode_cog_overview(assets[scl_asset_name], nrows, ncols, timeout=timeout, is_scl=True)
        else:
            # Approximate SCL from NDVI if SCL asset is absent
            ndvi_approx = (bands_out["B8"] - bands_out["B4"]) / np.clip(bands_out["B8"] + bands_out["B4"], 1e-6, None)
            scl_out = np.where(ndvi_approx > 0.30, 4, 5).astype(np.uint8)

        return bands_out, scl_out

    def fetch_bhilwara_dataset(
        self,
        nrows: int = 160,
        ncols: int = 240,
        force_fallback: bool = False,
        seed: int = 42,
        max_cloud: float = DEFAULT_MAX_CLOUD,
        date_range: str = DEFAULT_DATETIME,
        download_bands: bool = True,
    ) -> dict[str, Any]:
        """
        Fetches Sentinel-2 imagery for Bhilwara Pegmatite Belt and combines it
        with multi-modal geological, geophysical, and geochemical evidential layers.

        Guaranteed NEVER to raise unhandled network exceptions; on any failure or
        when force_fallback=True, gracefully falls back to synthetic_generator.
        """
        if force_fallback:
            logger.info("[STAC] force_fallback=True requested. Seamlessly using synthetic benchmark.")
            return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed)

        try:
            scenes = self.search_scenes(
                bbox=BHILWARA_BBOX,
                date_range=date_range,
                max_cloud=max_cloud,
                limit=5,
            )
            if not scenes:
                logger.warning("[STAC] No scenes found matching criteria. Falling back to synthetic benchmark.")
                return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed)

            # Select scene with minimal cloud cover
            best_scene = sorted(scenes, key=lambda s: s.get("cloud_cover", 100.0))[0]

            # Assemble base district layers (DEM, aeromag, geochemistry, structures, occurrences)
            dataset = build_bhilwara_benchmark(nrows=nrows, ncols=ncols, seed=seed)

            if download_bands:
                try:
                    bands, scl = self.fetch_bands(best_scene, nrows=nrows, ncols=ncols, timeout=float(self.timeout))
                    dataset["bands"] = bands
                    dataset["scl"] = scl
                except Exception as band_err:
                    logger.warning(
                        f"[STAC] Unable to decode live band rasters ({type(band_err).__name__}: {band_err}). "
                        "Falling back to synthetic benchmark."
                    )
                    return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed)

            dataset["data_source"] = "real_stac"
            dataset["stac_scene_id"] = best_scene.get("id", "unknown")
            dataset["stac_cloud_cover"] = best_scene.get("cloud_cover", 0.0)
            dataset["stac_datetime"] = best_scene.get("datetime", "")
            dataset["stac_endpoint"] = best_scene.get("stac_endpoint", self.endpoint)

            # Ensure Band 6 is present
            if "B6" not in dataset["bands"]:
                dataset["bands"]["B6"] = (
                    0.5 * (dataset["bands"]["B4"] + dataset["bands"]["B8"])
                ).astype(np.float32)

            return dataset

        except (requests.RequestException, TimeoutError, ConnectionError, ImportError, Exception) as exc:
            logger.warning(
                f"[STAC] Exception encountered during STAC query/ingestion ({type(exc).__name__}: {exc}). "
                "Seamlessly falling back to synthetic benchmark."
            )
            return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed, district="bhilwara")

    def fetch_katghora_dataset(
        self,
        nrows: int = 160,
        ncols: int = 240,
        force_fallback: bool = False,
        seed: int = 42,
        max_cloud: float = DEFAULT_MAX_CLOUD,
        date_range: str = "2024-03-01/2024-05-31",
        download_bands: bool = True,
    ) -> dict[str, Any]:
        """
        Fetches Sentinel-2 imagery for Katghora Lithium-REE Block & Korba District,
        Chhattisgarh, and combines it with multi-modal evidential layers.
        """
        if force_fallback:
            logger.info("[STAC] force_fallback=True requested. Using Katghora synthetic benchmark.")
            return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed, district="katghora")

        try:
            scenes = self.search_scenes(
                bbox=KATGHORA_BBOX,
                date_range=date_range,
                max_cloud=max_cloud,
                limit=5,
            )
            if not scenes:
                logger.warning("[STAC] No scenes found for Katghora. Falling back to synthetic benchmark.")
                return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed, district="katghora")

            best_scene = sorted(scenes, key=lambda s: s.get("cloud_cover", 100.0))[0]
            dataset = build_katghora_benchmark(nrows=nrows, ncols=ncols, seed=seed)

            if download_bands:
                try:
                    bands, scl = self.fetch_bands(best_scene, nrows=nrows, ncols=ncols, timeout=float(self.timeout))
                    dataset["bands"] = bands
                    dataset["scl"] = scl
                except Exception as band_err:
                    logger.warning(f"[STAC] Katghora band decoding error ({band_err}). Using synthetic benchmark.")
                    return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed, district="katghora")

            dataset["data_source"] = "real_stac"
            dataset["stac_scene_id"] = best_scene.get("id", "unknown")
            dataset["stac_cloud_cover"] = best_scene.get("cloud_cover", 0.0)
            dataset["stac_datetime"] = best_scene.get("datetime", "")
            dataset["stac_endpoint"] = best_scene.get("stac_endpoint", self.endpoint)

            if "B6" not in dataset["bands"]:
                dataset["bands"]["B6"] = (
                    0.5 * (dataset["bands"]["B4"] + dataset["bands"]["B8"])
                ).astype(np.float32)

            return dataset
        except Exception as exc:
            logger.warning(f"[STAC] Katghora query error ({exc}). Using synthetic benchmark.")
            return _build_fallback_dataset(nrows=nrows, ncols=ncols, seed=seed, district="katghora")

    def fetch_district_dataset(
        self,
        district: str = "katghora",
        nrows: int = 160,
        ncols: int = 240,
        force_fallback: bool = False,
        seed: int = 42,
        max_cloud: float = DEFAULT_MAX_CLOUD,
        date_range: Optional[str] = None,
        download_bands: bool = True,
    ) -> dict[str, Any]:
        """Unified dispatcher to fetch dataset by district name."""
        d_clean = district.lower().strip()
        if "katghora" in d_clean or "korba" in d_clean or "chhattisgarh" in d_clean:
            dr = date_range or "2024-03-01/2024-05-31"
            return self.fetch_katghora_dataset(
                nrows=nrows, ncols=ncols, force_fallback=force_fallback, seed=seed,
                max_cloud=max_cloud, date_range=dr, download_bands=download_bands,
            )
        else:
            dr = date_range or DEFAULT_DATETIME
            return self.fetch_bhilwara_dataset(
                nrows=nrows, ncols=ncols, force_fallback=force_fallback, seed=seed,
                max_cloud=max_cloud, date_range=dr, download_bands=download_bands,
            )


def fetch_bhilwara_dataset(
    nrows: int = 160,
    ncols: int = 240,
    force_fallback: bool = False,
    seed: int = 42,
    endpoint: str = PLANETARY_COMPUTER_STAC_URL,
    backup_endpoint: Optional[str] = EARTH_SEARCH_STAC_URL,
    timeout: int = 15,
    download_bands: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to fetch Bhilwara dataset with automated STAC querying
    and zero-crash offline fallback.
    """
    client = Sentinel2STACClient(
        endpoint=endpoint,
        backup_endpoint=backup_endpoint,
        timeout=timeout,
    )
    return client.fetch_bhilwara_dataset(
        nrows=nrows,
        ncols=ncols,
        force_fallback=force_fallback,
        seed=seed,
        download_bands=download_bands,
    )


def fetch_katghora_dataset(
    nrows: int = 160,
    ncols: int = 240,
    force_fallback: bool = False,
    seed: int = 42,
    endpoint: str = PLANETARY_COMPUTER_STAC_URL,
    backup_endpoint: Optional[str] = EARTH_SEARCH_STAC_URL,
    timeout: int = 15,
    download_bands: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to fetch Katghora, Korba district dataset with automated STAC
    querying and zero-crash offline fallback.
    """
    client = Sentinel2STACClient(
        endpoint=endpoint,
        backup_endpoint=backup_endpoint,
        timeout=timeout,
    )
    return client.fetch_katghora_dataset(
        nrows=nrows,
        ncols=ncols,
        force_fallback=force_fallback,
        seed=seed,
        download_bands=download_bands,
    )


def fetch_district_dataset(
    district: str = "katghora",
    nrows: int = 160,
    ncols: int = 240,
    force_fallback: bool = False,
    seed: int = 42,
    endpoint: str = PLANETARY_COMPUTER_STAC_URL,
    backup_endpoint: Optional[str] = EARTH_SEARCH_STAC_URL,
    timeout: int = 15,
    download_bands: bool = True,
) -> dict[str, Any]:
    """
    Convenience dispatcher function to fetch any district dataset by name.
    """
    client = Sentinel2STACClient(
        endpoint=endpoint,
        backup_endpoint=backup_endpoint,
        timeout=timeout,
    )
    return client.fetch_district_dataset(
        district=district,
        nrows=nrows,
        ncols=ncols,
        force_fallback=force_fallback,
        seed=seed,
        download_bands=download_bands,
    )


# Alias for backwards and cross-module compatibility
fetch_bhilwara_sentinel2_dataset = fetch_bhilwara_dataset

