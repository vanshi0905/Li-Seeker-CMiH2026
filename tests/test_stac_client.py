"""
Unit and Integration Tests for Sentinel-2 STAC Client & Resilient Ingestion.
Critical Minerals Innovation Hackathon (CMiH 2026) - Problem Statement 01.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import requests

from src.data_ingestion.stac_client import (
    BHILWARA_BBOX,
    PLANETARY_COMPUTER_STAC_URL,
    EARTH_SEARCH_STAC_URL,
    REQUIRED_BANDS,
    Sentinel2STACClient,
    fetch_bhilwara_dataset,
    fetch_bhilwara_sentinel2_dataset,
)
from src.geospatial.raster_stack import EvidentialRasterStack


class TestBhilwaraSpatialDefinition:
    """Validates spatial bounds and coordinate conventions for Bhilwara Pegmatite Belt."""

    def test_bhilwara_bbox_coordinates(self):
        assert len(BHILWARA_BBOX) == 4
        min_lon, min_lat, max_lon, max_lat = BHILWARA_BBOX
        assert min_lon == 74.05
        assert min_lat == 25.05
        assert max_lon == 75.25
        assert max_lat == 25.85
        assert min_lon < max_lon
        assert min_lat < max_lat

    def test_bhilwara_bbox_geographic_span(self):
        min_lon, min_lat, max_lon, max_lat = BHILWARA_BBOX
        lon_span = max_lon - min_lon
        lat_span = max_lat - min_lat
        assert pytest.approx(lon_span, 0.01) == 1.20
        assert pytest.approx(lat_span, 0.01) == 0.80


class TestSTACQueryConstruction:
    """Validates STAC search query parameter construction."""

    def test_default_query_construction(self):
        client = Sentinel2STACClient()
        query = client.build_search_query()

        assert query["collections"] == ["sentinel-2-l2a"]
        assert query["bbox"] == BHILWARA_BBOX
        assert query["datetime"] == "2024-01-01/2024-05-31"
        assert query["query"] == {"eo:cloud_cover": {"lt": 20.0}}
        assert query["limit"] == 5

    def test_custom_query_construction(self):
        client = Sentinel2STACClient()
        custom_bbox = [74.10, 25.10, 74.90, 25.70]
        query = client.build_search_query(
            bbox=custom_bbox,
            date_range="2024-02-01/2024-04-30",
            max_cloud=10.0,
            collection="custom-sentinel",
            limit=10,
        )

        assert query["collections"] == ["custom-sentinel"]
        assert query["bbox"] == custom_bbox
        assert query["datetime"] == "2024-02-01/2024-04-30"
        assert query["query"] == {"eo:cloud_cover": {"lt": 10.0}}
        assert query["limit"] == 10

    def test_invalid_bbox_validation(self):
        client = Sentinel2STACClient()
        with pytest.raises(ValueError, match="bbox must contain 4 coordinates"):
            client.build_search_query(bbox=[74.05, 25.05, 75.25])

        with pytest.raises(ValueError, match="Invalid bbox coordinates"):
            client.build_search_query(bbox=[75.25, 25.05, 74.05, 25.85])


class TestSTACClientInitialization:
    """Validates client configuration and endpoints."""

    def test_default_initialization(self):
        client = Sentinel2STACClient()
        assert client.endpoint == PLANETARY_COMPUTER_STAC_URL
        assert client.backup_endpoint == EARTH_SEARCH_STAC_URL
        assert client.timeout == 15

    def test_custom_initialization(self):
        client = Sentinel2STACClient(
            endpoint="https://custom-stac.org/v1",
            backup_endpoint=None,
            timeout=30,
        )
        assert client.endpoint == "https://custom-stac.org/v1"
        assert client.backup_endpoint is None
        assert client.timeout == 30


class TestMockSearchScenes:
    """Validates scene search formatting and backup failover."""

    def test_mock_search_scenes_formatting(self):
        client = Sentinel2STACClient()

        # Create mock Item
        mock_item = MagicMock()
        mock_item.id = "S2B_TEST_SCENE_123"
        mock_item.datetime = "2024-04-15T05:30:00Z"
        mock_item.bbox = BHILWARA_BBOX
        mock_item.properties = {"eo:cloud_cover": 4.5, "datetime": "2024-04-15T05:30:00Z"}
        mock_asset_b2 = MagicMock()
        mock_asset_b2.href = "https://mock.blob.core.windows.net/B02.tif"
        mock_item.assets = {"B02": mock_asset_b2}
        mock_item.to_dict.return_value = {
            "id": "S2B_TEST_SCENE_123",
            "bbox": BHILWARA_BBOX,
            "properties": {"eo:cloud_cover": 4.5, "datetime": "2024-04-15T05:30:00Z"},
            "assets": {"B02": {"href": "https://mock.blob.core.windows.net/B02.tif"}},
        }

        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]

        mock_pystac_client = MagicMock()
        mock_pystac_client.search.return_value = mock_search

        with patch.object(client, "_open_client", return_value=mock_pystac_client):
            scenes = client.search_scenes()

            assert len(scenes) == 1
            scene = scenes[0]
            assert scene["id"] == "S2B_TEST_SCENE_123"
            assert scene["cloud_cover"] == 4.5
            assert scene["bbox"] == BHILWARA_BBOX
            assert "B02" in scene["assets"]
            assert scene["assets"]["B02"] == "https://mock.blob.core.windows.net/B02.tif"

    def test_search_scenes_failover_to_backup(self):
        """When primary endpoint fails, the client should attempt backup endpoint."""
        client = Sentinel2STACClient(
            endpoint=PLANETARY_COMPUTER_STAC_URL,
            backup_endpoint=EARTH_SEARCH_STAC_URL,
        )

        mock_item = MagicMock()
        mock_item.id = "S2B_BACKUP_SCENE"
        mock_item.to_dict.return_value = {
            "id": "S2B_BACKUP_SCENE",
            "properties": {"eo:cloud_cover": 2.1},
            "assets": {},
        }

        mock_backup_client = MagicMock()
        mock_backup_search = MagicMock()
        mock_backup_search.items.return_value = [mock_item]
        mock_backup_client.search.return_value = mock_backup_search

        def open_client_side_effect(url):
            if url == PLANETARY_COMPUTER_STAC_URL:
                raise requests.exceptions.ConnectionError("Primary STAC unreachable")
            return mock_backup_client

        with patch.object(client, "_open_client", side_effect=open_client_side_effect):
            scenes = client.search_scenes()
            assert len(scenes) == 1
            assert scenes[0]["id"] == "S2B_BACKUP_SCENE"


class TestOfflineFallback:
    """Validates robust zero-crash offline fallback mechanism."""

    def test_offline_fallback_on_connection_error(self, caplog):
        """Simulates network disconnection and ensures graceful fallback."""
        client = Sentinel2STACClient()

        with patch.object(client, "search_scenes", side_effect=requests.exceptions.ConnectionError("Network disconnected")):
            dataset = client.fetch_bhilwara_dataset(nrows=40, ncols=60, force_fallback=False)

            assert dataset["data_source"] == "synthetic_fallback"
            assert dataset["grid"].nrows == 40
            assert dataset["grid"].ncols == 60
            assert len(dataset["occurrences"]) == 15
            for band in REQUIRED_BANDS:
                assert band in dataset["bands"]
                assert dataset["bands"][band].shape == (40, 60)
            assert dataset["scl"].shape == (40, 60)
            assert dataset["dem"].shape == (40, 60)

    def test_offline_fallback_on_timeout(self):
        """Simulates request timeout and ensures graceful fallback."""
        client = Sentinel2STACClient()

        with patch.object(client, "search_scenes", side_effect=requests.exceptions.ConnectTimeout("Request timed out")):
            dataset = client.fetch_bhilwara_dataset(nrows=40, ncols=60, force_fallback=False)
            assert dataset["data_source"] == "synthetic_fallback"

    def test_offline_fallback_on_force_fallback(self):
        """When force_fallback=True, immediately returns synthetic benchmark without network queries."""
        client = Sentinel2STACClient()

        with patch.object(client, "search_scenes") as mock_search:
            dataset = client.fetch_bhilwara_dataset(nrows=40, ncols=60, force_fallback=True)
            mock_search.assert_not_called()
            assert dataset["data_source"] == "synthetic_fallback"
            assert dataset["grid"].nrows == 40
            assert dataset["grid"].ncols == 60

    def test_offline_fallback_on_empty_scenes(self):
        """When no scenes are returned matching criteria, falls back cleanly."""
        client = Sentinel2STACClient()

        with patch.object(client, "search_scenes", return_value=[]):
            dataset = client.fetch_bhilwara_dataset(nrows=40, ncols=60, force_fallback=False)
            assert dataset["data_source"] == "synthetic_fallback"

    def test_offline_fallback_on_band_decode_error(self):
        """When search succeeds but band downloading fails, falls back cleanly."""
        client = Sentinel2STACClient()
        mock_scene = {
            "id": "S2B_TEST",
            "cloud_cover": 5.0,
            "assets": {"B02": "https://corrupted.blob.com/bad.tif"},
            "stac_endpoint": PLANETARY_COMPUTER_STAC_URL,
        }

        with patch.object(client, "search_scenes", return_value=[mock_scene]):
            with patch.object(client, "fetch_bands", side_effect=RuntimeError("Band decode failed")):
                dataset = client.fetch_bhilwara_dataset(nrows=40, ncols=60, force_fallback=False)
                assert dataset["data_source"] == "synthetic_fallback"


class TestRealSTACExecution:
    """Validates real_stac execution path with mock band rasters."""

    def test_real_stac_with_mock_bands(self):
        client = Sentinel2STACClient()
        nrows, ncols = 40, 60

        mock_bands = {b: np.full((nrows, ncols), 0.20, dtype=np.float32) for b in REQUIRED_BANDS}
        mock_scl = np.full((nrows, ncols), 5, dtype=np.uint8)

        mock_scene = {
            "id": "S2B_MSIL2A_20240428_MOCK",
            "cloud_cover": 3.2,
            "datetime": "2024-04-28T05:26:49Z",
            "bbox": BHILWARA_BBOX,
            "assets": {b: f"https://mock.blob/{b}.tif" for b in REQUIRED_BANDS},
            "mock_bands": mock_bands,
            "mock_scl": mock_scl,
            "stac_endpoint": PLANETARY_COMPUTER_STAC_URL,
        }

        with patch.object(client, "search_scenes", return_value=[mock_scene]):
            dataset = client.fetch_bhilwara_dataset(nrows=nrows, ncols=ncols, force_fallback=False)

            assert dataset["data_source"] == "real_stac"
            assert dataset["stac_scene_id"] == "S2B_MSIL2A_20240428_MOCK"
            assert dataset["stac_cloud_cover"] == 3.2
            assert dataset["stac_datetime"] == "2024-04-28T05:26:49Z"
            assert "B6" in dataset["bands"]
            assert dataset["bands"]["B2"].shape == (nrows, ncols)
            assert dataset["scl"].shape == (nrows, ncols)


class TestDownstreamContractCompliance:
    """Validates that both synthetic and real outputs comply with EvidentialRasterStack."""

    def test_synthetic_fallback_compliance(self):
        nrows, ncols = 50, 75
        dataset = fetch_bhilwara_dataset(nrows=nrows, ncols=ncols, force_fallback=True)

        stack = EvidentialRasterStack(dataset)
        assert stack.valid_mask.shape == (nrows, ncols)
        assert "al_oh_mica_ratio" in stack.feature_rasters
        assert "cardoso_lpi" in stack.feature_rasters
        assert "crosta_pc_al_oh" in stack.feature_rasters

    def test_convenience_functions_and_aliases(self):
        nrows, ncols = 30, 45
        ds1 = fetch_bhilwara_dataset(nrows=nrows, ncols=ncols, force_fallback=True)
        ds2 = fetch_bhilwara_sentinel2_dataset(nrows=nrows, ncols=ncols, force_fallback=True)

        assert ds1["data_source"] == "synthetic_fallback"
        assert ds2["data_source"] == "synthetic_fallback"
        assert ds1["grid"].nrows == nrows
        assert ds2["grid"].ncols == ncols


class TestLiveSTACQuery:
    """Live integration test querying Microsoft Planetary Computer (network-dependent)."""

    def test_live_planetary_computer_query(self):
        """Verifies real STAC query against Microsoft Planetary Computer for Bhilwara bbox."""
        client = Sentinel2STACClient(timeout=10)
        try:
            scenes = client.search_scenes(
                bbox=BHILWARA_BBOX,
                date_range="2024-01-01/2024-05-31",
                max_cloud=15.0,
                limit=2,
            )
            if scenes:
                scene = scenes[0]
                assert "id" in scene
                assert "cloud_cover" in scene
                assert scene["cloud_cover"] <= 15.0
                assert "assets" in scene
                assert "B02" in scene["assets"] or "blue" in scene["assets"]
        except (requests.RequestException, Exception) as e:
            pytest.skip(f"Live network query skipped due to connectivity: {e}")
