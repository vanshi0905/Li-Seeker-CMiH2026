"""
Unit tests for Katghora Block (Korba District, Chhattisgarh) dataset.
"""
import json, os, numpy as np, pytest
from src.geospatial.synthetic_generator import build_katghora_benchmark, build_district_benchmark
from src.data_ingestion.stac_client import fetch_district_dataset, fetch_katghora_dataset, KATGHORA_BBOX
from src.geospatial.raster_stack import EvidentialRasterStack

def test_katghora_ground_truth_integrity():
    gt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ground_truth', 'katghora_pegmatites.json')
    assert os.path.exists(gt_path)
    with open(gt_path, 'r') as f:
        data = json.load(f)
    assert data['district'] == 'Korba'
    assert len(data['occurrences']) == 10
    main_block = next((occ for occ in data['occurrences'] if occ['id'] == 'KORB-01'), None)
    assert main_block is not None
    assert pytest.approx(main_block['latitude'], 0.01) == 22.5025

def test_katghora_benchmark_generation():
    ds = build_katghora_benchmark(nrows=30, ncols=45, seed=42)
    assert ds['district'] == 'Katghora'
    assert ds['grid'].nrows == 30
    assert len(ds['occurrences']) == 10
    for b in ['B2', 'B3', 'B4', 'B6', 'B8', 'B8A', 'B11', 'B12']:
        assert b in ds['bands']
        assert not np.isnan(ds['bands'][b]).any()

def test_katghora_raster_stack_27_layers():
    ds = build_katghora_benchmark(nrows=30, ncols=45, seed=42)
    stack = EvidentialRasterStack(ds, ndvi_threshold=0.35)
    assert len(stack.feature_names) == 27
    X, mask, feats = stack.get_feature_matrix(apply_mask=True)
    assert X.shape[1] == 27
    for expected_layer in [
        "radiometric_k_pct", "radiometric_u_th_ratio", "radiometric_ternary_index",
        "bouguer_gravity_mgal", "gravity_gradient_fvd", "aster_quartz_index_qi",
        "sar_cband_vh_db", "sar_roughness_ratio"
    ]:
        assert expected_layer in stack.feature_names
        assert not np.isnan(stack.feature_rasters[expected_layer]).any()


def test_katghora_target_extractor_naming():
    from src.evaluation.target_extractor import extract_prospective_targets
    from src.geospatial.raster_ops import GeoGrid

    grid = GeoGrid(min_lat=22.25, max_lat=22.75, min_lon=82.25, max_lon=82.75, nrows=20, ncols=30)
    prospectivity = np.zeros((20, 30), dtype=float)
    prospectivity[5:10, 5:10] = 0.95

    targets, geojson = extract_prospective_targets(prospectivity, grid, threshold=0.70, district_name="katghora")
    assert len(targets) >= 1
    assert geojson["type"] == "FeatureCollection"
    assert geojson["name"] == "Katghora_Lithium_Exploration_Targets"


def test_katghora_export_deliverables_integration(tmp_path):
    from src.evaluation.metrics import compute_prediction_area_plot
    from src.evaluation.target_extractor import extract_prospective_targets
    from src.models.feature_importance import compute_feature_rankings
    from src.export.exporter import export_prospectivity_deliverables

    ds = build_katghora_benchmark(nrows=20, ncols=30, seed=42)
    grid = ds['grid']
    occurrences = ds['occurrences']
    prob_map = np.random.uniform(0.2, 0.8, (20, 30)).astype(np.float32)

    pa_metrics = compute_prediction_area_plot(prob_map, occurrences, grid, n_steps=20)
    targets, geojson = extract_prospective_targets(prob_map, grid, threshold=0.5, district_name="katghora")
    rankings_df = compute_feature_rankings(np.ones(3)/3, ["ngcm_k_rb_ratio", "crosta_pc_al_oh", "elevation_dem"])

    # Test output directory confinement: subfolder inside project output
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_katghora_sub')
    paths = export_prospectivity_deliverables(
        prob_map, grid, pa_metrics, geojson, rankings_df, out_dir, district_prefix="katghora"
    )
    assert os.path.exists(paths["geotiff"])
    assert "katghora" in paths["geotiff"]
    assert os.path.exists(paths["geojson"])
    assert "katghora" in paths["geojson"]
    assert os.path.exists(paths["pa_plot_png"])


def test_cmih2_folder_integrity():
    cmih2_dir = os.path.join(os.path.dirname(__file__), '..', 'CMIH 2')
    if not os.path.exists(cmih2_dir):
        pytest.skip("CMIH 2 folder omitted from lightweight GitHub distribution")

    # Mandatory files check
    required_files = [
        "katghora_pegmatites.json",
        "katghora_occurrences.csv",
        "katghora_evidential_stack.npz",
        "katghora_lithium_prospectivity.tif",
        "katghora_drill_targets.geojson",
        "feature_rankings.csv",
        "prospectivity_metrics.json",
        "prediction_area_plot.png",
        "katghora_dataset_metadata.json",
        "fetch_katghora_stac_data.py",
        "IMPLEMENTATION_PLAN_KATGHORA_DATASET.md",
        "TEAM_MEETING_GUIDE.md",
        "CMIH_PPT_SUBMISSION_PLAN.md",
        "KATGHORA_DATASET_CATALOG.md",
    ]
    for rf in required_files:
        p = os.path.join(cmih2_dir, rf)
        assert os.path.exists(p), f"Missing required file in CMIH 2: {rf}"
        assert os.path.getsize(p) > 0, f"File in CMIH 2 is empty: {rf}"

    # Verify GeoJSON naming in CMIH 2
    with open(os.path.join(cmih2_dir, "katghora_drill_targets.geojson"), "r") as f:
        gj = json.load(f)
    assert gj["name"] == "Katghora_Lithium_Exploration_Targets"

    # Verify NPZ contents
    npz = np.load(os.path.join(cmih2_dir, "katghora_evidential_stack.npz"))
    assert npz["min_lat"] == 22.25
    assert npz["max_lat"] == 22.75
    assert npz["min_lon"] == 82.25
    assert npz["max_lon"] == 82.75
    assert "ngcm_k_rb_ratio" in npz
    assert "crosta_pc_al_oh" in npz
    assert "elevation_dem" in npz
    assert "radiometric_ternary_index" in npz
    assert "bouguer_gravity_mgal" in npz
    assert "aster_quartz_index_qi" in npz
    assert "sar_cband_vh_db" in npz

