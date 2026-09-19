"""
Export and Serialization Engine for Prospectivity Deliverables.
Hardened with strict directory confinement to prevent path traversal.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ..geospatial.raster_ops import export_geotiff, GeoGrid


def export_prospectivity_deliverables(prospectivity_map: np.ndarray, grid: GeoGrid,
                                      pa_metrics: dict, target_geojson: dict,
                                      feature_rankings_df, output_dir: str,
                                      district_prefix: str = "bhilwara",
                                      uncertainty_map: np.ndarray = None):
    """
    Exports all required CMiH hackathon deliverables to disk.
    Strictly confines outputs to PROJECT_ROOT / "output" to prevent path traversal
    and ensure directory confinement compliance (PROJECT.md Feature F13 & ORIGINAL_REQUEST §R4).
    """
    if "\x00" in str(output_dir):
        raise ValueError(f"Embedded null byte detected in output_dir: {output_dir}")

    # Canonical base paths for confinement verification
    project_root = Path(__file__).resolve().parents[2]
    project_output = (project_root / "output").resolve()

    raw_path = Path(output_dir)

    if raw_path.is_absolute():
        resolved_dir = raw_path.resolve()
        if not resolved_dir.is_relative_to(project_output):
            raise ValueError(
                f"Directory Confinement Violation: Absolute output_dir '{output_dir}' "
                f"resolved to '{resolved_dir}', which is outside allowed directory '{project_output}'."
            )
    else:
        # Check if relative path was meant relative to project_root (e.g. 'output' or 'output/sub')
        c1 = (project_root / raw_path).resolve()
        if c1.is_relative_to(project_output):
            resolved_dir = c1
        else:
            # Check if relative path was meant relative to project_output (e.g. 'subrun')
            c2 = (project_output / raw_path).resolve()
            if c2.is_relative_to(project_output):
                resolved_dir = c2
            else:
                raise ValueError(
                    f"Directory Confinement Violation: Relative output_dir '{output_dir}' "
                    f"escapes allowed directory '{project_output}'."
                )

    os.makedirs(resolved_dir, exist_ok=True)

    pref = district_prefix.lower().strip() if district_prefix else "bhilwara"

    # 1. GeoTIFF Prospectivity Heatmap
    tif_path = os.path.join(str(resolved_dir), f"{pref}_lithium_prospectivity.tif")
    export_geotiff(prospectivity_map, tif_path, grid)
    # Maintain default bhilwara path for backwards compatibility if different
    if pref != "bhilwara":
        export_geotiff(prospectivity_map, os.path.join(str(resolved_dir), "bhilwara_lithium_prospectivity.tif"), grid)

    # 1b. Predictive Uncertainty GeoTIFF
    unc_tif_path = None
    if uncertainty_map is not None:
        unc_tif_path = os.path.join(str(resolved_dir), f"{pref}_uncertainty_map.tif")
        export_geotiff(uncertainty_map, unc_tif_path, grid)
        if pref != "bhilwara":
            export_geotiff(uncertainty_map, os.path.join(str(resolved_dir), "bhilwara_uncertainty_map.tif"), grid)

    # 2. GeoJSON Exploration Targets
    geojson_path = os.path.join(str(resolved_dir), f"{pref}_drill_targets.geojson")
    with open(geojson_path, "w") as f:
        json.dump(target_geojson, f, indent=2)
    if pref != "bhilwara":
        with open(os.path.join(str(resolved_dir), "bhilwara_drill_targets.geojson"), "w") as f:
            json.dump(target_geojson, f, indent=2)

    # 3. Metrics JSON
    metrics_path = os.path.join(str(resolved_dir), "prospectivity_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(pa_metrics, f, indent=2)

    # 4. Feature Rankings CSV
    csv_path = os.path.join(str(resolved_dir), "feature_rankings.csv")
    feature_rankings_df.to_csv(csv_path, index=False)

    # 5. Prediction-Area (P-A) Plot Image
    pa_fig_path = os.path.join(str(resolved_dir), "prediction_area_plot.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pa_metrics["area_percentages"], pa_metrics["deposit_capture_rates"],
            label="Deposit Capture Rate (Pd)", color="#1b9e77", lw=2.5)
    ax.plot(pa_metrics["area_percentages"], 100.0 - np.array(pa_metrics["area_percentages"]),
            label="100% - Area Proportion (100-Pa)", color="#d95f02", linestyle="--", lw=2.0)

    cross = pa_metrics["crossing_point"]
    ax.scatter([cross["area_percentage"]], [cross["deposit_capture_percentage"]],
               color="red", s=100, zorder=5,
               label=f"Crossing Point (Th={cross['optimal_threshold']}, Capture={cross['deposit_capture_percentage']}%)")

    if "katghora" in pref or "korba" in pref:
        dist_title = "Katghora Block (Korba District)"
    elif "bhilwara" in pref:
        dist_title = "Bhilwara District"
    elif "mandya" in pref:
        dist_title = "Mandya District"
    else:
        dist_title = f"{pref.title()} District"
    ax.set_title(f"Prediction-Area (P-A) Plot: {dist_title} LCT Pegmatites", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cumulative Prospective Area (%)", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(pa_fig_path, dpi=200)
    plt.close(fig)

    out_paths = {
        "geotiff": tif_path,
        "geojson": geojson_path,
        "metrics_json": metrics_path,
        "feature_rankings_csv": csv_path,
        "pa_plot_png": pa_fig_path
    }
    if unc_tif_path is not None:
        out_paths["uncertainty_geotiff"] = unc_tif_path

    return out_paths
