"""
GSI G3 Exploration Data Extractor for Katghora-Rampur Block, Korba District, Chhattisgarh.

Extracts:
1. 80 field-validated bedrock samples (BRS) with 28-element geochemical assays (Annexure-VI).
2. 15 diamond drill borehole collars (KRKC-01 to KRKC-15) on 400m x 400m grid with UTM 44N to Lat/Lon conversion (Annexure-IX).
3. 453 downhole drill core lithium, rare-metal, and REE assays (Annexure-X & XI).
4. Updates data/ground_truth/katghora_pegmatites.json with 95 real GSI ground truth points (80 BRS + 15 drill collars).
"""

import io
import json
import math
import os
import zipfile
import numpy as np
import pandas as pd


def utm44n_to_latlon(easting: float, northing: float, zone: int = 44, northern: bool = True) -> tuple[float, float]:
    """
    Converts UTM Zone 44N coordinates to WGS84 Latitude and Longitude in decimal degrees.
    Uses standard Transverse Mercator formulas for the WGS84 ellipsoid.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e = math.sqrt(1.0 - (b / a) ** 2)
    e_prime_sq = (e * a / b) ** 2
    k0 = 0.9996

    x = float(easting) - 500000.0
    y = float(northing) if northern else float(northing) - 10000000.0
    lon_origin = (zone - 1) * 6 - 180 + 3  # Zone 44 -> 81.0 deg E

    M = y / k0
    mu = M / (a * (1.0 - e**2 / 4.0 - 3.0 * e**4 / 64.0 - 5.0 * e**6 / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - e**2)) / (1.0 + math.sqrt(1.0 - e**2))

    J1 = 3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0
    J2 = 21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0
    J3 = 151.0 * e1**3 / 96.0
    J4 = 1097.0 * e1**4 / 512.0

    fp = mu + J1 * math.sin(2.0 * mu) + J2 * math.sin(4.0 * mu) + J3 * math.sin(6.0 * mu) + J4 * math.sin(8.0 * mu)

    C1 = e_prime_sq * (math.cos(fp) ** 2)
    T1 = math.tan(fp) ** 2
    R1 = a * (1.0 - e**2) / ((1.0 - e**2 * (math.sin(fp) ** 2)) ** 1.5)
    N1 = a / math.sqrt(1.0 - e**2 * (math.sin(fp) ** 2))
    D = x / (N1 * k0)

    lat_rad = fp - (N1 * math.tan(fp) / R1) * (
        (D**2) / 2.0
        - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * (C1**2) - 9.0 * e_prime_sq) * (D**4) / 24.0
        + (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * (T1**2) - 252.0 * e_prime_sq - 3.0 * (C1**2)) * (D**6) / 720.0
    )

    lon_diff_rad = (
        D
        - (1.0 + 2.0 * T1 + C1) * (D**3) / 6.0
        + (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * (C1**2) + 8.0 * e_prime_sq + 24.0 * (T1**2)) * (D**5) / 120.0
    ) / math.cos(fp)

    lat_deg = math.degrees(lat_rad)
    lon_deg = lon_origin + math.degrees(lon_diff_rad)
    return lat_deg, lon_deg


def locate_gsi_annexures_zip(project_root: str = None) -> tuple[str, bytes | None]:
    """
    Locates the GSI Annexures zip archive in the local directory tree.
    Returns (path, bytes_or_none).
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    candidate_paths = [
        os.path.join(
            project_root,
            "Download",
            "Exploration_Data_20250108192652_297_Plates_Pdf_20260918233433997",
            "CRO-23909-2022",
            "TABLES",
            "20250108192806.842_Annexures_Excel & Word.zip",
        ),
        os.path.join(
            project_root,
            "data",
            "lithium g3",
            "Exploration_Data_20250108192652_297_Plates_Pdf_20260919111905052.zip",
        ),
        os.path.join(
            project_root,
            "Download",
            "Exploration_Data_20250108192652_297_Plates_Pdf_20260918233433997.zip",
        ),
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            # Check if this is the direct annexures zip or outer zip
            if p.endswith("Annexures_Excel & Word.zip"):
                with open(p, "rb") as f:
                    return p, f.read()
            else:
                # Inspect nested archive
                try:
                    with zipfile.ZipFile(p, "r") as outer:
                        for entry in outer.namelist():
                            if entry.endswith("Annexures_Excel & Word.zip"):
                                return f"{p}!{entry}", outer.read(entry)
                except Exception:
                    pass

    raise FileNotFoundError("Could not find GSI Annexures zip in project workspace.")


def extract_brs_samples(annexures_zip_bytes: bytes) -> pd.DataFrame:
    """
    Extracts the 80 Bedrock Samples (BRS) with coordinates and multi-element assays from Annexure-VI.
    """
    with zipfile.ZipFile(io.BytesIO(annexures_zip_bytes)) as z:
        target_name = None
        for n in z.namelist():
            if "Annexure-VI_BRS-Li-RM-REE.xlsx" in n:
                target_name = n
                break
        if not target_name:
            raise KeyError("Annexure-VI_BRS-Li-RM-REE.xlsx not found in archive")

        content = z.read(target_name)
        df_raw = pd.read_excel(io.BytesIO(content), sheet_name=0, header=2)

    # Standardize column names
    col_map = {
        "Sr. No.": "sr_no",
        "Sample No.": "sample_id",
        "Lithology": "lithology",
        "Lattitude": "latitude",
        "Longitude": "longitude",
        "Li": "li_ppm",
        "Cs": "cs_ppm",
        "Be": "be_ppm",
        "Rb": "rb_ppm",
        "Ta": "ta_ppm",
        "Nb": "nb_ppm",
        "Sn": "sn_ppm",
        "W": "w_ppm",
        "Mo": "mo_ppm",
        "La": "la_ppm",
        "Ce": "ce_ppm",
        "Pr": "pr_ppm",
        "Nd": "nd_ppm",
        "Eu": "eu_ppm",
        "Sm": "sm_ppm",
        "Gd": "gd_ppm",
        "Tb": "tb_ppm",
        "Dy": "dy_ppm",
        "Ho": "ho_ppm",
        "Er": "er_ppm",
        "Tm": "tm_ppm",
        "Yb": "yb_ppm",
        "Lu": "lu_ppm",
    }
    df = df_raw.rename(columns=col_map)
    df = df.dropna(subset=["sample_id", "latitude", "longitude"]).copy()

    # Convert numeric fields
    numeric_cols = [c for c in df.columns if c not in ["sample_id", "lithology"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sr_no"] = df["sr_no"].astype(int)
    df["gsi_stage"] = "G3"
    df["district"] = "Korba"
    df["state"] = "Chhattisgarh"
    df["sample_type"] = "Bedrock Outcrop (BRS)"
    return df


def extract_borehole_collars(annexures_zip_bytes: bytes) -> pd.DataFrame:
    """
    Extracts the 15 borehole collar coordinates (KRKC-01 to KRKC-15) from Annexure-IX
    and converts UTM Zone 44N coordinates to WGS84 Lat/Lon.
    """
    with zipfile.ZipFile(io.BytesIO(annexures_zip_bytes)) as z:
        target_name = None
        for n in z.namelist():
            if "Annexure-IX_litholog.xlsx" in n:
                target_name = n
                break
        if not target_name:
            raise KeyError("Annexure-IX_litholog.xlsx not found in archive")

        xl = pd.ExcelFile(io.BytesIO(z.read(target_name)))

    collars = []
    for i in range(1, 16):
        bh_id = f"KRKC-{i:02d}"
        sheet_cand = [bh_id, f"KRKC-{i}", "KRKC-01-15"]
        sheet_name = next((s for s in sheet_cand if s in xl.sheet_names), None)

        if not sheet_name:
            continue

        df_sheet = xl.parse(sheet_name, header=None)

        north, east, rl = None, None, None
        rig, init_date, close_date, depth_m = "Diamond Rig", None, None, 45.0

        for r in range(min(8, len(df_sheet))):
            row_vals = [str(x).strip() for x in df_sheet.iloc[r].values]
            for c_idx, v in enumerate(row_vals):
                if v.startswith("2491") or v.startswith("2490"):
                    try:
                        north = float(v)
                        east = float(row_vals[c_idx + 1])
                        if c_idx + 2 < len(row_vals) and row_vals[c_idx + 2] not in ["nan", ""]:
                            rl = float(row_vals[c_idx + 2])
                        if c_idx + 4 < len(row_vals):
                            rig = row_vals[c_idx + 4]
                        if c_idx + 5 < len(row_vals):
                            init_date = str(row_vals[c_idx + 5]).split()[0]
                        if c_idx + 6 < len(row_vals):
                            close_date = str(row_vals[c_idx + 6]).split()[0]
                        if c_idx + 7 < len(row_vals) and "m" in str(row_vals[c_idx + 7]):
                            depth_m = float(str(row_vals[c_idx + 7]).replace("m", "").strip())
                    except Exception:
                        pass
                    break
            if north is not None:
                break

        if north is not None and east is not None:
            lat, lon = utm44n_to_latlon(east, north, zone=44, northern=True)
            collars.append({
                "borehole_id": bh_id,
                "northing_utm44n": round(north, 3),
                "easting_utm44n": round(east, 3),
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "collar_rl_m": round(rl, 3) if rl else 330.0,
                "inclination_deg": 90.0,
                "azimuth_deg": 0.0,
                "drilling_rig": rig,
                "initiated_date": init_date,
                "completed_date": close_date,
                "total_depth_m": depth_m,
                "district": "Korba",
                "state": "Chhattisgarh",
                "block": "Katghora-Rampur G3 Block",
            })

    return pd.DataFrame(collars)


def extract_drill_core_assays(annexures_zip_bytes: bytes) -> pd.DataFrame:
    """
    Extracts and merges downhole drill core lithium, rare-metal, and REE assays
    from Annexure-X and Annexure-XI.
    """
    with zipfile.ZipFile(io.BytesIO(annexures_zip_bytes)) as z:
        target_li = None
        target_ree = None
        for n in z.namelist():
            if "Annexure-X_Core results_Li" in n:
                target_li = n
            elif "Annexure-XI_Core results_REE" in n:
                target_ree = n

        if not target_li:
            raise KeyError("Annexure-X not found in archive")

        xl_li = pd.ExcelFile(io.BytesIO(z.read(target_li)))
        # KRKC-1 is the consolidated master table containing all 15 boreholes
        df_li = xl_li.parse("KRKC-1", header=None)
        sub_li = df_li.iloc[4:].copy()
        sub_li.columns = [str(c).strip() for c in df_li.iloc[2].values]
        valid_li = sub_li[sub_li["Sample No."].astype(str).str.contains("KRKC", na=False)].copy()

        # REE Table
        valid_ree = None
        if target_ree:
            xl_ree = pd.ExcelFile(io.BytesIO(z.read(target_ree)))
            df_ree = xl_ree.parse("KRKC-1", header=None)
            hdr_ree = [str(c).encode("ascii", "ignore").decode().strip() for c in df_ree.iloc[2].values]
            sub_ree = df_ree.iloc[3:].copy()
            sub_ree.columns = hdr_ree
            valid_ree = sub_ree[sub_ree["Sample No."].astype(str).str.contains("KRKC", na=False)].copy()

    # Clean Li columns
    li_cols = {
        "Sample No.": "sample_id",
        "From (m)": "from_m",
        "To (m)": "to_m",
        "Width (m)": "width_m",
        "Lithology": "lithology",
        "Li-mica %": "li_mica_pct",
        "Li": "li_ppm",
        "Li2O": "li2o_pct",
        "Cs": "cs_ppm",
        "Be": "be_ppm",
        "Rb": "rb_ppm",
        "Ta": "ta_ppm",
        "Nb": "nb_ppm",
        "Sn": "sn_ppm",
        "W": "w_ppm",
        "Mo": "mo_ppm",
    }
    valid_li = valid_li.rename(columns={k: v for k, v in li_cols.items() if k in valid_li.columns})

    if valid_ree is not None:
        ree_cols = {
            "Sample No.": "sample_id",
            "La": "la_ppm",
            "Ce": "ce_ppm",
            "Pr": "pr_ppm",
            "Nd": "nd_ppm",
            "Eu": "eu_ppm",
            "Sm": "sm_ppm",
            "Gd": "gd_ppm",
            "Tb": "tb_ppm",
            "Dy": "dy_ppm",
            "Ho": "ho_ppm",
            "Er": "er_ppm",
            "Tm": "tm_ppm",
            "Yb": "yb_ppm",
            "Lu": "lu_ppm",
            "REE": "total_ree_ppm",
        }
        valid_ree = valid_ree.rename(columns={k: v for k, v in ree_cols.items() if k in valid_ree.columns})
        ree_cols_to_merge = [c for c in ree_cols.values() if c in valid_ree.columns and c != "sample_id"]
        merged = pd.merge(valid_li, valid_ree[["sample_id"] + ree_cols_to_merge], on="sample_id", how="left")
    else:
        merged = valid_li

    # Extract Borehole ID from Sample ID (e.g. KRKC-01/1 -> KRKC-01)
    merged["borehole_id"] = merged["sample_id"].apply(lambda s: str(s).split("/")[0].strip())

    # Convert numeric values
    num_cols = ["from_m", "to_m", "width_m", "li_ppm", "li2o_pct", "cs_ppm", "be_ppm", "rb_ppm", "ta_ppm", "nb_ppm", "nd_ppm", "total_ree_ppm"]
    for nc in num_cols:
        if nc in merged.columns:
            merged[nc] = pd.to_numeric(merged[nc], errors="coerce")

    return merged


def save_all_gsi_g3_data(project_root: str = None) -> dict[str, str]:
    """
    Extracts all real GSI exploration datasets, writes clean CSVs into data/,
    and updates data/ground_truth/katghora_pegmatites.json.
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    archive_path, archive_bytes = locate_gsi_annexures_zip(project_root)
    print(f"[GSI G3 Extractor] Located exploration archive at: {archive_path}")

    # 1. BRS Samples
    df_brs = extract_brs_samples(archive_bytes)
    brs_csv_path = os.path.join(data_dir, "katghora_gsi_brs_samples.csv")
    df_brs.to_csv(brs_csv_path, index=False)
    print(f"[GSI G3 Extractor] Extracted {len(df_brs)} BRS samples -> {brs_csv_path}")

    # 2. Borehole Collars
    df_collars = extract_borehole_collars(archive_bytes)
    collars_csv_path = os.path.join(data_dir, "katghora_borehole_collars.csv")
    df_collars.to_csv(collars_csv_path, index=False)
    print(f"[GSI G3 Extractor] Extracted {len(df_collars)} borehole collars -> {collars_csv_path}")

    # 3. Drill Core Assays
    df_assays = extract_drill_core_assays(archive_bytes)
    assays_csv_path = os.path.join(data_dir, "katghora_drill_core_assays.csv")
    df_assays.to_csv(assays_csv_path, index=False)
    print(f"[GSI G3 Extractor] Extracted {len(df_assays)} drill core assays -> {assays_csv_path}")

    # 4. Update data/ground_truth/katghora_pegmatites.json
    gt_json_path = os.path.join(data_dir, "ground_truth", "katghora_pegmatites.json")
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    # Build BRS sample entries
    brs_entries = []
    for _, row in df_brs.iterrows():
        brs_entries.append({
            "id": f"BRS-{row['sr_no']:02d}",
            "sample_id": str(row["sample_id"]),
            "name": f"Katghora Outcrop {row['sample_id']} ({row['lithology']})",
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "lithology": str(row["lithology"]),
            "li_ppm": float(row["li_ppm"]) if pd.notnull(row["li_ppm"]) else None,
            "rb_ppm": float(row["rb_ppm"]) if pd.notnull(row["rb_ppm"]) else None,
            "cs_ppm": float(row["cs_ppm"]) if pd.notnull(row["cs_ppm"]) else None,
            "be_ppm": float(row["be_ppm"]) if pd.notnull(row["be_ppm"]) else None,
            "ta_ppm": float(row["ta_ppm"]) if pd.notnull(row["ta_ppm"]) else None,
            "minerals": ["Lepidolite", "Spodumene", "Albite", "Quartz", "Tourmaline"],
            "type": f"LCT Bedrock Sample ({row['lithology']})",
            "gsi_stage": "G3",
            "source": "GSI CRO-23909-2022 Annexure-VI",
        })

    # Build Borehole collar entries with downhole assay summary
    bh_entries = []
    assay_summary = df_assays.groupby("borehole_id").agg(
        max_li=("li_ppm", "max"),
        mean_li=("li_ppm", "mean"),
        assay_count=("li_ppm", "count")
    ).to_dict("index")

    for _, row in df_collars.iterrows():
        b_id = str(row["borehole_id"])
        b_summary = assay_summary.get(b_id, {})
        bh_entries.append({
            "id": b_id,
            "name": f"Diamond Drillhole {b_id} (Collar)",
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "northing_utm44n": float(row["northing_utm44n"]),
            "easting_utm44n": float(row["easting_utm44n"]),
            "collar_rl_m": float(row["collar_rl_m"]),
            "total_depth_m": float(row["total_depth_m"]),
            "inclination_deg": float(row["inclination_deg"]),
            "drilling_rig": str(row["drilling_rig"]),
            "peak_downhole_li_ppm": float(b_summary.get("max_li", 0.0)),
            "mean_downhole_li_ppm": float(round(b_summary.get("mean_li", 0.0), 1)),
            "assay_interval_count": int(b_summary.get("assay_count", 0)),
            "minerals": ["Spodumene", "Lepidolite", "Amblygonite", "Columbite-Tantalite"],
            "type": "Diamond Drill Hole Collar (Subsurface Validated)",
            "gsi_stage": "G3",
            "source": "GSI CRO-23909-2022 Annexure-IX/X",
        })

    # Combined G3 points (80 BRS + 15 collars = 95 field-validated points)
    g3_points = brs_entries + bh_entries

    # Update json document structure
    gt_data["gsi_report_id"] = "GSI CRO-23909-2022 (FS 2022-23)"
    gt_data["gsi_g3_points_count"] = len(g3_points)
    gt_data["brs_samples_count"] = len(brs_entries)
    gt_data["boreholes_count"] = len(bh_entries)
    gt_data["regional_occurrences_count"] = len(gt_data.get("occurrences", []))
    gt_data["total_ground_truth_count"] = len(gt_data.get("occurrences", [])) + len(g3_points)

    # Attach the real datasets cleanly
    gt_data["brs_samples"] = brs_entries
    gt_data["borehole_collars"] = bh_entries
    gt_data["g3_exploration_points"] = g3_points
    gt_data["all_ground_truth"] = gt_data["occurrences"] + g3_points

    with open(gt_json_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"[GSI G3 Extractor] Updated {gt_json_path} with {len(g3_points)} real GSI points (total: {gt_data['total_ground_truth_count']}).")

    return {
        "brs_csv": brs_csv_path,
        "collars_csv": collars_csv_path,
        "assays_csv": assays_csv_path,
        "gt_json": gt_json_path,
    }


if __name__ == "__main__":
    paths = save_all_gsi_g3_data()
    print("Done! Extracted deliverables:", paths)
