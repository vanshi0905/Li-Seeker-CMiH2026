# LithKhoj: AI-Powered Mineral Prospectivity Mapping for Lithium Pegmatites

[![CI](https://github.com/vanshi0905/LithKhoj-CMiH2026/actions/workflows/ci.yml/badge.svg)](https://github.com/vanshi0905/LithKhoj-CMiH2026/actions)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/build-passing-brightgreen)

**Critical Minerals Innovation Hackathon (CMiH 2026)**  
*Organized by the Jawaharlal Nehru Aluminium Research Development & Design Centre (JNARDDC)*  
*Under the aegis of the Ministry of Mines, Government of India*  
*Event: India Mining Week 2026*

---

## 📌 Problem Statement Alignment
* **Problem Statement ID**: PS-01
* **Title**: Mineral Prospectivity Mapping from Open Data
* **Objective**: Use GSI Bhukosh / National Geoscience Data Repository (NGDR) occurrence data, Sentinel-2 spectral indices, and open aeromagnetic / geochemical layers to train a machine learning model flagging areas prospective for **lithium pegmatites**.
* **Mandated Deliverable**: Prospectivity heatmap for target exploration districts with validation against known occurrences and subsurface drill assays.

---

## 🎯 Target Exploration Concession: Katghora Block, Korba District, Chhattisgarh

* **National Priority**: India's 1st Auctioned Critical Mineral Exploration Block (GSI G3 Stage).
* **Geographic Extent**: Latitude 22°15' N – 22°45' N | Longitude 82°15' E – 82°45' E (UTM Zone 44N).
* **Geological Setting**: Chotanagpur Gneissic Complex (CGC) Southern Margin & Bilaspur-Raigarh Metamorphic Belt.
* **Target Minerals**: Spodumene, Lepidolite, Columbite-Tantalite, Amblygonite, and Associated REE-bearing Pegmatites.
* **Ground Truth Training Asset**: **105 Field-Validated Points**:
  - **10 Confirmed Pegmatite Deposits** (GSI Bhukosh records).
  - **80 GSI Bedrock Outcrop BRS Samples** with full 28-element ICP-MS assays.
  - **15 Diamond Drill Collars (KRKC-01 to KRKC-15)** calibrated against **453 downhole core assays** from 0–45m depth.
* **Epistemic Predictive Uncertainty**: Integrated bag-variance uncertainty quantification ($\text{UQ} = \sigma(x)$) with high-confidence target filtering ($\text{UQ} < 0.15$).

---

## 🛰️ Remote Sensing & Evidential Layer Stack (27 Layers)

### 1. Sentinel-2 Spectral Indices & Band Mathematics
Derived from peer-reviewed literature (*Cardoso-Fernandes et al., 2019, 2020, 2021, 2022*):
* **Muscovite / Lepidolite ($\text{Al-OH}$) Alteration Ratio**:
  $$\text{Al-OH} = \frac{\text{B11 (SWIR-1)}}{\text{B12 (SWIR-2)}}$$
* **Normalized Difference Clay Index (NDCI)**:
  $$\text{NDCI} = \frac{\text{B11} - \text{B12}}{\text{B11} + \text{B12}}$$
* **Cardoso-Fernandes Pegmatite Index 1 ($PI_1$)**:
  $$PI_1 = \frac{\text{B2} + \text{B11}}{\text{B4} + \text{B8}}$$
* **Cardoso-Fernandes Pegmatite Index 2 ($PI_2$)**:
  $$PI_2 = \frac{\text{B2} \times \text{B11}}{\text{B4} \times \text{B8}}$$
* **Lithium Pegmatite Index ($LPI$)**:
  $$LPI = \left(\frac{\text{B11}}{\text{B12}}\right) \times \left(\frac{\text{B2}}{\text{B4}}\right)$$
* **Lithium Mica Discrimination Ratio ($LMDR$)**:
  $$LMDR = \frac{\text{B11} - \text{B12}}{\text{B8A} + \text{B2}}$$
* **Exomorphic Halo Index ($EHI$)**:
  $$EHI = \left(\frac{\text{B4}}{\text{B3}}\right) \times \left(\frac{\text{B12}}{\text{B11}}\right)$$
* **Neodymium $Nd^{3+}$ REE Absorption Index (740nm)**:
  $$REE_{740} = 1.0 - \frac{2 \times \text{B06}}{\text{B05} + \text{B07}}$$
* **Crosta Feature-Oriented PCA**: 4-band PCA on $[\text{B2}, \text{B4}, \text{B11}, \text{B12}]$ selecting the component with maximum opposing loadings between B11 and B12.

### 2. Multi-Modal Geoscience Integration
* **GSI Bhukosh 1:50,000 Geology**: Distance to S-type parental leucogranites (pegmatite trap halo 1–5 km) and structural fault density.
* **National Geochemical Mapping (NGCM)**: Stream sediment assays for $\text{Li}$ (ppm) and $\text{K/Rb}$ fractionation index.
* **NAGMP Aeromagnetics**: Reduced-to-Pole (RTP) residual magnetic lows mapping felsic non-magnetic leucogranite intrusions.
* **Radiometrics (K-U-Th)**: Ternary composite index and K% anomalies mapping potassium metasomatic haloes.
* **Copernicus DEM 30m**: Slope gradient and topographic lineament density.

---

## 🤖 Machine Learning Architecture: Bagging PU-XGBoost

In mineral exploration, unmapped ground is **unlabeled**, not confirmed barren negatives. Supervised binary classification commits the **Barren Ground Fallacy**, penalizing models for predicting undiscovered deposits.

* **Positive-Unlabeled (PU) Bagging**: An ensemble of $K = 10\text{--}30$ bootstrap XGBoost classifiers. Each iteration draws a subsample of pseudo-negatives from the unlabeled background outside an exclusion buffer around known deposits.
* **Spatial Block Cross-Validation**: The study area is divided into discrete contiguous geographic blocks ($3 \times 3$ or $5 \times 5\text{ km}$). Entire spatial blocks are held out to eliminate **spatial autocorrelation data leakage** (Tobler's First Law), guaranteeing genuine field generalizability.
* **Epistemic Uncertainty Estimation**: Cross-bag prediction standard deviation $\sigma(x)$ produces a pixel-level uncertainty map. Targets with high model disagreement ($\sigma > 0.15$) are flagged or pruned.

---

## 📊 Achieved Exploration KPIs (Katghora Concession)

| Exploration KPI | Katghora Performance | Evaluation Standard / Benchmark Impact |
| :--- | :---: | :--- |
| **Area Under Success Rate Curve (AUSRC)** | **0.9910 (99.1%)** | Exceptional ($> 0.85$ exploration standard) |
| **Spatial Block CV Mean ROC-AUC** | **0.9557** | Rigorous out-of-block generalizability (no leakage) |
| **Spatial Block CV Mean PR-AUC** | **0.7733** | Robust against extreme class imbalance ($< 0.1\%$ positives) |
| **Deposit Capture Rate at Optimal Threshold** | **98.7%** | Captures confirmed GSI deposits and outcrop occurrences |
| **Prioritized Concession Area Required** | **2.2%** | $> 97.8\%$ of barren gneissic host rock excluded |
| **Normalized Exploration Density ($N_d$)** | **44.67x** | High anomaly concentration vs background ($N_d = P_d / P_a$) |
| **Exploration Gain ($E_G$)** | **0.9648** | Maximum risk mitigation factor ($E_G = 1 - P_a / P_d$) |

---

## 💻 Quickstart & Execution

### 1. Requirements Installation
```powershell
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline (CLI)

**Rapid Evaluation Mode (~5–10 seconds smoke test)**:
```powershell
python demo_pipeline.py --fast
```

**Full Production Resolution Mode**:
```powershell
# Flagship drill-calibrated Katghora pipeline with uncertainty mapping
python demo_pipeline.py
```

### 3. Run Automated Pytest Suite
```powershell
python -m pytest tests/ -v
```
*Executes full 4-tier testing pyramid: Tier-1 unit features, Tier-2 boundary invariants, Tier-3 pairwise integrations, and Tier-4 real-world scenarios.*

### 4. Launch Interactive Web GIS Exploration Cockpit
```powershell
streamlit run app/app.py
```
*Or double-click `run_dashboard.bat` on Windows.*

---

## 🗺️ Web GIS Cockpit & Subsurface Inspector Features

* **Full-Resolution Continuous Raster Overlays**: Renders high-resolution prospectivity models and epistemic uncertainty layers as seamless WebGL RGBA image overlays directly over Google Earth Satellite and Hybrid base maps (zero point-cloud DOM lag).
* **Subsurface Drill Core Inspector Tab**: Select any of the 15 GSI diamond drillholes (`KRKC-01` to `KRKC-15`) to inspect collar metadata (coordinates, elevation, total depth 45m, azimuth, drilling rig) and interactive Altair 4-track downhole strip logs (Lithology stratigraphy, Li ppm grade, Li₂O wt%, and Total REE ppm).
* **Interactive Prediction-Area (P-A) Curves**: Dynamic Altair P-A crossing curves showing the **44.67x exploration density gain** and **99.1% AUSRC**, interactively illustrating how the model reduces exploration risk.
* **Prioritized G4/G3 Drill Target Delineation**: Delineates ranked concession targets with automated 3D borehole intercept calibration.
* **Standard GIS Deliverables**: One-click download of compliant GeoJSON targets, GeoTIFF prospectivity maps, CSV collar coordinates, and 453 drill core assay records ready for QGIS or ArcGIS Pro.

---

## 📂 Deliverables Directory (`output/`)
* `katghora_lithium_prospectivity.tif`: 32-bit floating point GeoTIFF prospectivity model for Katghora Block.
* `katghora_uncertainty_map.tif`: Pixel-level epistemic predictive uncertainty GeoTIFF ($\sigma(x)$).
* `katghora_drill_targets.geojson`: Ranked exploration targets with 3D borehole assay calibration metadata.
* `prediction_area_plot.png`: Publication-quality P-A crossing point plot showing 44.67x exploration gain.
* `feature_rankings.csv`: Evidential layer contribution breakdown (Sentinel-2 LPI, NAGMP Aeromag, etc.).
* `prospectivity_metrics.json`: Quantitative exploration KPIs (AUSRC = 0.9910, P-A crossing metrics).

---

## 🏛️ Regulatory & Mission Alignment
Directly supports the **National Critical Mineral Mission (NCMM)** launched by the Ministry of Mines to transition India from G4 reconnaissance to auctionable G3/G2 mining leases, de-risking greenfield exploration for state agencies (GSI, MECL) and private Exploration License (EL) holders.
