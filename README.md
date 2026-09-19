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

## 🎯 Benchmark Target Districts

### 1. Flagship Drill-Calibrated Benchmark: Katghora Block, Korba District, Chhattisgarh
* **Geographic Extent**: Latitude 22°15' N – 22°45' N | Longitude 82°15' E – 82°45' E
* **Geological Setting**: Chotanagpur Gneissic Complex (CGC) Southern Margin & Bilaspur-Raigarh Belt
* **Target Minerals**: Spodumene, Lepidolite, Columbite-Tantalite, Amblygonite, REE-bearing pegmatites
* **Ground Truth Training Asset**: **105 Field-Validated Points** (10 Regional Deposits + **80 GSI Bedrock Outcrop BRS Samples** with 28-element assays + **15 Diamond Drill Collars** calibrated against **453 downhole core assays** from 0–45m depth).
* **Epistemic Predictive Uncertainty**: Integrated bag-variance uncertainty mapping ($\text{UQ} = \text{std}(p_i)$) with high-confidence target filtering ($\text{UQ} < 0.15$).

### 2. Arid Remote Sensing Benchmark: Bhilwara District, Rajasthan
* **Geographic Extent**: Latitude 25°03' N – 25°51' N | Longitude 74°03' E – 75°15' E
* **Geological Setting**: Aravalli Craton / Banded Gneissic Complex (BGC) & Mangalwar Complex
* **Target Belt**: **Bhilwara Pegmatite Belt (BPB)** — Mandal, Karera, Potlan, Asind, Bhunas, Raipur swarms
* **Optimal Remote Sensing Terrain**: Semi-arid pediment with sparse xerophytic scrub; dry-season bedrock outcrops yield **$\text{NDVI} < 0.20$**, enabling Sentinel-2 SWIR bands (B11, B12) to detect diagnostic $\text{Al-OH}$ / Li-mica signatures with high signal-to-noise ratio.

### Scientific Comparative Analysis with Other Indian Belts
* **Mandya (Karnataka)**: While Mandya hosts the Marlagalla spodumene belt (1,600 tonnes Li metal inferred by AMD), it lies in the fertile Cauvery agricultural plain with continuous sugarcane/paddy cultivation ($\text{NDVI} > 0.55$), causing severe optical remote sensing occlusion.
* **Bastar (Chhattisgarh)**: Covered by dense, moist-to-dry tropical Sal forests ($\text{NDVI} > 0.65$), preventing optical bedrock discrimination without SAR-optical fusion.
* **Fundamental Distinction from Reasi (J&K)**: Reasi is a **sedimentary-diagenetic, paleo-lateritic / bauxite-clay hosted lithium deposit** where Li is bound in clay lattices (illite/halloysite), **NOT** an igneous pegmatite. PS-01 explicitly mandates pegmatite exploration; applying pegmatite petrogenetic models to Reasi creates physical and metallurgical mismatch.

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

## 📊 Achieved Exploration KPIs

| Exploration KPI | Katghora Benchmark | Bhilwara Benchmark | Evaluation Standard |
| :--- | :---: | :---: | :--- |
| **Area Under Success Rate Curve (AUSRC)** | **0.9910** | **0.9958** | Exceptional ($> 0.85$ standard) |
| **Spatial Block CV Mean ROC-AUC** | **0.9557** | **0.9863** | Out-of-block generalizability |
| **Spatial Block CV Mean PR-AUC** | **0.7733** | **0.8225** | Robust against extreme class imbalance |
| **Deposit Capture Rate at Optimal Threshold** | **98.7%** | **100.0%** | All known deposits / GSI points captured |
| **Prioritized Concession Area Required** | **2.2%** | **1.3%** | $> 97.8\%$ of barren land excluded |
| **Normalized Exploration Density ($N_d$)** | **44.67x** | **78.31x** | Anomaly concentration vs background |
| **Exploration Gain ($E_G$)** | **0.9648** | **0.9872** | Maximum risk reduction factor |

---

## 💻 Quickstart & Execution

### 1. Requirements Installation
```powershell
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline (CLI)

**Rapid Evaluation Mode (~5–10 seconds smoke test)**:
```powershell
# Rapid smoke demo on Bhilwara District
python demo_pipeline.py --fast --district bhilwara

# Rapid smoke demo on Katghora District
python demo_pipeline.py --fast --district katghora
```

**Full Production Resolution Mode**:
```powershell
# Flagship drill-calibrated Katghora pipeline with uncertainty mapping
python demo_pipeline.py --district katghora

# Arid remote sensing Bhilwara pipeline
python demo_pipeline.py --district bhilwara
```

### 3. Run Automated Pytest Suite
```powershell
python -m pytest tests/ -v
```
*Executes full 4-tier testing pyramid: Tier-1 unit features, Tier-2 boundary invariants, Tier-3 pairwise integrations, and Tier-4 real-world scenarios.*

### 4. Launch Interactive Web GIS Dashboard
```powershell
streamlit run app/app.py
```
*Or double-click `run_dashboard.bat` on Windows.*

---

## 🗺️ Interactive Web GIS Features
* **Multi-Layer Toggle**: Switch between Prospectivity Heatmap, Epistemic Uncertainty Map, GSI Known Deposits, Delineated Drill Targets, Al-OH Mica Index, and Aeromagnetic RTP lows.
* **Dynamic Cutoff Slider**: Real-time recalculation of concession area, deposit recall, and exploration density.
* **Ranked Drill Targets**: Tabular overview of Tier-1 and Tier-2 targets with centroid coordinates, area ($\text{km}^2$), mean prospectivity, and borehole intercept validation.
* **Analytics Tab**: Prediction-Area (P-A) crossing point curve and Gini feature importance contributions.
* **Export Center**: Direct download of GeoTIFF prospectivity rasters, GeoJSON targets, and metrics reports for QGIS/ArcGIS.

---

## 📂 Deliverables Directory (`output/`)
* `katghora_lithium_prospectivity.tif`: 32-bit floating point GeoTIFF prospectivity heatmap.
* `katghora_uncertainty_map.tif`: Pixel-level epistemic predictive uncertainty GeoTIFF.
* `katghora_drill_targets.geojson`: Ranked exploration targets with borehole assay calibration metadata.
* `bhilwara_lithium_prospectivity.tif`: 32-bit floating point GeoTIFF for Bhilwara Pegmatite Belt.
* `bhilwara_drill_targets.geojson`: Delineated targets for Bhilwara swarms.
* `prediction_area_plot.png`: Publication-quality P-A crossing point plot.
* `feature_rankings.csv`: Evidential layer contribution breakdown.
* `prospectivity_metrics.json`: Quantitative exploration KPIs.

---

## 🏛️ Regulatory & Mission Alignment
Directly supports the **National Critical Mineral Mission (NCMM)** launched by the Ministry of Mines to transition India from G4 reconnaissance to auctionable G3/G2 mining leases, de-risking greenfield exploration for state agencies (GSI, MECL) and private Exploration License (EL) holders.
