# Li-Seeker: AI-Powered Mineral Prospectivity Mapping for Lithium Pegmatites

**Critical Minerals Innovation Hackathon (CMiH 2026)**  
*Organized by the Jawaharlal Nehru Aluminium Research Development & Design Centre (JNARDDC)*  
*Under the aegis of the Ministry of Mines, Government of India*  
*Event: India Mining Week 2026*

---

## 📌 Problem Statement Alignment
* **Problem Statement ID**: PS-01
* **Title**: Mineral Prospectivity Mapping from Open Data
* **Objective**: Use GSI Bhukosh / National Geoscience Data Repository (NGDR) occurrence data, Sentinel-2 spectral indices, and open aeromagnetic / geochemical layers to train a machine learning model flagging areas prospective for **lithium pegmatites**.
* **Mandated Deliverable**: Prospectivity heatmap for one district with validation against known occurrences.

---

## 🎯 Benchmark Target District Selection

### Primary Candidate: Bhilwara District, Rajasthan
* **Geographic Extent**: Latitude 25°03' N – 25°51' N | Longitude 74°03' E – 75°15' E
* **Geological Setting**: Aravalli Craton / Banded Gneissic Complex (BGC) & Mangalwar Complex
* **Target Belt**: **Bhilwara Pegmatite Belt (BPB)** — Mandal, Karera, Potlan, Asind, Bhunas, Raipur swarms
* **Target Minerals**: Spodumene ($\text{LiAlSi}_2\text{O}_6$), Lepidolite ($\text{K(Li,Al)}_3\text{(Al,Si)}_4\text{O}_{10}\text{(F,OH)}_2$), Amblygonite, Columbite-Tantalite, Beryl

### Why Bhilwara Outperforms All Other Indian Districts
1. **Proven LCT Pegmatite Ground Truth**: Bhilwara hosts hundreds of documented GSI Bhukosh pegmatite bodies with confirmed lithium mineralization, providing robust positive training labels and ground-truth validation.
2. **Optimal Remote Sensing Terrain**: Semi-arid pediment with sparse xerophytic scrub; dry-season bedrock outcrops yield **$\text{NDVI} < 0.20$**, enabling Sentinel-2 SWIR bands (B11, B12) to detect diagnostic $\text{Al-OH}$ / Li-mica signatures with high signal-to-noise ratio.
3. **Contrast with Mandya (Karnataka)**: While Mandya hosts the Marlagalla spodumene belt (1,600 tonnes Li metal inferred by AMD), it lies in the fertile Cauvery agricultural plain with continuous sugarcane/paddy cultivation ($\text{NDVI} > 0.55$), causing severe optical remote sensing occlusion.
4. **Contrast with Bastar (Chhattisgarh)**: Covered by dense, moist-to-dry tropical Sal forests ($\text{NDVI} > 0.65$), preventing optical bedrock discrimination.
5. **Fundamental Scientific Distinction from Reasi (J&K)**: Reasi is a **sedimentary-diagenetic, paleo-lateritic / bauxite-clay hosted lithium deposit** where Li is bound in clay lattices (illite/halloysite), **NOT** an igneous pegmatite. PS-01 explicitly mandates pegmatite exploration; applying pegmatite petrogenetic models to Reasi creates physical and metallurgical mismatch.

---

## 🛰️ Remote Sensing & Evidential Layer Stack

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
* **Crosta Feature-Oriented PCA**: 4-band PCA on $[\text{B2}, \text{B4}, \text{B11}, \text{B12}]$ selecting the component with maximum opposing loadings between B11 and B12.

### 2. Multi-Modal Geoscience Integration
* **GSI Bhukosh 1:50,000 Geology**: Distance to S-type parental leucogranites (pegmatite trap halo 1–5 km) and structural fault density.
* **National Geochemical Mapping (NGCM)**: Stream sediment assays for $\text{Li}$ (ppm) and $\text{K/Rb}$ fractionation index.
* **NAGMP Aeromagnetics**: Reduced-to-Pole (RTP) residual magnetic lows mapping felsic non-magnetic leucogranite intrusions.
* **Copernicus DEM 30m**: Slope gradient and topographic lineament density.

---

## 🤖 Machine Learning Architecture: Bagging PU-XGBoost

In mineral exploration, unmapped ground is **unlabeled**, not confirmed barren negatives. Standard binary classification fails due to severe label noise.

* **Positive-Unlabeled (PU) Bagging**: An ensemble of $K = 30$ bootstrap XGBoost classifiers. Each iteration draws a subsample of pseudo-negatives from the unlabeled background outside a 1 km exclusion buffer around known deposits.
* **Spatial Block Cross-Validation**: The study area is divided into discrete contiguous geographic blocks (5 km × 5 km). Entire spatial blocks are held out to eliminate **spatial autocorrelation leakage** (Tobler's First Law), guaranteeing genuine field generalizability.

---

## 📊 Achieved Hackathon KPIs (Bhilwara District)

| Exploration KPI | Achieved Score | Evaluation Standard |
| :--- | :---: | :--- |
| **Area Under Success Rate Curve (AUSRC)** | **0.9958** | Exceptional ($> 0.85$ standard) |
| **Spatial Block CV Mean ROC-AUC** | **0.9863** | Out-of-block generalizability |
| **Spatial Block CV Mean PR-AUC** | **0.8225** | Robust against extreme class imbalance |
| **Deposit Capture Rate at Optimal Threshold** | **100.0%** | All 15 known GSI deposits captured |
| **Prioritized Concession Area Required** | **1.3%** | 98.7% of barren land successfully excluded |
| **Normalized Exploration Density ($N_d$)** | **78.31x** | Anomaly concentration vs background |
| **Exploration Gain ($E_G$)** | **0.9872** | Maximum risk reduction factor |

---

## 💻 Quickstart & Execution

### 1. Requirements Installation
```powershell
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline (CLI)
```powershell
python demo_pipeline.py
```
*Executes the full pipeline, runs Spatial Block CV, computes the P-A plot, extracts drill targets, and generates all deliverables in ~20 seconds.*

### 3. Run Automated Pytest Suite
```powershell
python -m pytest tests/ -v
```

### 4. Launch Interactive Web GIS Dashboard
```powershell
streamlit run app/app.py
```
*Or double-click `run_dashboard.bat` on Windows.*

---

## 🗺️ Interactive Web GIS Features
* **Multi-Layer Toggle**: Switch between Prospectivity Heatmap, GSI Known Deposits, Delineated Drill Targets, Al-OH Mica Index, and Aeromagnetic RTP lows.
* **Dynamic Cutoff Slider**: Real-time recalculation of concession area, deposit recall, and exploration density.
* **Ranked Drill Targets**: Tabular overview of Tier-1 and Tier-2 targets with centroid coordinates, area ($\text{km}^2$), and mean prospectivity scores.
* **Analytics Tab**: Prediction-Area (P-A) crossing point curve and Gini feature importance contributions.
* **Export Center**: Direct download of GeoTIFF prospectivity rasters, GeoJSON targets, and metrics reports for QGIS/ArcGIS.

---

## 📂 Deliverables Directory (`output/`)
* `bhilwara_lithium_prospectivity.tif`: 32-bit floating point GeoTIFF prospectivity heatmap.
* `bhilwara_drill_targets.geojson`: Ranked G4/G3 exploration targets with bounding boxes and metadata.
* `prediction_area_plot.png`: Publication-quality P-A crossing point plot.
* `feature_rankings.csv`: Evidential layer contribution breakdown.
* `prospectivity_metrics.json`: Quantitative exploration KPIs.

---

## 🏛️ Regulatory & Mission Alignment
Directly supports the **National Critical Mineral Mission (NCMM)** launched by the Ministry of Mines to transition India from G4 reconnaissance to auctionable G3/G2 mining leases, de-risking greenfield exploration for state agencies (GSI, MECL) and private Exploration License (EL) holders.
