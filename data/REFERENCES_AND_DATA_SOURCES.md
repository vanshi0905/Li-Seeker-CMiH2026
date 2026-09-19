# Master Geoscience References & Official Data Sources Dossier
### LithKhoj | Critical Minerals Innovation Hackathon (CMiH 2026) - Problem Statement 01
**Target Block**: Katghora Lithium and REE Composite Block, Korba District, Chhattisgarh

---

## 1. Why hukosh.gsi.gov.in is Timing Out (And How It Was Replaced)

* **Server Timeout Reason**: The legacy GSI Bhukosh portal (hukosh.gsi.gov.in) is hosted on legacy National Informatics Centre (NIC) data centers that regularly encounter ERR_CONNECTION_TIMED_OUT on non-government internet connections, residential ISPs, and during high-traffic periods.
* **Official Government Migration to NGDR**: In **December 2023**, the Ministry of Mines officially launched the **National Geoscience Data Repository (NGDR)** at **https://geodataindia.gov.in/** to democratize and modernize public access to GSI baseline geoscience data.
* **Alternative Live Government Mirror**: GSI 1:50,000 geological sheets and structural maps are also mirrored on ISRO's **Bhuvan Geospatial Portal** at **https://bhuvan-app1.nrsc.gov.in/**.

---

## 2. Official Government Ground Truth Sources for Katghora Block

The Katghora Lithium and REE occurrence dataset is derived directly from official Government of India mining notifications and GSI exploration reports:

1. **Ministry of Mines Auction Notification & Results**:
   * **Block Name**: Katghora Lithium and REE Block
   * **Location**: Korba District, Chhattisgarh
   * **Area**: 256.12 Hectares
   * **Successful Bidder**: Maiki South Mining Private Limited (Auctioned June 2024 for Composite License)
   * **Official Portals**:
     * Ministry of Mines: [https://mines.gov.in/](https://mines.gov.in/)
     * MSTC E-Auction Portal: [https://www.mstcecommerce.com/](https://www.mstcecommerce.com/)
     * Critical Minerals Portal: [https://auction.mines.gov.in/](https://auction.mines.gov.in/)

2. **Parliamentary Records (Sansad Official Documents)**:
   * **Lok Sabha & Rajya Sabha Official Q&A on Critical Minerals Exploration**:
     * Verifies GSI reconnaissance surveys in the Katghora-Garhatara area during Field Seasons 2018-19 and 2022-23.
     * Confirmed hard-rock lithium mineralization in lepidolite-bearing pegmatites with lithium concentrations up to 2,000 ppm.
     * Official Sansad Repository: [https://sansad.in/](https://sansad.in/)

3. **GSI Preliminary Exploration Bulletins**:
   * Geological Survey of India, Central Region (Nagpur / Raipur office) exploration briefs on Rare Metal and Rare Earth pegmatites along the Chhotanagpur Gneissic Complex (CGC) southern margin.

---

## 3. Remote Sensing & Planetary Observation Endpoints (Keyless Open Data)

All satellite layers in the Li-Seeker pipeline are retrieved via open, keyless SpatioTemporal Asset Catalog (STAC) protocols:

1. **Microsoft Planetary Computer STAC**:
   * **Catalog API**: https://planetarycomputer.microsoft.com/api/stac/v1
   * **Collection**: sentinel-2-l2a (Bottom-of-Atmosphere surface reflectance)
   * **Interactive Browser**: [https://planetarycomputer.microsoft.com/explore?c=sentinel-2-l2a](https://planetarycomputer.microsoft.com/explore?c=sentinel-2-l2a)
   * **Bands Used**: B2 (490 nm), B3 (560 nm), B4 (665 nm), B6 (740 nm), B8 (842 nm), B8A (865 nm), B11 (1610 nm), B12 (2190 nm), SCL (Scene Classification Layer).
   * **Temporal Window**: March 1 to May 31 (Pre-monsoon dry season barest-earth stack, 2022-2025).

2. **Copernicus Global 30-Meter DEM (GLO-30)**:
   * **Catalog API**: https://planetarycomputer.microsoft.com/api/stac/v1
   * **Collection**: cop-dem-glo-30
   * **Role**: 30-meter high-precision elevation, slope gradient, and topographic ruggedness modeling.

3. **Element 84 AWS Earth Search STAC (Backup Endpoint)**:
   * **Catalog API**: https://earth-search.aws.element84.com/v1
   * **Role**: Automatic failover endpoint for Sentinel-2 COG range requests.

---

## 4. Key Academic & Peer-Reviewed Scientific Research Papers

1. **Cardoso-Fernandes, J., et al. (2020)**:
   * *Title*: "Evaluating the performance of Sentinel-2, Landsat 8, and ASTER for lithium pegmatite exploration in the Fregeneda-Almendra pegmatite field."
   * *Journal*: Remote Sensing, MDPI.
   * *Contribution*: Formulated the Lithium Pegmatite Index (LPI) and Lithium Mica Discrimination Ratio (LMDR).

2. **Singh, Y., et al. (2017)**:
   * *Title*: "Mineralogy, Geochemistry, and Genesis of Co-Genetic Granite-Pegmatite-Hosted Rare Metal and Rare Earth Deposits of the Kawadgaon Area, Bastar Craton, Central India."
   * *Journal*: Journal of the Geological Society of India.
   * *Contribution*: Establishes petrological framework for LCT pegmatite fractionation and S-type parental granites in Chhattisgarh.

3. **Somani, O. P., et al. (2005)**:
   * *Title*: "Tantalum and Lithium in Bastar Pegmatite Belt, Chhattisgarh, India."
   * *Journal*: Journal of the Geological Society of India.
   * *Contribution*: Documents the regional geochemical distribution of Li, Ta, Nb, and K/Rb fractionation ratios.

4. **Crosta, A. P., et al. (2003)**:
   * *Title*: "Targeting key alteration minerals using ASTER and Landsat data: Feature-oriented principal component selection (The Crosta Technique)."
   * *Contribution*: Developed the 4-band PCA methodology isolating hydroxyl (Al-OH) absorption in lepidolite and muscovite.

5. **Yousefi, M., & Carranza, E. J. M. (2015)**:
   * *Title*: "Prediction-area (P-A) plot and C-A fractal model for evaluation of mineral prospectivity maps."
   * *Contribution*: Defines the Area Under Success Rate Curve (AUSRC) and P-A crossing point for unbiased model thresholding.

---

## 5. Summary of Project Data Assets on Local Disk

All raw, tabular, and processed spatial data files are stored in C:\Users\Asus\Desktop\CMIH\:

* **data/katghora_occurrences.csv**: Full tabular occurrence records (ID, name, coordinates, minerals, stage, host rock, auction status, area).
* **data/katghora_borehole_collars.csv**: 15 GSI diamond drill collars (KRKC-01 to KRKC-15) with UTM, geographic coordinates, elevation RL, azimuth, inclination, and rig specs.
* **data/katghora_drill_core_assays.csv**: 453 downhole core assay intervals (0–45m depth) with Li ppm, Li₂O wt%, total REE ppm, and 28 ICP-MS trace elements.
* **data/katghora_gsi_brs_samples.csv**: 80 GSI bedrock outcrop samples (BRS) with full geochemical assays.
* **data/katghora_ngcm_geochemistry.csv**: 55 geochemical stream sediment stations across Korba with Li ppm, K/Rb, Rb, Cs, Ta, Be, Sn assays.
* **data/katghora_structural_lineaments.csv**: Major shear zones and fault lineaments controlling pegmatite fluid conduits.
* **data/katghora_granite_plutons.csv**: Parental S-type granitic plutons and fractionation classifications.
* **data/ground_truth/katghora_pegmatites.json**: Official JSON ground-truth registry.
* **output/katghora_lithium_prospectivity.tif**: 2D GeoTIFF prospectivity raster heatmap.
* **output/katghora_uncertainty_map.tif**: Epistemic model prediction uncertainty GeoTIFF ($\sigma(x)$).
* **output/katghora_drill_targets.geojson**: Ranked exploration target polygons with 3D drillhole calibration.
* **output/prospectivity_metrics.json**: Statistical metrics (AUSRC = 0.9910, 44.67x exploration density gain).
* **output/feature_rankings.csv**: All evidential layers ranked by Gini feature importance.
* **output/prediction_area_plot.png**: High-resolution P-A plot chart.
