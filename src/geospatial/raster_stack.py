"""
Multi-Modal Evidential Feature Stacker.

Integrates Sentinel-2 spectral indices, Crosta PCA, structural geology,
aeromagnetics, and NGCM geochemistry into an aligned tabular feature matrix.
"""

import numpy as np
from ..remote_sensing.indices import compute_spectral_feature_cube
from ..remote_sensing.crosta_pca import run_crosta_al_oh_pca
from ..remote_sensing.preprocessing import apply_scl_mask, filter_bare_rock_pediment


class EvidentialRasterStack:
    """
    Manages multi-source raster alignment, feature matrix extraction, and mask filtering.
    """
    def __init__(self, dataset: dict, ndvi_threshold: float = 0.28):
        self.grid = dataset['grid']
        self.occurrences = dataset.get('all_ground_truth') or dataset.get('occurrences', [])
        self.bands = dataset['bands']
        self.scl = dataset['scl']
        self.dem = dataset['dem']
        self.slope = dataset['slope']
        self.fault_dist_km = dataset['fault_dist_km']
        self.lineament_density = dataset['lineament_density']
        self.granite_dist_km = dataset['granite_dist_km']
        self.aeromag_rtp = dataset['aeromag_rtp']
        self.ngcm_li = dataset['ngcm_li']
        self.ngcm_k_rb = dataset['ngcm_k_rb']

        # 1. Compute Sentinel-2 Spectral Indices
        self.spectral_features = compute_spectral_feature_cube(self.bands)

        # 2. Validity Mask: Cloud/shadow free + Bare-rock pediment (low NDVI)
        scl_valid = apply_scl_mask(self.scl)
        bare_valid = filter_bare_rock_pediment(self.spectral_features['ndvi'], ndvi_threshold=ndvi_threshold)
        self.valid_mask = scl_valid & bare_valid

        # 3. Crosta Technique PCA on Al-OH bands [B2, B4, B11, B12]
        self.pc_al_oh, self.best_pc_idx, self.pca_loadings, self.pca_var_ratio = run_crosta_al_oh_pca(
            self.bands['B2'], self.bands['B4'], self.bands['B11'], self.bands['B12'],
            mask=self.valid_mask
        )

        # 4. Assemble All Evidential Layers
        self.feature_rasters = {
            # Optical Spectral Indices (Cardoso-Fernandes & Alteration)
            "al_oh_mica_ratio": self.spectral_features['al_oh_ratio'],
            "ndci_clay": self.spectral_features['ndci'],
            "cardoso_pi_1": self.spectral_features['pi_1'],
            "cardoso_pi_2": self.spectral_features['pi_2'],
            "cardoso_lpi": self.spectral_features['lpi'],
            "cardoso_lmdr": self.spectral_features['lmdr'],
            "cardoso_ehi": self.spectral_features['ehi'],
            "fe3_oxide": self.spectral_features['fe3_ratio'],
            "crosta_pc_al_oh": self.pc_al_oh,

            # REE & Rare-Metal Spectral Exploration Layers
            "ree_composite_index": self.spectral_features['ree_index'],
            "ree_nd_absorption": self.spectral_features['ree_neodymium'],

            # Structural & Magmatic Controls
            "fault_distance_km": self.fault_dist_km,
            "lineament_density": self.lineament_density,
            "granite_contact_dist_km": self.granite_dist_km,

            # Geophysics & Geomorphology
            "aeromag_rtp_residual": self.aeromag_rtp,
            "elevation_dem": self.dem,
            "slope_deg": self.slope,

            # Geochemistry (NGCM Stream Sediments)
            "ngcm_li_ppm": self.ngcm_li,
            "ngcm_k_rb_ratio": self.ngcm_k_rb,
        }

        # Multi-Sensor & Advanced Geophysical Exploration Layers (Airborne Radiometrics, Gravity, ASTER TIR, SAR)
        if 'radiometric_k_pct' in dataset:
            self.feature_rasters["radiometric_k_pct"] = dataset['radiometric_k_pct']
        if 'radiometric_u_th_ratio' in dataset:
            self.feature_rasters["radiometric_u_th_ratio"] = dataset['radiometric_u_th_ratio']
        if 'radiometric_ternary_index' in dataset:
            self.feature_rasters["radiometric_ternary_index"] = dataset['radiometric_ternary_index']
        if 'bouguer_gravity_mgal' in dataset:
            self.feature_rasters["bouguer_gravity_mgal"] = dataset['bouguer_gravity_mgal']
        if 'gravity_gradient_fvd' in dataset:
            self.feature_rasters["gravity_gradient_fvd"] = dataset['gravity_gradient_fvd']
        if 'aster_quartz_index_qi' in dataset:
            self.feature_rasters["aster_quartz_index_qi"] = dataset['aster_quartz_index_qi']
        if 'sar_cband_vh_db' in dataset:
            self.feature_rasters["sar_cband_vh_db"] = dataset['sar_cband_vh_db']
        if 'sar_roughness_ratio' in dataset:
            self.feature_rasters["sar_roughness_ratio"] = dataset['sar_roughness_ratio']

        self.feature_names = list(self.feature_rasters.keys())

    def get_feature_matrix(self, apply_mask: bool = True):
        """
        Extracts 2D feature matrix X of shape (N_pixels, D_features).
        If apply_mask is True, returns only unmasked bare-rock/outcrop pixels.
        """
        mask = self.valid_mask if apply_mask else np.ones((self.grid.nrows, self.grid.ncols), dtype=bool)
        n_pixels = int(np.sum(mask))

        X = np.zeros((n_pixels, len(self.feature_names)), dtype=np.float32)
        for idx, feat_name in enumerate(self.feature_names):
            arr = self.feature_rasters[feat_name]
            # Replace NaNs with column medians for stability
            val = arr[mask]
            nan_sel = np.isnan(val)
            if np.any(nan_sel):
                med = np.nanmedian(val)
                val[nan_sel] = med
            X[:, idx] = val

        return X, mask, self.feature_names

    def get_ground_truth_pixel_indices(self):
        """
        Returns (row, col) coordinates of all positive ground-truth deposits.
        """
        pos_pixels = []
        for occ in self.occurrences:
            r, c = self.grid.coord_to_pixel(occ["latitude"], occ["longitude"])
            pos_pixels.append((r, c))
        return pos_pixels
