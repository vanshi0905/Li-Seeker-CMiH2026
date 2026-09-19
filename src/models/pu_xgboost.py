"""
Bagging Positive-Unlabeled (PU) Machine Learning Engine for Mineral Prospectivity Mapping.

Handles the fundamental absence of confirmed barren negative deposits using
ensemble bagging over unlabeled background pixels (Mordelet & Vert, 2014; Elkan & Noto, 2008).

Key Features:
- Tri-Model Diversity Ensemble: Dynamically blends XGBoost, CatBoost, and Random Forest
  when CatBoost is available, with seamless automatic fallback to XGBoost + Random Forest.
- Epistemic Predictive Uncertainty Quantification: Computes pixel-wise standard deviation
  across all bootstrap estimators: UQ(x, y) = std(p_i(x, y)).
- Dual-Raster Output: Delivers both mean posterior prospectivity P(y=1|x) and uncertainty rasters.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class BaggingPUMiner:
    """
    Bagging Positive-Unlabeled Classifier for LCT Pegmatite Prospectivity with
    Predictive Uncertainty Quantification and Tri-Model Ensembling.
    """
    def __init__(
        self,
        n_estimators: int = 25,
        neg_pos_ratio: float = 3.0,
        max_depth: int = 4,
        random_state: int = 42,
        use_xgboost: bool = True,
        ensemble_mode: str = "auto",
    ):
        """
        Parameters:
        -----------
        n_estimators: Number of bootstrap estimators in ensemble.
        neg_pos_ratio: Ratio of unlabeled pseudo-negatives to positive deposits per bag.
        max_depth: Tree maximum depth.
        random_state: Seed for reproducibility.
        use_xgboost: Backwards-compatible flag. If False, forces Random Forest only.
        ensemble_mode: "auto" / "tri_model" (XGBoost + CatBoost + RF), "xgboost", or "rf".
        """
        self.n_estimators = n_estimators
        self.neg_pos_ratio = neg_pos_ratio
        self.max_depth = max_depth
        self.random_state = random_state
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.ensemble_mode = ensemble_mode

        # Model storage
        self.models = []
        self.model_types = []
        self.feature_importances_ = None
        self.has_catboost = HAS_CATBOOST
        self.has_xgboost = HAS_XGBOOST

    def _determine_model_pool(self) -> list[str]:
        """Determines the cyclical sequence of model architectures to train."""
        if not self.use_xgboost:
            return ["rf"]

        if self.ensemble_mode == "xgboost" and HAS_XGBOOST:
            return ["xgboost"]
        elif self.ensemble_mode == "rf":
            return ["rf"]

        # Tri-Model ensemble with graceful fallback
        pool = []
        if HAS_XGBOOST:
            pool.append("xgboost")
        if HAS_CATBOOST:
            pool.append("catboost")
        pool.append("rf")
        return pool

    def fit(self, X_pos: np.ndarray, X_unlabeled: np.ndarray):
        """
        Trains ensemble of bootstrap classifiers over random unlabeled pseudo-negatives.

        Parameters:
        -----------
        X_pos: (N_pos, D) feature array of known positive deposits
        X_unlabeled: (N_unlabeled, D) feature array of background terrain pixels
        """
        rng = np.random.RandomState(self.random_state)
        n_pos = len(X_pos)
        n_unlabeled = len(X_unlabeled)
        n_neg_sample = int(min(n_pos * self.neg_pos_ratio, n_unlabeled))

        self.models = []
        self.model_types = []
        all_importances = []

        y_pos = np.ones(n_pos, dtype=int)
        y_neg = np.zeros(n_neg_sample, dtype=int)
        y_train = np.concatenate([y_pos, y_neg])

        model_pool = self._determine_model_pool()

        for i in range(self.n_estimators):
            # Subsample random pseudo-negatives from unlabeled background
            neg_indices = rng.choice(n_unlabeled, size=n_neg_sample, replace=False)
            X_neg = X_unlabeled[neg_indices]
            X_train = np.vstack([X_pos, X_neg])

            # Select model architecture in round-robin fashion from pool
            m_type = model_pool[i % len(model_pool)]
            seed = self.random_state + i

            if m_type == "xgboost":
                model = xgb.XGBClassifier(
                    n_estimators=60,
                    max_depth=self.max_depth,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=1,
                )
            elif m_type == "catboost":
                model = CatBoostClassifier(
                    iterations=60,
                    depth=min(self.max_depth, 6),
                    learning_rate=0.08,
                    verbose=0,
                    random_seed=seed,
                    thread_count=1,
                )
            else:  # Random Forest
                model = RandomForestClassifier(
                    n_estimators=60,
                    max_depth=self.max_depth,
                    random_state=seed,
                    n_jobs=1,
                )

            model.fit(X_train, y_train)
            self.models.append(model)
            self.model_types.append(m_type)

            # Collect feature importances normalized to sum to 1.0
            if hasattr(model, "feature_importances_"):
                imp = np.array(model.feature_importances_, dtype=np.float64)
            elif hasattr(model, "get_feature_importance"):
                imp = np.array(model.get_feature_importance(), dtype=np.float64)
            else:
                imp = np.ones(X_train.shape[1], dtype=np.float64)

            s = np.sum(imp)
            if s > 0:
                imp = imp / s
            all_importances.append(imp)

        self.feature_importances_ = np.mean(all_importances, axis=0)
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes both mean posterior probability and epistemic predictive uncertainty
        (pixel-wise standard deviation across bagging estimators):

        UQ(x, y) = std(p_i(x, y))

        Parameters:
        -----------
        X: (N, D) feature array

        Returns:
        --------
        mean_proba: (N,) array of mean prospectivity probabilities [0.0, 1.0]
        uncertainty: (N,) array of standard deviations across ensemble predictions [0.0, 0.5]
        """
        if not self.models:
            raise ValueError("Model has not been fitted yet.")

        n_samples = len(X)
        n_models = len(self.models)
        all_preds = np.zeros((n_models, n_samples), dtype=np.float32)

        for i, model in enumerate(self.models):
            all_preds[i] = model.predict_proba(X)[:, 1]

        mean_proba = np.mean(all_preds, axis=0)
        uncertainty = np.std(all_preds, axis=0)
        return mean_proba, uncertainty

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes mean posterior probability of mineral prospectivity P(y=1|x)
        averaged across all bootstrap models.
        """
        mean_proba, _ = self.predict_with_uncertainty(X)
        return mean_proba

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Computes predictive uncertainty (standard deviation across bootstrap models).
        """
        _, uncertainty = self.predict_with_uncertainty(X)
        return uncertainty

    def predict_rasters(
        self, X_all: np.ndarray, valid_mask: np.ndarray, nrows: int, ncols: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates 2D prospectivity map and 2D uncertainty raster across the spatial grid.

        Returns:
        --------
        prospectivity_map: 2D numpy array [0.0, 1.0], NaN on invalid/masked pixels
        uncertainty_map: 2D numpy array [0.0, 0.5], NaN on invalid/masked pixels
        """
        mean_proba, uncertainty = self.predict_with_uncertainty(X_all)

        prospectivity_map = np.full((nrows, ncols), np.nan, dtype=np.float32)
        uncertainty_map = np.full((nrows, ncols), np.nan, dtype=np.float32)

        if valid_mask.ndim == 1 and len(valid_mask) == nrows * ncols:
            mask_2d = valid_mask.reshape((nrows, ncols))
        else:
            mask_2d = valid_mask

        if len(mean_proba) == nrows * ncols:
            p_2d = mean_proba.reshape((nrows, ncols)).copy()
            p_2d[~mask_2d] = np.nan
            prospectivity_map = p_2d
            u_2d = uncertainty.reshape((nrows, ncols)).copy()
            u_2d[~mask_2d] = np.nan
            uncertainty_map = u_2d
        else:
            prospectivity_map[mask_2d] = mean_proba
            uncertainty_map[mask_2d] = uncertainty

        return prospectivity_map, uncertainty_map
