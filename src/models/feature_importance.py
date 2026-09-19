"""
Feature Importance & Geological Interpretability Engine.
"""

import numpy as np
import pandas as pd


def compute_feature_rankings(feature_importances: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Ranks evidential layers by their Gini/MDI contribution to the prospectivity model.
    """
    total = np.sum(feature_importances)
    norm_importances = feature_importances / total if total > 0 else feature_importances

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": norm_importances,
        "Percentage": norm_importances * 100.0
    })
    df = df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df
