"""
Unit tests for Bagging Positive-Unlabeled (PU) XGBoost/Random Forest model.
"""

import numpy as np
import pytest
from src.models.pu_xgboost import BaggingPUMiner


def test_pu_miner_fit_and_predict():
    np.random.seed(42)
    # Synthetic 5-dimensional feature space
    n_pos = 15
    n_unlabeled = 200
    n_features = 5

    # Positive deposits have elevated values in feature 0 and 1
    X_pos = np.random.normal(loc=3.0, scale=0.5, size=(n_pos, n_features))
    X_unlabeled = np.random.normal(loc=0.0, scale=1.0, size=(n_unlabeled, n_features))

    miner = BaggingPUMiner(n_estimators=5, neg_pos_ratio=2.0, max_depth=3, random_state=42)
    miner.fit(X_pos, X_unlabeled)

    assert miner.feature_importances_ is not None
    assert len(miner.feature_importances_) == n_features

    # Predict on unseen test data
    test_pos = np.random.normal(loc=3.0, scale=0.5, size=(5, n_features))
    test_neg = np.random.normal(loc=-1.0, scale=0.5, size=(5, n_features))

    prob_pos = miner.predict_proba(test_pos)
    prob_neg = miner.predict_proba(test_neg)

    assert np.all((prob_pos >= 0.0) & (prob_pos <= 1.0))
    assert np.all((prob_neg >= 0.0) & (prob_neg <= 1.0))
    # High-signal positive samples should have higher prospectivity than negative background
    assert np.mean(prob_pos) > np.mean(prob_neg)
