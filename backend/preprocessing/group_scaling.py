"""
backend/preprocessing/group_scaling.py

Group scaling for features with the same unit/system.

Implements:
  - Group features by unit/system
  - For each group, find the maximum standard deviation
  - Scale all features in the group using this maximum std
  - Keep individual means for each feature
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class GroupScaler(BaseEstimator, TransformerMixin):
    """
    Scale features by group using the maximum std in each group.

    For features sharing the same unit/system, scale them together:
    - Keep individual means for each feature
    - Use the maximum std in the group for scaling all features

    Parameters:
    -----------
    groups : dict, default=None
        Dictionary mapping group_name -> list of feature indices/names.
        If None, all features are treated as one group.

    copy : bool, default=True
        If False, try to avoid copying and do inplace scaling instead.

    Attributes:
    ----------
    group_means_ : dict
        Mean for each feature, keyed by feature index/name
    group_stds_ : dict
        Max std for each group, keyed by group name
    n_features_in_ : int
        Number of features seen during fit
    """

    def __init__(self, groups: Optional[Dict[str, List]] = None, copy: bool = True):
        self.groups = groups
        self.copy = copy

    def fit(self, X, y=None):
        """
        Compute the mean and max std for each group.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the scaling parameters.
        y : None
            Ignored, present for API compatibility.

        Returns:
        -------
        self : object
            Fitted scaler
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)
        self.n_features_in_ = X.shape[1]

        if self.groups is None:
            # Treat all features as one group
            self.groups = {"default": list(range(X.shape[1]))}

        self.group_means_ = {}
        self.group_stds_ = {}

        for group_name, feature_list in self.groups.items():
            # Resolve feature indices to actual column indices
            if isinstance(X, pd.DataFrame):
                indices = [X.columns.get_loc(f) if isinstance(f, str) else f for f in feature_list]
            else:
                indices = [int(f) for f in feature_list]

            # Compute means for each feature (individual)
            group_data = X[:, indices] if not isinstance(X, pd.DataFrame) else X.iloc[:, indices].values
            means = np.mean(group_data, axis=0)
            for i, idx in enumerate(indices):
                self.group_means_[idx] = means[i] if hasattr(means, '__iter__') else means

            # Find max std in the group
            stds = np.std(group_data, axis=0)
            max_std = np.max(stds) if len(stds) > 0 else 1.0
            if max_std < 1e-10:
                max_std = 1.0  # Avoid division by zero
            self.group_stds_[group_name] = max_std

        return self

    def transform(self, X):
        """
        Scale the data using group-based scaling.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform

        Returns:
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Transformed data
        """
        check_is_fitted(self, ["group_means_", "group_stds_"])
        X = check_array(X, dtype=np.float64, copy=self.copy)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but GroupScaler "
                f"was fitted with {self.n_features_in_} features"
            )

        X_scaled = X.copy()

        for group_name, feature_list in self.groups.items():
            if group_name not in self.group_stds_:
                continue

            max_std = self.group_stds_[group_name]
            indices = [int(f) for f in feature_list]

            for idx in indices:
                mean_val = self.group_means_.get(idx, 0.0)
                X_scaled[:, idx] = (X[:, idx] - mean_val) / max_std

        return X_scaled

    def inverse_transform(self, X):
        """
        Inverse transform the data back to original scale.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Transformed data

        Returns:
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original scale
        """
        check_is_fitted(self, ["group_means_", "group_stds_"])
        X = check_array(X, dtype=np.float64, copy=self.copy)

        X_original = X.copy()

        for group_name, feature_list in self.groups.items():
            if group_name not in self.group_stds_:
                continue

            max_std = self.group_stds_[group_name]
            indices = [int(f) for f in feature_list]

            for idx in indices:
                mean_val = self.group_means_.get(idx, 0.0)
                X_original[:, idx] = X[:, idx] * max_std + mean_val

        return X_original

    def get_feature_names_out(self, input_features=None):
        """Used to get feature names for sklearn compatibility."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            return input_features
        return [f"x{i}" for i in range(self.n_features_in_)]
