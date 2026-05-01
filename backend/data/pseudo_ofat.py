# backend/data/pseudo_ofat.py
"""
Pseudo-OFAT (One-Factor-At-a-Time) exploration and duplicate sample detection.

Implements:
  1. Pseudo-OFAT: Vary one factor at a time, keep others constant,
     to explore its effect on the target.
  2. Duplicate sample detection: Find samples with nearly identical features
     but different target values (likely experimental error or pseudo-OFAT opportunity).
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any
import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# 1. Pseudo-OFAT Exploration
# ──────────────────────────────────────────────────────

def generate_pseudo_ofat_suggestions(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    n_suggestions: int = 10,
    numeric_tolerance: float = 0.1,  # 10% tolerance for "constant"
) -> List[Dict[str, Any]]:
    """
    Generate pseudo-OFAT experimental suggestions.

    For each feature (factor), find opportunities to vary it while
    keeping other features constant, to explore its effect on target.

    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_cols: List of feature column names
        n_suggestions: Number of suggestions to return
        numeric_tolerance: Tolerance for considering values "constant" (as ratio of std)

    Returns:
        List of suggestion dicts with keys:
          - factor: Feature name to vary
          - base_sample: Index of base sample
          - current_value: Current factor value
          - suggested_values: List of values to try
          - other_features_constant: Dict of other feature values (should be kept constant)
          - current_target: Current target value
          - priority: Priority score (higher = more valuable)
    """
    suggestions = []
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    for factor in numeric_cols:
        # Group by "constant" other features
        other_numeric = [c for c in numeric_cols if c != factor]
        other_categorical = [c for c in categorical_cols if c != factor]

        # For each sample, find similar samples where only this factor varies
        for idx in df.index[:min(100, len(df))]:  # Limit to first 100 for performance
            base_row = df.iloc[idx]

            # Find samples where other features are "constant"
            mask = pd.Series(True, index=df.index)
            for col in other_numeric:
                col_std = df[col].std()
                if col_std > 1e-10:
                    tolerance = numeric_tolerance * col_std
                    mask &= (df[col] - base_row[col]).abs() <= tolerance
                else:
                    mask &= (df[col] == base_row[col])

            for col in other_categorical:
                mask &= (df[col] == base_row[col])

            # Exclude the base sample itself
            mask.iloc[idx] = False

            similar = df[mask]
            if len(similar) < 2:
                continue  # Not enough similar samples

            # Check if factor varies in similar samples
            factor_values = similar[factor].unique()
            if len(factor_values) < 2:
                continue  # Factor doesn't vary

            # Calculate target trend
            factor_target = similar.groupby(factor)[target_col].mean()
            if len(factor_target) < 2:
                continue

            # Priority: number of similar samples * variance of target
            priority = len(similar) * similar[target_col].var()

            suggestions.append({
                "factor": factor,
                "base_sample": int(idx),
                "base_features": {c: base_row[c] for c in feature_cols},
                "factor_values": sorted(factor_values.tolist()),
                "target_means": factor_target.to_dict(),
                "n_similar": len(similar),
                "priority": float(priority),
            })

    # Sort by priority and return top N
    suggestions.sort(key=lambda x: x["priority"], reverse=True)
    return suggestions[:n_suggestions]


def find_pseudo_ofat_opportunities(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    n_neighbors: int = 5,
    distance_threshold: float = 0.2,  # Normalized distance threshold
) -> List[Dict[str, Any]]:
    """
    Find pseudo-OFAT opportunities using nearest neighbor search.

    For each sample, find neighbors where only one feature differs significantly.
    These are opportunities to explore that feature's effect.

    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_cols: Feature column names
        n_neighbors: Number of neighbors to check
        distance_threshold: Threshold for "small" distance (normalized)

    Returns:
        List of opportunities
    """
    from sklearn.preprocessing import StandardScaler

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)

    opportunities = []
    for i in range(len(df)):
        for j_idx in range(1, len(indices[i])):  # Skip self (j_idx=0)
            neighbor_idx = indices[i][j_idx]
            dist = distances[i][j_idx]

            if dist > distance_threshold:
                break

            # Check how many features differ
            diff_features = []
            for k, col in enumerate(feature_cols):
                if abs(X[i, k] - X[neighbor_idx, k]) > 0.1:  # Normalized difference
                    diff_features.append(col)

            if len(diff_features) == 1:
                # Only one feature differs - pseudo-OFAT opportunity!
                factor = diff_features[0]
                opportunities.append({
                    "base_sample": int(i),
                    "neighbor_sample": int(neighbor_idx),
                    "factor": factor,
                    "base_value": float(df.iloc[i][factor]),
                    "neighbor_value": float(df.iloc[neighbor_idx][factor]),
                    "base_target": float(df.iloc[i][target_col]),
                    "neighbor_target": float(df.iloc[neighbor_idx][target_col]),
                    "distance": float(dist),
                })

    return opportunities


# ──────────────────────────────────────────────────────
# 2. Duplicate Sample Detection
# ──────────────────────────────────────────────────────

def detect_duplicate_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    numeric_threshold: float = 0.05,  # 5% of std as "same"
    categorical_exact: bool = True,
    target_diff_threshold: float = 0.1,  # Minimum target difference to flag
) -> Dict[str, Any]:
    """
    Detect duplicate or near-duplicate samples with different target values.

    These are likely:
      - Experimental errors (same conditions but different results)
      - Pseudo-OFAT opportunities (intentionally varying one factor)

    Args:
        df: Input DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        numeric_threshold: Threshold for numeric features (as ratio of std)
        categorical_exact: If True, categorical features must match exactly
        target_diff_threshold: Minimum target difference to flag

    Returns:
        Dict with keys:
          - duplicate_pairs: List of (idx1, idx2, similarity, target_diff)
          - suspicious_samples: List of sample indices with high duplicate potential
          - summary: Text summary
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    # Normalize features
    numeric_cols = [c for c in feature_cols if c in df.select_dtypes(include=[np.number]).columns]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Build feature matrix
    X_numeric = df[numeric_cols].values if numeric_cols else np.array([]).reshape(len(df), 0)
    X_categorical = np.array([]).reshape(len(df), 0)

    if categorical_cols and categorical_exact:
        # One-hot encode categorical
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(df[categorical_cols])

    # Combine (normalize numeric)
    if X_numeric.shape[1] > 0:
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X_numeric)
    X = np.hstack([X_numeric, X_categorical])

    # Compute pairwise similarities
    similarities = cosine_similarity(X)

    # Find duplicate pairs
    duplicate_pairs = []
    suspicious_indices = set()

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            sim = similarities[i, j]

            if sim > (1.0 - numeric_threshold):
                # Check target difference
                target_diff = abs(df.iloc[i][target_col] - df.iloc[j][target_col])

                if target_diff > target_diff_threshold:
                    duplicate_pairs.append({
                        "idx1": int(i),
                        "idx2": int(j),
                        "similarity": float(sim),
                        "target_diff": float(target_diff),
                        "sample1": df.iloc[i][feature_cols + [target_col]].to_dict(),
                        "sample2": df.iloc[j][feature_cols + [target_col]].to_dict(),
                    })
                    suspicious_indices.add(i)
                    suspicious_indices.add(j)

    # Summary
    summary = f"Found {len(duplicate_pairs)} duplicate pairs with target differences.\n"
    if duplicate_pairs:
        summary += "Top 3 most suspicious:\n"
        sorted_pairs = sorted(duplicate_pairs, key=lambda x: -x["similarity"])[:3]
        for p in sorted_pairs:
            summary += f"  - Samples {p['idx1']} & {p['idx2']}: " \
                      f"similarity={p['similarity']:.3f}, " \
                      f"target_diff={p['target_diff']:.3f}\n"

    return {
        "duplicate_pairs": duplicate_pairs,
        "suspicious_samples": list(suspicious_indices),
        "summary": summary,
        "n_pairs": len(duplicate_pairs),
    }


def find_near_duplicates(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_neighbors: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find near-duplicate samples using nearest neighbor search.

    Returns list of samples that have very similar feature values
    but potentially different target values.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    numeric_cols = [c for c in feature_cols if c in df.select_dtypes(include=[np.number]).columns]

    if not numeric_cols:
        return []

    X = StandardScaler().fit_transform(df[numeric_cols])

    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(df)),
                            algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)

    results = []
    for i in range(len(df)):
        neighbors = []
        for j_idx in range(1, len(indices[i])):  # Skip self
            neighbor_idx = indices[i][j_idx]
            dist = distances[i][j_idx]

            if dist > 0.1:  # Threshold for "near" duplicate
                break

            target_diff = abs(df.iloc[i][target_col] - df.iloc[neighbor_idx][target_col])
            if target_diff > 0.05:  # Meaningful target difference
                neighbors.append({
                    "neighbor_idx": int(neighbor_idx),
                    "distance": float(dist),
                    "target_diff": float(target_diff),
                })

        if neighbors:
            results.append({
                "sample_idx": int(i),
                "sample": df.iloc[i][feature_cols + [target_col]].to_dict(),
                "near_duplicates": neighbors,
            })

    return results


if __name__ == "__main__":
    # Simple test
    df = pd.DataFrame({
        "temp": [100, 100, 100, 150, 150, 100, 100],
        "pressure": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "solvent": ["A", "A", "A", "A", "A", "B", "A"],
        "yield": [0.8, 0.82, 0.3, 0.9, 0.91, 0.85, 0.81],
    })

    # Pseudo-OFAT
    suggestions = generate_pseudo_ofat_suggestions(df, "yield", ["temp", "pressure", "solvent"])
    print(f"Pseudo-OFAT suggestions: {len(suggestions)}")
    for s in suggestions[:3]:
        print(f"  - Factor: {s['factor']}, Base: {s['base_sample']}, "
              f"Values: {s['factor_values']}")

    # Duplicate detection
    result = detect_duplicate_samples(df, ["temp", "pressure", "solvent"], "yield")
    print(f"\nDuplicate detection: {result['n_pairs']} pairs")
    print(result["summary"])
