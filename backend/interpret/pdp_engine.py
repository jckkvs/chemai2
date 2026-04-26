"""
backend/interpret/pdp_engine.py — 精緻化版 (PDP計算コア)

メモリ効率の高いサンプリングと並列化を用いた部分依存プロット（PDP）の計算エンジン。
"""

from typing import List, Dict, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


def compute_partial_dependence_efficient(
    model,
    X: pd.DataFrame,
    features: Union[int, str, List[Union[int, str]]],
    grid_resolution: Optional[int] = None,
    sample_size: Optional[int] = None,
    n_jobs: int = 1,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Compute partial dependence with memory-efficient sampling and parallelization
    """
    n_features = len(features) if isinstance(features, list) else 1
    n_samples = len(X)
    
    if grid_resolution is None:
        base_resolution = 50
        resolution_factor = max(0.5, 1.0 - (n_features - 1) * 0.1)
        grid_resolution = int(base_resolution * resolution_factor)
        grid_resolution = max(10, min(100, grid_resolution))
        logger.debug(f"Auto-set grid_resolution={grid_resolution}")
    
    if sample_size is not None and n_samples > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X
    
    if isinstance(features, (int, str)):
        features = [features]
    
    feature_names = []
    for f in features:
        if isinstance(f, int):
            feature_names.append(X.columns[f])
        elif isinstance(f, str) and f in X.columns:
            feature_names.append(f)
    
    if not feature_names:
        return {}
    
    results = {}
    
    if n_jobs != 1 and len(feature_names) > 1:
        max_workers = min(n_jobs if n_jobs > 0 else mp.cpu_count(), len(feature_names))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single_pdp, model, X_sample, feat, grid_resolution, random_state): feat
                for feat in feature_names
            }
            for future in as_completed(futures):
                feat = futures[future]
                try:
                    grid, pdp_vals = future.result(timeout=300)
                    results[feat] = (grid, pdp_vals)
                except Exception as e:
                    logger.error(f"PDP failed for '{feat}': {e}")
                    results[feat] = (np.array([]), np.array([]))
    else:
        for feat in feature_names:
            try:
                grid, pdp_vals = _compute_single_pdp(model, X_sample, feat, grid_resolution, random_state)
                results[feat] = (grid, pdp_vals)
            except Exception as e:
                logger.error(f"PDP failed for '{feat}': {e}")
                results[feat] = (np.array([]), np.array([]))
    
    return results


def _compute_single_pdp(
    model,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PDP for a single feature"""
    feat_idx = [X.columns.get_loc(feature)]
    pdp_result = partial_dependence(
        model, X, features=feat_idx, grid_resolution=grid_resolution, kind='average'
    )
    
    grid_values = pdp_result['values'][0]
    pdp_values = pdp_result['average'][0]
    
    valid_mask = np.isfinite(pdp_values) & np.isfinite(grid_values)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    return grid_values[valid_mask], pdp_values[valid_mask]


def plot_partial_dependence_interactive(
    pdp_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    feature_units: Optional[Dict[str, str]] = None,
    title: str = "Partial Dependence Plots"
):
    """Generate interactive Plotly PDP visualization"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if not pdp_results: return None
    
    n_features = len(pdp_results)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(pdp_results.keys()))
    
    for idx, (feat, (grid, pdp_vals)) in enumerate(pdp_results.items()):
        if len(grid) == 0: continue
        row, col = idx // cols + 1, idx % cols + 1
        fig.add_trace(
            go.Scatter(x=grid, y=pdp_vals, mode='lines+markers', name=feat),
            row=row, col=col
        )
        unit = feature_units.get(feat) if feature_units else None
        fig.update_xaxes(title_text=f"{feat}{' ('+unit+')' if unit else ''}", row=row, col=col)
    
    fig.update_layout(title=title, height=300 * rows, width=400 * cols, showlegend=False, template='plotly_white')
    return fig
