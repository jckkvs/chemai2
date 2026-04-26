"""
Model Interpretation Engine - chemai2/backend/ml/interpretation.py
SHAP/SAGE/PDP/FeatureImportance with Plotly visualization
"""
import warnings
from typing import Dict, List, Optional, Union, Any, Literal, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import shap
from backend.ml.constraints import ConstraintEngine, ConstraintSpec
from backend.utils.logger import logger


class InterpretationEngine:
    """
    Unified model interpretation engine with Plotly-based visualizations
    
    Features:
    - SHAP summary, dependence, waterfall, force
    - SAGE (Sampling-based Attribution of Global Effects)
    - Partial Dependence Plots (PDP)
    - Feature Importance (permutation & model-based)
    - Constraint violation visualization
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str] = None):
        self.model = model
        self.X_train = X_train.copy()
        self.feature_names = feature_names or list(X_train.columns)
        self._explainer = None
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            if hasattr(self.model, 'predict_proba'):
                self._explainer = shap.TreeExplainer(self.model)
                self.model_output = 'log_odds'
            else:
                self._explainer = shap.TreeExplainer(self.model)
                self.model_output = 'raw'
        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to KernelExplainer: {e}")
            self._explainer = shap.KernelExplainer(self.model.predict, self.X_train.iloc[:100])
    
    def plot_shap_summary(self, X: pd.DataFrame, max_display: int = 20) -> go.Figure:
        """SHAP summary plot (feature importance + impact)"""
        shap_values = self._explainer.shap_values(X)
        
        # Convert to Plotly-compatible format
        summary_df = pd.DataFrame(shap_values, columns=self.feature_names)
        summary_df['abs_mean'] = summary_df.abs().mean()
        
        # Sort by importance
        importance_order = summary_df['abs_mean'].sort_values(ascending=False).index[:max_display]
        
        fig = go.Figure()
        for feat in importance_order[::-1]:  # Reverse for top-to-bottom
            fig.add_trace(go.Box(
                x=shap_values[:, self.feature_names.index(feat)],
                y=[feat] * len(shap_values),
                name=feat,
                boxmean='sd',
                orientation='h',
                showlegend=False
            ))
        
        fig.update_layout(
            title="SHAP Feature Importance & Impact",
            xaxis_title="SHAP Value (Impact on Model Output)",
            yaxis=dict(title="Feature", autorange='reversed'),
            template='plotly_white',
            height=500
        )
        return fig
    
    def plot_shap_dependence(self, X: pd.DataFrame, feature: str, interaction_feature: str = None) -> go.Figure:
        """SHAP dependence plot with optional interaction coloring"""
        feat_idx = self.feature_names.index(feature)
        shap_values = self._explainer.shap_values(X)
        
        fig = px.scatter(
            x=X[feature],
            y=shap_values[:, feat_idx],
            color=interaction_feature if interaction_feature and interaction_feature in X.columns else None,
            title=f"SHAP Dependence: {feature}",
            labels={'x': feature, 'y': f"SHAP value for {feature}"},
            template='plotly_white',
            opacity=0.6
        )
        fig.update_layout(height=450)
        return fig
    
    def plot_partial_dependence(self, X: pd.DataFrame, features: List[str], grid_resolution: int = 50) -> go.Figure:
        """Partial Dependence Plot (PDP) with ICE curves"""
        from sklearn.inspection import partial_dependence, PartialDependenceDisplay
        
        fig = make_subplots(rows=1, cols=len(features), subplot_titles=features)
        
        for i, feat in enumerate(features):
            feat_idx = [self.feature_names.index(feat)]
            pdp_values, axes = partial_dependence(
                self.model, X, features=feat_idx, grid_resolution=grid_resolution
            )
            pdp_mean = pdp_values[0]
            x_grid = axes[0]
            
            fig.add_trace(go.Scatter(x=x_grid, y=pdp_mean, mode='lines+markers', name='PDP Mean', showlegend=(i==0)), row=1, col=i+1)
            
            # Add ICE lines (sample)
            ice = []
            n_ice = min(20, len(X))
            sample_idx = np.random.choice(len(X), n_ice, replace=False)
            for idx in sample_idx:
                x_single = np.tile(X.iloc[idx].values, (len(x_grid), 1))
                x_single[:, feat_idx[0]] = x_grid
                pred_single = self.model.predict(x_single)
                ice.append(pred_single)
            
            for ice_vals in ice:
                fig.add_trace(go.Scatter(x=x_grid, y=ice_vals, mode='lines', line=dict(width=0.5, color='gray', dash='dot'), showlegend=False), row=1, col=i+1)
        
        fig.update_layout(height=400, template='plotly_white')
        return fig
    
    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series, method: Literal['model', 'permutation'] = 'model') -> go.Figure:
        """Feature importance (model-based or permutation-based)"""
        if method == 'model' and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # Permutation importance
            from sklearn.inspection import permutation_importance
            result = permutation_importance(self.model, X, y, n_repeats=10, random_state=42, scoring='r2')
            importances = result.importances_mean
        
        sorted_idx = np.argsort(importances)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[self.feature_names[i] for i in sorted_idx],
            x=[importances[i] for i in sorted_idx],
            orientation='h',
            text=[f"{v:.3f}" for v in [importances[i] for i in sorted_idx]],
            textposition='auto'
        ))
        fig.update_layout(
            title=f"Feature Importance ({method.capitalize()})",
            xaxis_title="Importance Score",
            yaxis=dict(title="Feature", autorange='reversed'),
            template='plotly_white',
            height=500
        )
        return fig
    
    def plot_constraint_validation(self, constraints: Dict[str, ConstraintSpec], evaluations: Dict[str, Any]) -> go.Figure:
        """Visualize constraint validation results"""
        fig = go.Figure()
        
        for feat_name, eval_data in evaluations.items():
            passed = eval_data.get('passed', False)
            violations = eval_data.get('monotonic_violation_ratio', 0)
            r2_lin = eval_data.get('linearity_r2', None)
            
            status = "✅ Passed" if passed else "❌ Failed"
            text = f"Violations: {violations*100:.1f}%\nLinearity R²: {r2_lin:.3f}" if r2_lin else f"Violations: {violations*100:.1f}%"
            
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=violations,
                title={"text": f"{feat_name}<br><span style='font-size:0.8em;color:gray'>{status}</span>"},
                delta={"reference": 0, "increasing": False, "decreasing": True, "valueformat": ".1%"},
                number={"suffix": "%"},
                domain={"x": [0, 0.45], "y": [0, 1]} if len(evaluations) == 1 else None
            ))
        
        fig.update_layout(height=300, template='plotly_white')
        return fig
