"""
EDA Visualization Engine - chemai2/backend/eda/analyzer.py
Plotly-based exploratory data analysis with automatic type detection
"""
import warnings
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from backend.core.config import settings
from backend.utils.logger import logger


@dataclass
class EDAConfig:
    """Configuration for EDA visualization"""
    target_column: Optional[str] = None
    task_type: Literal['regression', 'classification', 'unsupervised'] = 'unsupervised'
    max_points_plot: int = 5000  # Downsample for performance
    random_state: int = 42
    color_scheme: str = 'Plotly'
    include_outliers: bool = True
    show_marginals: bool = True


class EDAAnalyzer:
    """
    Comprehensive EDA engine using Plotly for interactive visualizations
    
    Supports:
    - Correlation heatmap (numeric + categorical encoding)
    - Pairwise scatter/histogram matrix
    - Distribution plots with KDE/Box/Violin
    - Missing value matrix & bar chart
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    """
    
    def __init__(self, config: EDAConfig = None):
        self.config = config or EDAConfig()
        self._rng = np.random.default_rng(self.config.random_state)
    
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Generate interactive correlation heatmap with automatic encoding"""
        # Encode categorical/binary to numeric for correlation
        df_num = df.copy()
        for col in df_num.columns:
            if df_num[col].dtype == 'object' or df_num[col].dtype.name == 'category':
                # Target encoding or simple label encoding for correlation
                unique_vals = df_num[col].dropna().unique()
                if len(unique_vals) <= 2:
                    df_num[col] = df_num[col].map({v: i for i, v in enumerate(unique_vals)}).astype(float)
                else:
                    # Drop high-cardinality categorical from correlation
                    df_num = df_num.drop(columns=[col])
        
        # Downsample if too large
        if len(df_num) > self.config.max_points_plot:
            df_num = df_num.sample(self.config.max_points_plot, random_state=self.config.random_state)
        
        corr = df_num.corr(method='pearson')
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.where(mask)
        
        fig = px.imshow(
            corr_masked,
            labels=dict(color="Correlation Coefficient"),
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Correlation Matrix (Pearson)",
            template='plotly_white'
        )
        fig.update_layout(width=800, height=700)
        return fig
    
    def generate_pairplot(self, df: pd.DataFrame, columns: List[str] = None) -> go.Figure:
        """Generate pairwise scatter/histogram matrix (up to 6 columns for performance)"""
        cols = columns or df.select_dtypes(include=['number']).columns.tolist()[:6]
        if len(cols) > 6:
            cols = cols[:6]
            warnings.warn(f"Limited pairplot to first 6 numeric columns: {cols}")
        
        df_plot = df[cols].copy()
        if len(df_plot) > 2000:
            df_plot = df_plot.sample(2000, random_state=self.config.random_state)
        
        fig = px.scatter_matrix(
            df_plot,
            dimensions=cols,
            title="Pairwise Scatter Plot Matrix",
            template='plotly_white',
            opacity=0.7
        )
        fig.update_traces(diagonal_visible=True)
        fig.update_layout(width=900, height=900)
        return fig
    
    def generate_distribution_plots(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, go.Figure]:
        """Generate distribution plots (histogram + KDE + box) per column"""
        cols = columns or df.select_dtypes(include=['number']).columns.tolist()
        figures = {}
        
        for col in cols:
            data = df[col].dropna()
            if len(data) < 10:
                continue
                
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f"{col} - Histogram", f"{col} - Box Plot",
                                f"{col} - Violin", f"{col} - QQ Plot"],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram + KDE
            fig.add_trace(go.Histogram(x=data, name='Histogram', nbinsx=30), row=1, col=1)
            kde_x = np.linspace(data.min(), data.max(), 100)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            fig.add_trace(go.Scatter(x=kde_x, y=kde(kde_x), name='KDE', line=dict(color='red')), row=1, col=1)
            
            # Box plot
            fig.add_trace(go.Box(y=data, name='Box', showlegend=False), row=1, col=2)
            
            # Violin
            fig.add_trace(go.Violin(y=data, name='Violin', showlegend=False, points='all', pointpos=0), row=2, col=1)
            
            # QQ Plot
            import scipy.stats as stats
            stats.probplot(data, dist="norm", plot=lambda x, y: fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='QQ', showlegend=False), row=2, col=2))
            
            fig.update_layout(height=600, width=1000, showlegend=True)
            figures[col] = fig
        
        return figures
    
    def generate_missing_value_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Generate missing value matrix and summary bar chart"""
        missing = df.isnull()
        n_missing = missing.sum()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Missing Value Matrix", "Missing Value Count"],
            specs=[[{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Matrix
        fig.add_trace(
            go.Heatmap(z=missing.T.astype(int), x=df.index, y=df.columns, 
                       colorscale=[[0, 'white'], [1, 'red']], showscale=False),
            row=1, col=1
        )
        
        # Bar
        fig.add_trace(
            go.Bar(x=n_missing.index, y=n_missing.values, name='Missing Count'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, width=1200, title="Missing Value Analysis")
        return fig
    
    def generate_dimensionality_reduction(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, go.Figure]:
        """Generate PCA, t-SNE, and UMAP plots"""
        # Preprocess
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=['number']))
        
        # Downsample for t-SNE/UMAP
        n = min(self.config.max_points_plot, len(X_scaled))
        idx = self._rng.choice(len(X_scaled), size=n, replace=False)
        X_sample = X_scaled[idx]
        y_sample = y.iloc[idx] if y is not None else None
        
        figures = {}
        
        # PCA
        pca = PCA(n_components=2, random_state=self.config.random_state)
        X_pca = pca.fit_transform(X_sample)
        fig_pca = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y_sample, 
                             title=f"PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})",
                             labels={'x': f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", 
                                     'y': f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"},
                             template='plotly_white')
        figures['PCA'] = fig_pca
        
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=self.config.random_state, n_iter=1000)
        X_tsne = tsne.fit_transform(X_sample)
        fig_tsne = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y_sample,
                              title="t-SNE (Perplexity=30)", template='plotly_white')
        figures['t-SNE'] = fig_tsne
        
        # UMAP (requires umap-learn, fallback if not installed)
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=self.config.random_state, n_neighbors=15)
            X_umap = reducer.fit_transform(X_sample)
            fig_umap = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=y_sample,
                                  title="UMAP (n_neighbors=15)", template='plotly_white')
            figures['UMAP'] = fig_umap
        except ImportError:
            logger.warning("umap-learn not installed. Skipping UMAP visualization.")
        
        return figures
