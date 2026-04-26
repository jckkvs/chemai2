"""
Constraint Compliance Report Generator - backend/evaluation/constraint_report.py

Generates interactive reports for monotonicity/linearity constraint compliance
using Plotly visualizations and statistical summaries.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.utils.logger import logger

logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolationStats:
    """Statistics for constraint violations"""
    feature_name: str
    constraint_type: Literal['monotonic', 'linearity']
    direction: Optional[Literal['increasing', 'decreasing']] = None
    sigma_range: float = 3.0
    strength: Literal['strong', 'weak'] = 'weak'
    
    # Violation metrics
    total_checks: int = 0
    violations: int = 0
    violation_ratio: float = 0.0
    max_violation_magnitude: float = 0.0
    mean_violation_magnitude: float = 0.0
    
    # Linearity-specific
    r_squared: Optional[float] = None
    rmse: Optional[float] = None
    
    @property
    def compliance_rate(self) -> float:
        """Return compliance rate (1 - violation_ratio)"""
        return 1.0 - self.violation_ratio
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization"""
        return {
            'feature_name': self.feature_name,
            'constraint_type': self.constraint_type,
            'direction': self.direction,
            'sigma_range': self.sigma_range,
            'strength': self.strength,
            'compliance_rate': self.compliance_rate,
            'violation_ratio': self.violation_ratio,
            'violations': self.violations,
            'total_checks': self.total_checks,
            'r_squared': self.r_squared,
            'rmse': self.rmse,
        }


class ConstraintReportGenerator:
    """
    Generate comprehensive reports for constraint compliance evaluation
    
    Features:
    - Per-feature violation statistics
    - Interactive Plotly visualizations
    - Summary dashboard with compliance rates
    - Export to HTML/PDF/JSON
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        self.violation_stats: List[ConstraintViolationStats] = []
    
    def add_violation_stats(self, stats: ConstraintViolationStats):
        """Add violation statistics for a feature"""
        self.violation_stats.append(stats)
        logger.debug(f"Added stats for {stats.feature_name}: compliance={stats.compliance_rate:.1%}")
    
    def generate_summary_dataframe(self) -> pd.DataFrame:
        """Generate summary DataFrame of all constraint violations"""
        if not self.violation_stats:
            return pd.DataFrame()
        
        records = [s.to_dict() for s in self.violation_stats]
        df = pd.DataFrame(records)
        
        # Add derived columns
        df['severity'] = df['violation_ratio'].apply(
            lambda r: 'high' if r > 0.3 else 'medium' if r > 0.1 else 'low'
        )
        df['priority'] = df.apply(
            lambda row: 1 if row['strength'] == 'strong' and row['violation_ratio'] > 0.1 
                       else 2 if row['violation_ratio'] > 0.2 else 3,
            axis=1
        )
        
        return df.sort_values('priority')
    
    def plot_compliance_dashboard(self, 
                                df: Optional[pd.DataFrame] = None,
                                title: str = "Constraint Compliance Dashboard"
                               ) -> go.Figure:
        """
        Generate interactive compliance dashboard
        
        Args:
            df: Optional pre-computed summary DataFrame
            title: Dashboard title
        
        Returns:
            Plotly Figure object
        """
        if df is None:
            df = self.generate_summary_dataframe()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No constraint data available", showarrow=False)
            return fig
        
        # Create subplots: 2 rows x 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Compliance Rate by Feature",
                "Violation Ratio Distribution",
                "Strong vs Weak Constraints",
                "Compliance by σ-Range"
            ],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "box"}]]
        )
        
        # Plot 1: Compliance rate bar chart
        fig.add_trace(
            go.Bar(
                x=df['feature_name'],
                y=df['compliance_rate'] * 100,
                marker_color=df['compliance_rate'].apply(
                    lambda r: 'green' if r > 0.95 else 'orange' if r > 0.8 else 'red'
                ),
                name="Compliance %",
                hovertemplate="<b>%{x}</b><br>Compliance: %{y:.1f}%<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Plot 2: Violation ratio histogram
        fig.add_trace(
            go.Histogram(
                x=df['violation_ratio'] * 100,
                nbinsx=20,
                marker_color='steelblue',
                name="Violation %",
                hovertemplate="Violations: %{x:.1f}%<br>Count: %{y}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Plot 3: Strong vs Weak pie chart
        strength_counts = df['strength'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=strength_counts.index,
                values=strength_counts.values,
                name="Constraint Strength",
                hole=0.3,
                marker_colors=['#2E86AB', '#A23B72']
            ),
            row=2, col=1
        )
        
        # Plot 4: Compliance by sigma range box plot
        if 'sigma_range' in df.columns and df['sigma_range'].nunique() > 1:
            fig.add_trace(
                go.Box(
                    x=df['sigma_range'].astype(str),
                    y=df['compliance_rate'] * 100,
                    name="Compliance by σ",
                    marker_color='#06A77D',
                    boxmean=True
                ),
                row=2, col=2
            )
        
        # Layout updates
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            height=700,
            showlegend=False,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Axis labels
        fig.update_xaxes(title_text="Feature", row=1, col=1)
        fig.update_yaxes(title_text="Compliance (%)", row=1, col=1)
        fig.update_xaxes(title_text="Violation Ratio (%)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        return fig
    
    def plot_violation_detail(self, 
                            feature_name: str,
                            x_values: np.ndarray,
                            y_pred: np.ndarray,
                            constraint_type: Literal['monotonic', 'linearity'],
                            direction: Optional[str] = None,
                            title: Optional[str] = None
                           ) -> go.Figure:
        """
        Generate detailed violation plot for a single feature
        
        Shows the relationship between feature value and prediction,
        highlighting constraint violations.
        """
        if len(x_values) != len(y_pred):
            raise ValueError("x_values and y_pred must have same length")
        
        # Sort by x for monotonicity checking
        sorted_idx = np.argsort(x_values)
        x_sorted = x_values[sorted_idx]
        y_sorted = y_pred[sorted_idx]
        
        # Calculate violations
        violations = np.zeros(len(x_sorted), dtype=bool)
        if constraint_type == 'monotonic' and direction and len(x_sorted) > 1:
            dy = np.diff(y_sorted)
            dx = np.diff(x_sorted)
            mask = dx > 1e-10  # Avoid division by zero
            
            if direction == 'increasing':
                violations[1:][mask] = dy[mask] < -1e-8
            elif direction == 'decreasing':
                violations[1:][mask] = dy[mask] > 1e-8
        
        # Create figure
        fig = go.Figure()
        
        # Main scatter plot
        fig.add_trace(
            go.Scatter(
                x=x_sorted,
                y=y_sorted,
                mode='markers',
                name='Predictions',
                marker=dict(
                    size=6,
                    color=['red' if v else 'blue' for v in violations],
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGray')
                ),
                hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>Violation: %{marker.color}<extra></extra>"
            )
        )
        
        # Add trend line for linearity constraint
        if constraint_type == 'linearity' and len(x_sorted) >= 3:
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(x_sorted, y_sorted)
            y_trend = slope * x_sorted + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_trend,
                    mode='lines',
                    name=f'Linear Fit (R²={r_value**2:.3f})',
                    line=dict(color='green', width=2, dash='dash')
                )
            )
        
        # Add moving average for monotonicity visualization
        if constraint_type == 'monotonic' and len(x_sorted) >= 10:
            window = min(20, len(x_sorted) // 5)
            y_smooth = pd.Series(y_sorted).rolling(window, center=True, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_smooth,
                    mode='lines',
                    name=f'Moving Avg (w={window})',
                    line=dict(color='orange', width=2)
                )
            )
        
        # Layout
        title_text = title or f"{constraint_type.capitalize()} Constraint: {feature_name}"
        if direction:
            title_text += f" ({direction})"
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5),
            xaxis_title=feature_name,
            yaxis_title="Model Prediction",
            template='plotly_white',
            height=500,
            hovermode='closest',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Add violation count annotation
        n_violations = np.sum(violations)
        if n_violations > 0:
            fig.add_annotation(
                text=f"⚠️ {n_violations} violations detected",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,0,0,0.1)",
                bordercolor="red",
                borderwidth=1
            )
        
        return fig
    
    def export_report(self, 
                     format: Literal['html', 'json', 'csv'] = 'html',
                     filename: Optional[str] = None
                    ) -> str:
        """
        Export constraint report to specified format
        
        Args:
            format: Output format (html/json/csv)
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to generated file
        """
        import json
        import os
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"constraint_report_{timestamp}.{format}"
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, filename)
        else:
            filepath = filename
        
        if format == 'json':
            # Export statistics as JSON
            data = {
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_features': len(self.violation_stats),
                    'avg_compliance': np.mean([s.compliance_rate for s in self.violation_stats]),
                    'min_compliance': min([s.compliance_rate for s in self.violation_stats], default=0),
                },
                'details': [s.to_dict() for s in self.violation_stats]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            # Export summary DataFrame as CSV
            df = self.generate_summary_dataframe()
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        elif format == 'html':
            # Export interactive dashboard as standalone HTML
            fig = self.plot_compliance_dashboard()
            fig.write_html(filepath, include_plotlyjs='cdn', auto_open=False)
        
        logger.info(f"Constraint report exported to {filepath}")
        return filepath
