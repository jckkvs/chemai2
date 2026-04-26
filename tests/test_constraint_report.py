"""
Test: tests/test_constraint_report.py
Unit tests for constraint compliance report generation
"""
import pytest
import numpy as np
import pandas as pd
from backend.evaluation.constraint_report import (
    ConstraintViolationStats,
    ConstraintReportGenerator
)


class TestConstraintViolationStats:
    """Tests for ConstraintViolationStats dataclass"""
    
    def test_compliance_rate_calculation(self):
        """Verify compliance_rate = 1 - violation_ratio"""
        stats = ConstraintViolationStats(
            feature_name="test_feat",
            constraint_type="monotonic",
            direction="increasing",
            total_checks=100,
            violations=15
        )
        # Manually set violation_ratio for testing
        stats.violation_ratio = 0.15
        
        assert stats.compliance_rate == 0.85
        assert abs(stats.compliance_rate + stats.violation_ratio - 1.0) < 1e-10
    
    def test_to_dict_serialization(self):
        """Verify to_dict produces serializable output"""
        stats = ConstraintViolationStats(
            feature_name="mol_weight",
            constraint_type="linearity",
            sigma_range=2.5,
            strength="strong",
            total_checks=50,
            violations=3,
            r_squared=0.92,
            rmse=0.15
        )
        stats.violation_ratio = 0.06  # 3/50
        
        result = stats.to_dict()
        
        assert result['feature_name'] == "mol_weight"
        assert result['constraint_type'] == "linearity"
        assert result['compliance_rate'] == 0.94
        assert result['r_squared'] == 0.92
        assert result['rmse'] == 0.15
        assert 'severity' not in result  # Derived column not included in base dict


class TestConstraintReportGenerator:
    """Tests for ConstraintReportGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create test generator instance"""
        return ConstraintReportGenerator(output_dir="/tmp/test_reports")
    
    def test_add_and_summarize_stats(self, generator):
        """Verify stats can be added and summarized"""
        # Add test statistics
        generator.add_violation_stats(ConstraintViolationStats(
            feature_name="feat_a",
            constraint_type="monotonic",
            direction="increasing",
            total_checks=100,
            violations=5,
            violation_ratio=0.05
        ))
        generator.add_violation_stats(ConstraintViolationStats(
            feature_name="feat_b",
            constraint_type="linearity",
            strength="strong",
            total_checks=80,
            violations=20,
            violation_ratio=0.25,
            r_squared=0.78
        ))
        
        # Generate summary
        df = generator.generate_summary_dataframe()
        
        assert len(df) == 2
        assert set(df['feature_name']) == {"feat_a", "feat_b"}
        assert df.loc[df['feature_name'] == 'feat_a', 'compliance_rate'].iloc[0] == 0.95
        assert df.loc[df['feature_name'] == 'feat_b', 'severity'].iloc[0] == 'medium'
    
    def test_empty_generator(self, generator):
        """Verify behavior with no stats added"""
        df = generator.generate_summary_dataframe()
        assert df.empty
        
        fig = generator.plot_compliance_dashboard()
        # Should have annotation for empty state
        assert len(fig.layout.annotations) >= 1
    
    def test_plot_violation_detail_monotonic(self, generator):
        """Test detailed violation plot for monotonic constraint"""
        # Generate test data with known monotonic relationship + noise
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 0.5, 100)  # Mostly increasing with noise
        
        fig = generator.plot_violation_detail(
            feature_name="test_feature",
            x_values=x,
            y_pred=y,
            constraint_type="monotonic",
            direction="increasing",
            title="Test Monotonicity Plot"
        )
        
        # Verify figure structure
        assert len(fig.data) >= 1  # At least main scatter
        assert fig.layout.title.text == "Test Monotonicity Plot (increasing)"
        assert fig.layout.xaxis.title.text == "test_feature"
    
    def test_plot_violation_detail_linearity(self, generator):
        """Test detailed violation plot for linearity constraint"""
        # Generate test data with linear relationship
        np.random.seed(123)
        x = np.linspace(-5, 5, 80)
        y = 3 * x - 1 + np.random.normal(0, 0.3, 80)
        
        fig = generator.plot_violation_detail(
            feature_name="linear_feat",
            x_values=x,
            y_pred=y,
            constraint_type="linearity",
            title="Test Linearity Plot"
        )
        
        # Should have scatter + linear fit line
        assert len(fig.data) >= 2
        # Check that linear fit trace exists
        trace_names = [t.name for t in fig.data]
        assert any("Linear Fit" in name for name in trace_names)
    
    def test_export_json(self, generator, tmp_path):
        """Test JSON export functionality"""
        import json
        
        generator.add_violation_stats(ConstraintViolationStats(
            feature_name="export_test",
            constraint_type="monotonic",
            total_checks=50,
            violations=2,
            violation_ratio=0.04
        ))
        
        filepath = generator.export_report(format='json', filename=str(tmp_path / "report.json"))
        
        # Verify file exists and is valid JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'generated_at' in data
        assert data['summary']['total_features'] == 1
        assert data['details'][0]['feature_name'] == 'export_test'
    
    def test_export_csv(self, generator, tmp_path):
        """Test CSV export functionality"""
        generator.add_violation_stats(ConstraintViolationStats(
            feature_name="csv_test",
            constraint_type="linearity",
            r_squared=0.89,
            total_checks=30,
            violations=3
        ))
        
        filepath = generator.export_report(format='csv', filename=str(tmp_path / "report.csv"))
        
        # Verify file exists and can be read as CSV
        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert df.iloc[0]['feature_name'] == 'csv_test'
        assert df.iloc[0]['r_squared'] == 0.89
