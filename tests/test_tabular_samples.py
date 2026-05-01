"""
tests/test_tabular_samples.py

TDD for sample data processing, starting with tabular_50_safe.csv (no missing values).
Focus on non-LLM features first: data loading, type detection, preprocessing.
Then expand to other sample files.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from backend.data.loader import load_file
from backend.data.type_detector import TypeDetector


# ============================================================
# Fixtures: Primary Starting Point (tabular_50_safe.csv)
# ============================================================

@pytest.fixture
def safe_csv_path() -> Path:
    """Path to tabular_50_safe.csv (primary starting point, no missing values)."""
    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_safe.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_safe_df(safe_csv_path: Path) -> pd.DataFrame:
    """Load tabular_50_safe.csv (no missing values)."""
    return load_file(safe_csv_path)


@pytest.fixture
def safe_detection_result(loaded_safe_df: pd.DataFrame) -> tuple:
    """Run TypeDetector on loaded_safe_df and return (detector, result)."""
    dt = TypeDetector()
    res = dt.detect(loaded_safe_df)
    return dt, res


# ============================================================
# Fixtures: Edge Case (tabular_50_simple.csv with missing values)
# ============================================================

@pytest.fixture
def simple_csv_path() -> Path:
    """Path to tabular_50_simple.csv (has missing values for edge case testing)."""
    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_simple.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_simple_df(simple_csv_path: Path) -> pd.DataFrame:
    """Load tabular_50_simple.csv (has missing values)."""
    return load_file(simple_csv_path)


@pytest.fixture
def simple_detection_result(loaded_simple_df: pd.DataFrame) -> tuple:
    """Run TypeDetector on loaded_simple_df and return (detector, result)."""
    dt = TypeDetector()
    res = dt.detect(loaded_simple_df)
    return dt, res


# ============================================================
# Fixtures: Other Tabular Files
# ============================================================

@pytest.fixture
def tabular_1000_path() -> Path:
    """Path to tabular_1000_large.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_1000_large.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_1000_df(tabular_1000_path: Path) -> pd.DataFrame:
    """Load tabular_1000_large.csv."""
    return load_file(tabular_1000_path)


@pytest.fixture
def tabular_200_path() -> Path:
    """Path to tabular_200_complex.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_200_complex.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_200_df(tabular_200_path: Path) -> pd.DataFrame:
    """Load tabular_200_complex.csv."""
    return load_file(tabular_200_path)


# ============================================================
# Fixtures: SMILES Files
# ============================================================

@pytest.fixture
def smiles_25_reg_path() -> Path:
    """Path to smiles_25_regression.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_25_regression.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_smiles_25_reg(smiles_25_reg_path: Path) -> pd.DataFrame:
    """Load smiles_25_regression.csv."""
    return load_file(smiles_25_reg_path)


@pytest.fixture
def smiles_25_cls_path() -> Path:
    """Path to smiles_25_classification.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_25_classification.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_smiles_25_cls(smiles_25_cls_path: Path) -> pd.DataFrame:
    """Load smiles_25_classification.csv."""
    return load_file(smiles_25_cls_path)


@pytest.fixture
def smiles_100_reg_path() -> Path:
    """Path to smiles_100_regression.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_100_regression.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_smiles_100_reg(smiles_100_reg_path: Path) -> pd.DataFrame:
    """Load smiles_100_regression.csv."""
    return load_file(smiles_100_reg_path)


@pytest.fixture
def smiles_100_cls_path() -> Path:
    """Path to smiles_100_classification.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_100_classification.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_smiles_100_cls(smiles_100_cls_path: Path) -> pd.DataFrame:
    """Load smiles_100_classification.csv."""
    return load_file(smiles_100_cls_path)


# ============================================================
# T-010: Data Loading Tests (tabular_50_safe.csv - primary starting point)
# ============================================================

class TestTabular50SafeLoading:
    """T-010: Load tabular_50_safe.csv (no missing values) and verify basic properties."""

    def test_file_exists(self, safe_csv_path: Path) -> None:
        """Sample CSV file exists at expected path. (T-010-01)"""
        assert safe_csv_path.exists()

    def test_load_returns_dataframe(self, safe_csv_path: Path) -> None:
        """load_file returns a pandas DataFrame. (T-010-02)"""
        df = load_file(safe_csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_is_50_rows_9_columns(self, loaded_safe_df: pd.DataFrame) -> None:
        """Loaded DataFrame has 50 rows and 9 columns (8 features + 1 target). (T-010-03)"""
        assert loaded_safe_df.shape == (50, 9), f"Expected (50,9), got {loaded_safe_df.shape}"

    def test_column_names(self, loaded_safe_df: pd.DataFrame) -> None:
        """Columns are Feature_1 through Feature_8 plus Target. (T-010-04)"""
        expected = [f"Feature_{i}" for i in range(1, 9)] + ["Target"]
        assert list(loaded_safe_df.columns) == expected

    def test_no_duplicate_columns(self, loaded_safe_df: pd.DataFrame) -> None:
        """No duplicate column names. (T-010-05)"""
        assert len(loaded_safe_df.columns) == len(set(loaded_safe_df.columns))

    def test_all_features_are_numeric(self, loaded_safe_df: pd.DataFrame) -> None:
        """All Feature_1..Feature_8 are numeric (float). (T-010-06)"""
        for i in range(1, 9):
            col = f"Feature_{i}"
            assert pd.api.types.is_numeric_dtype(loaded_safe_df[col]), f"{col} is not numeric"

    def test_target_is_numeric(self, loaded_safe_df: pd.DataFrame) -> None:
        """Target column is numeric. (T-010-07)"""
        assert pd.api.types.is_numeric_dtype(loaded_safe_df["Target"])

    def test_no_missing_values(self, loaded_safe_df: pd.DataFrame) -> None:
        """No missing values in the safe dataset. (T-010-08)"""
        assert loaded_safe_df.isna().sum().sum() == 0, "tabular_50_safe.csv should have no missing values"

    def test_save_and_reload(self, loaded_safe_df: pd.DataFrame, tmp_path: Path) -> None:
        """Save DataFrame to CSV and reload, shape is preserved. (T-010-09)"""
        from backend.data.loader import save_dataframe
        out_path = tmp_path / "reloaded_safe.csv"
        save_dataframe(loaded_safe_df, out_path)
        reloaded = pd.read_csv(out_path)
        assert reloaded.shape == loaded_safe_df.shape


# ============================================================
# T-010-B: Data Loading Tests (tabular_50_simple.csv - with missing values)
# ============================================================

class TestTabular50SimpleLoading:
    """T-010-B: Load tabular_50_simple.csv (has missing values) for edge case testing."""

    def test_file_exists(self, simple_csv_path: Path) -> None:
        """Sample CSV file exists at expected path. (T-010-B-01)"""
        assert simple_csv_path.exists()

    def test_load_returns_dataframe(self, simple_csv_path: Path) -> None:
        """load_file returns a pandas DataFrame. (T-010-B-02)"""
        df = load_file(simple_csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_is_50_rows_9_columns(self, loaded_simple_df: pd.DataFrame) -> None:
        """Loaded DataFrame has 50 rows and 9 columns. (T-010-B-03)"""
        assert loaded_simple_df.shape == (50, 9), f"Expected (50,9), got {loaded_simple_df.shape}"

    def test_missing_values_count(self, loaded_simple_df: pd.DataFrame) -> None:
        """Expected missing values: Feature_1 has 1, Feature_2 has 2, Feature_3 has 3, total 6. (T-010-B-04)"""
        total_missing = loaded_simple_df.isna().sum().sum()
        assert total_missing == 6, f"Expected 6 missing values, got {total_missing}"
        assert loaded_simple_df["Feature_1"].isna().sum() == 1
        assert loaded_simple_df["Feature_2"].isna().sum() == 2
        assert loaded_simple_df["Feature_3"].isna().sum() == 3

    def test_missing_positions(self, loaded_simple_df: pd.DataFrame) -> None:
        """Missing values are at expected positions. (T-010-B-05)"""
        # Line 9 (data row 8) → iloc[7]: Feature_3 missing
        assert pd.isna(loaded_simple_df.iloc[7]["Feature_3"]), "Line9: Feature_3 missing"
        # Line 31 (data row 30) → iloc[29]: Feature_1 missing
        assert pd.isna(loaded_simple_df.iloc[29]["Feature_1"]), "Line31: Feature_1 missing"
        # Line 32 (data row 31) → iloc[30]: Feature_3 missing
        assert pd.isna(loaded_simple_df.iloc[30]["Feature_3"]), "Line32: Feature_3 missing"
        # Line 36 (data row 35) → iloc[34]: Feature_3 missing
        assert pd.isna(loaded_simple_df.iloc[34]["Feature_3"]), "Line36: Feature_3 missing"
        # Line 41 (data row 40) → iloc[39]: Feature_2 missing
        assert pd.isna(loaded_simple_df.iloc[39]["Feature_2"]), "Line41: Feature_2 missing"
        # Line 42 (data row 41) → iloc[40]: Feature_2 missing
        assert pd.isna(loaded_simple_df.iloc[40]["Feature_2"]), "Line42: Feature_2 missing"

    def test_no_missing_in_target(self, loaded_simple_df: pd.DataFrame) -> None:
        """Target column has no missing values. (T-010-B-06)"""
        assert loaded_simple_df["Target"].notna().all()


# ============================================================
# T-011: Type Detection Tests (tabular_50_safe.csv - no missing)
# ============================================================

class TestTabular50SafeTypeDetection:
    """T-011: Type detection on tabular_50_safe.csv (no missing values)."""

    def test_detection_result_valid(self, safe_detection_result: tuple) -> None:
        """TypeDetector returns a valid DetectionResult. (T-011-01)"""
        _, res = safe_detection_result
        from backend.data.type_detector import DetectionResult
        assert isinstance(res, DetectionResult)

    def test_all_features_detected_as_numeric(self, safe_detection_result: tuple) -> None:
        """All Feature_1..Feature_8 are detected as NUMERIC type. (T-011-02)"""
        _, res = safe_detection_result
        from backend.data.type_detector import ColumnType
        numeric_types = (ColumnType.NUMERIC_NORMAL, ColumnType.NUMERIC_LOG, ColumnType.NUMERIC_POWER)
        for i in range(1, 9):
            col = f"Feature_{i}"
            assert col in res.column_info, f"{col} not in detection result"
            assert res.column_info[col].col_type in numeric_types, \
                f"{col} detected as {res.column_info[col].col_type}, expected numeric"

    def test_target_detected_as_numeric(self, safe_detection_result: tuple) -> None:
        """Target column is detected as numeric. (T-011-03)"""
        _, res = safe_detection_result
        from backend.data.type_detector import ColumnType
        numeric_types = (ColumnType.NUMERIC_NORMAL, ColumnType.NUMERIC_LOG, ColumnType.NUMERIC_POWER)
        assert res.column_info["Target"].col_type in numeric_types

    def test_no_smiles_columns_detected(self, safe_detection_result: tuple) -> None:
        """No SMILES columns are detected. (T-011-04)"""
        _, res = safe_detection_result
        assert len(res.smiles_columns) == 0

    def test_null_rates_all_zero(self, safe_detection_result: tuple) -> None:
        """All null rates are zero (no missing values). (T-011-05)"""
        _, res = safe_detection_result
        for col in res.column_info:
            assert res.column_info[col].null_rate == 0.0, f"{col} has non-zero null rate"

    def test_get_numeric_columns_includes_all_features(self, safe_detection_result: tuple) -> None:
        """get_numeric_columns() returns all feature columns and target. (T-011-06)"""
        _, res = safe_detection_result
        numeric_cols = res.get_numeric_columns()
        for i in range(1, 9):
            assert f"Feature_{i}" in numeric_cols
        assert "Target" in numeric_cols


# ============================================================
# T-011-B: Type Detection Tests (tabular_50_simple.csv - with missing)
# ============================================================

class TestTabular50SimpleTypeDetection:
    """T-011-B: Type detection on tabular_50_simple.csv (has missing values)."""

    def test_null_rates_calculated(self, simple_detection_result: tuple) -> None:
        """Null rates are correctly calculated for columns with missing values. (T-011-B-01)"""
        _, res = simple_detection_result
        assert abs(res.column_info["Feature_1"].null_rate - 0.02) < 1e-6
        assert abs(res.column_info["Feature_2"].null_rate - 0.04) < 1e-6
        assert abs(res.column_info["Feature_3"].null_rate - 0.06) < 1e-6
        for i in range(4, 9):
            assert res.column_info[f"Feature_{i}"].null_rate == 0.0
        assert res.column_info["Target"].null_rate == 0.0


# ============================================================
# T-012: Tabular 1000 Large Tests
# ============================================================

class TestTabular1000Large:
    """T-012: Load tabular_1000_large.csv and verify properties."""

    def test_load_returns_dataframe(self, tabular_1000_path: Path) -> None:
        """load_file returns a DataFrame. (T-012-01)"""
        df = load_file(tabular_1000_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_1000_rows_11_columns(self, loaded_1000_df: pd.DataFrame) -> None:
        """Loaded DataFrame has 1000 rows and 11 columns. (T-012-02)"""
        assert loaded_1000_df.shape[1] == 11, f"Expected 11 columns, got {loaded_1000_df.shape[1]}"
        assert len(loaded_1000_df) == 1000, f"Expected 1000 rows, got {len(loaded_1000_df)}"

    def test_column_names(self, loaded_1000_df: pd.DataFrame) -> None:
        """Columns include Feature_1..8, Target, Sample_ID, Category. (T-012-03)"""
        cols = list(loaded_1000_df.columns)
        for i in range(1, 9):
            assert f"Feature_{i}" in cols
        assert "Target" in cols
        assert "Sample_ID" in cols
        assert "Category" in cols

    def test_sample_id_column_exists(self, loaded_1000_df: pd.DataFrame) -> None:
        """Sample_ID column exists and is non-numeric. (T-012-04)"""
        assert "Sample_ID" in loaded_1000_df.columns
        assert not pd.api.types.is_numeric_dtype(loaded_1000_df["Sample_ID"])

    def test_category_column_exists(self, loaded_1000_df: pd.DataFrame) -> None:
        """Category column exists and is categorical. (T-012-05)"""
        assert "Category" in loaded_1000_df.columns
        assert loaded_1000_df["Category"].nunique() <= 10


# ============================================================
# T-013: Tabular 200 Complex Tests
# ============================================================

class TestTabular200Complex:
    """T-013: Load tabular_200_complex.csv and verify properties."""

    def test_load_returns_dataframe(self, tabular_200_path: Path) -> None:
        """load_file returns a DataFrame. (T-013-01)"""
        df = load_file(tabular_200_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_200_rows_11_columns(self, loaded_200_df: pd.DataFrame) -> None:
        """Loaded DataFrame has 200 rows and 11 columns. (T-013-02)"""
        assert loaded_200_df.shape[1] == 11, f"Expected 11 columns, got {loaded_200_df.shape[1]}"
        assert len(loaded_200_df) == 200, f"Expected 200 rows, got {len(loaded_200_df)}"

    def test_has_missing_values(self, loaded_200_df: pd.DataFrame) -> None:
        """Has some missing values (check from data). (T-013-03)"""
        # Row 7 (0-based index 7) has Feature_2 missing
        assert pd.isna(loaded_200_df.iloc[7]["Feature_2"]), "Feature_2 should be missing at row 7"
        total_missing = loaded_200_df.isna().sum().sum()
        assert total_missing > 0, "Should have some missing values"

    def test_category_column(self, loaded_200_df: pd.DataFrame) -> None:
        """Category column exists with multiple classes. (T-013-04)"""
        assert "Category" in loaded_200_df.columns
        assert loaded_200_df["Category"].nunique() >= 2


# ============================================================
# T-014: Preprocessing Tests (tabular_50_safe.csv)
# ============================================================

class TestTabular50SafePreprocessing:
    """T-014: Preprocessing pipeline with tabular_50_safe.csv (no missing values)."""

    def test_build_pipeline(self, safe_detection_result: tuple) -> None:
        """Preprocessor builds a ColumnTransformer pipeline. (T-014-01)"""
        from backend.data.preprocessor import Preprocessor
        from sklearn.compose import ColumnTransformer
        _, res = safe_detection_result
        pp = Preprocessor()
        ct = pp.build(res, target_col="Target")
        assert isinstance(ct, ColumnTransformer)

    def test_fit_transform(self, loaded_safe_df: pd.DataFrame) -> None:
        """Pipeline fit_transform works with no missing values. (T-014-02)"""
        from backend.data.preprocessor import Preprocessor, PreprocessConfig
        from backend.data.type_detector import TypeDetector
        dt = TypeDetector()
        res = dt.detect(loaded_safe_df)
        config = PreprocessConfig(add_missing_indicator=False)
        pp = Preprocessor(config)
        ct = pp.build(res, target_col="Target")
        X = loaded_safe_df.drop(columns=["Target"])
        X_transformed = ct.fit_transform(X, loaded_safe_df["Target"])
        assert not np.isnan(X_transformed).any(), "Transformed data has NaN"

    def test_output_shape(self, loaded_safe_df: pd.DataFrame) -> None:
        """Output shape after preprocessing is (n_samples, n_features). (T-014-03)"""
        from backend.data.preprocessor import Preprocessor, PreprocessConfig
        from backend.data.type_detector import TypeDetector
        dt = TypeDetector()
        res = dt.detect(loaded_safe_df)
        config = PreprocessConfig(add_missing_indicator=True)
        pp = Preprocessor(config)
        ct = pp.build(res, target_col="Target")
        X = loaded_safe_df.drop(columns=["Target"])
        X_transformed = ct.fit_transform(X, loaded_safe_df["Target"])
        assert X_transformed.shape[0] == len(loaded_safe_df)
        assert X_transformed.shape[1] >= X.shape[1], "Should have at least as many features as input"

    def test_preprocessor_with_different_imputers(self, loaded_safe_df: pd.DataFrame) -> None:
        """Test different imputation strategies (even with no missing values). (T-014-04)"""
        from backend.data.preprocessor import Preprocessor, PreprocessConfig
        from backend.data.type_detector import TypeDetector
        dt = TypeDetector()
        res = dt.detect(loaded_safe_df)
        for imputer in ["mean", "median"]:
            config = PreprocessConfig(numeric_imputer=imputer, add_missing_indicator=False)
            pp = Preprocessor(config)
            ct = pp.build(res, target_col="Target")
            X = loaded_safe_df.drop(columns=["Target"])
            X_transformed = ct.fit_transform(X, loaded_safe_df["Target"])
            assert not np.isnan(X_transformed).any(), f"NaN with imputer {imputer}"

    def test_preprocessor_with_scalers(self, loaded_safe_df: pd.DataFrame) -> None:
        """Test different scaling strategies. (T-014-05)"""
        from backend.data.preprocessor import Preprocessor, PreprocessConfig
        from backend.data.type_detector import TypeDetector
        dt = TypeDetector()
        res = dt.detect(loaded_safe_df)
        for scaler in ["standard", "robust", "minmax", "quantile_uniform"]:
            config = PreprocessConfig(
                numeric_scaler=scaler,
                numeric_imputer="mean",
                add_missing_indicator=False
            )
            pp = Preprocessor(config)
            ct = pp.build(res, target_col="Target")
            X = loaded_safe_df.drop(columns=["Target"])
            X_transformed = ct.fit_transform(X, loaded_safe_df["Target"])
            assert X_transformed.shape[0] == len(loaded_safe_df), f"Shape mismatch with scaler {scaler}"


# ============================================================
# T-015: SMILES 25 Regression Tests
# ============================================================

class TestSmiles25Regression:
    """T-015: Load smiles_25_regression.csv and verify properties."""

    def test_load_returns_dataframe(self, smiles_25_reg_path: Path) -> None:
        """load_file returns a DataFrame. (T-015-01)"""
        df = load_file(smiles_25_reg_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_25_rows_3_columns(self, loaded_smiles_25_reg: pd.DataFrame) -> None:
        """Loaded DataFrame has 25 rows and 3 columns. (T-015-02)"""
        assert loaded_smiles_25_reg.shape == (25, 3), f"Expected (25,3), got {loaded_smiles_25_reg.shape}"

    def test_columns_exist(self, loaded_smiles_25_reg: pd.DataFrame) -> None:
        """Columns are Compound_Name, SMILES, logS. (T-015-03)"""
        cols = list(loaded_smiles_25_reg.columns)
        assert "Compound_Name" in cols
        assert "SMILES" in cols
        assert "logS" in cols

    def test_smiles_column_detected(self, loaded_smiles_25_reg: pd.DataFrame) -> None:
        """SMILES column is detected as SMILES type. (T-015-04)"""
        dt = TypeDetector()
        res = dt.detect(loaded_smiles_25_reg)
        assert "SMILES" in res.smiles_columns

    def test_target_column_is_numeric(self, loaded_smiles_25_reg: pd.DataFrame) -> None:
        """logS (target) is numeric. (T-015-05)"""
        assert pd.api.types.is_numeric_dtype(loaded_smiles_25_reg["logS"])

    def test_no_missing_values(self, loaded_smiles_25_reg: pd.DataFrame) -> None:
        """No missing values in the data. (T-015-06)"""
        assert loaded_smiles_25_reg.isna().sum().sum() == 0

    def test_smiles_strings_valid_format(self, loaded_smiles_25_reg: pd.DataFrame) -> None:
        """SMILES strings have valid format (contain common SMILES chars). (T-015-07)"""
        for smiles in loaded_smiles_25_reg["SMILES"]:
            assert any(c in smiles for c in ["C", "O", "N", "c", "("]), f"Invalid SMILES: {smiles}"


# ============================================================
# T-016: SMILES 25 Classification Tests
# ============================================================

class TestSmiles25Classification:
    """T-016: Load smiles_25_classification.csv and verify properties."""

    def test_load_returns_dataframe(self, smiles_25_cls_path: Path) -> None:
        """load_file returns a DataFrame. (T-016-01)"""
        df = load_file(smiles_25_cls_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_25_rows_3_columns(self, loaded_smiles_25_cls: pd.DataFrame) -> None:
        """Loaded DataFrame has 25 rows and 3 columns. (T-016-02)"""
        assert loaded_smiles_25_cls.shape == (25, 3), f"Expected (25,3), got {loaded_smiles_25_cls.shape}"

    def test_columns_exist(self, loaded_smiles_25_cls: pd.DataFrame) -> None:
        """Columns are Compound_Name, SMILES, Class. (T-016-03)"""
        cols = list(loaded_smiles_25_cls.columns)
        assert "Compound_Name" in cols
        assert "SMILES" in cols
        assert "Class" in cols

    def test_smiles_column_detected(self, loaded_smiles_25_cls: pd.DataFrame) -> None:
        """SMILES column is detected as SMILES type. (T-016-04)"""
        dt = TypeDetector()
        res = dt.detect(loaded_smiles_25_cls)
        assert "SMILES" in res.smiles_columns

    def test_target_is_integer(self, loaded_smiles_25_cls: pd.DataFrame) -> None:
        """Class (target) is integer type. (T-016-05)"""
        assert pd.api.types.is_numeric_dtype(loaded_smiles_25_cls["Class"])

    def test_no_missing_values(self, loaded_smiles_25_cls: pd.DataFrame) -> None:
        """No missing values in the data. (T-016-06)"""
        assert loaded_smiles_25_cls.isna().sum().sum() == 0

    def test_class_values_are_binary(self, loaded_smiles_25_cls: pd.DataFrame) -> None:
        """Class values are 0 or 1 (binary classification). (T-016-07)"""
        unique_vals = loaded_smiles_25_cls["Class"].unique()
        for v in unique_vals:
            assert v in [0, 1], f"Unexpected class value: {v}"


# ============================================================
# T-017: SMILES 100 Regression Tests
# ============================================================

class TestSmiles100Regression:
    """T-017: Load smiles_100_regression.csv and verify properties."""

    def test_load_returns_dataframe(self, smiles_100_reg_path: Path) -> None:
        """load_file returns a DataFrame. (T-017-01)"""
        df = load_file(smiles_100_reg_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_100_rows_3_columns(self, loaded_smiles_100_reg: pd.DataFrame) -> None:
        """Loaded DataFrame has 100 rows and 3 columns. (T-017-02)"""
        assert loaded_smiles_100_reg.shape[0] == 100, f"Expected 100 rows, got {loaded_smiles_100_reg.shape[0]}"
        assert loaded_smiles_100_reg.shape[1] == 3, f"Expected 3 columns, got {loaded_smiles_100_reg.shape[1]}"

    def test_columns_exist(self, loaded_smiles_100_reg: pd.DataFrame) -> None:
        """Columns are Compound_Name, SMILES, logS. (T-017-03)"""
        cols = list(loaded_smiles_100_reg.columns)
        assert "Compound_Name" in cols
        assert "SMILES" in cols
        assert "logS" in cols

    def test_smiles_column_detected(self, loaded_smiles_100_reg: pd.DataFrame) -> None:
        """SMILES column is detected as SMILES type. (T-017-04)"""
        dt = TypeDetector()
        res = dt.detect(loaded_smiles_100_reg)
        assert "SMILES" in res.smiles_columns

    def test_target_is_numeric(self, loaded_smiles_100_reg: pd.DataFrame) -> None:
        """logS (target) is numeric. (T-017-05)"""
        assert pd.api.types.is_numeric_dtype(loaded_smiles_100_reg["logS"])

    def test_no_missing_values(self, loaded_smiles_100_reg: pd.DataFrame) -> None:
        """No missing values in the data. (T-017-06)"""
        assert loaded_smiles_100_reg.isna().sum().sum() == 0


# ============================================================
# T-018: SMILES 100 Classification Tests
# ============================================================

class TestSmiles100Classification:
    """T-018: Load smiles_100_classification.csv and verify properties."""

    def test_load_returns_dataframe(self, smiles_100_cls_path: Path) -> None:
        """load_file returns a DataFrame. (T-018-01)"""
        df = load_file(smiles_100_cls_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_100_rows_3_columns(self, loaded_smiles_100_cls: pd.DataFrame) -> None:
        """Loaded DataFrame has 100 rows and 3 columns. (T-018-02)"""
        assert loaded_smiles_100_cls.shape[0] == 100, f"Expected 100 rows, got {loaded_smiles_100_cls.shape[0]}"
        assert loaded_smiles_100_cls.shape[1] == 3, f"Expected 3 columns, got {loaded_smiles_100_cls.shape[1]}"

    def test_columns_exist(self, loaded_smiles_100_cls: pd.DataFrame) -> None:
        """Columns are Compound_Name, SMILES, Class. (T-018-03)"""
        cols = list(loaded_smiles_100_cls.columns)
        assert "Compound_Name" in cols
        assert "SMILES" in cols
        assert "Class" in cols

    def test_smiles_column_detected(self, loaded_smiles_100_cls: pd.DataFrame) -> None:
        """SMILES column is detected as SMILES type. (T-018-04)"""
        dt = TypeDetector()
        res = dt.detect(loaded_smiles_100_cls)
        assert "SMILES" in res.smiles_columns

    def test_target_is_integer(self, loaded_smiles_100_cls: pd.DataFrame) -> None:
        """Class (target) is integer type. (T-018-05)"""
        assert pd.api.types.is_numeric_dtype(loaded_smiles_100_cls["Class"])

    def test_no_missing_values(self, loaded_smiles_100_cls: pd.DataFrame) -> None:
        """No missing values in the data. (T-018-06)"""
        assert loaded_smiles_100_cls.isna().sum().sum() == 0


# ============================================================
# T-019: Mixture 30 Simple Tests
# ============================================================

@pytest.fixture
def mixture_30_simple_path() -> Path:
    """Path to mixture_30_simple.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_30_simple.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_mixture_30_simple(mixture_30_simple_path: Path) -> pd.DataFrame:
    """Load mixture_30_simple.csv."""
    return load_file(mixture_30_simple_path)


class TestMixture30Simple:
    """T-019: Load mixture_30_simple.csv (mixture data with 3 compounds)."""

    def test_load_returns_dataframe(self, mixture_30_simple_path: Path) -> None:
        """load_file returns a DataFrame. (T-019-01)"""
        df = load_file(mixture_30_simple_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_30_rows_13_columns(self, loaded_mixture_30_simple: pd.DataFrame) -> None:
        """Loaded DataFrame has 30 rows and 13 columns (including Notes). (T-019-02)"""
        assert loaded_mixture_30_simple.shape[0] == 30, f"Expected 30 rows, got {loaded_mixture_30_simple.shape[0]}"
        assert loaded_mixture_30_simple.shape[1] == 13, f"Expected 13 columns, got {loaded_mixture_30_simple.shape[1]}"

    def test_has_smiles_columns(self, loaded_mixture_30_simple: pd.DataFrame) -> None:
        """Has SMILES columns for 3 compounds. (T-019-03)"""
        cols = list(loaded_mixture_30_simple.columns)
        assert "Compound_1_SMILES" in cols
        assert "Compound_2_SMILES" in cols
        assert "Compound_3_SMILES" in cols

    def test_has_wt_columns(self, loaded_mixture_30_simple: pd.DataFrame) -> None:
        """Has weight percentage columns. (T-019-04)"""
        cols = list(loaded_mixture_30_simple.columns)
        assert "Compound_1_WT%" in cols
        assert "Compound_2_WT%" in cols
        assert "Compound_3_WT%" in cols
        assert "Total_WT%" in cols

    def test_has_target_property(self, loaded_mixture_30_simple: pd.DataFrame) -> None:
        """Has Target_Property column. (T-019-05)"""
        assert "Target_Property" in loaded_mixture_30_simple.columns
        assert pd.api.types.is_numeric_dtype(loaded_mixture_30_simple["Target_Property"])

    def test_smiles_columns_detected(self, loaded_mixture_30_simple: pd.DataFrame) -> None:
        """SMILES columns are detected as SMILES type. (T-019-06)"""
        dt = TypeDetector()
        res = dt.detect(loaded_mixture_30_simple)
        assert "Compound_1_SMILES" in res.smiles_columns
        assert "Compound_2_SMILES" in res.smiles_columns
        assert "Compound_3_SMILES" in res.smiles_columns

    def test_total_wt_percent(self, loaded_mixture_30_simple: pd.DataFrame) -> None:
        """Total_WT% is close to 100. (T-019-07)"""
        # Total_WT% should be around 100 for each row
        for idx in range(min(5, len(loaded_mixture_30_simple))):
            wt = loaded_mixture_30_simple.iloc[idx]["Total_WT%"]
            assert abs(wt - 100) < 1.0, f"Total_WT% should be ~100, got {wt}"

    def test_no_missing_values(self, loaded_mixture_30_simple: pd.DataFrame) -> None:
        """No missing values in the data. (T-019-08)"""
        assert loaded_mixture_30_simple.isna().sum().sum() == 0


# ============================================================
# T-020: Mixture 30 Regression Tests
# ============================================================

@pytest.fixture
def mixture_30_reg_path() -> Path:
    """Path to mixture_30_regression.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_30_regression.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_mixture_30_reg(mixture_30_reg_path: Path) -> pd.DataFrame:
    """Load mixture_30_regression.csv."""
    return load_file(mixture_30_reg_path)


class TestMixture30Regression:
    """T-020: Load mixture_30_regression.csv."""

    def test_load_returns_dataframe(self, mixture_30_reg_path: Path) -> None:
        """load_file returns a DataFrame. (T-020-01)"""
        df = load_file(mixture_30_reg_path)
        assert isinstance(df, pd.DataFrame)

    def test_has_target_property(self, loaded_mixture_30_reg: pd.DataFrame) -> None:
        """Has Target_Property column (numeric). (T-020-02)"""
        assert "Target_Property" in loaded_mixture_30_reg.columns
        assert pd.api.types.is_numeric_dtype(loaded_mixture_30_reg["Target_Property"])

    def test_smiles_columns_detected(self, loaded_mixture_30_reg: pd.DataFrame) -> None:
        """SMILES columns are detected. (T-020-03)"""
        dt = TypeDetector()
        res = dt.detect(loaded_mixture_30_reg)
        assert len(res.smiles_columns) >= 3, f"Expected >=3 SMILES columns, got {res.smiles_columns}"


# ============================================================
# T-021: Mixture 3_30 Regression Tests
# ============================================================

@pytest.fixture
def mixture_3_30_reg_path() -> Path:
    """Path to mixture_3_30_regression.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_3_30_regression.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_mixture_3_30_reg(mixture_3_30_reg_path: Path) -> pd.DataFrame:
    """Load mixture_3_30_regression.csv."""
    return load_file(mixture_3_30_reg_path)


class TestMixture330Regression:
    """T-021: Load mixture_3_30_regression.csv."""

    def test_load_returns_dataframe(self, mixture_3_30_reg_path: Path) -> None:
        """load_file returns a DataFrame. (T-021-01)"""
        df = load_file(mixture_3_30_reg_path)
        assert isinstance(df, pd.DataFrame)

    def test_has_target_property(self, loaded_mixture_3_30_reg: pd.DataFrame) -> None:
        """Has Target_Property column. (T-021-02)"""
        assert "Target_Property" in loaded_mixture_3_30_reg.columns


# ============================================================
# T-022: Mixture 50 Debug Numeric Tests
# ============================================================

@pytest.fixture
def mixture_50_debug_path() -> Path:
    """Path to mixture_50_debug_numeric.csv."""
    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_50_debug_numeric.csv"
    assert p.exists(), f"Sample file not found: {p}"
    return p


@pytest.fixture
def loaded_mixture_50_debug(mixture_50_debug_path: Path) -> pd.DataFrame:
    """Load mixture_50_debug_numeric.csv."""
    return load_file(mixture_50_debug_path)


class TestMixture50DebugNumeric:
    """T-022: Load mixture_50_debug_numeric.csv (has SMILES columns too)."""

    def test_load_returns_dataframe(self, mixture_50_debug_path: Path) -> None:
        """load_file returns a DataFrame. (T-022-01)"""
        df = load_file(mixture_50_debug_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_50_rows_17_columns(self, loaded_mixture_50_debug: pd.DataFrame) -> None:
        """Loaded DataFrame has 50 rows and 17 columns. (T-022-02)"""
        assert loaded_mixture_50_debug.shape[0] == 50, f"Expected 50 rows, got {loaded_mixture_50_debug.shape[0]}"
        assert loaded_mixture_50_debug.shape[1] == 17, f"Expected 17 columns, got {loaded_mixture_50_debug.shape[1]}"

    def test_has_smiles_columns(self, loaded_mixture_50_debug: pd.DataFrame) -> None:
        """Has SMILES columns for compounds. (T-022-03)"""
        cols = list(loaded_mixture_50_debug.columns)
        assert "Compound_1_SMILES" in cols
        assert "Compound_2_SMILES" in cols
        assert "Compound_3_SMILES" in cols

    def test_numeric_columns_are_numeric(self, loaded_mixture_50_debug: pd.DataFrame) -> None:
        """Numeric columns (WT%, conditions, target) are numeric. (T-022-04)"""
        # Exclude non-numeric columns: Sample_ID, Name columns, SMILES columns
        non_numeric = ["Sample_ID",
                        "Compound_1_Name", "Compound_2_Name", "Compound_3_Name",
                        "Compound_1_SMILES", "Compound_2_SMILES", "Compound_3_SMILES"]
        for col in loaded_mixture_50_debug.columns:
            if col not in non_numeric:
                assert pd.api.types.is_numeric_dtype(loaded_mixture_50_debug[col]), f"{col} is not numeric"

    def test_has_target_column(self, loaded_mixture_50_debug: pd.DataFrame) -> None:
        """Has target column (Boiling_Point_C_Target). (T-022-05)"""
        assert "Boiling_Point_C(Target)" in loaded_mixture_50_debug.columns
        assert pd.api.types.is_numeric_dtype(loaded_mixture_50_debug["Boiling_Point_C(Target)"])


# ============================================================
# T-023: Debug Directory - Basic Loading Tests
# ============================================================

# Parameterized test for debug CSV files
debug_test_cases = [
    ("classification_balanced.csv", (100, 11), ["Feature_0"]),
    ("mixture_regression_debug.csv", (50, 19), ["Compound_1_SMILES"]),
    ("mixture_smiles_numeric.csv", (50, 19), ["Compound_1_SMILES"]),
    ("mixture_smiles_only.csv", (50, 14), ["Compound_1_SMILES"]),
    ("monotonicity_test.csv", (100, 4), ["MW"]),
    ("numeric_only_ml_features.csv", (5, 7), ["PCA_1"]),
    ("numeric_only_regression.csv", (5, 8), ["MolecularWeight"]),
    ("numeric_only_solubility.csv", (5, 7), ["MolecularWeight"]),
    ("simple_smiles_classification.csv", (10, 4), ["SMILES"]),
    ("simple_smiles_regression.csv", (10, 4), ["SMILES"]),
    ("timeseries_leak_test.csv", (50, 5), ["Date"]),
    ("xtb_dependency_test.csv", (20, 3), ["SMILES"]),
]


class TestDebugFiles:
    """T-023: Load all debug directory CSV files and verify basic properties."""

    @pytest.fixture(params=debug_test_cases, ids=lambda x: x[0])
    def debug_file_info(self, request):
        """Fixture providing debug file info: (filename, expected_shape, sample_cols)."""
        return request.param

    def test_load_debug_file(self, debug_file_info):
        """Load debug CSV file successfully. (T-023-01)"""
        filename, expected_shape, _ = debug_file_info
        p = Path(r"C:\Users\horie\chemai2_cc\data\samples\debug") / filename
        assert p.exists(), f"File not found: {p}"
        df = load_file(p)
        assert isinstance(df, pd.DataFrame)

    def test_shape_correct(self, debug_file_info):
        """Shape matches expected. (T-023-02)"""
        filename, expected_shape, _ = debug_file_info
        p = Path(r"C:\Users\horie\chemai2_cc\data\samples\debug") / filename
        df = load_file(p)
        assert df.shape == expected_shape, f"{filename}: expected {expected_shape}, got {df.shape}"

    def test_has_expected_columns(self, debug_file_info):
        """Has expected columns. (T-023-03)"""
        filename, _, sample_cols = debug_file_info
        p = Path(r"C:\Users\horie\chemai2_cc\data\samples\debug") / filename
        df = load_file(p)
        for col in sample_cols:
            assert col in df.columns, f"{filename}: column {col} not found"

    def test_no_duplicate_columns(self, debug_file_info):
        """No duplicate column names. (T-023-04)"""
        filename, _, _ = debug_file_info
        p = Path(r"C:\Users\horie\chemai2_cc\data\samples\debug") / filename
        df = load_file(p)
        assert len(df.columns) == len(set(df.columns)), f"{filename}: duplicate columns found"

    def test_smiles_detection_for_smiles_files(self, debug_file_info):
        """SMILES columns are detected in files that have them. (T-023-05)"""
        filename, _, sample_cols = debug_file_info
        if "SMILES" not in sample_cols[0]:
            pytest.skip("No SMILES columns in this file")
        p = Path(r"C:\Users\horie\chemai2_cc\data\samples\debug") / filename
        df = load_file(p)
        dt = TypeDetector()
        res = dt.detect(df)
        assert len(res.smiles_columns) > 0, f"{filename}: expected SMILES columns not detected"
