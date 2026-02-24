"""
tests/test_data.py

backend/data モジュールのユニットテスト。
TypeDetector, Preprocessor, Loader をテストする。
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.data.loader import (
    load_file,
    load_from_bytes,
    save_dataframe,
    get_supported_extensions,
)
from backend.data.type_detector import (
    TypeDetector,
    ColumnType,
    ColumnInfo,
    DetectionResult,
)
from backend.data.preprocessor import (
    Preprocessor,
    PreprocessConfig,
    LogTransformer,
    SinCosTransformer,
    build_full_pipeline,
)


# ============================================================
# テスト用データフィクスチャ
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """型判定テスト用のサンプルDataFrame。"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "numeric_normal": np.random.randn(n),                        # 正規
        "numeric_log": np.random.exponential(5, n),                  # 対数候補
        "binary_int": np.random.randint(0, 2, n),                    # バイナリ整数
        "cat_low": np.random.choice(["A", "B", "C"], n),             # カテゴリ低
        "cat_high": [f"item_{i}" for i in np.random.randint(0, 30, n)],  # カテゴリ高
        "smiles_col": ["CCO", "C", "CC", "CCC"] * (n // 4),         # SMILES
        "constant_col": [1.0] * n,                                   # 定数
    })


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """テスト用CSVファイルを作成する。"""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "target": [10.0, 20.0, 30.0],
    })
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    return p


# ============================================================
# T-001: TypeDetector テスト
# ============================================================

class TestTypeDetector:
    """T-001: 変数型自動判定のテスト。"""

    def test_detect_returns_detection_result(self, sample_df: pd.DataFrame) -> None:
        """TypeDetector.detect() が DetectionResult を返すこと。(T-001-01)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert isinstance(result, DetectionResult)

    def test_detect_numeric_normal(self, sample_df: pd.DataFrame) -> None:
        """正規分布の数値列が NUMERIC_NORMAL 判定されること。(T-001-02)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert result.column_info["numeric_normal"].col_type == ColumnType.NUMERIC_NORMAL

    def test_detect_numeric_log(self, sample_df: pd.DataFrame) -> None:
        """指数分布の数値列が NUMERIC_LOG 判定されること。(T-001-03)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert result.column_info["numeric_log"].col_type == ColumnType.NUMERIC_LOG

    def test_detect_binary(self, sample_df: pd.DataFrame) -> None:
        """0/1の整数列が BINARY 判定されること。(T-001-04)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert result.column_info["binary_int"].col_type == ColumnType.BINARY

    def test_detect_category_low(self, sample_df: pd.DataFrame) -> None:
        """少ないユニーク数のカテゴリが CATEGORY_LOW 判定されること。(T-001-05)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert result.column_info["cat_low"].col_type == ColumnType.CATEGORY_LOW

    def test_detect_category_high(self, sample_df: pd.DataFrame) -> None:
        """多いユニーク数のカテゴリが CATEGORY_HIGH 判定されること。(T-001-06)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert result.column_info["cat_high"].col_type == ColumnType.CATEGORY_HIGH

    def test_detect_smiles(self, sample_df: pd.DataFrame) -> None:
        """SMILES列が SMILES 判定されること。(T-001-07)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert result.column_info["smiles_col"].col_type == ColumnType.SMILES

    def test_detect_constant(self, sample_df: pd.DataFrame) -> None:
        """定数列が CONSTANT 判定されること。(T-001-08)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert result.column_info["constant_col"].col_type == ColumnType.CONSTANT

    def test_smiles_in_smiles_columns(self, sample_df: pd.DataFrame) -> None:
        """smiles_columns リストに SMILES列が含まれること。(T-001-09)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        assert "smiles_col" in result.smiles_columns

    def test_summary_table_shape(self, sample_df: pd.DataFrame) -> None:
        """summary_table() が列数と同じ行数のDataFrameを返すこと。(T-001-10)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        tbl = result.summary_table()
        assert len(tbl) == len(sample_df.columns)

    def test_get_numeric_columns(self, sample_df: pd.DataFrame) -> None:
        """get_numeric_columns() が数値列を返すこと。(T-001-11)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        nums = result.get_numeric_columns()
        assert "numeric_normal" in nums
        assert "cat_low" not in nums

    def test_datetime_detection(self) -> None:
        """datetime列が DATETIME 判定されること。(T-001-12)"""
        df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])})
        dt = TypeDetector()
        result = dt.detect(df)
        assert result.column_info["dt"].col_type == ColumnType.DATETIME

    def test_periodic_col_detection(self) -> None:
        """periodic_cols を指定した場合 PERIODIC 判定されること。(T-001-13)"""
        df = pd.DataFrame({"angle": [0.0, 90.0, 180.0, 270.0] * 5})
        dt = TypeDetector(periodic_cols=["angle"])
        result = dt.detect(df)
        assert result.column_info["angle"].col_type == ColumnType.PERIODIC

    def test_null_rate_calculation(self) -> None:
        """欠損率が正しく計算されること。(T-001-14)"""
        df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        dt = TypeDetector()
        result = dt.detect(df)
        assert abs(result.column_info["x"].null_rate - 0.4) < 1e-6


# ============================================================
# T-002: Loader テスト
# ============================================================

class TestLoader:
    """T-002: データローダーのテスト。"""

    def test_load_csv(self, csv_file: Path) -> None:
        """CSVファイルを正しく読み込めること。(T-002-01)"""
        df = load_file(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["a", "b", "target"]

    def test_load_excel(self, tmp_path: Path) -> None:
        """Excelファイルを読み込めること。(T-002-02)"""
        pytest.importorskip("openpyxl", reason="openpyxlが未インストール")
        df_orig = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = tmp_path / "test.xlsx"
        df_orig.to_excel(p, index=False)
        df = load_file(p)
        assert len(df) == 2

    def test_load_parquet(self, tmp_path: Path) -> None:
        """Parquetファイルを読み込めること。(T-002-03)"""
        df_orig = pd.DataFrame({"a": [10.0, 20.0], "b": [1, 2]})
        p = tmp_path / "test.parquet"
        df_orig.to_parquet(p, index=False)
        df = load_file(p)
        assert list(df.columns) == ["a", "b"]

    def test_load_json(self, tmp_path: Path) -> None:
        """JSONファイルを読み込めること。(T-002-04)"""
        df_orig = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]})
        p = tmp_path / "test.json"
        df_orig.to_json(p, orient="records", force_ascii=False)
        df = load_file(p)
        assert len(df) == 2

    def test_load_sqlite(self, tmp_path: Path) -> None:
        """SQLiteファイルを読み込めること。(T-002-05)"""
        import sqlite3
        p = tmp_path / "test.db"
        con = sqlite3.connect(p)
        pd.DataFrame({"id": [1, 2], "val": [0.1, 0.2]}).to_sql("data", con, index=False)
        con.close()
        df = load_file(p)
        assert "id" in df.columns

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """存在しないファイルで FileNotFoundError が上がること。(T-002-06)"""
        with pytest.raises(FileNotFoundError):
            load_file(tmp_path / "nonexistent.csv")

    def test_load_unsupported_ext(self, tmp_path: Path) -> None:
        """未対応拡張子で ValueError が上がること。(T-002-07)"""
        p = tmp_path / "test.xyz"
        p.write_text("hello")
        with pytest.raises(ValueError):
            load_file(p)

    def test_load_from_bytes_csv(self) -> None:
        """バイト列CSVを読み込めること。(T-002-08)"""
        content = b"a,b\n1,2\n3,4\n"
        df = load_from_bytes(content, "test.csv")
        assert len(df) == 2

    def test_save_csv(self, tmp_path: Path) -> None:
        """DataFrameをCSVで保存できること。(T-002-09)"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        p = tmp_path / "out.csv"
        result_path = save_dataframe(df, p)
        assert result_path.exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == 3

    def test_get_supported_extensions(self) -> None:
        """対応拡張子リストが非空であること。(T-002-10)"""
        exts = get_supported_extensions()
        assert isinstance(exts, list)
        assert ".csv" in exts


# ============================================================
# T-003: Preprocessor テスト
# ============================================================

class TestPreprocessor:
    """T-003: 前処理パイプライン構築のテスト。"""

    def test_build_returns_column_transformer(self, sample_df: pd.DataFrame) -> None:
        """build() が ColumnTransformer を返すこと。(T-003-01)"""
        from sklearn.compose import ColumnTransformer
        dt = TypeDetector()
        result = dt.detect(sample_df)
        pp = Preprocessor()
        ct = pp.build(result, target_col=None)
        assert isinstance(ct, ColumnTransformer)

    def test_fit_transform_shape(self, sample_df: pd.DataFrame) -> None:
        """fit_transform 後の出力をnumpy配列として取得できること。(T-003-02)"""
        dt = TypeDetector()
        result = dt.detect(sample_df)
        pp = Preprocessor(PreprocessConfig(exclude_smiles=True, exclude_constant=True))
        ct = pp.build(result, target_col=None)
        X_transformed = ct.fit_transform(sample_df)
        assert X_transformed.shape[0] == len(sample_df)

    def test_build_excludes_target_col(self, sample_df: pd.DataFrame) -> None:
        """前処理は target_col を除外すること。(T-003-03)"""
        # numeric_normal を target とした場合、変換後の列数が減ること
        dt = TypeDetector()
        result_with = dt.detect(sample_df.drop(columns=["smiles_col", "constant_col"]))
        result_without = dt.detect(sample_df.drop(columns=["smiles_col", "constant_col"]))

        pp1 = Preprocessor(PreprocessConfig(exclude_smiles=True, exclude_constant=True))
        pp2 = Preprocessor(PreprocessConfig(exclude_smiles=True, exclude_constant=True))

        ct1 = pp1.build(result_with, target_col=None)
        ct2 = pp2.build(result_without, target_col="numeric_normal")

        X1 = ct1.fit_transform(sample_df.drop(columns=["smiles_col", "constant_col"]))
        X2 = ct2.fit_transform(sample_df.drop(columns=["smiles_col", "constant_col"]))
        assert X2.shape[1] < X1.shape[1]

    def test_log_transformer(self) -> None:
        """LogTransformer が正しく変換・逆変換すること。(T-003-04)"""
        lt = LogTransformer()
        X = np.array([[1.0], [4.0], [9.0]])
        X_t = lt.transform(X)
        assert X_t.shape == X.shape
        X_inv = lt.inverse_transform(X_t)
        np.testing.assert_allclose(X_inv, X, atol=1e-6)

    def test_sincos_transformer(self) -> None:
        """SinCosTransformer が2列に変換すること。(T-003-05)"""
        sct = SinCosTransformer(period=360)
        X = np.array([[0.0], [90.0], [180.0], [270.0]])
        X_t = sct.transform(X)
        assert X_t.shape == (4, 2)
        # 0度のsin≈0, cos≈1
        np.testing.assert_allclose(X_t[0], [0.0, 1.0], atol=1e-6)

    def test_transformer_property_raises_before_build(self) -> None:
        """build()前にtransformerプロパティにアクセスするとRuntimeError。(T-003-06)"""
        pp = Preprocessor()
        with pytest.raises(RuntimeError):
            _ = pp.transformer

    def test_build_full_pipeline(self, sample_df: pd.DataFrame) -> None:
        """build_full_pipeline() がfitできること。(T-003-07)"""
        from sklearn.linear_model import LinearRegression
        dt = TypeDetector()
        X = sample_df.drop(columns=["smiles_col", "constant_col"])
        result = dt.detect(X.drop(columns=["numeric_normal"]))
        pipeline = build_full_pipeline(
            result,
            LinearRegression(),
            target_col=None,
            config=PreprocessConfig(exclude_smiles=True, exclude_constant=True),
        )
        y = sample_df["numeric_normal"].values
        X_fit = X.drop(columns=["numeric_normal"])
        pipeline.fit(X_fit, y)
        preds = pipeline.predict(X_fit)
        assert preds.shape == (len(y),)
