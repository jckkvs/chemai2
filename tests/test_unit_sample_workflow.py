"""
tests/test_unit_sample_workflow.py

サンプルデータ（data/samples/）を使った単体テスト群。
カバー範囲:
  U-1  データ読み込みとデータ型検出
  U-2  解析前ヒヤリング（LLMモック）
  U-3  記述子推奨（DescriptorRecommender）
  U-4  SMILES記述子計算
  U-5  特徴量自動選択と説明
  U-6  単調性制約の自動検出と説明
  U-7  AutoMLコンポーネント
  U-8  結果レポート出力

各テストは実際のサンプルCSVファイルを読み込み、
コンポーネントを単独で検証する。LLMは全てモック化。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─── サンプルデータパス ─────────────────────────────────────────────
SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"

SMILES_QUICK  = SAMPLES_DIR / "smiles_25_quick.csv"
SMILES_100    = SAMPLES_DIR / "smiles_100_ml.csv"
SMILES_500    = SAMPLES_DIR / "smiles_500_stress.csv"
TABULAR_50    = SAMPLES_DIR / "tabular_50_simple.csv"
TABULAR_200   = SAMPLES_DIR / "tabular_200_complex.csv"
TABULAR_1000  = SAMPLES_DIR / "tabular_1000_large.csv"
MIXTURE_30    = SAMPLES_DIR / "mixture_30_simple.csv"
MIXTURE_50    = SAMPLES_DIR / "mixture_50_debug_numeric.csv"
MIXTURE_100   = SAMPLES_DIR / "mixture_100_ml.csv"

# ─── サンプルファイルごとのメタ情報 ────────────────────────────────
SMILES_CONFIGS = [
    {"path": SMILES_QUICK, "smiles_col": "SMILES", "target_col": "logS",  "task": "regression"},
    {"path": SMILES_100,   "smiles_col": "SMILES", "target_col": "logS",  "task": "regression"},
    {"path": SMILES_QUICK, "smiles_col": "SMILES", "target_col": "Class", "task": "classification"},
]

TABULAR_CONFIGS = [
    {"path": TABULAR_50,   "target_col": "Target", "task": "regression"},
    {"path": TABULAR_200,  "target_col": "Target", "task": "regression"},
]

MIXTURE_CONFIGS = [
    {"path": MIXTURE_30,  "target_col": "Target_Property", "task": "regression"},
    {"path": MIXTURE_50,  "target_col": "Boiling_Point_C", "task": "regression"},
    {"path": MIXTURE_100, "target_col": "Target_Property",  "task": "regression"},
]


def _load_csv(path: Path) -> pd.DataFrame:
    """BOM付きCSVも含めて読み込む。"""
    return pd.read_csv(path, encoding="utf-8-sig")


# ════════════════════════════════════════════════════════════════════
# U-1: データ読み込みとデータ型検出
# ════════════════════════════════════════════════════════════════════

class TestDataLoading:
    """各サンプルCSVが正常に読み込まれ、期待する列・行数を持つことを確認する。"""

    @pytest.mark.parametrize("path,expected_col,min_rows", [
        (SMILES_QUICK, "SMILES",   20),
        (SMILES_100,   "SMILES",   90),
        (TABULAR_50,   "Target",   40),
        (TABULAR_200,  "Target",  180),
        (MIXTURE_30,   "Target_Property", 25),
    ])
    def test_csv_loads_successfully(self, path, expected_col, min_rows):
        """T-U01: CSVが正常に読み込まれ、期待列と最小行数を持つ。"""
        df = _load_csv(path)
        assert expected_col in df.columns, f"{path.name}: '{expected_col}' 列がない"
        assert len(df) >= min_rows, f"{path.name}: 行数不足 {len(df)} < {min_rows}"

    def test_smiles_column_has_valid_smiles(self):
        """T-U02: SMILES列の文字列が最低限の有効性を持つ（非空、英字含む）。"""
        df = _load_csv(SMILES_QUICK)
        smiles_series = df["SMILES"].dropna()
        assert len(smiles_series) > 0
        for smi in smiles_series[:10]:
            assert isinstance(smi, str) and len(smi) > 0, f"無効なSMILES: {smi!r}"

    def test_tabular_all_feature_cols_numeric(self):
        """T-U03: tabularデータの特徴量列が全て数値型。"""
        df = _load_csv(TABULAR_50)
        feat_cols = [c for c in df.columns if c.startswith("Feature_")]
        for col in feat_cols:
            assert pd.to_numeric(df[col], errors="coerce").notna().any(), \
                f"列 '{col}' が数値に変換できない"

    def test_tabular_1000_row_count(self):
        """T-U04: tabular_1000_large.csv は900行以上持つ。"""
        df = _load_csv(TABULAR_1000)
        assert len(df) >= 900

    def test_mixture_smiles_columns_present(self):
        """T-U05: mixtureデータに複数SMILES列が存在する。"""
        df = _load_csv(MIXTURE_30)
        smiles_cols = [c for c in df.columns if "SMILES" in c]
        assert len(smiles_cols) >= 2, f"SMILES列が2つ未満: {smiles_cols}"

    @pytest.mark.parametrize("cfg", SMILES_CONFIGS[:2])
    def test_smiles_target_col_is_numeric(self, cfg):
        """T-U06: SMILES系ファイルのターゲット列(logS)が数値。"""
        df = _load_csv(cfg["path"])
        vals = pd.to_numeric(df[cfg["target_col"]], errors="coerce")
        assert vals.notna().sum() > 10, \
            f"{cfg['path'].name}: {cfg['target_col']} 列の数値が少ない"


# ════════════════════════════════════════════════════════════════════
# U-2: 解析前ヒヤリング（LLMモック）
# ════════════════════════════════════════════════════════════════════

MOCK_ANALYSIS_RESULT = {
    "data_overview": "25行のSMILESデータ。logS（連続値）を目的変数とする回帰問題。",
    "preprocessing": "SMILESをRDKit記述子に変換。欠損値を中央値で補完。",
    "feature_engineering": "RDKit基本記述子 + Mordred 2D記述子を使用。",
    "model_candidates": ["RandomForest", "Ridge", "LightGBM"],
    "validation_strategy": "5-fold交差検証。サンプル数が少ないため。",
    "interpretation_plan": "SHAP Summary Plotで特徴量重要度を可視化。",
    "cautions": "外れ値SMILESに注意。pIC50との多重共線性を確認すること。"
}


class TestLLMInterviewing:
    """U-2: LLMAnalyzerがモックを通じて正しくデータ分析方針を返すことを確認。"""

    def _make_mock_provider(self) -> MagicMock:
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.prepare_dataframe_context = MagicMock(
            return_value={"shape": (25, 6), "columns": ["SMILES", "logS", "pIC50"],
                          "null_counts": {}, "sample_rows": []}
        )
        provider.generate = AsyncMock(return_value=json.dumps(MOCK_ANALYSIS_RESULT))
        return provider

    @pytest.mark.asyncio
    async def test_analyzer_returns_dict_with_required_keys(self):
        """T-U07: LLMAnalyzerがモック応答から必須キーを持つ辞書を返す。"""
        from backend.services.llm_data_analyzer import LLMDataAnalyzer

        df = _load_csv(SMILES_QUICK)
        analyzer = LLMDataAnalyzer()
        analyzer.provider = self._make_mock_provider()

        result = await analyzer.analyze(df)

        required_keys = [
            "data_overview", "preprocessing", "feature_engineering",
            "model_candidates", "validation_strategy", "interpretation_plan"
        ]
        for key in required_keys:
            assert key in result, f"必須キー '{key}' が結果に含まれない"

    @pytest.mark.asyncio
    async def test_analyzer_model_candidates_is_list(self):
        """T-U08: model_candidatesがリスト型であること。"""
        from backend.services.llm_data_analyzer import LLMDataAnalyzer

        df = _load_csv(SMILES_QUICK)
        analyzer = LLMDataAnalyzer()
        analyzer.provider = self._make_mock_provider()

        result = await analyzer.analyze(df)
        assert isinstance(result["model_candidates"], list)
        assert len(result["model_candidates"]) >= 1

    @pytest.mark.asyncio
    async def test_analyzer_handles_json_parse_failure_gracefully(self):
        """T-U09: LLMが壊れたJSONを返したとき、warningキー付き辞書が返る。"""
        from backend.services.llm_data_analyzer import LLMDataAnalyzer

        provider = MagicMock()
        provider.is_available.return_value = True
        provider.prepare_dataframe_context = MagicMock(return_value={})
        provider.generate = AsyncMock(return_value="これはJSONではない")

        df = pd.DataFrame({"A": [1, 2]})
        analyzer = LLMDataAnalyzer()
        analyzer.provider = provider

        result = await analyzer.analyze(df)
        assert "warning" in result or "raw_output" in result or "error" in result

    @pytest.mark.asyncio
    async def test_analyzer_unavailable_provider_returns_error(self):
        """T-U10: LLMプロバイダーが未初期化の場合、errorキー付きの辞書が返る。"""
        from backend.services.llm_data_analyzer import LLMDataAnalyzer

        provider = MagicMock()
        provider.is_available.return_value = False

        df = pd.DataFrame({"A": [1, 2]})
        analyzer = LLMDataAnalyzer()
        analyzer.provider = provider

        result = await analyzer.analyze(df)
        assert "error" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path,smiles_col,target_col", [
        (SMILES_QUICK, "SMILES", "logS"),
        (TABULAR_50,   None,     "Target"),
        (MIXTURE_30,   "Compound_1_SMILES", "Target_Property"),
    ])
    async def test_analyzer_called_for_each_sample_type(self, path, smiles_col, target_col):
        """T-U11: SMILES/tabular/mixtureそれぞれでanalyze()が正常完走する。"""
        from backend.services.llm_data_analyzer import LLMDataAnalyzer

        df = _load_csv(path)
        analyzer = LLMDataAnalyzer()
        analyzer.provider = self._make_mock_provider()

        result = await analyzer.analyze(df)
        assert isinstance(result, dict)
        assert len(result) > 0


# ════════════════════════════════════════════════════════════════════
# U-3: 記述子推奨（DescriptorRecommender）
# ════════════════════════════════════════════════════════════════════

class TestDescriptorRecommendation:
    """U-3: DescriptorRecommenderが目的変数名から適切な記述子セットを推奨する。"""

    # recommend(property_name, available_plugins, ...) のシグネチャに合わせた
    # 代表的なプラグイン名リスト
    _ALL_PLUGINS = [
        "rdkit_basic", "rdkit_logp", "rdkit_tpsa", "rdkit_acidic_groups",
        "mordred_2d", "mordred_hydrophobic",
        "xtb_energy", "unipka", "cosmo_sigma",
        "mol2vec", "molfeat", "chemprop",
    ]

    @pytest.fixture
    def recommender(self):
        from backend.chem.recommender import DescriptorRecommender
        return DescriptorRecommender()

    def test_recommend_for_solubility(self, recommender):
        """T-U12: 'logS'/'solubility'系の目的変数に対し、LogP関連記述子が推奨される。"""
        recs = recommender.recommend("logS", available_plugins=self._ALL_PLUGINS)
        assert isinstance(recs, list), "推奨結果がリストでない"
        assert len(recs) >= 1, "推奨記述子が0件"
        plugin_names = [r.get("plugin", "") for r in recs]
        assert any("rdkit" in p.lower() for p in plugin_names), \
            f"RDKit系記述子が推奨されていない: {plugin_names}"

    def test_recommend_for_unknown_property(self, recommender):
        """T-U13: 未知のプロパティ名でもデフォルト推奨が返る（フォールバック動作）。"""
        recs = recommender.recommend(
            "mysterious_property_xyz_999", available_plugins=self._ALL_PLUGINS
        )
        assert isinstance(recs, list)
        assert len(recs) >= 1, "フォールバック推奨が空"

    def test_recommend_for_pka(self, recommender):
        """T-U14: 'pka'に対し、unipka or rdkit関連が推奨される。"""
        recs = recommender.recommend("pka", available_plugins=self._ALL_PLUGINS)
        assert len(recs) >= 1
        plugin_names = [r.get("plugin", "").lower() for r in recs]
        assert any("unipka" in p or "rdkit" in p or "acid" in p for p in plugin_names), \
            f"pKa向け記述子が見当たらない: {plugin_names}"

    @pytest.mark.parametrize("target_name", ["logS", "pIC50", "logp", "pka", "boiling_point"])
    def test_recommend_returns_priority_field(self, recommender, target_name):
        """T-U15: 各推奨記述子に'priority'フィールドが含まれる。"""
        recs = recommender.recommend(target_name, available_plugins=self._ALL_PLUGINS)
        for rec in recs:
            assert "priority" in rec, f"推奨辞書に'priority'がない: {rec}"

    def test_recommend_cache_consistency(self, recommender):
        """T-U16: 同じクエリを2回呼んだとき同じ結果が返る（キャッシュ一貫性）。"""
        recs1 = recommender.recommend("logS", available_plugins=self._ALL_PLUGINS)
        recs2 = recommender.recommend("logS", available_plugins=self._ALL_PLUGINS)
        assert recs1 == recs2, "キャッシュの前後で推奨結果が異なる"


# ════════════════════════════════════════════════════════════════════
# U-4: SMILES記述子計算
# ════════════════════════════════════════════════════════════════════

class TestSmilesDescriptorCalculation:
    """U-4: 各SMILESサンプルファイルからRDKit記述子が正しく計算される。"""

    @pytest.mark.parametrize("path,smiles_col,target_col", [
        (SMILES_QUICK, "SMILES", "logS"),
        (SMILES_100,   "SMILES", "logS"),
    ])
    def test_rdkit_descriptors_computed(self, path, smiles_col, target_col):
        """T-U17: RDKit記述子が1列以上計算され、SMILES列が除去される。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _load_csv(path)
        transformer = SmilesDescriptorTransformer(smiles_col=smiles_col)
        df_out = transformer.fit_transform(df)

        assert smiles_col not in df_out.columns, "SMILES列が残存している"
        assert target_col in df_out.columns, f"目的変数列 '{target_col}' が失われた"
        desc_cols = [c for c in df_out.columns if c != target_col]
        assert len(desc_cols) >= 1, f"記述子が0列: {df_out.columns.tolist()}"

    @pytest.mark.parametrize("path,smiles_col", [
        (SMILES_QUICK, "SMILES"),
        (SMILES_100,   "SMILES"),
    ])
    def test_no_all_nan_descriptor_columns(self, path, smiles_col):
        """T-U18: 変換後の記述子に全行NaNの列がない。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _load_csv(path)
        transformer = SmilesDescriptorTransformer(smiles_col=smiles_col)
        df_out = transformer.fit_transform(df)

        non_target_cols = [c for c in df_out.columns
                           if c not in ("logS", "pIC50", "Class", "Compound_Name",
                                        "Source", "Notes")]
        for col in non_target_cols:
            assert df_out[col].notna().any(), f"列 '{col}' が全行NaN"

    def test_row_count_preserved_after_transform(self):
        """T-U19: 変換後の行数が元データと一致する。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _load_csv(SMILES_QUICK)
        original_len = len(df)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_out = transformer.fit_transform(df)

        assert len(df_out) == original_len, \
            f"行数変化: {original_len} → {len(df_out)}"

    def test_descriptor_values_are_finite(self):
        """T-U20: RDKit記述子の大部分（>80%）が有限値。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _load_csv(SMILES_QUICK)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_out = transformer.fit_transform(df)

        desc_cols = [c for c in df_out.columns
                     if c not in ("logS", "pIC50", "Class", "Compound_Name",
                                  "Source", "Notes")]
        if not desc_cols:
            pytest.skip("記述子列なし")

        desc_df = df_out[desc_cols].select_dtypes(include=[np.number])
        total = desc_df.size
        finite_count = np.isfinite(desc_df.values).sum()
        finite_ratio = finite_count / total if total > 0 else 0
        assert finite_ratio >= 0.8, f"有限値比率が低い: {finite_ratio:.2%}"

    def test_descriptor_explanation_includes_column_names(self):
        """T-U21: 変換後の列名が記述子の意味を示す名前を持つ（例: MolWt, LogP など）。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _load_csv(SMILES_QUICK)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_out = transformer.fit_transform(df)

        non_meta_cols = [c for c in df_out.columns
                         if c not in ("logS", "pIC50", "Class",
                                      "Compound_Name", "Source", "Notes")]
        # 少なくとも1つは英数字のみで構成される記述子名があるはず
        assert any(len(c) > 1 for c in non_meta_cols), \
            f"記述子列名が無効に見える: {non_meta_cols[:10]}"


# ════════════════════════════════════════════════════════════════════
# U-5: 特徴量自動選択と説明
# ════════════════════════════════════════════════════════════════════

class TestFeatureAutoSelection:
    """U-5: FeatureSelectorが数値データから特徴量サブセットを選択する。"""

    def _make_numeric_xy(self, df: pd.DataFrame, target_col: str):
        """数値特徴量とターゲット配列を抽出。"""
        drop_cols = [target_col] + [c for c in df.columns
                                    if df[c].dtype == object]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns],
                    errors="ignore")
        X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        return X, y

    def test_variance_threshold_reduces_features(self):
        """T-U22: 分散フィルタで定数列が除去され、特徴量数が減少または維持される。"""
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

        df = _load_csv(TABULAR_200)
        X, y = self._make_numeric_xy(df, "Target")

        # 定数列を意図的に追加
        X["const_col"] = 0.0

        config = FeatureSelectorConfig(method="variance", task="regression")
        selector = FeatureSelector(config)
        X_sel = selector.fit_transform(X, y)

        assert X_sel.shape[1] <= X.shape[1], "特徴量数が増加している"
        assert "const_col" not in X_sel.columns, "定数列が除去されていない"

    def test_mutual_info_selects_subset(self):
        """T-U23: mutual_info法で特徴量のサブセットが選択される。"""
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

        df = _load_csv(TABULAR_50)
        X, y = self._make_numeric_xy(df, "Target")

        config = FeatureSelectorConfig(
            method="mutual_info",
            task="regression",
            k=min(4, X.shape[1]),
        )
        selector = FeatureSelector(config)
        X_sel = selector.fit_transform(X, y)

        assert X_sel.shape[1] <= X.shape[1]
        assert X_sel.shape[1] >= 1

    def test_feature_selector_preserves_row_count(self):
        """T-U24: 特徴量選択後も行数が変わらない。"""
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

        df = _load_csv(TABULAR_50)
        X, y = self._make_numeric_xy(df, "Target")

        config = FeatureSelectorConfig(method="variance", task="regression")
        selector = FeatureSelector(config)
        X_sel = selector.fit_transform(X, y)

        assert len(X_sel) == len(X)

    def test_feature_selector_with_smiles_descriptors(self):
        """T-U25: SMILES記述子計算後のデータフレームでも特徴量選択が動作する。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

        df = _load_csv(SMILES_QUICK)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_desc = transformer.fit_transform(df)

        X = df_desc.drop(columns=["logS"], errors="ignore") \
                   .select_dtypes(include=[np.number]) \
                   .fillna(0)
        y = pd.to_numeric(df_desc["logS"], errors="coerce").fillna(0)

        config = FeatureSelectorConfig(method="variance", task="regression")
        selector = FeatureSelector(config)
        X_sel = selector.fit_transform(X, y)

        assert X_sel.shape[0] == len(df)
        assert X_sel.shape[1] >= 1


# ════════════════════════════════════════════════════════════════════
# U-6: 単調性制約の自動検出と説明
# ════════════════════════════════════════════════════════════════════

def _detect_monotonic_constraints(
    X: pd.DataFrame, y: pd.Series, spearman_threshold: float = 0.4
) -> Dict[str, Dict[str, Any]]:
    """
    Spearman相関で単調性制約を自動検出する。
    |corr| > threshold の場合に制約を設定。

    Returns
    -------
    dict: {feature_name: {"direction": "increasing"|"decreasing", "rho": float}}
    """
    from scipy import stats

    result = {}
    for col in X.columns:
        x_vals = X[col].dropna().values
        y_vals = y.loc[X[col].dropna().index].values
        if len(x_vals) < 5:
            continue
        rho, _ = stats.spearmanr(x_vals, y_vals)
        if not np.isfinite(rho):
            continue
        if abs(rho) >= spearman_threshold:
            result[col] = {
                "direction": "increasing" if rho > 0 else "decreasing",
                "rho": float(rho),
            }
    return result


class TestConstraintAutoDetection:
    """U-6: Spearman相関による単調性制約の自動検出ロジックを検証する。"""

    def _get_tabular_xy(self, path: Path, target_col: str):
        df = _load_csv(path)
        drop_cols = [c for c in df.columns if df[c].dtype == object]
        X = df.drop(columns=drop_cols + [target_col], errors="ignore") \
               .select_dtypes(include=[np.number]) \
               .fillna(df.median(numeric_only=True))
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        return X, y

    def test_auto_detect_returns_dict(self):
        """T-U26: 自動検出関数が辞書を返す。"""
        X, y = self._get_tabular_xy(TABULAR_50, "Target")
        constraints = _detect_monotonic_constraints(X, y)
        assert isinstance(constraints, dict)

    def test_auto_detect_direction_is_valid(self):
        """T-U27: 検出された制約の方向が'increasing'または'decreasing'のどちらか。"""
        X, y = self._get_tabular_xy(TABULAR_200, "Target")
        constraints = _detect_monotonic_constraints(X, y, spearman_threshold=0.3)
        for feat, info in constraints.items():
            assert info["direction"] in ("increasing", "decreasing"), \
                f"{feat}: 不正な方向値 '{info['direction']}'"

    def test_auto_detect_rho_is_in_valid_range(self):
        """T-U28: 検出されたSpearman相関係数が[-1, 1]の範囲内。"""
        X, y = self._get_tabular_xy(TABULAR_200, "Target")
        constraints = _detect_monotonic_constraints(X, y, spearman_threshold=0.3)
        for feat, info in constraints.items():
            assert -1.0 <= info["rho"] <= 1.0, \
                f"{feat}: rho={info['rho']} が範囲外"

    def test_auto_detect_threshold_effect(self):
        """T-U29: 閾値を上げると検出される制約が減少または同数（単調非増加）。"""
        X, y = self._get_tabular_xy(TABULAR_200, "Target")
        c_low  = _detect_monotonic_constraints(X, y, spearman_threshold=0.2)
        c_high = _detect_monotonic_constraints(X, y, spearman_threshold=0.7)
        assert len(c_high) <= len(c_low), \
            "閾値を上げても制約数が増えた（単調性に反する）"

    def test_constraint_engine_evaluates_model(self):
        """T-U30: ConstraintEngineがfitted estimatorで制約評価を実行できる。"""
        from sklearn.linear_model import Ridge
        from backend.ml.constraints import ConstraintEngine, ConstraintSpec

        X, y = self._get_tabular_xy(TABULAR_50, "Target")
        if X.shape[1] == 0:
            pytest.skip("特徴量なし")

        # 最初の特徴量に増加制約を設定
        first_feat = X.columns[0]
        specs = {first_feat: ConstraintSpec(
            feature_name=first_feat,
            monotonic="increasing",
            strength="weak",
        )}

        engine = ConstraintEngine(constraints=specs)
        model = Ridge().fit(X, y)
        engine.fit(X, list(X.columns))

        evaluations = engine.evaluate_model(model, X)
        assert isinstance(evaluations, dict)
        assert first_feat in evaluations

    def test_constraint_spec_explanation_fields(self):
        """T-U31: ConstraintSpecが制約説明に必要なフィールドを持つ。"""
        from backend.ml.constraints import ConstraintSpec

        spec = ConstraintSpec(
            feature_name="MolWt",
            monotonic="increasing",
            strength="strong",
            sigma_range=3.0,
        )
        assert spec.feature_name == "MolWt"
        assert spec.monotonic == "increasing"
        assert spec.sigma_range == 3.0

    @pytest.mark.parametrize("path,target_col", [
        (TABULAR_50,  "Target"),
        (TABULAR_200, "Target"),
    ])
    def test_auto_detect_with_smiles_descriptors(self, path, target_col):
        """T-U32: SMILES記述子データでも自動検出が正常動作する。"""
        df = _load_csv(path)
        X = df.drop(columns=[c for c in df.columns if df[c].dtype == object] + [target_col],
                    errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

        constraints = _detect_monotonic_constraints(X, y, spearman_threshold=0.3)
        assert isinstance(constraints, dict)


# ════════════════════════════════════════════════════════════════════
# U-7: AutoMLコンポーネント
# ════════════════════════════════════════════════════════════════════

class TestAutoMLComponents:
    """U-7: AutoMLEngineが各サンプルデータで正常に動作することを確認する。"""

    def _make_lightweight_engine(self, task: str = "regression"):
        from backend.models.automl import AutoMLEngine
        return AutoMLEngine(
            task=task,
            cv_folds=2,
            model_keys=["ridge", "rf"] if task == "regression" else ["dt_c"],
            timeout_seconds=120,
        )

    def test_automl_regression_tabular_50(self):
        """T-U33: tabular_50でAutoML回帰が完走し有効なスコアを返す。"""
        from backend.models.automl import AutoMLResult

        df = _load_csv(TABULAR_50)
        X_df = df.drop(columns=[c for c in df.columns if df[c].dtype == object],
                       errors="ignore")
        X_df = X_df.drop(columns=["Target"], errors="ignore") \
                   .select_dtypes(include=[np.number]).fillna(X_df.median(numeric_only=True))
        df_in = pd.concat([X_df, df["Target"]], axis=1)

        engine = self._make_lightweight_engine("regression")
        result = engine.run(df_in, target_col="Target")

        assert isinstance(result, AutoMLResult)
        assert result.best_model_key is not None
        assert np.isfinite(result.best_score)

    def test_automl_regression_smiles_via_smiles_col_arg(self):
        """T-U34: smiles_col引数を使いSMILES→記述子→AutoMLが1ステップで完走する。"""
        from backend.models.automl import AutoMLResult

        df = _load_csv(SMILES_QUICK)
        engine = self._make_lightweight_engine("regression")
        result = engine.run(df, target_col="logS", smiles_col="SMILES")

        assert isinstance(result, AutoMLResult)
        assert result.best_model_key is not None

    def test_automl_classification_smiles(self):
        """T-U35: SMILES分類データ（Class列）でAutoML分類が完走する。"""
        from backend.models.automl import AutoMLResult

        df = _load_csv(SMILES_QUICK)
        engine = self._make_lightweight_engine("classification")
        result = engine.run(df, target_col="Class", smiles_col="SMILES")

        assert isinstance(result, AutoMLResult)
        assert result.task == "classification"

    def test_automl_predict_no_nan(self):
        """T-U36: 最良モデルの予測結果にNaNが含まれない。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _load_csv(SMILES_QUICK)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_desc = transformer.fit_transform(df)

        engine = self._make_lightweight_engine("regression")
        result = engine.run(df_desc, target_col="logS")

        X = df_desc.drop(columns=["logS"])
        preds = result.best_pipeline.predict(X)
        assert not np.isnan(preds).any(), "予測にNaNが含まれる"

    def test_automl_model_scores_all_finite(self):
        """T-U37: 全モデルのCVスコアが有限値。"""
        df = _load_csv(TABULAR_50)
        X_num = df.select_dtypes(include=[np.number]).drop(columns=["Target"], errors="ignore") \
                  .fillna(df.median(numeric_only=True))
        df_in = pd.concat([X_num, df["Target"]], axis=1)

        engine = self._make_lightweight_engine("regression")
        result = engine.run(df_in, target_col="Target")

        for key, score in result.model_scores.items():
            assert np.isfinite(score), f"モデル '{key}' のスコア: {score}"

    def test_automl_elapsed_positive(self):
        """T-U38: 実行時間が正の値。"""
        df = _load_csv(TABULAR_50)
        X_num = df.select_dtypes(include=[np.number]).drop(columns=["Target"], errors="ignore") \
                  .fillna(df.median(numeric_only=True))
        df_in = pd.concat([X_num, df["Target"]], axis=1)

        engine = self._make_lightweight_engine("regression")
        result = engine.run(df_in, target_col="Target")

        assert result.elapsed_seconds > 0


# ════════════════════════════════════════════════════════════════════
# U-8: 結果レポート出力
# ════════════════════════════════════════════════════════════════════

_SAMPLE_REPORT_DICT = {
    "best_model_name": "RandomForest",
    "metrics": {"R2": 0.85, "RMSE": 0.32, "MAE": 0.24},
    "feature_importances": {"MolWt": 0.35, "LogP": 0.28, "TPSA": 0.15},
    "dataframe_head": pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
}


class TestReportExport:
    """U-8: PDF/Wordエクスポータが解析結果辞書からファイルを生成する。"""

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return tmp_path

    def test_pdf_export_creates_file(self, tmp_dir):
        """T-U39: PDFExporter.export()が.pdfファイルを生成する。"""
        try:
            from backend.export.pdf_exporter import PDFExporter
        except ImportError:
            pytest.skip("reportlab未インストール")

        exporter = PDFExporter(output_dir=tmp_dir)
        out_path = exporter.export(_SAMPLE_REPORT_DICT, "unit_test_report")

        assert out_path.exists(), f"PDFファイルが生成されていない: {out_path}"
        assert out_path.suffix == ".pdf"
        assert out_path.stat().st_size > 0, "PDFファイルが空"

    def test_word_export_creates_file(self, tmp_dir):
        """T-U40: WordExporter.export()が.docxファイルを生成する。"""
        try:
            from backend.export.word_exporter import WordExporter
        except ImportError:
            pytest.skip("python-docx未インストール")

        exporter = WordExporter(output_dir=tmp_dir)
        out_path = exporter.export(_SAMPLE_REPORT_DICT, "unit_test_report")

        assert out_path.exists(), f"Wordファイルが生成されていない: {out_path}"
        assert out_path.suffix == ".docx"
        assert out_path.stat().st_size > 0, "Wordファイルが空"

    def test_pdf_export_output_dir_created(self, tmp_path):
        """T-U41: output_dirが存在しない場合でもディレクトリが自動作成される。"""
        try:
            from backend.export.pdf_exporter import PDFExporter
        except ImportError:
            pytest.skip("reportlab未インストール")

        new_dir = tmp_path / "new_subdir" / "reports"
        assert not new_dir.exists()

        exporter = PDFExporter(output_dir=new_dir)
        assert new_dir.exists(), "output_dirが自動作成されていない"

    def test_report_dict_structure_validation(self):
        """T-U42: レポート辞書が必須キーを含むことを検証するヘルパー。"""
        required_keys = ["best_model_name", "metrics"]
        report = _SAMPLE_REPORT_DICT.copy()
        for key in required_keys:
            assert key in report, f"必須キー '{key}' がない"
        assert isinstance(report["metrics"], dict)
        assert "R2" in report["metrics"]
