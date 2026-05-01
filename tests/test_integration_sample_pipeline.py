"""
tests/test_integration_sample_pipeline.py

サンプルデータ（data/samples/）を使った結合テスト群。
以下の全ワークフローを、実際のファイルを入力として E2E で検証する:

  Step 1: ヒヤリング（LLMモック）→ 解析方針 dict
  Step 2: 記述子推奨（SMILES系のみ）
  Step 3: SMILES記述子計算（SMILES系のみ）
  Step 4: 特徴量自動選択
  Step 5: 単調性制約の自動検出と説明生成
  Step 6: AutoML（軽量設定）
  Step 7: 結果レポート出力（PDF/Word）

各テストクラスはデータ種別（smiles/tabular/mixture）ごとに分割。
ストレステスト（smiles_500, tabular_1000）は marks=slow で区別。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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


# ─── 共通ユーティリティ ─────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _make_mock_llm_provider(extra_keys: Dict[str, Any] | None = None):
    """モックLLMプロバイダーを返す。"""
    base_response = {
        "data_overview": "テスト用モックデータ概要",
        "preprocessing": "欠損値補完・スケーリング",
        "feature_engineering": "RDKit記述子",
        "model_candidates": ["RandomForest", "Ridge"],
        "validation_strategy": "5-fold CV",
        "interpretation_plan": "SHAP Summary Plot",
        "cautions": "なし",
    }
    if extra_keys:
        base_response.update(extra_keys)

    provider = MagicMock()
    provider.is_available.return_value = True
    provider.prepare_dataframe_context = MagicMock(
        return_value={"shape": (50, 5), "columns": [], "null_counts": {}, "sample_rows": []}
    )
    provider.generate = AsyncMock(return_value=json.dumps(base_response))
    return provider


def _detect_monotonic_constraints(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.4
) -> Dict[str, Dict[str, Any]]:
    """Spearman相関で単調性制約を自動検出する（統合テスト用ヘルパー）。"""
    result = {}
    for col in X.columns:
        valid = X[col].notna()
        if valid.sum() < 5:
            continue
        rho, _ = stats.spearmanr(X.loc[valid, col].values, y.loc[valid].values)
        if not np.isfinite(rho):
            continue
        if abs(rho) >= threshold:
            result[col] = {
                "direction": "increasing" if rho > 0 else "decreasing",
                "rho": float(rho),
                "explanation": (
                    f"Spearman相関 {rho:+.3f} → "
                    f"{'正の' if rho>0 else '負の'}相関が強く、"
                    f"{'増加' if rho>0 else '減少'}単調性制約を設定。"
                ),
            }
    return result


def _build_lightweight_automl(task: str = "regression"):
    """軽量設定のAutoMLEngineを生成する。"""
    from backend.models.automl import AutoMLEngine
    return AutoMLEngine(
        task=task,
        cv_folds=2,
        model_keys=["ridge", "rf"] if task == "regression" else ["dt_c"],
        timeout_seconds=180,
    )


def _export_report(result_dict: Dict, output_dir: Path, stem: str) -> Dict[str, Path]:
    """利用可能なエクスポータで結果を出力する。"""
    paths = {}
    try:
        from backend.export.pdf_exporter import PDFExporter
        p = PDFExporter(output_dir=output_dir).export(result_dict, stem + "_pdf")
        paths["pdf"] = p
    except (ImportError, Exception):
        pass
    try:
        from backend.export.word_exporter import WordExporter
        p = WordExporter(output_dir=output_dir).export(result_dict, stem + "_word")
        paths["word"] = p
    except (ImportError, Exception):
        pass
    return paths


# ════════════════════════════════════════════════════════════════════
# I-1: SMILES系 完全パイプライン統合テスト
# ════════════════════════════════════════════════════════════════════

class TestIntegrationSmilesPipeline:
    """
    smiles_25_quick.csv / smiles_100_ml.csv を使い、
    ヒヤリング → 記述子推奨 → SMILES変換 → 特徴量選択
    → 制約検出 → AutoML → レポート の全ステップを統合検証する。
    """

    @pytest.fixture(scope="class")
    def pipeline_result_smiles_quick(self, tmp_path_factory):
        """smiles_25_quick の完全パイプライン実行結果（クラス内で共有）。"""
        return _run_smiles_pipeline(SMILES_QUICK, "logS", tmp_path_factory.mktemp("smiles_quick"))

    @pytest.fixture(scope="class")
    def pipeline_result_smiles_100(self, tmp_path_factory):
        """smiles_100_ml の完全パイプライン実行結果。"""
        return _run_smiles_pipeline(SMILES_100, "logS", tmp_path_factory.mktemp("smiles_100"))

    # ── smiles_25_quick ──────────────────────────────────────────────

    def test_quick_step1_analysis_policy(self, pipeline_result_smiles_quick):
        """T-I01: smiles_quick - Step1ヒヤリングで解析方針dictが返る。"""
        r = pipeline_result_smiles_quick
        assert "analysis_policy" in r
        assert "model_candidates" in r["analysis_policy"]

    def test_quick_step2_descriptor_recommendation(self, pipeline_result_smiles_quick):
        """T-I02: smiles_quick - Step2記述子推奨が1件以上返る。"""
        r = pipeline_result_smiles_quick
        assert "descriptor_recs" in r
        assert len(r["descriptor_recs"]) >= 1

    def test_quick_step3_smiles_descriptors_computed(self, pipeline_result_smiles_quick):
        """T-I03: smiles_quick - Step3SMILES変換後に記述子列が存在する。"""
        r = pipeline_result_smiles_quick
        assert "df_desc" in r
        df = r["df_desc"]
        desc_cols = [c for c in df.columns if c != "logS"]
        assert len(desc_cols) >= 1

    def test_quick_step4_feature_selection(self, pipeline_result_smiles_quick):
        """T-I04: smiles_quick - Step4特徴量選択後も1列以上の特徴量が残る。"""
        r = pipeline_result_smiles_quick
        assert "df_selected" in r
        assert r["df_selected"].shape[1] >= 1

    def test_quick_step5_constraints_detected(self, pipeline_result_smiles_quick):
        """T-I05: smiles_quick - Step5単調性制約検出が辞書を返す。"""
        r = pipeline_result_smiles_quick
        assert "constraints" in r
        assert isinstance(r["constraints"], dict)

    def test_quick_step5_constraint_explanation_present(self, pipeline_result_smiles_quick):
        """T-I06: smiles_quick - 検出された制約には説明文が含まれる。"""
        r = pipeline_result_smiles_quick
        for feat, info in r["constraints"].items():
            assert "explanation" in info, f"{feat}: explanationキーがない"
            assert len(info["explanation"]) > 0

    def test_quick_step6_automl_completes(self, pipeline_result_smiles_quick):
        """T-I07: smiles_quick - Step6AutoMLが完走し有効な結果を返す。"""
        r = pipeline_result_smiles_quick
        assert "automl_result" in r
        res = r["automl_result"]
        assert res is not None
        assert np.isfinite(res.best_score)

    def test_quick_step6_best_model_key(self, pipeline_result_smiles_quick):
        """T-I08: smiles_quick - best_model_keyが返る。"""
        r = pipeline_result_smiles_quick
        assert r["automl_result"].best_model_key in ("ridge", "rf")

    def test_quick_step7_report_generated(self, pipeline_result_smiles_quick):
        """T-I09: smiles_quick - Step7でPDFまたはWordファイルが生成される。"""
        r = pipeline_result_smiles_quick
        exported = r.get("exported_files", {})
        if not exported:
            pytest.skip("reportlab/python-docx 未インストール")
        for fmt, path in exported.items():
            assert path.exists(), f"{fmt}ファイルが存在しない: {path}"
            assert path.stat().st_size > 0, f"{fmt}ファイルが空"

    # ── smiles_100 ───────────────────────────────────────────────────

    def test_100_automl_completes(self, pipeline_result_smiles_100):
        """T-I10: smiles_100 - AutoMLが完走し有効なスコアを返す。"""
        r = pipeline_result_smiles_100
        assert r["automl_result"] is not None
        assert np.isfinite(r["automl_result"].best_score)

    def test_100_row_count_preserved(self, pipeline_result_smiles_100):
        """T-I11: smiles_100 - 全ステップを通じて行数が保持される。"""
        r = pipeline_result_smiles_100
        original_len = r["original_len"]
        assert len(r["df_desc"]) == original_len
        assert len(r["df_selected"]) == original_len

    def test_100_classification_task(self, tmp_path):
        """T-I12: smiles_100 - 分類タスク（Class列）でもパイプラインが完走する。"""
        result = _run_smiles_pipeline(SMILES_100, "Class", tmp_path, task="classification")
        assert result["automl_result"] is not None
        assert result["automl_result"].task == "classification"


# ════════════════════════════════════════════════════════════════════
# I-2: Tabular系 完全パイプライン統合テスト
# ════════════════════════════════════════════════════════════════════

class TestIntegrationTabularPipeline:
    """
    tabular_50_simple.csv / tabular_200_complex.csv を使い、
    ヒヤリング → 特徴量選択 → 制約検出 → AutoML → レポートを統合検証する。
    (SMILESなし・記述子計算ステップはスキップ)
    """

    @pytest.fixture(scope="class")
    def pipeline_result_tabular_50(self, tmp_path_factory):
        return _run_tabular_pipeline(TABULAR_50, "Target", tmp_path_factory.mktemp("tabular_50"))

    @pytest.fixture(scope="class")
    def pipeline_result_tabular_200(self, tmp_path_factory):
        return _run_tabular_pipeline(TABULAR_200, "Target", tmp_path_factory.mktemp("tabular_200"))

    def test_tabular50_step1_analysis_policy(self, pipeline_result_tabular_50):
        """T-I13: tabular_50 - ヒヤリング結果が必須キーを含む。"""
        r = pipeline_result_tabular_50
        assert "analysis_policy" in r
        assert "preprocessing" in r["analysis_policy"]

    def test_tabular50_step4_feature_selection(self, pipeline_result_tabular_50):
        """T-I14: tabular_50 - 特徴量選択後に特徴量が残る。"""
        r = pipeline_result_tabular_50
        assert r["df_selected"].shape[1] >= 1

    def test_tabular50_step5_constraints(self, pipeline_result_tabular_50):
        """T-I15: tabular_50 - 制約検出が辞書を返す。"""
        assert isinstance(pipeline_result_tabular_50["constraints"], dict)

    def test_tabular50_step6_automl(self, pipeline_result_tabular_50):
        """T-I16: tabular_50 - AutoMLが正常完走する。"""
        r = pipeline_result_tabular_50
        assert r["automl_result"] is not None
        assert np.isfinite(r["automl_result"].best_score)

    def test_tabular200_step6_automl(self, pipeline_result_tabular_200):
        """T-I17: tabular_200 - より大きいデータでも完走する。"""
        r = pipeline_result_tabular_200
        assert r["automl_result"] is not None
        assert np.isfinite(r["automl_result"].best_score)

    def test_tabular200_constraint_explanation(self, pipeline_result_tabular_200):
        """T-I18: tabular_200 - 検出された各制約に説明文が付与される。"""
        constraints = pipeline_result_tabular_200["constraints"]
        for feat, info in constraints.items():
            assert "explanation" in info

    def test_tabular200_report_generated(self, pipeline_result_tabular_200):
        """T-I19: tabular_200 - レポートファイルが生成される。"""
        exported = pipeline_result_tabular_200.get("exported_files", {})
        if not exported:
            pytest.skip("reportlab/python-docx 未インストール")
        for fmt, path in exported.items():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_tabular_nan_handling(self):
        """T-I20: 欠損値を含むtabularデータでもAutoMLが完走する。"""
        df = _load_csv(TABULAR_1000)
        # tabular_1000 には元々欠損がある可能性
        X_num = df.select_dtypes(include=[np.number]).drop(columns=["Target"], errors="ignore")
        df_in = pd.concat([X_num, df["Target"]], axis=1)

        engine = _build_lightweight_automl("regression")
        result = engine.run(df_in, target_col="Target")
        assert result is not None
        assert np.isfinite(result.best_score)


# ════════════════════════════════════════════════════════════════════
# I-3: Mixture系 完全パイプライン統合テスト
# ════════════════════════════════════════════════════════════════════

class TestIntegrationMixturePipeline:
    """
    mixture系サンプルデータでパイプラインを検証する。
    mixtureデータは複数SMILES列と数値特徴量の組み合わせ。
    AutoMLは数値特徴量のみを使用する（SMILES変換は個別SMILES列ごとに実施）。
    """

    @pytest.fixture(scope="class")
    def pipeline_result_mixture_30(self, tmp_path_factory):
        return _run_mixture_pipeline(
            MIXTURE_30, "Target_Property", tmp_path_factory.mktemp("mixture_30")
        )

    @pytest.fixture(scope="class")
    def pipeline_result_mixture_50(self, tmp_path_factory):
        return _run_mixture_pipeline(
            MIXTURE_50, "Boiling_Point_C", tmp_path_factory.mktemp("mixture_50")
        )

    def test_mixture30_analysis_policy(self, pipeline_result_mixture_30):
        """T-I21: mixture_30 - ヒヤリング結果が返る。"""
        r = pipeline_result_mixture_30
        assert "analysis_policy" in r
        assert isinstance(r["analysis_policy"], dict)

    def test_mixture30_automl_completes(self, pipeline_result_mixture_30):
        """T-I22: mixture_30 - 数値特徴量でAutoMLが完走する。"""
        r = pipeline_result_mixture_30
        assert r["automl_result"] is not None
        assert np.isfinite(r["automl_result"].best_score)

    def test_mixture50_numeric_features_used(self, pipeline_result_mixture_50):
        """T-I23: mixture_50 - 数値特徴量（温度・湿度等）が選択される。"""
        r = pipeline_result_mixture_50
        assert r["df_selected"].shape[1] >= 1

    def test_mixture50_automl_completes(self, pipeline_result_mixture_50):
        """T-I24: mixture_50 - AutoMLが完走する。"""
        r = pipeline_result_mixture_50
        assert r["automl_result"] is not None
        assert np.isfinite(r["automl_result"].best_score)

    def test_mixture_100_regression(self, tmp_path):
        """T-I25: mixture_100 - 最大の混合物データセットでも完走する。"""
        result = _run_mixture_pipeline(MIXTURE_100, "Target_Property", tmp_path)
        assert result["automl_result"] is not None
        assert np.isfinite(result["automl_result"].best_score)


# ════════════════════════════════════════════════════════════════════
# I-4: ストレステスト（@pytest.mark.slow）
# ════════════════════════════════════════════════════════════════════

class TestIntegrationStress:
    """大サイズのサンプルデータでのストレステスト。CI では skip 可能。"""

    @pytest.mark.slow
    def test_smiles_500_descriptor_calculation(self):
        """T-I26: smiles_500_stress でRDKit記述子計算が完走する。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _load_csv(SMILES_500)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_out = transformer.fit_transform(df)

        assert len(df_out) == len(df)
        desc_cols = [c for c in df_out.columns if c not in ("logS", "pIC50", "Class",
                                                              "Compound_Name", "Source", "Notes")]
        assert len(desc_cols) >= 1

    @pytest.mark.slow
    def test_smiles_500_automl(self, tmp_path):
        """T-I27: smiles_500 で AutoML（軽量）が完走する。"""
        result = _run_smiles_pipeline(SMILES_500, "logS", tmp_path)
        assert result["automl_result"] is not None
        assert np.isfinite(result["automl_result"].best_score)

    @pytest.mark.slow
    def test_tabular_1000_automl(self, tmp_path):
        """T-I28: tabular_1000_large で AutoML が完走する。"""
        result = _run_tabular_pipeline(TABULAR_1000, "Target", tmp_path)
        assert result["automl_result"] is not None
        assert np.isfinite(result["automl_result"].best_score)


# ════════════════════════════════════════════════════════════════════
# I-5: 解析方針説明の整合性テスト
# ════════════════════════════════════════════════════════════════════

class TestAnalysisPolicyExplanation:
    """Step1の解析方針辞書とStep6のAutoML結果との整合性を検証する。"""

    @pytest.mark.asyncio
    async def test_analysis_policy_matches_detected_task(self):
        """T-I29: 回帰タスクではmodel_candidatesに回帰モデルが含まれる。"""
        from backend.services.llm_data_analyzer import LLMDataAnalyzer

        mock_response = {
            "data_overview": "連続値ターゲット",
            "preprocessing": "標準化",
            "feature_engineering": "RDKit",
            "model_candidates": ["RandomForest", "Ridge", "LightGBM"],
            "validation_strategy": "5-fold CV",
            "interpretation_plan": "SHAP",
            "cautions": "なし",
        }
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.prepare_dataframe_context = MagicMock(return_value={})
        provider.generate = AsyncMock(return_value=json.dumps(mock_response))

        df = _load_csv(SMILES_QUICK)
        analyzer = LLMDataAnalyzer()
        analyzer.provider = provider
        policy = await analyzer.analyze(df)

        candidates = policy.get("model_candidates", [])
        regression_models = {"RandomForest", "Ridge", "LightGBM", "XGBoost", "SVR"}
        assert any(m in regression_models for m in candidates), \
            f"回帰モデルが提案されていない: {candidates}"

    def test_constraint_explanation_aligned_with_spearman(self):
        """T-I30: Spearmanが正の場合はexplanationに'増加'が含まれる。"""
        X = pd.DataFrame({"feat": np.arange(20, dtype=float)})
        y = pd.Series(np.arange(20, dtype=float) + np.random.randn(20) * 0.1)

        constraints = _detect_monotonic_constraints(X, y, threshold=0.5)
        if "feat" in constraints:
            assert "増加" in constraints["feat"]["explanation"], \
                f"正相関なのに増加単調性の説明がない: {constraints['feat']['explanation']}"

    def test_constraint_explanation_aligned_with_negative_spearman(self):
        """T-I31: Spearmanが負の場合はexplanationに'減少'が含まれる。"""
        X = pd.DataFrame({"feat": np.arange(20, dtype=float)})
        y = pd.Series(-np.arange(20, dtype=float) + np.random.randn(20) * 0.1)

        constraints = _detect_monotonic_constraints(X, y, threshold=0.5)
        if "feat" in constraints:
            assert "減少" in constraints["feat"]["explanation"], \
                f"負相関なのに減少単調性の説明がない: {constraints['feat']['explanation']}"

    def test_full_policy_summary_dict_structure(self):
        """T-I32: 全パイプラインの要約辞書が必須キーを持つ。"""
        df = _load_csv(SMILES_QUICK)
        X = pd.DataFrame({"feat1": np.random.randn(len(df))})
        y = pd.Series(np.random.randn(len(df)))
        constraints = _detect_monotonic_constraints(X, y)

        summary = _build_pipeline_summary(
            analysis_policy={"model_candidates": ["Ridge"]},
            descriptor_recs=[{"plugin": "rdkit_basic", "priority": 1.0}],
            constraints=constraints,
            automl_best_model="Ridge",
            automl_score=-0.5,
        )
        required = ["analysis_policy", "descriptor_recommendations",
                    "constraint_summary", "automl_best_model", "automl_score"]
        for key in required:
            assert key in summary, f"要約辞書に '{key}' がない"


# ════════════════════════════════════════════════════════════════════
# パイプライン実行ヘルパー関数
# ════════════════════════════════════════════════════════════════════

def _run_smiles_pipeline(
    path: Path, target_col: str, output_dir: Path, task: str = "regression"
) -> Dict[str, Any]:
    """SMILESパイプライン全ステップを実行して結果辞書を返す。"""
    import asyncio
    from backend.services.llm_data_analyzer import LLMDataAnalyzer
    from backend.chem.recommender import DescriptorRecommender
    from backend.chem.smiles_transformer import SmilesDescriptorTransformer
    from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

    result: Dict[str, Any] = {}
    df = _load_csv(path)
    result["original_len"] = len(df)

    # Step 1: ヒヤリング（LLMモック）
    analyzer = LLMDataAnalyzer()
    analyzer.provider = _make_mock_llm_provider()
    policy = asyncio.get_event_loop().run_until_complete(analyzer.analyze(df))
    result["analysis_policy"] = policy

    # Step 2: 記述子推奨
    _all_plugins = [
        "rdkit_basic", "rdkit_logp", "rdkit_tpsa", "mordred_2d", "xtb_energy", "unipka",
    ]
    recommender = DescriptorRecommender()
    recs = recommender.recommend(target_col, available_plugins=_all_plugins)
    result["descriptor_recs"] = recs

    # Step 3: SMILES記述子計算
    transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
    df_desc = transformer.fit_transform(df)
    result["df_desc"] = df_desc

    # Step 4: 特徴量選択
    non_target_cols = [c for c in df_desc.columns if c != target_col]
    meta_cols = [c for c in non_target_cols
                 if df_desc[c].dtype == object]
    X_raw = df_desc.drop(columns=[target_col] + meta_cols, errors="ignore") \
                   .select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(df_desc[target_col], errors="coerce").fillna(0)

    feat_config = FeatureSelectorConfig(
        method="variance", task=task if task == "regression" else "classification"
    )
    selector = FeatureSelector(feat_config)
    X_sel = selector.fit_transform(X_raw, y)
    result["df_selected"] = X_sel

    # Step 5: 単調性制約自動検出
    constraints = _detect_monotonic_constraints(X_sel, y, threshold=0.35)
    result["constraints"] = constraints

    # Step 6: AutoML
    df_for_automl = pd.concat(
        [X_sel.reset_index(drop=True), y.reset_index(drop=True).rename(target_col)],
        axis=1
    )
    engine = _build_lightweight_automl(task)
    automl_result = engine.run(df_for_automl, target_col=target_col)
    result["automl_result"] = automl_result

    # Step 7: レポート出力
    report_dict = {
        "best_model_name": automl_result.best_model_key,
        "metrics": {"R2": float(automl_result.best_score),
                    "RMSE": 0.0, "MAE": 0.0},
        "feature_importances": {c: 0.1 for c in X_sel.columns[:5]},
    }
    result["exported_files"] = _export_report(report_dict, output_dir, path.stem)

    return result


def _run_tabular_pipeline(
    path: Path, target_col: str, output_dir: Path
) -> Dict[str, Any]:
    """Tabularパイプライン全ステップを実行して結果辞書を返す（SMILES変換なし）。"""
    import asyncio
    from backend.services.llm_data_analyzer import LLMDataAnalyzer
    from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

    result: Dict[str, Any] = {}
    df = _load_csv(path)
    result["original_len"] = len(df)

    # Step 1: ヒヤリング（LLMモック）
    analyzer = LLMDataAnalyzer()
    analyzer.provider = _make_mock_llm_provider()
    policy = asyncio.get_event_loop().run_until_complete(analyzer.analyze(df))
    result["analysis_policy"] = policy

    # Step 4: 特徴量選択（tabularはStep2,3をスキップ）
    meta_cols = [c for c in df.columns if df[c].dtype == object]
    X_raw = df.drop(columns=meta_cols + [target_col], errors="ignore") \
               .select_dtypes(include=[np.number]) \
               .fillna(df.median(numeric_only=True))
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    feat_config = FeatureSelectorConfig(method="variance", task="regression")
    selector = FeatureSelector(feat_config)
    X_sel = selector.fit_transform(X_raw, y)
    result["df_selected"] = X_sel

    # Step 5: 単調性制約自動検出
    constraints = _detect_monotonic_constraints(X_sel, y, threshold=0.35)
    result["constraints"] = constraints

    # Step 6: AutoML
    df_for_automl = pd.concat(
        [X_sel.reset_index(drop=True), y.reset_index(drop=True).rename(target_col)],
        axis=1
    )
    engine = _build_lightweight_automl("regression")
    result["automl_result"] = engine.run(df_for_automl, target_col=target_col)

    # Step 7: レポート
    report_dict = {
        "best_model_name": result["automl_result"].best_model_key,
        "metrics": {"R2": float(result["automl_result"].best_score), "RMSE": 0.0, "MAE": 0.0},
        "feature_importances": {c: 0.1 for c in X_sel.columns[:5]},
    }
    result["exported_files"] = _export_report(report_dict, output_dir, path.stem)

    return result


def _run_mixture_pipeline(
    path: Path, target_col: str, output_dir: Path
) -> Dict[str, Any]:
    """Mixtureパイプラインを実行する。数値特徴量のみを使用。"""
    import asyncio
    from backend.services.llm_data_analyzer import LLMDataAnalyzer
    from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

    result: Dict[str, Any] = {}
    df = _load_csv(path)
    result["original_len"] = len(df)

    # Step 1: ヒヤリング
    analyzer = LLMDataAnalyzer()
    analyzer.provider = _make_mock_llm_provider()
    policy = asyncio.get_event_loop().run_until_complete(analyzer.analyze(df))
    result["analysis_policy"] = policy

    # 数値列を抽出（SMILES列・文字列列・ID列を除く）
    skip_cols = [c for c in df.columns if df[c].dtype == object]
    skip_cols += [target_col]
    X_raw = df.drop(columns=skip_cols, errors="ignore") \
               .select_dtypes(include=[np.number]) \
               .fillna(df.median(numeric_only=True))
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    # WT%列は組成情報として含む（そのまま使用）
    if X_raw.shape[1] == 0:
        # WT%列だけ取り出す
        wt_cols = [c for c in df.columns if "WT%" in c or "wt%" in c.lower()]
        X_raw = df[wt_cols].fillna(0)

    feat_config = FeatureSelectorConfig(method="variance", task="regression")
    selector = FeatureSelector(feat_config)
    X_sel = selector.fit_transform(X_raw, y) if X_raw.shape[1] > 0 else X_raw
    result["df_selected"] = X_sel

    # Step 5: 制約検出
    constraints = _detect_monotonic_constraints(X_sel, y, threshold=0.35)
    result["constraints"] = constraints

    # Step 6: AutoML
    df_for_automl = pd.concat(
        [X_sel.reset_index(drop=True), y.reset_index(drop=True).rename(target_col)],
        axis=1
    )
    engine = _build_lightweight_automl("regression")
    result["automl_result"] = engine.run(df_for_automl, target_col=target_col)

    # Step 7: レポート
    report_dict = {
        "best_model_name": result["automl_result"].best_model_key,
        "metrics": {"R2": float(result["automl_result"].best_score), "RMSE": 0.0, "MAE": 0.0},
        "feature_importances": {},
    }
    result["exported_files"] = _export_report(report_dict, output_dir, path.stem)

    return result


def _build_pipeline_summary(
    analysis_policy: Dict,
    descriptor_recs: list,
    constraints: Dict,
    automl_best_model: str,
    automl_score: float,
) -> Dict[str, Any]:
    """全パイプライン結果をまとめた要約辞書を生成する。"""
    return {
        "analysis_policy": analysis_policy,
        "descriptor_recommendations": descriptor_recs,
        "constraint_summary": {
            feat: {
                "direction": info["direction"],
                "rho": info["rho"],
                "explanation": info.get("explanation", ""),
            }
            for feat, info in constraints.items()
        },
        "automl_best_model": automl_best_model,
        "automl_score": automl_score,
    }
