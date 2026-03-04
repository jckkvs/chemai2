"""
tests/test_smiles_pipeline.py

SMILES列を含むデータでAutoMLエンジンの end-to-end テスト。
学習 → predict(生SMILES列込みX) が列不一致なく動くかを確認する。
"""
import pytest
import pandas as pd
import numpy as np
from backend.models.automl import AutoMLEngine


# --- テスト用データ ---
SMILES_DATA = [
    ("CCO",       -0.31),
    ("CC",        -1.89),
    ("CCC",       -0.70),
    ("CCCC",      -0.38),
    ("c1ccccc1",   2.13),
    ("CCN",       -1.03),
    ("OCC",       -1.13),
    ("CCCO",      -0.53),
    ("CCCCO",     -0.25),
    ("c1ccccc1C",  2.73),
    ("c1ccccc1O",  1.46),
    ("CC(=O)O",   -0.17),
    ("CNC",       -1.00),
    ("CCF",        0.18),
    ("CCCl",       0.60),
    ("OC(=O)C",   -0.17),
    ("CCOCC",      0.89),
    ("CCOC(=O)C",  0.73),
    ("c1ccncc1",   0.65),
    ("NCC",       -1.14),
]

def _make_df(n: int = 20) -> pd.DataFrame:
    rows = [SMILES_DATA[i % len(SMILES_DATA)] for i in range(n)]
    return pd.DataFrame(rows, columns=["smiles", "logS"])


class TestSmilesAutoMLPipeline:
    """SMILES列使用時のAutoMLパイプラインのend-to-endテスト。"""

    def test_fit_predict_with_smiles_col(self):
        """学習後に生SMILES列含みのXでpredict()が列不一致エラーなく.動くこと。"""
        df = _make_df(20)
        engine = AutoMLEngine(task="regression", cv_folds=2, timeout_seconds=120)
        result = engine.run(df, target_col="logS", smiles_col="smiles")

        pipeline = result.best_pipeline

        # 元のデータ（SMILES列含む）からターゲット除去
        X_raw = df.drop(columns=["logS"])
        
        # predict は SMILES列込みのXを受け付けられる必要がある
        y_pred = pipeline.predict(X_raw)
        assert len(y_pred) == len(df), "予測値の数がサンプル数と一致しません"
        assert not np.any(np.isnan(y_pred)), "予測値にNaNが含まれています"

    def test_fit_predict_no_smiles_col(self):
        """SMILES列を指定しない場合でもpredictが動くこと（通常の数値のみ）。"""
        df = pd.DataFrame({
            "feat1": np.random.rand(30),
            "feat2": np.random.rand(30),
            "target": np.random.rand(30),
        })
        engine = AutoMLEngine(task="regression", cv_folds=2, timeout_seconds=60)
        result = engine.run(df, target_col="target")
        X_raw = df.drop(columns=["target"])
        y_pred = result.best_pipeline.predict(X_raw)
        assert len(y_pred) == len(df)

    def test_pipeline_steps_include_smiles_transformer(self):
        """SMILES列がある場合、Pipelineの先頭にSmiles変換ステップが含まれること。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        df = _make_df(20)
        engine = AutoMLEngine(task="regression", cv_folds=2, timeout_seconds=120)
        result = engine.run(df, target_col="logS", smiles_col="smiles")
        step_names = [name for name, _ in result.best_pipeline.steps]
        assert "smiles" in step_names, (
            f"PipelineにSMILES変換ステップがありません: {step_names}"
        )
        assert isinstance(result.best_pipeline.named_steps["smiles"],
                          SmilesDescriptorTransformer)
