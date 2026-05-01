# -*- coding: utf-8 -*-
"""
tests/test_smiles_25_regression.py

smiles_25_regression.csv を使ったテスト駆動開発。
SMILES特徴量化 → MLモデル → 単調性制約 → 評価。
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# ヘルパー：SMILES → 特徴量
# ──────────────────────────────────────────────

def _load_smiles_data():
    """smiles_25_regression.csv をロードして特徴量化"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return None, None, None

    df = pd.read_csv("data/samples/smiles_25_regression.csv")
    X_list = []
    y_list = []
    valid_smiles = []

    for smi, target in zip(df["SMILES"].tolist(), df["logS"].values):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
        ]
        X_list.append(features)
        y_list.append(target)
        valid_smiles.append(smi)

    if not X_list:
        return None, None, None

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return X, y, valid_smiles


# ──────────────────────────────────────────────
# 1. データロード・基本確認
# ──────────────────────────────────────────────

class TestSmilesDataLoading:
    """smiles_25_regression.csv のロードと基本確認"""

    def test_load_shape(self):
        df = pd.read_csv("data/samples/smiles_25_regression.csv")
        assert df.shape[0] == 25
        assert "SMILES" in df.columns
        assert "logS" in df.columns

    def test_no_missing_smiles(self):
        df = pd.read_csv("data/samples/smiles_25_regression.csv")
        assert df["SMILES"].isnull().sum() == 0

    def test_target_range(self):
        df = pd.read_csv("data/samples/smiles_25_regression.csv")
        y = df["logS"].values
        assert len(y) == 25
        assert np.std(y) > 0


# ──────────────────────────────────────────────
# 2. SMILES 特徴量化（RDKit）
# ──────────────────────────────────────────────

class TestSmilesFeaturization:
    """SMILES → 数値特徴量の変換テスト"""

    def test_rdkit_descriptors(self):
        """RDKit による記述子計算"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
        except ImportError:
            pytest.skip("RDKit未インストール")

        df = pd.read_csv("data/samples/smiles_25_regression.csv")
        smiles_list = df["SMILES"].tolist()
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            features.append([mw, logp, hbd, hba])
        assert len(features) > 15
        arr = np.array(features)
        assert arr.shape[1] == 4
        assert np.all(np.isfinite(arr))

    def test_rdkit_builtin_adapter(self):
        """backend/chem/descriptors/_builtins のRDKitアダプター"""
        try:
            from backend.chem.descriptors._builtins.rdkit_physicochemical import RDKitPhysicochemical
        except ImportError:
            pytest.skip("RDKit未インストール")

        df = pd.read_csv("data/samples/smiles_25_regression.csv")
        smiles_list = df["SMILES"].tolist()
        adapter = RDKitPhysicochemical()
        result = adapter.compute(smiles_list)
        assert result.features is not None
        assert result.features.shape[0] == len(smiles_list)
        assert result.features.shape[1] > 0

    def test_smiles_to_features_pipeline(self):
        """SMILES → 特徴量 → 前処理の一連の流れ"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
        except ImportError:
            pytest.skip("RDKit未インストール")

        df = pd.read_csv("data/samples/smiles_25_regression.csv")
        X_list = []
        y_list = []
        for smi, target in zip(df["SMILES"].tolist(), df["logS"].values):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            features = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
            ]
            X_list.append(features)
            y_list.append(target)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 5
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape


# ──────────────────────────────────────────────
# 3. 基本MLモデル（SMILES特徴量）
# ──────────────────────────────────────────────

class TestBasicMLWithSmiles:
    """SMILES特徴量を使った基本MLモデルのテスト"""

    def test_rf_with_smiles_features(self):
        """RandomForest + SMILES特徴量"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        r2 = r2_score(y_test, y_pred)
        assert r2 > -2.0

    def test_svr_with_smiles_features(self):
        """SVR + SMILES特徴量"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        from sklearn.svm import SVR
        model = SVR(kernel="rbf")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

    def test_kernel_ridge_with_smiles_features(self):
        """KernelRidge + SMILES特徴量"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        from sklearn.kernel_ridge import KernelRidge
        model = KernelRidge(kernel="rbf")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)


# ──────────────────────────────────────────────
# 4. 単調性制約（SMILES特徴量）
# ──────────────────────────────────────────────

class TestMonotonicWithSmiles:
    """SMILES特徴量 + 単調性制約のテスト"""

    def test_monotonic_svr_with_smiles(self):
        """MonotonicSVR + SMILES特徴量"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        from backend.models.monotonic_kernel_models import MonotonicSVR
        model = MonotonicSVR(
            monotonic_features=[0],  # MolWtで単調制約
            constraint_strength=1.0,
        )
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        assert len(y_pred) == len(y)

    def test_monotonic_wrapper_rf_with_smiles(self):
        """MonotonicConstraintRegressor + RF + SMILES特徴量"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        from sklearn.ensemble import RandomForestRegressor
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        n_features = X.shape[1]
        base = RandomForestRegressor(n_estimators=10, random_state=42)
        model = MonotonicConstraintRegressor(
            base_estimator=base,
            monotonic_constraints=(1,) + (0,) * (n_features - 1),
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_monotonic_gpr_with_smiles(self):
        """MonotonicGPR + SMILES特徴量"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        from backend.models.monotonic_kernel_models import MonotonicGPR
        model = MonotonicGPR(
            monotonic_features=[0],  # MolWtで単調制約
            constraint_strength=1.0,
        )
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        assert len(y_pred) == len(y)


# ──────────────────────────────────────────────
# 5. Tree Kernels / RFRKernel（SMILES特徴量）
# ──────────────────────────────────────────────

class TestTreeKernelsWithSmiles:
    """SMILES特徴量 + Tree Kernel / RFRKernel"""

    def test_random_forest_kernel_with_smiles(self):
        """RandomForestKernel + KernelRidge + SMILES特徴量"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        from backend.models.tree_kernels import RandomForestKernel
        from sklearn.kernel_ridge import KernelRidge
        kernel = RandomForestKernel(n_trees=10, max_depth=5, random_state=42)
        kernel.fit(X, y)
        kr = KernelRidge(kernel=kernel)
        kr.fit(X, y)
        y_pred = kr.predict(X)
        assert len(y_pred) == len(y)

    @pytest.mark.skip(reason="MonotonicConstrainedKernelはsklearn Kernelを継承していないためGPRで直接使用不可。要修正。")
    def test_monotonic_tree_kernel_with_smiles(self):
        """MonotonicConstrainedKernel + GPR + SMILES（要修正）"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        from backend.models.monotonic_kernel import MonotonicConstrainedKernel
        from sklearn.gaussian_process import GaussianProcessRegressor
        constrained_kernel = MonotonicConstrainedKernel(
            monotonic_features=[0],
            constraint_strength=1.0,
        )
        constrained_kernel.fit(X, y)
        gpr = GaussianProcessRegressor(kernel=constrained_kernel)
        gpr.fit(X, y)
        y_pred = gpr.predict(X)
        assert len(y_pred) == len(y)


# ──────────────────────────────────────────────
# 6. パイプライン統合（SMILES）
# ──────────────────────────────────────────────

class TestSmilesPipelineIntegration:
    """SMILES → 特徴量化 → 学習 → 予測の統合テスト"""

    def test_end_to_end_smiles_ml(self):
        """SMILES特徴量 → RF学習 → 予測"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_end_to_end_smiles_monotonic(self):
        """SMILES特徴量 → 単調制約RF → 予測"""
        X, y, _ = _load_smiles_data()
        if X is None:
            pytest.skip("RDKit未インストール")
        from sklearn.ensemble import RandomForestRegressor
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        n_features = X.shape[1]
        model = MonotonicConstraintRegressor(
            base_estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            monotonic_constraints=(1,) + (0,) * (n_features - 1),
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))
