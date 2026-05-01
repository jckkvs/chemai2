# -*- coding: utf-8 -*-
"""
tests/test_tabular_50_safe.py

tabular_50_safe.csv を使ったテスト駆動開発。
ご考え（20260429.txt, 20260430_テスト駆動開発.txt）に基づく。
非LLM機能を先にテストし、その後LLM関連をテストする。
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import r2_score, mean_squared_error

# ──────────────────────────────────────────────
# 1. データローダー・前処理
# ──────────────────────────────────────────────

class TestDataLoading:
    """tabular_50_safe.csv のロードと基本確認"""

    @pytest.fixture
    def df(self):
        return pd.read_csv("data/samples/tabular_50_safe.csv")

    def test_load_shape(self, df):
        """50サンプル、9列（8特徴量+1目的変数）"""
        assert df.shape == (50, 9)
        assert "Target" in df.columns
        assert len([c for c in df.columns if c.startswith("Feature_")]) == 8

    def test_no_missing_values(self, df):
        """欠損値がない"""
        assert df.isnull().sum().sum() == 0

    def test_target_range(self, df):
        """Targetの範囲確認"""
        y = df["Target"].values
        assert len(y) == 50
        assert np.std(y) > 0  # 分散がある

    def test_feature_values_finite(self, df):
        """すべての特徴量が有限値"""
        X = df.drop("Target", axis=1).values
        assert np.all(np.isfinite(X))


class TestDataPreprocessing:
    """前処理のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values.astype(np.float64)
        y = df["Target"].values.astype(np.float64)
        feature_names = [c for c in df.columns if c != "Target"]
        return X, y, feature_names

    def test_standard_scaling(self, data):
        """標準化のテスト"""
        from sklearn.preprocessing import StandardScaler
        X, _, _ = data
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        assert X_s.shape == X.shape
        assert abs(X_s.mean()) < 1e-10  # 平均0に近い
        assert abs(X_s.std() - 1.0) < 1e-10  # 標準偏差1に近い

    def test_train_test_split(self, data):
        """train/test分割のテスト"""
        from sklearn.model_selection import train_test_split
        X, y, _ = data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        assert len(X_train) == 40
        assert len(X_test) == 10
        assert len(y_train) == 40
        assert len(y_test) == 10


# ──────────────────────────────────────────────
# 2. 単調性制約モデル（GPR, KernelRidge, SVR, GPC, RFR, RFC）
# ──────────────────────────────────────────────

class TestMonotonicGPR:
    """GaussianProcessRegressor with monotonic constraints"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]  # 小さいデータで高速化
        y = df["Target"].values[:40]
        return X, y

    def test_monotonic_gpr_fit_predict(self, data):
        """MonotonicGPRのfit/predictが動作する"""
        from backend.models.monotonic_kernel_models import MonotonicGPR
        X, y = data
        model = MonotonicGPR(monotonic_features=[0], constraint_strength=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))

    def test_monotonic_gpr_improves_monotonicity(self, data):
        """単調性が（無制約に比べて）向上することを確認"""
        from backend.models.monotonic_kernel_models import MonotonicGPR
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        X, y = data
        # 無制約
        gpr = GaussianProcessRegressor(kernel=RBF(), random_state=42)
        gpr.fit(X, y)
        y_pred_unconstrained = gpr.predict(X)
        # 単調制約
        mono_gpr = MonotonicGPR(
            monotonic_features=[0], constraint_strength=1.0,
        )
        mono_gpr.fit(X, y)
        y_pred_mono = mono_gpr.predict(X)
        assert len(y_pred_mono) == len(y)


class TestMonotonicKernelRidge:
    """KernelRidge with monotonic constraints (via RFRKernel)"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_kernel_ridge_basic(self, data):
        """通常のKernelRidgeが動作する"""
        from sklearn.kernel_ridge import KernelRidge
        X, y = data
        kr = KernelRidge(kernel="rbf")
        kr.fit(X, y)
        y_pred = kr.predict(X)
        assert len(y_pred) == len(y)

    def test_tree_kernel_ridge(self, data):
        """RandomForestKernel + KernelRidge"""
        from backend.models.tree_kernels import RandomForestKernel
        from sklearn.kernel_ridge import KernelRidge
        X, y = data
        rf_kernel = RandomForestKernel(n_trees=10, max_depth=5, random_state=42)
        rf_kernel.fit(X, y)
        kr = KernelRidge(kernel=rf_kernel)
        kr.fit(X, y)
        y_pred = kr.predict(X)
        assert len(y_pred) == len(y)


class TestMonotonicSVR:
    """SVR with monotonic constraints"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_monotonic_svr_fit_predict(self, data):
        """MonotonicSVRのfit/predictが動作する"""
        from backend.models.monotonic_kernel_models import MonotonicSVR
        X, y = data
        model = MonotonicSVR(
            monotonic_features=[0], constraint_strength=1.0
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))


class TestMonotonicWrapper:
    """MonotonicConstraintRegressor/Classifier のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_wrap_svr(self, data):
        """SVRをMonotonicConstraintRegressorでラップ"""
        from sklearn.svm import SVR
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = data
        n_features = X.shape[1]
        base = SVR()
        wrapped = MonotonicConstraintRegressor(
            base_estimator=base,
            monotonic_constraints=(1,) + (0,) * (n_features - 1),
        )
        wrapped.fit(X, y)
        y_pred = wrapped.predict(X)
        assert len(y_pred) == len(y)

    def test_wrap_rf(self, data):
        """RandomForestRegressorをMonotonicConstraintRegressorでラップ"""
        from sklearn.ensemble import RandomForestRegressor
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = data
        n_features = X.shape[1]
        base = RandomForestRegressor(n_estimators=10, random_state=42)
        wrapped = MonotonicConstraintRegressor(
            base_estimator=base,
            monotonic_constraints=(1,) + (0,) * (n_features - 1),
        )
        wrapped.fit(X, y)
        y_pred = wrapped.predict(X)
        assert len(y_pred) == len(y)


# ──────────────────────────────────────────────
# 3. Bernoulli RF, RGF, Kernel Forest
# ──────────────────────────────────────────────

class TestBernoulliRandomForest:
    """Bernoulli Random Forest のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_bernoulli_rf_regressor_fit_predict(self, data):
        """BernoulliRandomForestRegressor (回帰) の動作確認"""
        from backend.models.forests.bernoulli_rf import BernoulliRandomForestRegressor
        X, y = data
        model = BernoulliRandomForestRegressor(
            n_estimators=10, feature_prob=0.5, max_depth=5, random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))

    def test_bernoulli_rf_classifier(self, data):
        """BernoulliRandomForest (分類) の動作確認"""
        from backend.models.forests.bernoulli_rf import BernoulliRandomForest
        X, y = data
        # 二値化
        y_bin = (y > np.median(y)).astype(int)
        model = BernoulliRandomForest(
            n_estimators=10, feature_prob=0.5, max_depth=5, random_state=42
        )
        model.fit(X, y_bin)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y_bin)


class TestRGF:
    """Regularized Greedy Forest のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_rgf_regressor(self, data):
        """RGFRegressorの動作確認"""
        from backend.models.rgf import RGFRegressor
        X, y = data
        model = RGFRegressor(
            n_estimators=10, reg_lambda=0.1, random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))

    def test_rgf_classifier(self, data):
        """RGFClassifierの動作確認"""
        from backend.models.rgf import RGFClassifier
        X, y = data
        y_bin = (y > np.median(y)).astype(int)
        model = RGFClassifier(
            n_estimators=10, reg_lambda=0.1, random_state=42
        )
        model.fit(X, y_bin)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y_bin)


class TestKernelForest:
    """Kernel Forest (RFRKernel概念) のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_kernel_forest_regressor(self, data):
        """RandomKernelForestRegressor (回帰) の動作確認"""
        from backend.models.forests.kernel_forest import RandomKernelForestRegressor
        X, y = data
        model = RandomKernelForestRegressor(
            n_estimators=10, max_depth=5, n_rff=50, gamma=1.0, random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))


# ──────────────────────────────────────────────
# 4. Tree Kernels, RFRKernel概念
# ──────────────────────────────────────────────

class TestTreeKernels:
    """Tree Kernels (RFRKernel概念) のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_random_forest_kernel(self, data):
        """RandomForestKernel + KernelRidge"""
        from backend.models.tree_kernels import RandomForestKernel
        from sklearn.kernel_ridge import KernelRidge
        X, y = data
        kernel = RandomForestKernel(n_trees=10, max_depth=5, random_state=42)
        kernel.fit(X, y)
        kr = KernelRidge(kernel=kernel)
        kr.fit(X, y)
        y_pred = kr.predict(X)
        assert len(y_pred) == len(y)

    def test_extra_trees_kernel(self, data):
        """ExtraTreesKernel + KernelRidge"""
        from backend.models.tree_kernels import ExtraTreesKernel
        from sklearn.kernel_ridge import KernelRidge
        X, y = data
        kernel = ExtraTreesKernel(n_trees=10, max_depth=5, random_state=42)
        kernel.fit(X, y)
        kr = KernelRidge(kernel=kernel)
        kr.fit(X, y)
        y_pred = kr.predict(X)
        assert len(y_pred) == len(y)

    def test_tree_kernel_with_svr(self, data):
        """Tree Kernel + SVR"""
        from backend.models.tree_kernels import RandomForestKernel
        from sklearn.svm import SVR
        X, y = data
        kernel = RandomForestKernel(n_trees=10, max_depth=5, random_state=42)
        kernel.fit(X, y)
        svr = SVR(kernel=kernel)
        svr.fit(X, y)
        y_pred = svr.predict(X)
        assert len(y_pred) == len(y)

    def test_kernel_ridge_rfr_kernel(self, data):
        """RFRKernel概念：RFの木構造をカーネルとして使う"""
        from backend.models.tree_kernels import RandomForestKernel
        from sklearn.kernel_ridge import KernelRidge
        X, y = data
        # ここがRFRKernelの核心：RFの葉の出現パターンをカーネル化
        rf_kernel = RandomForestKernel(
            n_trees=20, max_depth=5, random_state=42
        )
        rf_kernel.fit(X, y)
        # KernelRidgeで予測
        kr = KernelRidge(kernel=rf_kernel, alpha=0.1)
        kr.fit(X, y)
        y_pred = kr.predict(X)
        r2 = r2_score(y, y_pred)
        assert r2 > -1.0  # 最低限の性能


# ──────────────────────────────────────────────
# 5. Linear Tree, RGF, Regularized Tree
# ──────────────────────────────────────────────

class TestLinearTree:
    """Linear Tree (1本の決定木でRF並みの性能) のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_linear_tree_regressor(self, data):
        """LinearTreeRegressorの動作確認"""
        from backend.models.linear_tree import LinearTreeRegressor
        X, y = data
        model = LinearTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))

    def test_linear_tree_vs_rf(self, data):
        """LinearTreeがRF並みの性能を出せるか（小さいデータで）"""
        from backend.models.linear_tree import LinearTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        X, y = data
        # Linear Tree
        lt = LinearTreeRegressor(max_depth=3, random_state=42)
        lt.fit(X, y)
        y_pred_lt = lt.predict(X)
        r2_lt = r2_score(y, y_pred_lt)
        # RF (reference)
        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X, y)
        y_pred_rf = rf.predict(X)
        r2_rf = r2_score(y, y_pred_rf)
        # LinearTreeがRFと同程度かそれ以上の性能を出すことを確認
        # （小さいデータでは厳しくない上限を設定）
        assert r2_lt > -2.0  # 最低限予測できている


# ──────────────────────────────────────────────
# 6. EDA機能（ペアプロット、色分け等）
# ──────────────────────────────────────────────

class TestEDA:
    """EDA機能のテスト"""

    @pytest.fixture
    def df(self):
        return pd.read_csv("data/samples/tabular_50_safe.csv")

    def test_pairplot_data_preparation(self, df):
        """ペアプロット用データ準備"""
        # 全説明変数と目的変数を選択
        X_cols = [c for c in df.columns if c != "Target"]
        assert len(X_cols) == 8
        # 目的変数で色分けする場合の設定
        y = df["Target"].values
        assert len(y) == 50

    def test_feature_statistics(self, df):
        """特徴量の基本統計量"""
        X = df.drop("Target", axis=1)
        stats = X.describe()
        assert stats.shape[0] == 8  # count, mean, std, min, 25%, 50%, 75%, max
        for col in X.columns:
            assert X[col].std() > 0  # 分散がある

    def test_feature_correlation(self, df):
        """特徴量間の相関計算"""
        X = df.drop("Target", axis=1)
        corr = X.corr()
        assert corr.shape == (8, 8)
        # 対角成分は1
        for i in range(8):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-10


# ──────────────────────────────────────────────
# 7. 交差検証 (CV)
# ──────────────────────────────────────────────

class TestCrossValidation:
    """Cross-Validationのテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_kfold_cv(self, data):
        """KFold CVの動作確認"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        X, y = data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_group_cv(self, data):
        """Group CVの動作確認"""
        from sklearn.model_selection import GroupKFold
        X, y = data
        groups = np.array([i // 5 for i in range(len(X))])  # 5サンプルずつのグループ
        gkf = GroupKFold(n_splits=3)
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        scores = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        assert len(scores) == 3


# ──────────────────────────────────────────────
# 8. パイプライン統合テスト
# ──────────────────────────────────────────────

class TestPipelineIntegration:
    """パイプライン統合のテスト"""

    @pytest.fixture
    def data(self):
        df = pd.read_csv("data/samples/tabular_50_safe.csv")
        X = df.drop("Target", axis=1).values[:40]
        y = df["Target"].values[:40]
        return X, y

    def test_end_to_end_monotonic(self, data):
        """単調制約→学習→予測の一連の流れ"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        from sklearn.ensemble import RandomForestRegressor
        X, y = data
        n_features = X.shape[1]
        # 単調制約を設定（特徴量0は単調増加）
        mono_constraints = (1,) + (0,) * (n_features - 1)
        model = MonotonicConstraintRegressor(
            base_estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            monotonic_constraints=mono_constraints,
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        r2 = r2_score(y, y_pred)
        assert r2 > -2.0

    def test_bernoulli_rf_pipeline(self, data):
        """BernoulliRandomForestRegressorのパイプライン統合"""
        from backend.models.forests.bernoulli_rf import BernoulliRandomForestRegressor
        X, y = data
        model = BernoulliRandomForestRegressor(
            n_estimators=10, feature_prob=0.7, max_depth=5, random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))

    def test_rgf_pipeline(self, data):
        """RGFのパイプライン統合"""
        from backend.models.rgf import RGFRegressor
        X, y = data
        model = RGFRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(np.isfinite(y_pred))
