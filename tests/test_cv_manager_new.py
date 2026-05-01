import pytest
import numpy as np
import pandas as pd
from backend.models.cv_manager import CVConfig, get_cv, WalkForwardSplit
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, GroupKFold, StratifiedKFold

def test_cv_aliases():
    # loo
    cfg = CVConfig(cv_key="loo")
    cv = get_cv(cfg)
    assert isinstance(cv, LeaveOneOut)

    # logo
    cfg = CVConfig(cv_key="logo")
    cv = get_cv(cfg)
    assert isinstance(cv, LeaveOneGroupOut)

    # groupfold
    cfg = CVConfig(cv_key="groupfold", n_splits=3)
    cv = get_cv(cfg)
    assert isinstance(cv, GroupKFold)
    assert cv.get_n_splits() == 3

    # stratifiedfold
    cfg = CVConfig(cv_key="stratifiedfold", n_splits=4)
    cv = get_cv(cfg)
    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == 4

    # walkthrough
    cfg = CVConfig(cv_key="walkthrough", n_splits=5, extra_params={"gap": 10})
    cv = get_cv(cfg)
    assert isinstance(cv, WalkForwardSplit)
    assert cv.n_splits == 5
    assert cv.gap == 10

def test_type_conversion():
    # 文字列から数値への変換テスト (UI入力を想定)
    cfg = CVConfig(
        cv_key="walkthrough",
        extra_params={
            "n_splits": "7",  # 文字列
            "gap": "5",       # 文字列
            "min_train_size": "100"
        }
    )
    cv = get_cv(cfg)
    assert cv.n_splits == 7
    assert cv.gap == 5
    assert cv.min_train_size == 100

def test_bool_conversion():
    # 文字列から真偽値への変換
    cfg = CVConfig(
        cv_key="kfold",
        extra_params={"shuffle": "true"}
    )
    cv = get_cv(cfg)
    assert cv.shuffle is True

    cfg = CVConfig(
        cv_key="kfold",
        extra_params={"shuffle": "False", "random_state": 42}
    )
    cv = get_cv(cfg)
    # random_state があるので shuffle=True に自動変換されるはず
    assert cv.shuffle is True


# ============================================================
# T-007: tabular_50_safe.csv を使った交差検証統合テスト
# ============================================================

class TestTabular50SafeCV:
    """T-007: tabular_50_safe.csv を使った交差検証のテスト。"""

    @pytest.fixture
    def tabular_data(self) -> tuple:
        """tabular_50_safe.csv を読み込んで X, y を返す。"""
        from pathlib import Path
        from backend.data.loader import load_file
        df = load_file(Path("data/samples/tabular_50_safe.csv"))
        X = df.drop(columns=["Target"])
        y = df["Target"].values
        return X, y

    def test_kfold_with_tabular(self, tabular_data) -> None:
        """KFold が tabular_50_safe.csv で正しく動作すること。(T-007-01)"""
        X, y = tabular_data
        cfg = CVConfig(cv_key="kfold", n_splits=5, extra_params={"shuffle": True, "random_state": 42})
        cv = get_cv(cfg)

        # 5-fold CV が正しく分割されること
        splits = list(cv.split(X, y))
        assert len(splits) == 5

        # 各分割で train と test のサイズを確認
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == 50
            assert len(test_idx) == 10  # 50 / 5 = 10

    def test_groupkfold_with_tabular(self, tabular_data) -> None:
        """GroupKFold が tabular_50_safe.csv で正しく動作すること。(T-007-02)"""
        X, y = tabular_data
        # グループを作成（10グループ×5サンプル）
        groups = np.repeat(np.arange(10), 5)

        cfg = CVConfig(cv_key="groupfold", n_splits=5)
        cv = get_cv(cfg)

        splits = list(cv.split(X, y, groups))
        assert len(splits) == 5

        # 各分割でグループが重複しないことを確認
        for train_idx, test_idx in splits:
            train_groups = groups[train_idx]
            test_groups = groups[test_idx]
            assert len(np.intersect1d(train_groups, test_groups)) == 0

    def test_leaveoneout_with_tabular(self, tabular_data) -> None:
        """LeaveOneOut が tabular_50_safe.csv で正しく動作すること。(T-007-03)"""
        X, y = tabular_data
        cfg = CVConfig(cv_key="loo")
        cv = get_cv(cfg)

        splits = list(cv.split(X, y))
        assert len(splits) == 50  # 50サンプルなので50分割

        # 各分割で1つだけテスト、残りが学習
        for train_idx, test_idx in splits:
            assert len(test_idx) == 1
            assert len(train_idx) == 49

    def test_cv_score_calculation(self, tabular_data) -> None:
        """CV スコアが正しく計算されること。(T-007-04)"""
        X, y = tabular_data
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        cfg = CVConfig(cv_key="kfold", n_splits=5, extra_params={"shuffle": True, "random_state": 42})
        cv = get_cv(cfg)

        scores = cross_val_score(Ridge(), X, y, cv=cv, scoring="r2")
        assert len(scores) == 5
        for score in scores:
            assert isinstance(score, float)
            assert not np.isnan(score)

    def test_stratifiedkfold_raises_for_regression(self, tabular_data) -> None:
        """回帰タスクで StratifiedKFold を使うとエラーになること。(T-007-05)"""
        X, y = tabular_data
        # 回帰タスクでは StratifiedKFold は使用不可
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=5)
        # 回帰タスク（連続値）で StratifiedKFold.split() を呼ぶと ValueError
        with pytest.raises(ValueError, match="continuous"):
            list(cv.split(X, y))

    def test_cv_with_different_fold_numbers(self, tabular_data) -> None:
        """異なるフォールド数で正しく動作すること。(T-007-06)"""
        X, y = tabular_data

        for n_splits in [2, 5, 10]:
            cfg = CVConfig(cv_key="kfold", n_splits=n_splits,
                          extra_params={"shuffle": True, "random_state": 42})
            cv = get_cv(cfg)
            splits = list(cv.split(X, y))
            assert len(splits) == n_splits

    def test_cv_score_with_different_models(self, tabular_data) -> None:
        """異なるモデルで CV スコアが計算されること。(T-007-07)"""
        X, y = tabular_data
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        cfg = CVConfig(cv_key="kfold", n_splits=5,
                        extra_params={"shuffle": True, "random_state": 42})
        cv = get_cv(cfg)

        for model in [Ridge(), Lasso(alpha=1.0), RandomForestRegressor(n_estimators=10, random_state=42)]:
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            assert len(scores) == 5
            assert all(not np.isnan(s) for s in scores)
