"""
GroupLASSO実装
グループ単位での特徴量選択を行う正則化回帰
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
import warnings


class GroupLasso(BaseEstimator, RegressorMixin):
    """
    GroupLASSO: グループ単位でのスパース回帰
    
    各グループの特徴量をまとめて選択/除外する正則化を行う。
    sklearnのLassoを拡張し、グループインデックスを使用して実装。
    
    Parameters
    ----------
    alpha : float
        正則化パラメータ（大きいほどスパースに）
    groups : List[int]
        各特徴量に対応するグループインデックス
        例: [0, 0, 1, 2, 2] → 特徴量0,1はグループ0、特徴量2はグループ1、特徴量3,4はグループ2
    fit_intercept : bool
        切片をフィットするか
    max_iter : int
        最大反復回数
    tol : float
        収束判定閾値
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        groups: Optional[List[int]] = None,
        fit_intercept: bool = True,
        max_iter: int = 10000,
        tol: float = 1e-4
    ):
        self.alpha = alpha
        self.groups = groups
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        
        # 学習済みパラメータ
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.selected_groups_ = None
        self.selected_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        GroupLASSOモデルを学習
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            説明変数
        y : ndarray, shape (n_samples,)
            目的変数
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # グループ設定
        if self.groups is None:
            # 未設定の場合は各特徴量を個別グループとして扱う（通常のLASSO）
            self.groups_ = list(range(n_features))
        else:
            self.groups_ = np.asarray(self.groups)
            if len(self.groups_) != n_features:
                raise ValueError(f"groupsの長さ({len(self.groups_)})が特徴量数({n_features})と一致しません")
        
        # グループごとのインデックスを整理
        unique_groups = np.unique(self.groups_)
        group_to_features = {g: np.where(self.groups_ == g)[0] for g in unique_groups}
        
        # 初期化
        self.coef_ = np.zeros(n_features)
        
        # 座標降下法で最適化（簡易実装）
        # 注: 本番ではより高度な最適化アルゴリズム（例: proximal gradient）を使用推奨
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for group_id in unique_groups:
                feat_indices = group_to_features[group_id]
                
                # 現在の予測
                residual = y - X @ self.coef_
                
                # グループの勾配
                grad = X[:, feat_indices].T @ residual
                
                # GroupLASSOのproximal operator
                group_norm = np.linalg.norm(grad)
                threshold = self.alpha * np.sqrt(len(feat_indices))
                
                if group_norm > threshold:
                    # shrinkage
                    shrink_factor = 1 - threshold / group_norm
                    self.coef_[feat_indices] += shrink_factor * grad / n_samples
                else:
                    # ゼロに設定（グループ全体を除外）
                    self.coef_[feat_indices] = 0
            
            # 収束判定
            if np.linalg.norm(self.coef_ - coef_old) < self.tol:
                break
        
        # 切片の計算
        if self.fit_intercept:
            self.intercept_ = y.mean() - X @ self.coef_
        else:
            self.intercept_ = 0.0
        
        # 選択されたグループと特徴量を記録
        self.selected_groups_ = []
        self.selected_features_ = []
        
        for group_id in unique_groups:
            feat_indices = group_to_features[group_id]
            if np.any(self.coef_[feat_indices] != 0):
                self.selected_groups_.append(int(group_id))
                self.selected_features_.extend(feat_indices.tolist())
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def get_selected_features(self) -> List[int]:
        """選択された特徴量のインデックスを取得"""
        return self.selected_features_
    
    def get_selected_groups(self) -> List[int]:
        """選択されたグループのインデックスを取得"""
        return self.selected_groups_
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        特徴量選択のマスクを取得
        
        Parameters
        ----------
        indices : bool
            Trueなら選択されたインデックスを、Falseならブールマスクを返す
        """
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        
        if indices:
            return np.where(mask)[0]
        return mask
    
    def score(self, X, y, sample_weight=None):
        """R²スコアを計算"""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)


# ============================================================
# GroupLASSO用のラッパー関数
# ============================================================

def train_group_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: Dict[str, List[str]],
    alpha: float = 1.0,
    **kwargs
) -> Dict:
    """
    GroupLASSOでモデル学習
    
    Parameters
    ----------
    X : DataFrame
        説明変数
    y : Series
        目的変数
    feature_groups : Dict[str, List[str]]
        グループ定義
    alpha : float
        正則化パラメータ
    **kwargs : dict
        GroupLASSOのその他のパラメータ
    
    Returns
    -------
    result : dict
        学習結果とメトリクス
    """
    from backend.preprocessing.group_scaler import create_group_indices
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import time
    
    start_time = time.time()
    
    # 特徴量名とグループインデックス
    feature_names = X.columns.tolist()
    group_indices = create_group_indices(feature_names, feature_groups)
    
    # GroupLASSOモデル作成
    model = GroupLasso(
        alpha=alpha,
        groups=group_indices,
        **kwargs
    )
    
    # 学習
    model.fit(X.values, y.values)
    
    # 予測
    y_pred = model.predict(X.values)
    
    # 評価指標
    metrics = {
        "R2": r2_score(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred),
        "n_selected_features": len(model.get_selected_features()),
        "n_selected_groups": len(model.get_selected_groups()),
        "selected_features": [feature_names[i] for i in model.get_selected_features()],
        "selected_groups": model.get_selected_groups()
    }
    
    return {
        "success": True,
        "model": model,
        "metrics": metrics,
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "training_time": time.time() - start_time
    }
