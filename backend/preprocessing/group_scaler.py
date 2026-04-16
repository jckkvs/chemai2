"""
グループ単位での標準化スケーラー
同じグループ内の特徴量は、最大の標準偏差で統一してスケーリングする
GroupLASSO等のグループ特徴量選択にも対応
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Optional, Union
import warnings


class GroupStandardScaler(BaseEstimator, TransformerMixin):
    """
    グループ単位での標準化
    
    各グループ内で：
    - 各特徴量は自分の平均で中心化
    - 全特徴量の標準偏差の最大値で統一してスケーリング
    
    Parameters
    ----------
    feature_groups : Dict[str, List[str]]
        グループ定義。例：{"temperature": ["Temp_A", "Temp_B", "Temp_C"]}
    default_scale_method : str
        グループに属さない特徴量の処理方法
        - "individual": 個別に標準化（デフォルト）
        - "none": スケーリングしない
        - "global_max": 全特徴量の最大stdでスケーリング
    with_mean : bool
        中心化（平均引き）を行うか（デフォルト: True）
    with_std : bool
        標準偏差でスケーリングするか（デフォルト: True）
    """
    
    def __init__(
        self,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        default_scale_method: str = "individual",
        with_mean: bool = True,
        with_std: bool = True
    ):
        self.feature_groups = feature_groups or {}
        self.default_scale_method = default_scale_method
        self.with_mean = with_mean
        self.with_std = with_std
        
        # 学習済みパラメータ
        self.group_means_: Dict[str, Dict[str, float]] = {}
        self.group_stds_: Dict[str, float] = {}
        self.feature_to_group_: Dict[str, str] = {}
        self.global_max_std_: Optional[float] = None
        self.ungrouped_means_: Dict[str, float] = {}
        self.ungrouped_stds_: Dict[str, float] = {}
        self.n_features_in_: int = 0
        self.feature_names_in_: Optional[List[str]] = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """
        グループごとの平均と最大標準偏差を計算
        
        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            学習データ
        y : Ignored
            互換性のため
        """
        # DataFrameに変換
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_names_in_ = df.columns.tolist()
        else:
            df = pd.DataFrame(X)
            self.feature_names_in_ = [f"feature_{i}" for i in range(df.shape[1])]
        
        self.n_features_in_ = df.shape[1]
        
        # 全特徴量の最大stdを計算（global_max用）
        all_stds = df.std(axis=0, ddof=0).values
        self.global_max_std_ = float(np.max(all_stds)) if len(all_stds) > 0 else 1.0
        
        # 各グループのパラメータを計算
        for group_name, features in self.feature_groups.items():
            # グループに属する特徴量のみ抽出（存在するもののみ）
            valid_features = [f for f in features if f in df.columns]
            
            if not valid_features:
                warnings.warn(f"グループ '{group_name}' の特徴量が見つかりません: {features}")
                continue
            
            # 各特徴量の平均を計算
            self.group_means_[group_name] = {}
            group_std_values = []
            
            for feature in valid_features:
                # 平均を保存
                self.group_means_[group_name][feature] = float(df[feature].mean())
                
                # 標準偏差を計算（ddof=0 for population std）
                std_val = float(df[feature].std(ddof=0))
                group_std_values.append(std_val)
                
                # 特徴量→グループのマッピング
                self.feature_to_group_[feature] = group_name
            
            # グループ内の最大標準偏差
            if group_std_values:
                self.group_stds_[group_name] = max(group_std_values)
        
        # グループに属さない特徴量の処理（individual用）
        if self.default_scale_method == "individual":
            all_grouped_features = set()
            for features in self.feature_groups.values():
                all_grouped_features.update(features)
            
            ungrouped_features = [f for f in df.columns if f not in all_grouped_features]
            
            for feature in ungrouped_features:
                self.ungrouped_means_[feature] = float(df[feature].mean())
                self.ungrouped_stds_[feature] = float(df[feature].std(ddof=0))
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        グループ単位で標準化
        
        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            変換対象データ
        
        Returns
        -------
        X_scaled : DataFrame
            標準化されたデータ
        """
        # DataFrameに変換
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # 出力用DataFrame
        df_scaled = df.copy().astype(float)
        
        # 各グループを処理
        for group_name, features in self.feature_groups.items():
            if group_name not in self.group_means_:
                continue
            
            group_means = self.group_means_[group_name]
            group_std = self.group_stds_[group_name]
            
            # グループに属する特徴量を標準化
            for feature in features:
                if feature not in df.columns:
                    continue
                
                if feature in group_means:
                    # 中心化（平均引き）
                    if self.with_mean:
                        centered = df[feature] - group_means[feature]
                    else:
                        centered = df[feature]
                    
                    # グループの最大stdでスケーリング
                    if self.with_std and group_std > 1e-10:
                        df_scaled[feature] = centered / group_std
                    elif self.with_std:
                        df_scaled[feature] = centered
                    else:
                        df_scaled[feature] = centered
        
        # グループに属さない特徴量の処理
        all_grouped_features = set()
        for features in self.feature_groups.values():
            all_grouped_features.update(features)
        
        ungrouped_features = [f for f in df.columns if f not in all_grouped_features]
        
        if self.default_scale_method == "individual":
            # 個別に標準化
            for feature in ungrouped_features:
                if feature not in df.columns:
                    continue
                    
                mean_val = self.ungrouped_means_.get(feature, df[feature].mean())
                std_val = self.ungrouped_stds_.get(feature, df[feature].std(ddof=0))
                
                if self.with_mean:
                    centered = df[feature] - mean_val
                else:
                    centered = df[feature]
                
                if self.with_std and std_val > 1e-10:
                    df_scaled[feature] = centered / std_val
                elif self.with_std:
                    df_scaled[feature] = centered
                else:
                    df_scaled[feature] = centered
        
        elif self.default_scale_method == "global_max":
            # 全体の最大stdでスケーリング
            if self.global_max_std_ is not None and self.global_max_std_ > 1e-10:
                for feature in ungrouped_features:
                    if feature not in df.columns:
                        continue
                    mean_val = df[feature].mean()
                    
                    if self.with_mean:
                        centered = df[feature] - mean_val
                    else:
                        centered = df[feature]
                    
                    df_scaled[feature] = centered / self.global_max_std_
        
        # "none" の場合は何もしない（元の値を維持）
        
        return df_scaled
    
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        標準化を元に戻す
        
        Parameters
        ----------
        X : DataFrame
            標準化されたデータ
        
        Returns
        -------
        X_original : DataFrame
            元のスケーリングに戻したデータ
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        df_original = df.copy().astype(float)
        
        # 各グループを逆変換
        for group_name, features in self.feature_groups.items():
            if group_name not in self.group_means_:
                continue
            
            group_means = self.group_means_[group_name]
            group_std = self.group_stds_[group_name]
            
            for feature in features:
                if feature not in df.columns:
                    continue
                
                if feature in group_means:
                    # 逆変換: (scaled * std) + mean
                    if self.with_std and group_std > 1e-10:
                        unscaled = df[feature] * group_std
                    else:
                        unscaled = df[feature]
                    
                    if self.with_mean:
                        df_original[feature] = unscaled + group_means[feature]
                    else:
                        df_original[feature] = unscaled
        
        # グループに属さない特徴量の逆変換
        all_grouped_features = set()
        for features in self.feature_groups.values():
            all_grouped_features.update(features)
        
        ungrouped_features = [f for f in df.columns if f not in all_grouped_features]
        
        if self.default_scale_method == "individual":
            for feature in ungrouped_features:
                if feature not in df.columns:
                    continue
                    
                mean_val = self.ungrouped_means_.get(feature, df[feature].mean())
                std_val = self.ungrouped_stds_.get(feature, df[feature].std(ddof=0))
                
                if self.with_std and std_val > 1e-10:
                    unscaled = df[feature] * std_val
                else:
                    unscaled = df[feature]
                
                if self.with_mean:
                    df_original[feature] = unscaled + mean_val
                else:
                    df_original[feature] = unscaled
        
        elif self.default_scale_method == "global_max":
            if self.global_max_std_ is not None:
                for feature in ungrouped_features:
                    if feature not in df.columns:
                        continue
                    mean_val = df[feature].mean()
                    
                    if self.with_std:
                        unscaled = df[feature] * self.global_max_std_
                    else:
                        unscaled = df[feature]
                    
                    if self.with_mean:
                        df_original[feature] = unscaled + mean_val
                    else:
                        df_original[feature] = unscaled
        
        return df_original
    
    def get_feature_names_out(self, input_features=None):
        """特徴量名を取得（sklearn Pipeline互換）"""
        if input_features is None and self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_)
        elif input_features is not None:
            return np.array(input_features)
        return None
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """グループ定義を取得"""
        return self.feature_groups.copy()
    
    def get_group_for_feature(self, feature: str) -> Optional[str]:
        """特定の特徴量が属するグループを取得"""
        return self.feature_to_group_.get(feature)
    
    def get_params(self, deep=True):
        """パラメータ取得（sklearn互換）"""
        return {
            'feature_groups': self.feature_groups,
            'default_scale_method': self.default_scale_method,
            'with_mean': self.with_mean,
            'with_std': self.with_std
        }
    
    def set_params(self, **params):
        """パラメータ設定（sklearn互換）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# ============================================================
# グループ定義のヘルパー関数
# ============================================================

def create_temperature_group(feature_names: List[str], keywords: List[str] = None) -> Dict[str, List[str]]:
    """
    温度関連の特徴量を自動でグループ化
    
    例: ["Temp_A", "Temp_B", "Temperature_1"] → {"temperature": [...]}
    """
    if keywords is None:
        keywords = ["temp", "temperature", "temprature"]
    
    temp_features = [f for f in feature_names if any(kw in f.lower() for kw in keywords)]
    
    if temp_features:
        return {"temperature": temp_features}
    return {}


def create_pressure_group(feature_names: List[str], keywords: List[str] = None) -> Dict[str, List[str]]:
    """圧力関連を自動グループ化"""
    if keywords is None:
        keywords = ["press", "pressure"]
    
    pressure_features = [f for f in feature_names if any(kw in f.lower() for kw in keywords)]
    
    if pressure_features:
        return {"pressure": pressure_features}
    return {}


def create_concentration_group(feature_names: List[str], keywords: List[str] = None) -> Dict[str, List[str]]:
    """濃度関連を自動グループ化"""
    if keywords is None:
        keywords = ["conc", "concentration", "concentrat"]
    
    conc_features = [f for f in feature_names if any(kw in f.lower() for kw in keywords)]
    
    if conc_features:
        return {"concentration": conc_features}
    return {}


def create_ph_group(feature_names: List[str]) -> Dict[str, List[str]]:
    """pH関連を自動グループ化"""
    ph_features = [f for f in feature_names if "ph" in f.lower()]
    
    if len(ph_features) > 1:
        return {"ph": ph_features}
    return {}


def auto_detect_groups(df: pd.DataFrame, verbose: bool = False) -> Dict[str, List[str]]:
    """
    特徴量名から自動的にグループを検出
    
    Returns
    -------
    groups : Dict[str, List[str]]
        自動検出されたグループ
    """
    feature_names = df.columns.tolist()
    groups = {}
    
    # 温度グループ
    temp_groups = create_temperature_group(feature_names)
    if temp_groups and verbose:
        print(f"[AutoDetect] 温度グループ検出: {temp_groups}")
    groups.update(temp_groups)
    
    # 圧力グループ
    pressure_groups = create_pressure_group(feature_names)
    if pressure_groups and verbose:
        print(f"[AutoDetect] 圧力グループ検出: {pressure_groups}")
    groups.update(pressure_groups)
    
    # pHグループ
    ph_groups = create_ph_group(feature_names)
    if ph_groups and verbose:
        print(f"[AutoDetect] pHグループ検出: {ph_groups}")
    groups.update(ph_groups)
    
    # 濃度グループ
    conc_groups = create_concentration_group(feature_names)
    if conc_groups and verbose:
        print(f"[AutoDetect] 濃度グループ検出: {conc_groups}")
    groups.update(conc_groups)
    
    return groups


# ============================================================
# GroupLASSO用のグループインデックス作成ヘルパー
# ============================================================

def create_group_indices(feature_names: List[str], feature_groups: Dict[str, List[str]]) -> List[int]:
    """
    GroupLASSO用のグループインデックス配列を作成
    
    例:
    feature_names = ["Temp_A", "Temp_B", "MW", "LogP"]
    feature_groups = {"temperature": ["Temp_A", "Temp_B"]}
    
    戻り値: [0, 0, 1, 2]
    → Temp_AとTemp_Bは同じグループ(0)、MWはグループ1、LogPはグループ2
    
    Parameters
    ----------
    feature_names : List[str]
        特徴量名のリスト
    feature_groups : Dict[str, List[str]]
        グループ定義
    
    Returns
    -------
    group_indices : List[int]
        各特徴量に対応するグループインデックス
    """
    group_indices = []
    group_id_map = {}
    next_group_id = 0
    
    # 各特徴量に対してグループを割り当て
    for feature in feature_names:
        assigned = False
        
        # 定義済みグループを検索
        for group_name, features in feature_groups.items():
            if feature in features:
                if group_name not in group_id_map:
                    group_id_map[group_name] = next_group_id
                    next_group_id += 1
                group_indices.append(group_id_map[group_name])
                assigned = True
                break
        
        # グループに属さない場合は個別グループ
        if not assigned:
            group_indices.append(next_group_id)
            next_group_id += 1
    
    return group_indices
