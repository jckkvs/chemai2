"""
backend/data/flexible_explorer.py

Data Sandbox Backend — 非破壊的なデータ加工・探索エンジン
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler

class FlexibleDataView:
    """
    ユーザーの探索的な加工（Log, Z-score, Binning）を反映した
    一時的なデータビューを管理するクラス。
    """
    def __init__(self, df: pd.DataFrame):
        self.source_df = df.copy()
        self.current_df = df.copy()
        self.transformations: Dict[str, str] = {} # col -> type
        self.selection_mask: Optional[pd.Series] = None

    def apply_transform(self, column: str, transform_type: str):
        """カラムに変換を適用する。
        
        Args:
            column: 対象カラム名
            transform_type: 'none', 'log10', 'zscore', 'binning'
        """
        if column not in self.source_df.columns:
            return

        self.transformations[column] = transform_type
        self._rebuild()

    def _rebuild(self):
        """現在の変換設定に基づいてcurrent_dfを再構築する。"""
        new_df = self.source_df.copy()
        
        for col, ttype in self.transformations.items():
            if ttype == "log10":
                # 0以下の値は微小値で置換またはNaN
                vals = new_df[col].values
                new_df[col] = np.log10(np.where(vals > 0, vals, 1e-9))
            elif ttype == "zscore":
                scaler = StandardScaler()
                new_df[col] = scaler.fit_transform(new_df[[col]].fillna(new_df[col].median()))
            elif ttype == "binning":
                new_df[col] = pd.qcut(new_df[col], q=10, labels=False, duplicates='drop')
        
        self.current_df = new_df

    def get_data(self) -> pd.DataFrame:
        return self.current_df

    def set_selection(self, indices: List[int]):
        """選択範囲を指定する。"""
        mask = pd.Series(False, index=self.source_df.index)
        mask.iloc[indices] = True
        self.selection_mask = mask

    def get_selected_data(self) -> pd.DataFrame:
        if self.selection_mask is None:
            return pd.DataFrame()
        return self.source_df[self.selection_mask]
