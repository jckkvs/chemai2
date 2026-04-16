"""
backend/ml/error_analysis.py

Error Analysis Engine — 予測誤差の要因分析
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

class ErrorAnalyzer:
    """
    誤差の原因を特定し、改善策を提案する。
    """
    def __init__(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, smiles_col: Optional[str] = None):
        self.df = df
        self.y_true = np.asarray(y_true).ravel()
        self.y_pred = np.asarray(y_pred).ravel()
        self.smiles_col = smiles_col
        self.residuals = self.y_true - self.y_pred
        self.abs_error = np.abs(self.residuals)

    def get_worst_samples(self, top_n: int = 10) -> pd.DataFrame:
        """誤差の大きいサンプルを取得する。"""
        indices = np.argsort(self.abs_error)[::-1][:top_n]
        worst_df = self.df.iloc[indices].copy()
        worst_df["actual"] = self.y_true[indices]
        worst_df["predicted"] = self.y_pred[indices]
        worst_df["abs_error"] = self.abs_error[indices]
        return worst_df

    def analyze_chemical_similarity(self, worst_samples_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        誤差の大きいサンプル間の化学的類似性を評価し、"難解な部分構造"のヒントを探す。
        """
        if not HAS_RDKIT or self.smiles_col is None:
            return []
        
        smiles_list = worst_samples_df[self.smiles_col].tolist()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list if isinstance(s, str)]
        mols = [m for m in mols if m is not None]
        
        if len(mols) < 2: return []
        
        # Morgan Fingerprint
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
        
        similar_clusters = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim > 0.6: # 類似度が高い場合
                    similar_clusters.append({
                        "idx1": i,
                        "idx2": j,
                        "similarity": sim,
                        "smiles1": smiles_list[i],
                        "smiles2": smiles_list[j]
                    })
        return similar_clusters

    def suggest_next_steps(self) -> List[str]:
        """誤差傾向に基づく改善策の提案。"""
        suggestions = []
        
        # 1. 系統的なバイアスのチェック
        mean_res = np.mean(self.residuals)
        if abs(mean_res) > 0.1 * np.std(self.y_true):
            suggestions.append("モデルが全体的に過小または過大評価しています。ターゲットの正規化やスケーリングを再確認してください。")
            
        # 2. 外れ値の数
        outlier_threshold = 3 * np.std(self.abs_error)
        n_outliers = np.sum(self.abs_error > outlier_threshold)
        if n_outliers > 0:
            suggestions.append(f"{n_outliers}個の極端な誤差サンプルがあります。これらが実験誤差（Conflict）でないか「Data Sandbox」で確認してください。")
            
        # 3. 類似した高誤差サンプル
        # (分析結果に応じて追加)
        
        return suggestions
