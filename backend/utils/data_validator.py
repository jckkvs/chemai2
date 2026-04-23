# backend/utils/data_validator.py
import logging
from typing import Optional, Tuple, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class DataValidator:
    """データ読み込み状態の検証・診断・レポート生成を担当する。"""
    
    @staticmethod
    def validate_dataframe(df: Optional[pd.DataFrame]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        DataFrameの妥当性を検証し、詳細情報を返す
        
        Returns:
            (is_valid, message, details)
        """
        details = {
            "is_none": df is None,
            "is_empty": False,
            "shape": None,
            "columns": [],
            "has_smiles": False,
            "memory_usage": None
        }
        
        if df is None:
            return False, "DataFrameがNoneです。データが読み込まれていません。", details
        
        try:
            details["shape"] = df.shape
            details["columns"] = list(df.columns)
            details["is_empty"] = df.empty
            
            # memory_usage の計算（空でない場合のみ）
            if not df.empty:
                details["memory_usage"] = df.memory_usage(deep=True).sum() / 1024  # KB
            else:
                details["memory_usage"] = 0
            
            # SMILES列の検出
            smiles_candidates = ['smiles', 'SMILES', 'Smiles', 'structure', 'Structure']
            details["has_smiles"] = any(col in df.columns for col in smiles_candidates)
            
            if df.empty:
                return False, "DataFrameは空です（行数0）。", details
            
            if len(df.columns) == 0:
                return False, "DataFrameに列がありません。", details
            
            return True, f"データ有効: {df.shape[0]}行 × {df.shape[1]}列", details
            
        except Exception as e:
            logger.error(f"DataFrame検証エラー: {e}")
            return False, f"データ検証中にエラーが発生: {str(e)}", details
    
    @staticmethod
    def generate_diagnostic_report(df: Optional[pd.DataFrame]) -> str:
        """診断レポートを生成（デバッグ用）"""
        is_valid, message, details = DataValidator.validate_dataframe(df)
        
        report = [
            "=== データ読み込み診断レポート ===",
            f"状態: {'✅ 有効' if is_valid else '❌ 無効'}",
            f"メッセージ: {message}",
            f"Noneチェック: {details['is_none']}",
            f"空データチェック: {details['is_empty']}",
            f"形状: {details['shape']}",
            f"列数: {len(details['columns'])}",
            f"列名: {details['columns'][:5]}{'...' if len(details['columns']) > 5 else ''}",
            f"SMILES列存在: {details['has_smiles']}",
            f"メモリ使用量: {details['memory_usage']:.2f} KB" if details['memory_usage'] else "メモリ使用量: N/A"
        ]
        
        return "\n".join(report)
