"""
backend/data/memory_utils.py
メモリ使用量削減のためのデータ処理ユーティリティ。
"""
import gc
import logging
import pandas as pd
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

def process_large_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    大規模データフレームの処理（メモリ節約版）。
    型変換によるダウンキャスト、欠損値の多い列の削除、ガベージコレクションを実行します。
    """
    if df is None:
        return None
    
    try:
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # 1. 欠損値があまりにも多い列（例: 90%以上）を削除
        limit_ratio = 0.9
        columns_to_drop = [col for col in df.columns if df[col].isna().sum() / len(df) > limit_ratio]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"欠損値が多すぎるため {len(columns_to_drop)} 列を削除しました。")
        
        # 2. データ型を最適化（ダウンキャスト）
        # 整数列
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # 浮動小数点列
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        # 3. ガベージコレクション
        gc.collect()
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"メモリ最適化完了: {initial_memory:.2f}MB -> {final_memory:.2f}MB")
        
        return df
        
    except Exception as e:
        logger.warning(f"メモリ最適化中にエラーが発生しました（処理を継続します）: {e}")
        return df
