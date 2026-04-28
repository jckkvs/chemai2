"""
backend/data/data_cleaner.py
LLMを活用したデータ整形・クリーニング支援
- 誤字修正
- セル結合の展開
- 列名の正規化
- 整形用Pythonコードの自動生成
"""
import re
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CleaningSuggestion:
    """クリーニング提案のデータクラス"""
    issue_type: str
    description: str
    suggested_code: str
    confidence: float  # 0.0 ~ 1.0
    auto_applicable: bool = False


class DataCleanerLLM:
    """
    LLM機能を活用したデータクリーニング支援クラス
    ローカルLLM / 外部API / プロンプト出力 の3モード対応
    """
    
    def __init__(self, mode: str = 'prompt_only', 
                 api_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: str = 'local'):
        """
        Args:
            mode: 'local' | 'api' | 'prompt_only'
                - local: ローカルLLMで直接実行
                - api: 外部API経由で実行
                - prompt_only: 実行せず、プロンプトのみ生成（セキュア環境向け）
            api_endpoint: APIエンドポイント（mode='api'時）
            api_key: APIキー（mode='api'時）
            model_name: 使用するモデル名
        """
        self.mode = mode
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_name = model_name
        self._cleaning_history: List[Dict] = []
    
    def analyze_data_issues(self, df: pd.DataFrame, 
                           sample_rows: int = 20) -> List[CleaningSuggestion]:
        """
        データの問題点を分析し、LLM支援でクリーニング提案を生成
        """
        suggestions = []
        
        # 1. 列名の整形提案
        suggestions.extend(self._suggest_column_cleaning(df))
        
        # 2. 欠損値処理提案
        suggestions.extend(self._suggest_missing_value_handling(df))
        
        # 3. 型変換提案
        suggestions.extend(self._suggest_type_conversion(df))
        
        # 4. 誤字・表記ゆれ検出提案（簡易ルールベース＋LLM補完）
        suggestions.extend(self._suggest_typo_correction(df, sample_rows))
        
        # 5. 結合セル展開提案
        suggestions.extend(self._suggest_merged_cell_handling(df))
        
        # 各提案にLLMによるコード生成を追加（mode依存）
        for suggestion in suggestions:
            if self.mode != 'prompt_only':
                suggestion.suggested_code = self._generate_cleaning_code(
                    suggestion, df.head(sample_rows)
                )
        
        return suggestions
    
    def _suggest_column_cleaning(self, df: pd.DataFrame) -> List[CleaningSuggestion]:
        """列名の整形提案"""
        suggestions = []
        
        # 空白・特殊文字を含む列名
        problematic_cols = [col for col in df.columns 
                          if pd.isna(col) or str(col).strip() == '' or re.search(r'[\s\(\)\[\]\/\\]', str(col))]
        
        if problematic_cols:
            code_lines = ["# 列名の正規化", "df.columns = df.columns.astype(str).str.strip()", 
                         "df.columns = df.columns.str.replace(r'[\\s\\(\\)\\[\\]\\/\\\\]+', '_', regex=True)",
                         "df.columns = [col if col else f'Column_{i}' for i, col in enumerate(df.columns)]"]
            
            suggestions.append(CleaningSuggestion(
                issue_type='column_name',
                description=f'列名に空白・特殊文字が含まれています: {problematic_cols[:5]}',
                suggested_code='\n'.join(code_lines),
                confidence=0.95,
                auto_applicable=True
            ))
        
        # 日本語列名の英語化提案（オプション）
        jp_cols = [col for col in df.columns if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', str(col))]
        if jp_cols:
            suggestions.append(CleaningSuggestion(
                issue_type='column_name_jp',
                description=f'日本語列名を検出: {jp_cols[:5]}。英語名への変換を提案可能',
                suggested_code=self._generate_jp_to_en_mapping_code(jp_cols),
                confidence=0.7,
                auto_applicable=False
            ))
        
        return suggestions
    
    def _suggest_missing_value_handling(self, df: pd.DataFrame) -> List[CleaningSuggestion]:
        """欠損値処理提案"""
        suggestions = []
        missing_stats = df.isnull().sum()
        
        # 高頻度欠損列
        high_missing = missing_stats[missing_stats > len(df) * 0.3]
        if not high_missing.empty:
            suggestions.append(CleaningSuggestion(
                issue_type='missing_high',
                description=f'欠損率30%超の列: {list(high_missing.index)}。削除を検討',
                suggested_code=f"df = df.drop(columns={list(high_missing.index)})",
                confidence=0.8,
                auto_applicable=True
            ))
        
        # 数値列の欠損：中央値補完提案
        num_missing = missing_stats[df.select_dtypes(include=[np.number]).columns]
        if num_missing.any():
            code_lines = ["# 数値列の欠損値を中央値で補完", 
                         "for col in df.select_dtypes(include=[np.number]).columns:",
                         "    if df[col].isnull().any():",
                         "        df[col] = df[col].fillna(df[col].median())"]
            suggestions.append(CleaningSuggestion(
                issue_type='missing_numeric',
                description=f'数値列に欠損: {list(num_missing[num_missing>0].index)}',
                suggested_code='\n'.join(code_lines),
                confidence=0.85,
                auto_applicable=True
            ))
        
        return suggestions
    
    def _suggest_type_conversion(self, df: pd.DataFrame) -> List[CleaningSuggestion]:
        """データ型変換提案"""
        suggestions = []
        
        # 数値として扱いたい文字列列の検出
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            if sample.empty: continue
            # 数値っぽい文字列（通貨記号・カンマ除去で数値になる）
            if sample.astype(str).str.replace(r'[¥$,，\s]', '', regex=True).str.match(r'^-?\d+\.?\d*$').all():
                suggestions.append(CleaningSuggestion(
                    issue_type='type_conversion',
                    description=f'列 "{col}" は数値に変換可能（通貨記号・カンマを含む）',
                    suggested_code=f"df['{col}'] = pd.to_numeric(df['{col}'].astype(str).str.replace(r'[¥$,，]', '', regex=True), errors='coerce')",
                    confidence=0.9,
                    auto_applicable=True
                ))
        
        # 日付列の検出
        date_patterns = [r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', r'\d{1,2}/\d{1,2}/\d{4}']
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(50)
            if sample.astype(str).str.match('|'.join(date_patterns)).any():
                suggestions.append(CleaningSuggestion(
                    issue_type='date_conversion',
                    description=f'列 "{col}" に日付形式の値を検出',
                    suggested_code=f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')",
                    confidence=0.75,
                    auto_applicable=True
                ))
        
        return suggestions
    
    def _suggest_typo_correction(self, df: pd.DataFrame, 
                                sample_rows: int) -> List[CleaningSuggestion]:
        """誤字・表記ゆれ検出提案（簡易ルール＋LLM補完）"""
        suggestions = []
        
        # カテゴリカル列の類似値検出
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 100:  # 高カーディナリティはスキップ
                values = df[col].dropna().unique()
                # 簡易類似度チェック（編集距離）
                similar_pairs = []
                for i, v1 in enumerate(values[:50]):  # 計算量削減
                    for v2 in values[i+1:50]:
                        if self._edit_distance(str(v1), str(v2)) <= 2 and len(str(v1)) > 3:
                            similar_pairs.append((v1, v2))
                
                if similar_pairs:
                    suggestions.append(CleaningSuggestion(
                        issue_type='typo',
                        description=f'列 "{col}" に類似する表記: {similar_pairs[:3]}',
                        suggested_code=self._generate_typo_fix_code(col, similar_pairs[:5]),
                        confidence=0.6,
                        auto_applicable=False  # 要確認
                    ))
        
        return suggestions
    
    def _suggest_merged_cell_handling(self, df: pd.DataFrame) -> List[CleaningSuggestion]:
        """結合セル展開提案"""
        suggestions = []
        
        # 前方に同じ値が連続する列（結合セルの可能性）
        for col in df.columns:
            if df[col].notna().sum() < len(df) * 0.5:  # 欠損が多い列
                # 前方填充で値が増えるかチェック
                filled = df[col].ffill()
                if filled.notna().sum() > df[col].notna().sum() * 1.5:
                    suggestions.append(CleaningSuggestion(
                        issue_type='merged_cells',
                        description=f'列 "{col}" に結合セルの疑い（前方填充で値が増加）',
                        suggested_code=f"# 結合セルの展開（前方填充）\ndf['{col}'] = df['{col}'].ffill()",
                        confidence=0.7,
                        auto_applicable=True
                    ))
        
        return suggestions
    
    def _generate_cleaning_code(self, suggestion: CleaningSuggestion, 
                               sample_df: pd.DataFrame) -> str:
        """LLMを使用してクリーニングコードを生成（mode依存）"""
        if self.mode == 'prompt_only':
            # codeは空のまま、プロンプト生成用に維持
            return suggestion.suggested_code
        
        # TODO: 実際のLLM呼び出し実装
        # ここでは簡易的に元コードを返す
        return suggestion.suggested_code
    
    def _generate_jp_to_en_mapping_code(self, jp_columns: List[str]) -> str:
        """日本語列名→英語列名のマッピングコード生成"""
        # 簡易マッピング例（実際はLLMで生成）
        mapping = {
            '売上': 'sales', '数量': 'quantity', '日付': 'date',
            '商品名': 'product_name', '顧客ID': 'customer_id'
        }
        code_lines = ["# 日本語列名→英語列名マッピング", "column_mapping = {"]
        for col in jp_columns:
            en_name = mapping.get(col, col)
            code_lines.append(f"    '{col}': '{en_name}',")
        code_lines.append("}\ndf = df.rename(columns=column_mapping)")
        return '\n'.join(code_lines)
    
    def _generate_typo_fix_code(self, column: str, 
                               similar_pairs: List[Tuple[str, str]]) -> str:
        """表記ゆれ修正コード生成"""
        code_lines = [f"# 列 '{column}' の表記ゆれ修正"]
        for v1, v2 in similar_pairs:
            code_lines.append(f"# '{v1}' / '{v2}' → どちらに統一しますか？")
            code_lines.append(f"# df['{column}'] = df['{column}'].replace('{v2}', '{v1}')")
        return '\n'.join(code_lines)
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """簡易編集距離計算"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    def generate_external_prompt(self, df: pd.DataFrame, 
                                issue_description: str) -> str:
        """
        外部チャット用プロンプトを生成
        セキュア環境で高精度LLMを使いたい場合に、このプロンプトをコピーして使用
        """
        sample = df.head(10).to_csv(index=False)
        
        prompt = f"""# ChemAI Data Cleaning Assistant

あなたは化学データ分析の専門家です。以下のデータの問題点を指摘し、
pandasを使用してクリーニングするPythonコードを生成してください。

## データの概要
- 行数: {len(df)}, 列数: {len(df.columns)}
- 問題の説明: {issue_description}

## データサンプル（先頭10行）
```csv
{sample}
```

## 出力形式
1. 検出された問題点のリスト
2. 各問題に対する修正コード（pandas使用）
3. 修正後のデータ確認コード

## 制約事項
- 化学データ（物性値、実験条件、分子記述子等）を扱っている可能性を考慮
- 数値の単位変換や有効数字には注意
- 欠損値処理は分析目的に応じて提案（削除/補完/フラグ追加）
- コードにはコメントを必ず付与

## 出力例
### 問題点
1. 列"温度_℃"に文字列"室温"が混在

### 修正コード
```python
# 温度列のクリーニング
def parse_temperature(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str) and '室温' in val:
        return 25.0  # 室温を25℃と仮定
    return pd.to_numeric(val, errors='coerce')

df['温度_℃'] = df['温度_℃'].apply(parse_temperature)
```

### 確認コード
```python
print(df['温度_℃'].describe())
print(df['温度_℃'].isnull().sum())
```
"""
        return prompt
    
    def apply_cleaning(self, df: pd.DataFrame, 
                      suggestions: List[CleaningSuggestion],
                      auto_apply: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        提案されたクリーニングを適用
        auto_apply=True: confidenceが高い提案を自動適用
        auto_apply=False: 全提案をレビュー後、手動で選択適用
        """
        result_df = df.copy()
        applied = []
        skipped = []
        
        for suggestion in suggestions:
            if auto_apply and suggestion.auto_applicable and suggestion.confidence >= 0.8:
                try:
                    # コードを実行（簡易eval - 実際はより安全な方法で）
                    # 本番では exec/eval は避け、AST解析やサンドボックス環境を推奨
                    local_vars = {'df': result_df, 'pd': pd, 'np': np}
                    exec(suggestion.suggested_code, {"__builtins__": {}}, local_vars)
                    result_df = local_vars['df']
                    applied.append(asdict(suggestion))
                    logger.info(f"クリーニング適用: {suggestion.issue_type}")
                except Exception as e:
                    logger.warning(f"クリーニング適用失敗: {suggestion.issue_type}, error={e}")
                    skipped.append({**asdict(suggestion), 'error': str(e)})
            else:
                skipped.append(asdict(suggestion))
        
        return result_df, {'applied': applied, 'skipped': skipped}
