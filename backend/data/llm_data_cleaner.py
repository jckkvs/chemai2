"""
backend/data/llm_data_cleaner.py

LLM を使用したデータ整形・クリーニングコード生成モジュール。

機能:
  - 整形されていない CSV/Excel データを分析
  - LLM がデータクリーニング用の Python コードを生成
  - セル結合,誤字,欠損値の自動修正コードを生成
  - 外部 LLM（ChatGPT, Claude 等）用のプロンプトも生成可能
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backend.llm.provider import LLMRequest, LLMResponse
from backend.llm import get_llm_provider

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """データ品質レポート."""
    is_clean: bool = True                    # きれいなデータか
    issues: list[str] = field(default_factory=list)  # 検出された問題
    suggestions: list[str] = field(default_factory=list)  # 修正提案
    merged_cells_detected: bool = False      # セル結合 detected
    encoding_issues: bool = False            # エンコーディング問題
    structural_issues: list[str] = field(default_factory=list)  # 構造的な問題
    
    @property
    def needs_cleaning(self) -> bool:
        """クリーニングが必要か."""
        return not self.is_clean or len(self.issues) > 0


def analyze_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """
    DataFrame の品質を分析し,問題を特定する。
    
    Args:
        df: 分析対象の DataFrame
        
    Returns:
        DataQualityReport オブジェクト
    """
    report = DataQualityReport()
    
    # 1. 基本的な統計情報
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    
    # 2. 欠損値の分析
    missing_count = df.isna().sum().sum()
    missing_rate = missing_count / total_cells if total_cells > 0 else 0
    
    if missing_rate > 0.3:
        report.issues.append(f"欠損率が高い ({missing_rate:.1%})")
        report.suggestions.append("欠損値の多い行または列を削除することを検討")
        report.is_clean = False
    
    # 3. 列名のチェック
    col_issues = []
    for col in df.columns:
        if pd.isna(col) or str(col).strip() == "":
            col_issues.append(f"空の列名があります")
            report.is_clean = False
        if str(col) != str(col).strip():
            col_issues.append(f"列名 '{col}' に前後の空白があります")
            report.suggestions.append(f"列名から空白を除去: df.columns = df.columns.str.strip()")
    
    # 4. データ型の混在チェック
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            types_in_col = set(type(v).__name__ for v in non_null.head(100))
            if len(types_in_col) > 2:
                report.issues.append(f"列 '{col}' に複数のデータ型が混在: {types_in_col}")
                report.is_clean = False
    
    # 5. 構造的な問題の検出
    # ヘッダー行が複数ある可能性
    if n_rows > 2:
        first_row_types = set(type(v).__name__ for v in df.iloc[0])
        second_row_types = set(type(v).__name__ for v in df.iloc[1])
        if first_row_types == second_row_types and 'str' in first_row_types:
            report.structure_issues.append("ヘッダー行が複数ある可能性があります")
            report.suggestions.append("skiprows パラメータでヘッダー行を調整")
    
    # 6. 明らかなフッター・注釈行の検出
    if n_rows > 5:
        last_few_rows = df.tail(3)
        if last_few_rows.isna().sum().sum() / (3 * n_cols) > 0.8:
            report.structure_issues.append("末尾に注釈行またはフッターがある可能性があります")
            report.suggestions.append("df.iloc[:-n] で不要な行を削除")
    
    # 7. 数値列に文字列が含まれているか
    for col in df.select_dtypes(include=['number']).columns:
        non_numeric = df[col].apply(lambda x: not isinstance(x, (int, float)) and pd.notna(x)).sum()
        if non_numeric > 0:
            report.issues.append(f"数値列 '{col}' に {non_numeric} 件の非数値が含まれています")
            report.suggestions.append(f"pd.to_numeric(df['{col}'], errors='coerce') で変換")
            report.is_clean = False
    
    # 8. 重複ヘッダーの検出
    header_duplicates = df.columns.duplicated().sum()
    if header_duplicates > 0:
        report.issues.append(f"{header_duplicates} 個の重複した列名があります")
        report.is_clean = False
    
    # 総合判定
    if not report.issues and not report.structure_issues:
        report.is_clean = True
        report.suggestions.append("データはきれいに整形されています")
    
    logger.info(f"Data quality analysis: clean={report.is_clean}, issues={len(report.issues)}")
    return report


def generate_cleaning_code(
    df: pd.DataFrame,
    report: DataQualityReport,
    provider_name: str = "stub",
    use_external_llm: bool = False,
) -> tuple[str, str]:
    """
    データクリーニング用の Python コードを LLM で生成する。
    
    Args:
        df: 元の DataFrame
        report: データ品質レポート
        provider_name: 使用する LLM プロバイダー名
        use_external_llm: 外部 LLM 用プロンプトを生成するか
        
    Returns:
        (生成されたコード, 外部 LLM 用プロンプト) のタプル
    """
    # コンテキスト作成
    context = _build_context(df, report)
    
    if use_external_llm:
        # 外部 LLM 用のプロンプトを生成
        external_prompt = _build_external_cleaning_prompt(context)
        return "", external_prompt
    
    # 内部 LLM でコード生成
    try:
        provider = get_llm_provider(provider_name)
        
        system_prompt = _CLEANING_SYSTEM_PROMPT
        user_prompt = _build_user_prompt(context)
        
        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.2,
        )
        
        response = provider.generate(request)
        code = _extract_code_from_response(response.content)
        
        logger.info(f"Generated cleaning code using {provider_name}")
        return code, ""
        
    except Exception as e:
        logger.error(f"LLM code generation failed: {e}")
        # フォールバック：ルールベースのクリーニングコードを生成
        fallback_code = _generate_rule_based_cleaning_code(df, report)
        return fallback_code, ""


def _build_context(df: pd.DataFrame, report: DataQualityReport) -> dict:
    """LLM に渡すコンテキスト情報を構築."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_per_col": df.isna().sum().to_dict(),
        "sample_rows": df.head(3).to_dict(orient="records"),
        "issues": report.issues,
        "suggestions": report.suggestions,
        "structural_issues": report.structure_issues,
    }


def _build_user_prompt(context: dict) -> str:
    """ユーザープロンプトを構築."""
    issues_str = "\n".join(f"- {issue}" for issue in context["issues"])
    suggestions_str = "\n".join(f"- {s}" for s in context["suggestions"])
    
    prompt = f"""データフレームのクリーニングコードを生成してください。

## データ概要
- 行数：{context['shape'][0]}
- 列数：{context['shape'][1]}
- 列名：{', '.join(str(c) for c in context['columns'][:10])}{'...' if len(context['columns']) > 10 else ''}

## 検出された問題
{issues_str if issues_str else '特になし'}

## 構造的な問題
{chr(10).join(context['structural_issues']) if context['structural_issues'] else '特になし'}

## 既存の提案
{suggestions_str if suggestions_str else 'なし'}

上記の問題を解決する Python コードを生成してください。
pandas を使用し,元の DataFrame を受け取ってクリーニング済みの DataFrame を返す関数を作成してください."""
    
    return prompt


_CLEANING_SYSTEM_PROMPT = """あなたは ChemAI ML Studio のデータクリーニング専門家です。

## 役割
ユーザーの CSV/Excel データを機械学習に適した形式に整形する Python コードを生成してください。

## 出力形式
以下の形式の Python コードを**コードブロックなしで**生成してください：

```python
import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"データをクリーニングする関数\"\"\"
    # コピーを作成
    df_clean = df.copy()
    
    # ここにクリーニングロジックを記述
    # 例：
    # - 列名の空白除去
    # - 欠損値の処理
    # - データ型の変換
    # - 不要な行の削除
    # - セル結合の解消
    
    return df_clean
```

## 制約
- 外部 API や危険なシステムコールは使用禁止
- エラーハンドリングを適切に行う
- 各ステップでコメントを付ける
- 元のデータを破壊しない（copy() を使用する）
- コードブロック (```python) は付けない
- 説明文は不要。コードのみ出力する
"""


def _build_external_cleaning_prompt(context: dict) -> str:
    """外部 LLM（ChatGPT, Claude 等）に渡すための完全なプロンプト."""
    issues_str = "\n".join(f"- {issue}" for issue in context["issues"])
    
    prompt = f"""# ChemAI ML Studio データクリーニングコード生成依頼

あなたは化学データ解析の専門家として,整形されていない実験データを機械学習に適した形式に整理する Python コードを作成してください。

## データの状況
- サイズ：{context['shape'][0]}行 × {context['shape'][1]}列
- 列名：{', '.join(str(c) for c in context['columns'][:15])}{'...' if len(context['columns']) > 15 else ''}

## 検出されている問題点
{issues_str if issues_str else '特に大きな問題はありません'}

## サンプルデータ（最初の 3 行）
{pd.DataFrame(context['sample_rows']).to_string()}

## 求める出力
以下の形式の Python 関数を作成してください：

```python
import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    実験データをクリーニングする関数
    
    実施すること:
    1. 列名の正規化（空白除去,統一）
    2. 欠損値の適切な処理
    3. データ型の修正
    4. 不要な行・列の削除
    5. セル結合やフォーマット問題の修正
    \"\"\"
    df_clean = df.copy()
    
    # ここに具体的なクリーニング処理を記述
    
    return df_clean
```

## 注意事項
- RDKit や化学専門ライブラリは不要 (汎用的なデータクリーニングのみ)
- 各処理ステップにコメントを付ける
- エラーが発生しやすい箇所には try-except を入れる
- 元の DataFrame を変更せず,コピーを操作する
- 最終的に対数表形式 (tidy data) になるようにする

コードのみを出力し,説明文は含めないでください."""
    
    return prompt


def _extract_code_from_response(content: str) -> str:
    """レスポンスから Python コードを抽出."""
    import re
    
    # コードブロック記法があれば除去
    code_block_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # コードブロック記法がなければ全体を返す（前後の空白を除去）
    lines = content.strip().split('\n')
    # 説明文の可能性のある行をフィルタ
    code_lines = []
    for line in lines:
        if line.strip() and not line.strip().startswith(('##', '# 求', '# 注意', '求める出力')):
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


def _generate_rule_based_cleaning_code(df: pd.DataFrame, report: DataQualityReport) -> str:
    """フォールバック用のルールベースクリーニングコードを生成."""
    operations = []
    
    # 列名の空白除去
    operations.append("""    # 列名の空白を除去
    df_clean.columns = df_clean.columns.str.strip()""")
    
    # 欠損値が多い列の削除
    high_missing_cols = [col for col, count in df.isna().sum().items() 
                         if count / len(df) > 0.5]
    if high_missing_cols:
        cols_str = ', '.join(f"'{c}'" for c in high_missing_cols[:5])
        operations.append(f"""    # 欠損率が 50% 以上の列を削除
    cols_to_drop = [{cols_str}]
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')""")
    
    # 数値列の強制変換
    operations.append("""    # 数値列を強制的に変換
    for col in df_clean.select_dtypes(include=['object']).columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
        except Exception:
            pass""")
    
    # 重複行の削除
    operations.append("""    # 完全に重複した行を削除
    df_clean = df_clean.drop_duplicates()""")
    
    # 空の行・列の削除
    operations.append("""    # 完全に空の行と列を削除
    df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')""")
    
    code = f"""import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"ルールベースのデータクリーニング関数\"\"\"
    df_clean = df.copy()
    
{chr(10).join(operations)}
    
    # インデックスをリセット
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean
"""
    return code


def execute_cleaning_code(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    生成されたクリーニングコードを実行する。
    
    Args:
        code: 実行する Python コード
        df: 元の DataFrame
        
    Returns:
        (クリーニング後の DataFrame, ログメッセージ)
    """
    import traceback
    
    local_ns = {"df": df.copy(), "pd": pd, "np": __import__("numpy")}
    
    try:
        exec(code, {}, local_ns)
        
        if "clean_dataframe" not in local_ns:
            raise ValueError("clean_dataframe 関数が定義されていません")
        
        result_df = local_ns["clean_dataframe"](df)
        
        if not isinstance(result_df, pd.DataFrame):
            raise ValueError("関数が DataFrame を返していません")
        
        log_msg = f"クリーニング完了：{df.shape} → {result_df.shape}"
        logger.info(log_msg)
        return result_df, log_msg
        
    except Exception as e:
        error_msg = f"クリーニング実行エラー：{str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return df, error_msg
