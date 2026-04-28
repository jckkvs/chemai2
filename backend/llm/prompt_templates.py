"""
backend/llm/prompt_templates.py
LLM用プロンプトテンプレート集
- データクリーニング用
- 解析方針立案用
- コード生成用
- 外部チャット用
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """プロンプトテンプレートのデータクラス"""
    name: str
    description: str
    template: str
    variables: List[str]
    category: str  # 'cleaning' | 'analysis' | 'code' | 'external'
    
    def format(self, **kwargs) -> str:
        """テンプレートに変数を埋め込んでフォーマット"""
        # 未指定変数は空文字で置換
        for var in self.variables:
            if var not in kwargs:
                kwargs[var] = ''
        return self.template.format(**kwargs)


# ============================================================================
#  プロンプトテンプレート定義
# ============================================================================

PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    'data_cleaning_basic': PromptTemplate(
        name='data_cleaning_basic',
        description='基本的なデータクリーニング用プロンプト',
        template="""あなたはPythonとpandasの専門家です。以下のデータの問題点を指摘し、
クリーニングするコードを生成してください。

## データ情報
- 行数: {n_rows}, 列数: {n_cols}
- 列名: {columns}

## 問題の説明
{issue_description}

## データサンプル（先頭{sample_size}行）
```
{data_sample}
```

## 出力形式
1. 検出された問題点（箇条書き）
2. 修正コード（pandas使用、コメント付き）
3. 修正確認コード

## 制約
- 化学データを扱っている可能性を考慮
- 数値の単位・有効数字に注意
- コードは安全で再現可能に
""",
        variables=['n_rows', 'n_cols', 'columns', 'issue_description', 'sample_size', 'data_sample'],
        category='cleaning'
    ),
    
    'analysis_plan_generation': PromptTemplate(
        name='analysis_plan_generation',
        description='解析方針立案用プロンプト',
        template="""あなたはデータサイエンスとケモインフォマティクスの専門家です。
以下のデータに対して、最適な分析方針を提案してください。

## データ概要
- タスクタイプ: {task_type}
- 目的変数: {target_column}
- 特徴量数: {n_features}
- ユーザーの目的: {user_goal}

## 分析方針に含める項目
1. 推奨される前処理ステップ（理由付き）
2. 適切な機械学習モデル（複数候補）
3. 評価指標の選択基準
4. 注意点・リスク要因
5. 次のステップの提案

## 出力形式
Markdown形式で、見出しを使って構造化して出力してください。
専門用語は必要に応じて簡潔に説明してください。
""",
        variables=['task_type', 'target_column', 'n_features', 'user_goal'],
        category='analysis'
    ),
    
    'code_generation_safe': PromptTemplate(
        name='code_generation_safe',
        description='安全なコード生成用プロンプト（サンドボックス前提）',
        template="""あなたはPythonコード生成の専門家です。以下の要件を満たすコードを生成してください。

## 要件
{requirements}

## 使用可能なライブラリ
{allowed_libraries}

## 制約事項（必須遵守）
1. 外部ネットワークアクセスなし
2. ファイルシステム書き込みは指定パスのみ
3. 無限ループ・再帰の防止
4. メモリ使用量の制限考慮
5. エラーハンドリングを必ず実装

## 出力形式
```python
# コードここから
{code}
# コードここまで
```

コードの後に、使用方法と想定される出力を簡潔に説明してください。
""",
        variables=['requirements', 'allowed_libraries', 'code'],
        category='code'
    ),
    
    'external_chat_prompt': PromptTemplate(
        name='external_chat_prompt',
        description='外部チャット（ChatGPT等）に貼り付け用の完全プロンプト',
        template="""# ChemAI Data Analysis Assistant

あなたは化学データ分析の専門家アシスタントです。
以下のコンテキストとデータに基づいて、分析支援を行ってください。

## セッション情報
- アプリ: ChemAI ML Studio
- ユーザーの分析目的: {user_goal}
- 現在のステップ: {current_step}

## 分析対象データ
{data_summary}

## 現在の問題・質問
{user_question}

## 期待する回答形式
{expected_output_format}

## 制約事項
- 化学적妥当性を最優先
- 数値計算は単位・有効数字に注意
- コードを提示する場合は、必ず説明を付与
- 不確かな情報は「不明」と明記

## 出力例
### 分析提案
[具体的な提案]

### 推奨コード
```python
[コード]
```

### 次のステップ
[アクション項目]
""",
        variables=['user_goal', 'current_step', 'data_summary', 'user_question', 'expected_output_format'],
        category='external'
    ),
    
    'error_debugging': PromptTemplate(
        name='error_debugging',
        description='エラー発生時のデバッグ支援用プロンプト',
        template="""Pythonコードでエラーが発生しました。原因を特定し、修正コードを提案してください。

## エラー情報
```
{error_message}
```

## 問題のコード
```python
{problem_code}
```

## 実行コンテキスト
- 入力データ形状: {input_shape}
- 使用ライブラリ: {libraries}
- 期待する動作: {expected_behavior}

## 出力形式
1. エラー原因の分析
2. 修正コード（差分または完全版）
3. 再発防止のアドバイス
""",
        variables=['error_message', 'problem_code', 'input_shape', 'libraries', 'expected_behavior'],
        category='code'
    )
}


def get_template(name: str) -> Optional[PromptTemplate]:
    """テンプレートを名前で取得"""
    return PROMPT_TEMPLATES.get(name)


def list_templates(category: Optional[str] = None) -> List[str]:
    """テンプレート一覧を取得（カテゴリでフィルタ可能）"""
    if category:
        return [name for name, tpl in PROMPT_TEMPLATES.items() if tpl.category == category]
    return list(PROMPT_TEMPLATES.keys())


def create_external_prompt(user_goal: str, data_summary: str, 
                          user_question: str, current_step: str = 'data_upload') -> str:
    """
    外部チャット用の完全プロンプトを生成
    セキュア環境で高精度LLMを使いたい場合に使用
    """
    template = get_template('external_chat_prompt')
    if not template:
        return "エラー: テンプレートが見つかりません"
    
    return template.format(
        user_goal=user_goal,
        current_step=current_step,
        data_summary=data_summary,
        user_question=user_question,
        expected_output_format="Markdown形式で、見出しを使って構造化"
    )
