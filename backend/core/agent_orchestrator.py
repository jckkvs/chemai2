"""
backend/core/agent_orchestrator.py
非コーディング研究者向け自律解析エージェント
- 自然言語で解析指示 → 自動コード生成・実行・可視化
- 化学ドメイン知識を組み込んだプロンプト設計
"""
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRequest:
    """解析リクエスト"""
    user_query: str  # 自然言語クエリ
    data_context: Dict  # 読み込み済みデータ情報
    goal_type: str  # 'prediction', 'visualization', 'comparison', 'optimization'
    constraints: Dict  # 制約条件（時間、精度、リソース等）


@dataclass
class AgentResponse:
    """エージェント応答"""
    plan: str  # 解析方針の説明
    generated_code: str  # 生成されたPythonコード
    expected_output: str  # 期待される出力の説明
    confidence: float
    requires_confirmation: bool  # ユーザー確認が必要か
    next_steps: List[str]


class MIAgentOrchestrator:
    """
    マテリアルズインフォマティクス特化エージェント
    非コーディングユーザー向けに解析を自律実行
    """
    
    # 化学ドメイン特化プロンプトテンプレート
    MI_PROMPT_TEMPLATE = """あなたはマテリアルズインフォマティクスの専門家アシスタントです。
ユーザーはコーディングができません。自然言語の指示から、以下の手順で解析を実行してください。

## ユーザーの指示
{user_query}

## 利用可能なデータ
{data_summary}

## 実行手順
1. 解析目的の明確化（回帰/分類/可視化/比較）
2. 適切な前処理の提案（欠損値、正規化、記述子生成）
3. 機械学習モデルの選択（化学データに強いモデル優先）
4. 評価指標の設定（RMSE, R², 混同行列等）
5. 可視化方法の提案（Plotly使用、インタラクティブ推奨）

## 制約事項
- ハイパーパラメータ探索は不要（デフォルト値で十分）
- Optunaは使用しない
- RAGは構築しない
- 生成コードは即実行可能に（import文を含む完全なスクリプト）
- エラーハンドリングを必ず含める
- 化学的妥当性を最優先（単位、有効数字、物性の物理的意味）

## 出力形式
### 解析方針
[簡潔な説明]

### 生成コード
```python
# 完全な実行可能コード
{code}
```

### 実行後の確認方法
[ユーザーが結果を確認する手順]

### 次のステップ提案
[追加分析の提案]
"""
    
    def __init__(self, llm_client, model_selector):
        """
        Args:
            llm_client: LLM推論クライアント（llama.cpp/vLLM/外部API）
            model_selector: ModelSelectorインスタンス
        """
        self.llm = llm_client
        self.selector = model_selector
        self._session_history: List[Dict] = []
    
    def process_request(self, request: AnalysisRequest) -> AgentResponse:
        """解析リクエストを処理"""
        # 1. データ要約を生成
        data_summary = self._summarize_data(request.data_context)
        
        # 2. プロンプトを構築
        prompt = self.MI_PROMPT_TEMPLATE.format(
            user_query=request.user_query,
            data_summary=data_summary,
            code="{自動生成}"
        )
        
        # 3. LLMにコード生成を依頼
        # 注意: 実際のコード生成はLLMの出力をパースして抽出
        # ここではモックまたはクライアント呼び出しを想定
        if hasattr(self.llm, 'generate'):
            llm_output = self.llm.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1  # 再現性重視
            )
        else:
            # フォールバック（テスト用）
            llm_output = "### 解析方針\nテスト方針\n### 生成コード\n```python\nprint('Hello MI')\n```"
        
        # 4. 出力をパースして構造化
        plan, code, confirmation_needed = self._parse_llm_output(llm_output)
        
        # 5. 信頼度評価（簡易）
        confidence = 0.85 if 'RandomForest' in code or 'LightGBM' in code else 0.7
        
        return AgentResponse(
            plan=plan,
            generated_code=code,
            expected_output="予測結果のDataFrameとPlotly可視化",
            confidence=confidence,
            requires_confirmation=confirmation_needed,
            next_steps=[
                "コードを実行して結果を確認",
                "必要に応じてパラメータ調整",
                "追加データのアップロードで精度向上"
            ]
        )
    
    def _summarize_data(self, context: Dict) -> str:
        """データコンテキストを要約"""
        import pandas as pd
        if 'dataframe' in context:
            df = context['dataframe']
            if isinstance(df, pd.DataFrame):
                return f"行数: {len(df)}, 列数: {len(df.columns)}, 列名: {list(df.columns)[:10]}"
        elif 'text' in context:
            return f"テキストデータ: {len(str(context['text']))}文字"
        return "データ読み込み済み"
    
    def _parse_llm_output(self, output: str) -> tuple:
        """LLM出力をパースして計画・コード・確認フラグを抽出"""
        lines = output.split('\n')
        plan, code = [], []
        in_code = False
        
        for line in lines:
            if '### 解析方針' in line:
                continue
            elif '### 生成コード' in line:
                in_code = True
                continue
            elif '```python' in line:
                continue
            elif '```' in line and in_code:
                in_code = False
                continue
            elif '### ' in line and in_code:
                in_code = False
            
            if in_code:
                code.append(line)
            elif not any(x in line for x in ['###', '```']):
                plan.append(line)
        
        # 確認が必要かの簡易判定
        needs_confirm = any(kw in output.lower() for kw in ['確認', 'review', 'check', 'uncertain'])
        
        return '\n'.join(plan).strip(), '\n'.join(code).strip(), needs_confirm
