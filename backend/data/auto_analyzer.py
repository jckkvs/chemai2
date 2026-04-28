"""
backend/data/auto_analyzer.py
きれいなデータに対する自動解析方針立案
- 目的変数の自動検出
- 分析タスクの分類（回帰/分類/クラスタリング）
- 適切な前処理・モデル・評価指標の提案
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AnalysisTask(Enum):
    """分析タスクの種類"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"
    CLUSTERING = "clustering"
    EDA_ONLY = "eda_only"
    UNKNOWN = "unknown"


@dataclass
class AnalysisPlan:
    """解析方針のデータクラス"""
    task_type: AnalysisTask
    target_column: Optional[str]
    feature_columns: List[str]
    recommended_preprocessing: List[str]
    recommended_models: List[str]
    recommended_metrics: List[str]
    confidence: float
    notes: List[str] = field(default_factory=list)
    requires_user_input: bool = False
    user_questions: List[str] = field(default_factory=list)


class AutoAnalyzer:
    """
    自動解析方針立案エンジン
    データの特徴から最適な分析パイプラインを提案
    """
    
    # タスク別推奨モデル（chemai2で利用可能なもの）
    MODEL_RECOMMENDATIONS = {
        AnalysisTask.REGRESSION: [
            'RandomForestRegressor', 'LightGBMRegressor', 'XGBRegressor',
            'LinearRegression', 'SVR', 'LinearTreeRegressor'
        ],
        AnalysisTask.BINARY_CLASSIFICATION: [
            'RandomForestClassifier', 'LightGBMClassifier', 'XGBClassifier',
            'LogisticRegression', 'SVC', 'LinearTreeClassifier'
        ],
        AnalysisTask.MULTI_CLASS_CLASSIFICATION: [
            'RandomForestClassifier', 'LightGBMClassifier', 'XGBClassifier',
            'LogisticRegression(multi_class)', 'SVC'
        ],
        AnalysisTask.CLUSTERING: [
            'KMeans', 'DBSCAN', 'AgglomerativeClustering', 'GaussianMixture'
        ]
    }
    
    # 評価指標の推奨
    METRIC_RECOMMENDATIONS = {
        AnalysisTask.REGRESSION: ['rmse', 'mae', 'r2', 'mape'],
        AnalysisTask.BINARY_CLASSIFICATION: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        AnalysisTask.MULTI_CLASS_CLASSIFICATION: ['accuracy', 'f1_weighted', 'confusion_matrix'],
        AnalysisTask.CLUSTERING: ['silhouette_score', 'davies_bouldin_score', 'inertia']
    }
    
    # 前処理の推奨ルール
    PREPROCESSING_RULES = {
        'scale_numeric': '数値特徴量の標準化/正規化',
        'encode_categorical': 'カテゴリカル変数のエンコーディング',
        'handle_missing': '欠損値処理（中央値/最頻値/モデル予測）',
        'remove_constant': '定数列の削除',
        'remove_high_corr': '高相関特徴量の除去（多重共線性対策）',
        'feature_selection': '特徴量選択（相互情報量/寄与度ベース）'
    }
    
    def __init__(self, domain: str = 'chemistry'):
        """
        Args:
            domain: 対象ドメイン（'chemistry', 'general', 'material'等）
                   ドメイン固有のヒューリスティクスを適用
        """
        self.domain = domain
        self._domain_keywords = {
            'chemistry': ['smiles', 'logp', 'tpsa', 'homo', 'lumo', 'pka', 'solubility'],
            'material': ['conductivity', 'tensile', 'thermal', 'density'],
        }
    
    def create_analysis_plan(self, df: pd.DataFrame,
                            target_hint: Optional[str] = None,
                            user_goal: Optional[str] = None) -> AnalysisPlan:
        """
        データから解析方針を自動立案
        
        Args:
            df: 分析対象データフレーム
            target_hint: ユーザーが指定した目的変数のヒント（列名部分一致）
            user_goal: ユーザーの分析目的（自然言語）
        
        Returns:
            AnalysisPlan: 立案された解析方針
        """
        # 1. 目的変数候補の検出
        target_col = self._detect_target_column(df, target_hint, user_goal)
        
        # 2. 分析タスクの分類
        task_type = self._classify_task(df, target_col)
        
        # 3. 特徴量列の選定
        feature_cols = self._select_features(df, target_col, task_type)
        
        # 4. 推奨前処理の生成
        preprocessing = self._recommend_preprocessing(df, feature_cols, task_type)
        
        # 5. 推奨モデル・評価指標の選択
        models = self.MODEL_RECOMMENDATIONS.get(task_type, [])
        metrics = self.METRIC_RECOMMENDATIONS.get(task_type, [])
        
        # 6. 信頼度スコアの計算
        confidence = self._calculate_confidence(df, target_col, task_type)
        
        # 7. ユーザー確認が必要な項目の抽出
        user_questions = self._generate_user_questions(df, target_col, task_type, user_goal)
        
        return AnalysisPlan(
            task_type=task_type,
            target_column=target_col,
            feature_columns=feature_cols,
            recommended_preprocessing=preprocessing,
            recommended_models=models,
            recommended_metrics=metrics,
            confidence=confidence,
            notes=self._generate_notes(df, target_col, task_type),
            requires_user_input=len(user_questions) > 0,
            user_questions=user_questions
        )
    
    def _detect_target_column(self, df: pd.DataFrame,
                             target_hint: Optional[str],
                             user_goal: Optional[str]) -> Optional[str]:
        """目的変数候補を自動検出"""
        candidates = []
        
        # ユーザーヒントがある場合は優先
        if target_hint:
            matches = [col for col in df.columns if target_hint.lower() in str(col).lower()]
            if matches:
                return matches[0]
        
        # ユーザーゴールからキーワード抽出
        if user_goal and self.domain in self._domain_keywords:
            keywords = self._domain_keywords[self.domain]
            for kw in keywords:
                matches = [col for col in df.columns if kw in str(col).lower()]
                candidates.extend(matches)
        
        # 自動推論ルール
        for col in df.columns:
            # 数値列で、一意値が多い → 回帰目的変数候補
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() > len(df) * 0.3 and df[col].nunique() > 20:
                    candidates.append((col, 'regression', 0.7))
            
            # 2値・少数カテゴリ → 分類目的変数候補
            elif df[col].nunique() <= 10:
                task = 'binary' if df[col].nunique() == 2 else 'multi'
                candidates.append((col, task, 0.6))
            
            # 列名に「目的」「結果」「target」「label」を含む
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ['target', 'label', '目的', '結果', 'outcome']):
                candidates.append((col, 'unknown', 0.9))
        
        if not candidates:
            return None
        
        # スコアでソートして最高スコアを返す
        if candidates and isinstance(candidates[0], tuple):
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[0][0]
        elif candidates:
            return candidates[0]
        return None
    
    def _classify_task(self, df: pd.DataFrame, 
                      target_col: Optional[str]) -> AnalysisTask:
        """分析タスクを分類"""
        if target_col is None or target_col not in df.columns:
            return AnalysisTask.EDA_ONLY
        
        target = df[target_col]
        
        # 欠損が多い場合は不明
        if target.isnull().sum() / len(target) > 0.5:
            return AnalysisTask.UNKNOWN
        
        # 数値型 → 回帰
        if pd.api.types.is_numeric_dtype(target):
            if target.nunique() <= 2:
                return AnalysisTask.BINARY_CLASSIFICATION
            elif target.nunique() <= 10:
                # 少数値だが数値 → ユーザー確認が必要
                return AnalysisTask.UNKNOWN
            return AnalysisTask.REGRESSION
        
        # カテゴリカル型
        n_unique = target.nunique()
        if n_unique == 2:
            return AnalysisTask.BINARY_CLASSIFICATION
        elif n_unique <= 20:
            return AnalysisTask.MULTI_CLASS_CLASSIFICATION
        elif n_unique > len(df) * 0.5:
            # 高カーディナリティ → クラスタリングまたはEDA
            return AnalysisTask.CLUSTERING
        
        return AnalysisTask.UNKNOWN
    
    def _select_features(self, df: pd.DataFrame,
                        target_col: Optional[str],
                        task_type: AnalysisTask) -> List[str]:
        """特徴量列を選定"""
        exclude_patterns = ['id', 'idx', 'index', 'no.', '番号', 'smiles', 'mol']
        
        features = []
        for col in df.columns:
            if col == target_col:
                continue
            if any(pat in str(col).lower() for pat in exclude_patterns):
                continue
            # 定数列は除外
            if df[col].nunique() <= 1:
                continue
            features.append(col)
        
        return features
    
    def _recommend_preprocessing(self, df: pd.DataFrame,
                                feature_cols: List[str],
                                task_type: AnalysisTask) -> List[str]:
        """前処理ステップを推奨"""
        recommendations = []
        
        # 数値特徴量のスケーリング（線形モデル・距離ベースに重要）
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols and task_type in [AnalysisTask.REGRESSION, AnalysisTask.CLUSTERING]:
            recommendations.append('scale_numeric')
        
        # カテゴリカル変数のエンコーディング
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if cat_cols:
            recommendations.append('encode_categorical')
        
        # 欠損値処理
        if df[feature_cols].isnull().any().any():
            recommendations.append('handle_missing')
        
        # 定数列除去
        if any(df[c].nunique() <= 1 for c in feature_cols):
            recommendations.append('remove_constant')
        
        # 高相関除去（化学記述子では多重共線性が頻発）
        if len(numeric_cols) > 10:
            recommendations.append('remove_high_corr')
        
        # 特徴量選択（高次元データの場合）
        if len(feature_cols) > 100:
            recommendations.append('feature_selection')
        
        return recommendations
    
    def _calculate_confidence(self, df: pd.DataFrame,
                             target_col: Optional[str],
                             task_type: AnalysisTask) -> float:
        """提案の信頼度を計算（0.0〜1.0）"""
        score = 0.5  # ベーススコア
        
        if target_col:
            score += 0.2  # 目的変数検出で+
            if target_col in df.columns and df[target_col].isnull().sum() / len(df) < 0.1:
                score += 0.1  # 欠損少ないで+
        
        if task_type != AnalysisTask.UNKNOWN:
            score += 0.1
        
        # 化学ドメイン固有：SMILES列がある場合は信頼度アップ
        if any('smiles' in str(c).lower() for c in df.columns):
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_user_questions(self, df: pd.DataFrame,
                                target_col: Optional[str],
                                task_type: AnalysisTask,
                                user_goal: Optional[str]) -> List[str]:
        """ユーザーに確認すべき質問を生成"""
        questions = []
        
        if target_col is None:
            questions.append("分析の目的変数（予測したい項目）はどれですか？")
        
        if task_type == AnalysisTask.UNKNOWN:
            questions.append(f"列 '{target_col}' について、回帰分析と分類分析のどちらを希望しますか？")
        
        if user_goal:
            # ゴールと検出タスクが一致しない場合
            if '分類' in user_goal and task_type == AnalysisTask.REGRESSION:
                questions.append("分類分析をご希望ですが、目的変数が数値です。カテゴリ変換しますか？")
            elif '予測' in user_goal and task_type in [AnalysisTask.EDA_ONLY, AnalysisTask.CLUSTERING]:
                questions.append("教師あり学習をご希望ですが、目的変数が検出されませんでした。指定しますか？")
        
        # 化学ドメイン固有質問
        if self.domain == 'chemistry' and any('smiles' in str(c).lower() for c in df.columns):
            questions.append("分子構造（SMILES）から記述子を自動生成しますか？（量子化学計算含む）")
        
        return questions
    
    def _generate_notes(self, df: pd.DataFrame,
                       target_col: Optional[str],
                       task_type: AnalysisTask) -> List[str]:
        """分析上の注意点を生成"""
        notes = []
        
        if target_col and target_col in df.columns and df[target_col].isnull().sum() > 0:
            notes.append(f"目的変数に {df[target_col].isnull().sum()} 件の欠損があります。処理方針をご確認ください。")
        
        if task_type == AnalysisTask.REGRESSION and target_col and target_col in df.columns:
            if df[target_col].skew() > 2:
                notes.append("目的変数が右に歪んでいます。対数変換を検討してください。")
        
        if self.domain == 'chemistry':
            notes.append("化学データでは、実験条件（温度・溶媒・触媒等）を特徴量に含めることが重要です。")
        
        return notes
    
    def refine_plan_with_feedback(self, plan: AnalysisPlan,
                                 user_feedback: Dict) -> AnalysisPlan:
        """
        ユーザーフィードバックに基づき解析方針を修正
        """
        # 目的変数の変更
        if 'target_column' in user_feedback:
            plan.target_column = user_feedback['target_column']
            # Re-classify task based on new target
            # Note: We don't have df here, so this is a simplified re-classification
            # In a real scenario, we would pass the dataframe back or use some cached metadata
            plan.task_type = AnalysisTask.UNKNOWN 
        
        # タスクタイプの変更
        if 'task_type' in user_feedback:
            plan.task_type = AnalysisTask(user_feedback['task_type'])
            plan.recommended_models = self.MODEL_RECOMMENDATIONS.get(plan.task_type, [])
            plan.recommended_metrics = self.METRIC_RECOMMENDATIONS.get(plan.task_type, [])
        
        # 特徴量の追加・除外
        if 'add_features' in user_feedback:
            plan.feature_columns.extend(user_feedback['add_features'])
        if 'remove_features' in user_feedback:
            plan.feature_columns = [c for c in plan.feature_columns 
                                   if c not in user_feedback['remove_features']]
        
        # 信頼度再計算
        plan.confidence = min(plan.confidence + 0.1, 1.0)  # ユーザー確認で信頼度アップ
        plan.requires_user_input = False
        plan.user_questions = []
        
        return plan
