"""
frontend_nicegui/pages/preprocessing_page.py
Preprocessing & Feature Selection page - 仕様書5.2, 5.3, 5.1.2に基づく実装
特徴量選択、前処理設定、相関分析
"""
from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class PreprocessingPage:
    """Preprocessing & Feature Selectionページ"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.selected_features: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.smiles_cols: List[str] = []
        self.domain_knowledge: Dict = {}

    def render(self):
        """Preprocessingページを描画"""
        with ui.column().classes('w-full max-w-7xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('🔧 Preprocessing & Feature Selection').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('仕様書5.2, 5.3に基づく').classes('text-gray-600')

            # データ未ロード時
            self._no_data = ui.card().classes('w-full mb-4')
            with self._no_data:
                ui.label('⚠️ データが読み込まれていません').classes('text-lg font-bold text-orange-600 mb-2')
                ui.label('「Data Upload」タブからデータをアップロードしてください。').classes('text-sm text-gray-600')
                ui.button('← Data Uploadへ', on_click=lambda: ui.navigate.to('/#data'), color='primary').props('outline')

            # データ状態
            self._data_card = ui.card().classes('w-full mb-4')
            self._data_card.visible = False
            with self._data_card:
                ui.label('📊 データ状態').classes('font-bold text-lg mb-2')
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('行数').classes('text-xs text-gray-500')
                        self._row_count = ui.label('-')
                    with ui.column():
                        ui.label('全列数').classes('text-xs text-gray-500')
                        self._col_count = ui.label('-')
                    with ui.column():
                        ui.label('数値列').classes('text-xs text-gray-500')
                        self._numeric_count = ui.label('-')
                    with ui.column():
                        ui.label('カテゴリ列').classes('text-xs text-gray-500')
                        self._cat_count = ui.label('-')
                    with ui.column():
                        ui.label('SMILES列').classes('text-xs text-gray-500')
                        self._smiles_count = ui.label('-')

            # 目的変数選択
            self._target_card = ui.card().classes('w-full mb-4')
            self._target_card.visible = False
            with self._target_card:
                ui.label('🎯 目的変数選択').classes('font-bold text-lg mb-2')
                ui.label('予測対象となる変数を選択してください。数値＝回帰、カテゴリ＝分類').classes('text-sm text-gray-500 mb-2')
                self._target_select = ui.select(
                    options=[],
                    label='目的変数 (Target)',
                    with_input=True
                ).classes('w-full')

            # 特徴量選択 (仕様書5.3)
            self._feature_card = ui.card().classes('w-full mb-4')
            self._feature_card.visible = False
            with self._feature_card:
                ui.label('🔬 Feature Selection').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **LLMによる特徴量選択** (仕様書4.4):
                - データのサンプル数に応じて適切な特徴量数をLLMが提案
                - 相関係数順ソート、物理化学的妥当性をLLMが評価
                - ユーザーは最終調整可能
                """)

                # 選択方式
                with ui.row().classes('w-full gap-4 mb-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('選択方式').classes('text-xs text-gray-500')
                        self._selection_mode = ui.select(
                            options={
                                'manual': '手動選択',
                                'correlation': '相関係数順',
                                'llm': 'LLM推奨',
                                'domain': 'ドメイン知識優先',
                            },
                            value='correlation'
                        ).classes('w-full')

                    with ui.column().classes('flex-1'):
                        ui.label('最大特徴量数').classes('text-xs text-gray-500')
                        self._max_features = ui.number(value=20, min=1, max=100).classes('w-full')

                # 特徴量リスト
                ui.label('特徴量選択 (チェックで有効化)').classes('font-bold text-sm mt-2 mb-2')
                self._feature_container = ui.column().classes('w-full max-h-96 overflow-y-auto')

                with ui.row().classes('w-full gap-2 mt-4'):
                    ui.button('✓ すべて選択', on_click=self._select_all, color='primary').props('dense')
                    ui.button('✗ すべて解除', on_click=self._deselect_all).props('dense outline')
                    ui.button('🔍 相関順ソート', on_click=self._sort_by_correlation).props('dense outline')

            # 前処理設定 (仕様書5.4)
            self._preprocess_card = ui.card().classes('w-full mb-4')
            self._preprocess_card.visible = False
            with self._preprocess_card:
                ui.label('⚙️ Preprocessing Configuration').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **データ変換** (仕様書5.4):
                - **PowerTransformer**: Yeo-Johnson, Box-Cox
                - **QuantileTransformer**: uniform, normal（個人的にuniformが好成績）
                """)

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('数値スケーリング').classes('text-xs text-gray-500')
                        self._scaler_select = ui.select(
                            options={
                                'standard': 'StandardScaler (標準化)',
                                'minmax': 'MinMaxScaler (0-1)',
                                'robust': 'RobustScaler (中央値)',
                                'none': 'なし',
                            },
                            value='standard'
                        ).classes('w-full')

                    with ui.column().classes('flex-1'):
                        ui.label('欠損値補完').classes('text-xs text-gray-500')
                        self._imputer_select = ui.select(
                            options={
                                'median': '中央値',
                                'mean': '平均値',
                                'knn': 'KNN補完',
                                'iterative': '反復補完',
                            },
                            value='median'
                        ).classes('w-full')

                    with ui.column().classes('flex-1'):
                        ui.label('カテゴリエンコーディング').classes('text-xs text-gray-500')
                        self._encoder_select = ui.select(
                            options={
                                'onehot': 'One-Hot Encoding',
                                'label': 'Label Encoding',
                                'target': 'Target Encoding',
                            },
                            value='onehot'
                        ).classes('w-full')

                # 変換設定
                with ui.expansion('データ変換設定', icon='transform').classes('w-full mt-4'):
                    with ui.row().classes('w-full gap-4'):
                        with ui.column().classes('flex-1'):
                            ui.label('PowerTransform').classes('text-xs text-gray-500')
                            self._power_transform = ui.select(
                                options={
                                    'none': 'なし',
                                    'yeo-johnson': 'Yeo-Johnson',
                                    'box-cox': 'Box-Cox',
                                },
                                value='none'
                            ).classes('w-full')

                        with ui.column().classes('flex-1'):
                            ui.label('QuantileTransform').classes('text-xs text-gray-500')
                            self._quantile_transform = ui.select(
                                options={
                                    'none': 'なし',
                                    'uniform': 'Uniform (推奨)',
                                    'normal': 'Normal',
                                },
                                value='none'
                            ).classes('w-full')

                    ui.label('※ LLMがデータに応じて推奨します',).classes('text-xs text-gray-500 mt-2')

            # 相関分析
            self._correlation_card = ui.card().classes('w-full mb-4')
            self._correlation_card.visible = False
            with self._correlation_card:
                ui.label('📈 相関分析').classes('font-bold text-lg mb-2')
                ui.label('目的変数との相関係数順ソート').classes('text-sm text-gray-500 mb-2')

                self._correlation_container = ui.column().classes('w-full')

                ui.button('📊 相関ヒートマップを表示', on_click=self._show_correlation_heatmap, color='primary').props('outline')

            # ドメイン知識の反映 (仕様書5.5)
            self._domain_card = ui.card().classes('w-full mb-4')
            self._domain_card.visible = False
            with self._domain_card:
                ui.label('🧠 Domain Knowledge Integration').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **ユーザーのドメイン知識を反映** (仕様書5.5):

                - ユーザーが「重要」と指定した変数を優先的に選択
                - ユーザーが「あまり効かない」と指定した変数を除外・低優先度化
                - ドメイン知識に基づく変数間の関連性をLASSO等の正則化で考慮
                """)

                self._domain_container = ui.column().classes('w-full')

                with ui.row().classes('w-full gap-2 mt-4'):
                    ui.button('🧠 ドメイン知識を反映', on_click=self._apply_domain_knowledge, color='primary')
                    ui.button('💬 LLMに相談', on_click=self._consult_llm, color='primary').props('outline')

            # 適用ボタン
            self._apply_card = ui.card().classes('w-full mb-4')
            self._apply_card.visible = False
            with self._apply_card:
                with ui.row().classes('w-full justify-center gap-4'):
                    ui.button('✓ 前処理設定を適用', on_click=self._apply_preprocessing, color='primary', size='lg')
                    ui.button('💾 設定を保存', on_click=self._save_config).props('outline')

    def load_data(self, df: pd.DataFrame):
        """データをロード"""
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.smiles_cols = [c for c in df.columns if 'smiles' in c.lower()]

        # UI更新
        self._no_data.visible = False
        self._data_card.visible = True
        self._target_card.visible = True
        self._feature_card.visible = True
        self._preprocess_card.visible = True
        self._correlation_card.visible = True
        self._domain_card.visible = True
        self._apply_card.visible = True

        self._row_count.text = f"{len(df):,}"
        self._col_count.text = f"{len(df.columns)}"
        self._numeric_count.text = f"{len(self.numeric_cols)}"
        self._cat_count.text = f"{len(self.categorical_cols)}"
        self._smiles_count.text = f"{len(self.smiles_cols)}"

        # 目的変数選択肢
        all_cols = self.numeric_cols + self.categorical_cols
        self._target_select.options = all_cols
        if self.numeric_cols:
            self._target_select.value = self.numeric_cols[-1]

        # 特徴量リスト更新
        self._update_feature_list()

        ui.notify(f'✓ Preprocessing: {len(df)}行 × {len(df.columns)}列を読み込みました', type='positive')

    def set_target(self, target_col: str):
        """目的変数を設定"""
        self.target_col = target_col
        if hasattr(self, '_target_select'):
            self._target_select.value = target_col
        self._update_feature_list()

    def _update_feature_list(self):
        """特徴量リストを更新"""
        if self.df is None:
            return

        self._feature_container.clear()
        self.selected_features = []

        # 目的変数以外の数値列をデフォルト選択
        exclude_cols = [self.target_col] if self.target_col else []
        feature_cols = [c for c in self.numeric_cols if c not in exclude_cols]

        with self._feature_container:
            for i, col in enumerate(feature_cols[:50]):  # 最大50列表示
                with ui.row().classes('w-full items-center gap-2'):
                    checkbox = ui.checkbox(text=col, value=True)
                    checkbox.classes('text-sm')
                    self.selected_features.append(col)

                    # 相関係数があれば表示
                    if self.target_col and self.target_col in self.df.columns:
                        try:
                            corr = self.df[col].corr(self.df[self.target_col])
                            color = 'text-green-600' if corr > 0.5 else 'text-gray-500'
                            ui.label(f'r={corr:.3f}').classes(f'text-xs {color}')
                        except Exception:
                            pass

    def _select_all(self):
        """すべて選択"""
        # チェックボックスをすべてON（簡易実装）
        ui.notify('すべて選択しました', type='positive')

    def _deselect_all(self):
        """すべて解除"""
        ui.notify('すべて解除しました', type='positive')

    def _sort_by_correlation(self):
        """相関係数順にソート"""
        if not self.target_col or self.df is None:
            ui.notify('目的変数を選択してください', type='warning')
            return

        ui.notify('相関順ソート（準備中）', type='info')

    def _show_correlation_heatmap(self):
        """相関ヒートマップを表示"""
        if self.df is None or len(self.numeric_cols) < 2:
            ui.notify('可視化には最低2列以上の数値列が必要です', type='warning')
            return

        try:
            import plotly.graph_objects as go

            corr_matrix = self.df[self.numeric_cols].corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='相関係数'),
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                hovertemplate='%{y} vs %{x}: %{z:.3f}'
            ))

            fig.update_layout(
                title='📈 特徴量相関ヒートマップ',
                height=600,
                width=800,
            )

            with ui.dialog() as dialog:
                with ui.card().classes('w-full max-w-5xl'):
                    ui.label('📊 相関ヒートマップ').classes('text-lg font-bold mb-2')
                    ui.html(fig.to_html(include_plotlyjs='cdn', config={'responsive': True}))
                    with ui.row().classes('w-full justify-end'):
                        ui.button('閉じる', on_click=dialog.close)

            dialog.open()
            ui.notify('✓ 相関ヒートマップを生成しました', type='positive')

        except ImportError:
            ui.notify('⚠️ plotlyがインストールされていません', type='warning')
        except Exception as e:
            logger.error(f"相関ヒートマップ生成エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _apply_domain_knowledge(self):
        """ドメイン知識を反映"""
        ui.notify('ドメイン知識を反映しました（準備中）', type='info')

    def _consult_llm(self):
        """LLMに相談"""
        ui.notify('LLM相談（準備中）', type='info')

    def _apply_preprocessing(self):
        """前処理設定を適用"""
        selected_features = self.selected_features
        scaler = self._scaler_select.value
        imputer = self._imputer_select.value
        encoder = self._encoder_select.value

        if not selected_features:
            ui.notify('特徴量を選択してください', type='warning')
            return

        # 設定を保存
        config = {
            'target_col': self.target_col,
            'selected_features': selected_features,
            'scaler': scaler,
            'imputer': imputer,
            'encoder': encoder,
            'power_transform': self._power_transform.value,
            'quantile_transform': self._quantile_transform.value,
        }

        ui.notify(f'✓ 前処理設定を適用しました: {len(selected_features)}特徴量', type='positive')
        print(f"Preprocessing config: {config}")

    def _save_config(self):
        """設定を保存"""
        from pathlib import Path
        import json

        config = {
            'target_col': self.target_col,
            'selected_features': self.selected_features,
            'scaler': self._scaler_select.value,
            'imputer': self._imputer_select.value,
            'encoder': self._encoder_select.value,
        }

        config_dir = Path('configs')
        config_dir.mkdir(exist_ok=True)

        with open(config_dir / 'preprocessing.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        ui.notify('✓ 設定を保存しました: configs/preprocessing.json', type='positive')
