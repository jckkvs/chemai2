"""
frontend_nicegui/pages/visualization_tab.py
データ可視化ページ - PCA, t-SNE, UMAP, 相関ヒートマップ
"""
from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VisualizationPage:
    """データ可視化ページ"""

    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.numeric_cols: list = []

    def render(self):
        """可視化ページを描画"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('📊 データ可視化').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('多角的なデータ解析による洞察').classes('text-gray-600')

            # データ確認セクション
            with ui.card().classes('w-full mb-4'):
                ui.label('📈 読み込み済みデータ').classes('font-bold text-lg mb-2')

                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('行数').classes('text-xs text-gray-500')
                        self._row_count = ui.label('-')
                    with ui.column():
                        ui.label('列数').classes('text-xs text-gray-500')
                        self._col_count = ui.label('-')
                    with ui.column():
                        ui.label('数値列').classes('text-xs text-gray-500')
                        self._numeric_count = ui.label('-')

                self._no_data_msg = ui.label('⚠️ データが読み込まれていません。「Data Upload」タブからデータをアップロードしてください。')
                self._no_data_msg.classes('text-orange-600 p-4 bg-orange-50 rounded mt-4')

            # 可視化タイプ選択
            self._viz_card = ui.card().classes('w-full mb-4')
            self._viz_card.visible = False

            with self._viz_card:
                ui.label('🎨 可視化タイプを選択').classes('font-bold text-lg mb-2')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('次元削減手法').classes('text-xs text-gray-500')
                        self._method_select = ui.radio(
                            options={
                                'pca': 'PCA (高速)',
                                'tsne': 't-SNE (非線形)',
                                'umap': 'UMAP (バランス型)',
                            },
                            value='pca'
                        ).props('dense')

                    with ui.column().classes('flex-1'):
                        ui.label('色分け列（オプション）').classes('text-xs text-gray-500')
                        self._color_col = ui.select(options=['なし'], value='なし').classes('w-full')

                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('📊 相関ヒートマップを表示', on_click=self._show_correlation).props('outline')
                    ui.button('🔄 次元削減プロットを生成', on_click=self._generate_reduction, color='primary')

            # 可視化結果エリア
            self._plot_card = ui.card().classes('w-full mt-4')
            self._plot_card.visible = False

            with self._plot_card:
                with ui.row().classes('w-full items-center mb-4'):
                    ui.label('✅ 可視化結果').classes('font-bold text-lg text-green-600')
                    ui.space()
                    self._plot_type_label = ui.label('')

                self._plot_container = ui.column().classes('w-full')

            # 進捗インジケータ
            self._progress_card = ui.card().classes('w-full mt-4')
            self._progress_card.visible = False

            with self._progress_card:
                ui.label('⏳ 処理中...').classes('font-bold')
                self._progress = ui.linear_progress(value=0, show_value=True)

    def load_data(self, data: pd.DataFrame):
        """データをロードしてUIを更新"""
        self.current_data = data
        self.numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # 統計情報を更新
        self._row_count.text = f"{len(data):,}"
        self._col_count.text = f"{len(data.columns)}"
        self._numeric_count.text = f"{len(self.numeric_cols)}"

        if len(self.numeric_cols) < 2:
            ui.notify('⚠️ 可視化には最低2列以上の数値列が必要です', type='warning')
            self._no_data_msg.visible = True
            self._viz_card.visible = False
            return

        # UI表示
        self._no_data_msg.visible = False
        self._viz_card.visible = True

        # 色分け列オプションを更新
        color_options = ['なし'] + categorical_cols
        self._color_col.options = color_options
        self._color_col.value = 'なし'

    def _show_correlation(self):
        """相関ヒートマップを表示"""
        if self.current_data is None or len(self.numeric_cols) < 2:
            ui.notify('数値列が不足しています', type='warning')
            return

        self._progress_card.visible = True
        self._progress.value = 30

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            # 相関行列を計算
            corr_matrix = self.current_data[self.numeric_cols].corr()
            self._progress.value = 60

            # ヒートマップを作成
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='相関係数'),
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}: %{z:.3f}'
            ))

            fig.update_layout(
                title='📈 特徴量相関ヒートマップ',
                height=600,
                width=800,
                xaxis_title='',
                yaxis_title='',
            )

            self._progress.value = 90

            # 結果を表示
            self._plot_container.clear()
            with self._plot_container:
                ui.html(fig.to_html(include_plotlyjs='cdn', config={'responsive': True}))

            self._plot_type_label.text = '相関ヒートマップ'
            self._plot_card.visible = True

            ui.notify('相関ヒートマップを生成しました', type='positive')
            self._progress.value = 100

        except Exception as e:
            logger.error(f"相関ヒートマップ生成エラー: {e}")
            ui.notify(f'エラー: {str(e)}', type='negative')
        finally:
            self._progress_card.visible = False

    def _generate_reduction(self):
        """次元削減プロットを生成"""
        if self.current_data is None or len(self.numeric_cols) < 2:
            ui.notify('数値列が不足しています', type='warning')
            return

        if len(self.numeric_cols) < 3 and self._method_select.value in ['tsne', 'umap']:
            ui.notify(f'⚠️ {self._method_select.value.upper()}には最低3列以上の特徴量が必要です', type='warning')
            return

        self._progress_card.visible = True
        self._progress.value = 20

        try:
            import plotly.express as px
            from sklearn.preprocessing import StandardScaler

            method = self._method_select.value
            X = self.current_data[self.numeric_cols].fillna(0)

            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self._progress.value = 40

            # 次元削減を実行
            if method == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X_scaled)
                explained_var = f"説明分散: {reducer.explained_variance_ratio_.sum():.1%}"
                title = f'📊 PCA投影 ({explained_var})'

            elif method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=min(30, len(X)//5))
                X_reduced = reducer.fit_transform(X_scaled)
                title = '📊 t-SNE投影'

            elif method == 'umap':
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    X_reduced = reducer.fit_transform(X_scaled)
                    title = '📊 UMAP投影'
                except ImportError:
                    ui.notify('⚠️ UMAPがインストールされていません。PCAを使用します', type='warning')
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2, random_state=42)
                    X_reduced = reducer.fit_transform(X_scaled)
                    title = '📊 PCA投影 (UMAP未インストール)'

            self._progress.value = 70

            # プロットを作成
            plot_df = pd.DataFrame({
                'x': X_reduced[:, 0],
                'y': X_reduced[:, 1],
            })

            # 色分け列を追加
            color_col = self._color_col.value
            if color_col != 'なし':
                plot_df['color'] = self.current_data[color_col].astype(str)
                fig = px.scatter(
                    plot_df,
                    x='x',
                    y='y',
                    color='color',
                    hover_name=plot_df.index,
                    title=title,
                    labels={'x': 'PC1/TSNE1/UMAP1', 'y': 'PC2/TSNE2/UMAP2'},
                    height=600,
                    width=800
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x='x',
                    y='y',
                    hover_name=plot_df.index,
                    title=title,
                    labels={'x': 'PC1/TSNE1/UMAP1', 'y': 'PC2/TSNE2/UMAP2'},
                    height=600,
                    width=800
                )

            fig.update_layout(
                hovermode='closest',
                plot_bgcolor='rgba(240,240,240,0.5)',
            )

            self._progress.value = 90

            # 結果を表示
            self._plot_container.clear()
            with self._plot_container:
                ui.html(fig.to_html(include_plotlyjs='cdn', config={'responsive': True}))

            self._plot_type_label.text = f'{method.upper()}投影'
            self._plot_card.visible = True

            ui.notify(f'{method.upper()}投影を生成しました', type='positive')
            self._progress.value = 100

        except Exception as e:
            logger.error(f"次元削減エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')
        finally:
            self._progress_card.visible = False
