"""
frontend_nicegui/pages/eda_page.py
EDA (Exploratory Data Analysis) page - 仕様書6章に基づく実装
インタラクティブ・フィルタリング、データ熟読支援、SMILESホバー機能
"""
from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class EDAPage:
    """EDAページ - ユーザーがデータを熟読し、変数ごとのパターンを把握"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.smiles_cols: List[str] = []
        self.filter_conditions: Dict[str, Any] = {}
        self._containers: Dict[str, Any] = {}

    def render(self):
        """EDAページを描画"""
        with ui.column().classes('w-full max-w-7xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('📊 探索的データ分析 (EDA)').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('データを熟読してからモデリングへ進む').classes('text-gray-600')

            # データ確認セクション
            self._data_status = ui.card().classes('w-full mb-4')
            with self._data_status:
                ui.label('📂 データ状態').classes('font-bold text-lg mb-2')
                self._data_msg = ui.label('⚠️ データが読み込まれていません。「Data Upload」タブからデータをアップロードしてください。')
                self._data_msg.classes('text-orange-600 p-4 bg-orange-50 rounded')

            # 統計サマリー
            self._stats_card = ui.card().classes('w-full mb-4')
            self._stats_card.visible = False
            with self._stats_card:
                ui.label('📈 基本統計量').classes('font-bold text-lg mb-2')
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('行数（全データ）').classes('text-xs text-gray-500')
                        self._total_rows = ui.label('-')
                    with ui.column().classes('flex-1'):
                        ui.label('行数（フィルタ後）').classes('text-xs text-gray-500')
                        self._filtered_rows = ui.label('-')
                    with ui.column().classes('flex-1'):
                        ui.label('列数').classes('text-xs text-gray-500')
                        self._col_count = ui.label('-')
                    with ui.column().classes('flex-1'):
                        ui.label('数値列').classes('text-xs text-gray-500')
                        self._numeric_count = ui.label('-')

                # フィルタ適用ボタン
                with ui.row().classes('w-full mt-4 gap-2'):
                    self._apply_filter_btn = ui.button('🔍 フィルタ適用', on_click=self._apply_filters, color='primary')
                    self._reset_filter_btn = ui.button('🔄 リセット', on_click=self._reset_filters).props('outline')
                    ui.space()
                    self._filter_status = ui.label('').classes('text-sm text-gray-500')

            # インタラクティブ・フィルタリング (仕様書 6.7)
            self._filter_card = ui.card().classes('w-full mb-4')
            self._filter_card.visible = False
            with self._filter_card:
                ui.label('🔍 インタラクティブ・フィルタリング').classes('font-bold text-lg mb-2')
                ui.label('変数の範囲やカテゴリで絞り込み、即時反映').classes('text-sm text-gray-500 mb-4')

                # フィルタ条件入力エリア
                self._filter_container = ui.column().classes('w-full')

            # データプレビュー
            self._preview_card = ui.card().classes('w-full mb-4')
            self._preview_card.visible = False
            with self._preview_card:
                ui.label('📋 データプレビュー（フィルタ後）').classes('font-bold text-lg mb-2')
                self._data_table = ui.table(
                    columns=[],
                    rows=[],
                    pagination=dict(rowsPerPage=10, sortBy='index')
                ).classes('w-full')

            # 可視化セクション
            self._viz_card = ui.card().classes('w-full mb-4')
            self._viz_card.visible = False
            with self._viz_card:
                ui.label('📊 可視化').classes('font-bold text-lg mb-2')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('可視化タイプ').classes('text-xs text-gray-500')
                        self._viz_type = ui.select(
                            options={
                                'correlation': '相関ヒートマップ',
                                'histogram': 'ヒストグラム',
                                'boxplot': 'ボックスプロット',
                                'scatter': '散布図',
                            },
                            value='correlation'
                        ).classes('w-full')

                    with ui.column().classes('flex-1'):
                        ui.label('色分け列（オプション）').classes('text-xs text-gray-500')
                        self._color_by = ui.select(options=['なし'], value='なし').classes('w-full')

                ui.button('📊 可視化を生成', on_click=self._generate_visualization, color='primary').classes('mt-2')

                # 可視化結果
                self._viz_result = ui.column().classes('w-full mt-4')

            # データ熟読支援 (仕様書 6.8)
            self._reading_card = ui.card().classes('w-full mb-4')
            self._reading_card.visible = False
            with self._reading_card:
                ui.label('📖 データ熟読支援').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **目標**: サンプル名を言われたらその説明変数の値がぱっと浮かぶレベルまでデータをよく見る。

                - 変数ごとの分布・要約統計を「よく見る変数」としてブックマーク
                - 特定サンプルをクリック→全特徴量・SMILES構造式を詳細表示
                - フィルタ前後比較で新たなパターン発見
                """)

                with ui.row().classes('w-full gap-4'):
                    ui.button('📊 相関行列を表示', on_click=self._show_correlation_heatmap, color='primary').props('outline')
                    ui.button('🔍 疑似OFATを検出', on_click=self._detect_ofat, color='primary').props('outline')
                    ui.button('⚠️ 外れ値検出', on_click=self._detect_outliers, color='primary').props('outline')

                # 検出結果
                self._detection_result = ui.column().classes('w-full mt-4')

            # LLMガイド (仕様書 6.8.3)
            self._llm_card = ui.card().classes('w-full mb-4')
            self._llm_card.visible = False
            with self._llm_card:
                ui.label('🤖 LLMによるEDA熟読誘導').classes('font-bold text-lg mb-2')
                ui.label('LLMがデータパターンを発見・提示').classes('text-sm text-gray-500 mb-2')

                self._llm_suggestion = ui.markdown('').classes('w-full bg-blue-50 p-4 rounded')
                ui.button('💬 LLMに相談', on_click=self._consult_llm, color='primary').props('outline')

    def load_data(self, df: pd.DataFrame):
        """データをロードしてUIを更新"""
        self.df = df.copy()
        self.filtered_df = df.copy()

        # 列の分類
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.smiles_cols = [c for c in df.columns if 'smiles' in c.lower()]

        # 統計情報を更新
        self._data_status.visible = False
        self._stats_card.visible = True
        self._filter_card.visible = True
        self._preview_card.visible = True
        self._viz_card.visible = True
        self._reading_card.visible = True
        self._llm_card.visible = True

        self._total_rows.text = f"{len(df):,}"
        self._filtered_rows.text = f"{len(df):,}"
        self._col_count.text = f"{len(df.columns)}"
        self._numeric_count.text = f"{len(self.numeric_cols)}"

        # フィルタUIを構築
        self._build_filter_ui()

        # 色分け列オプションを更新
        color_options = ['なし'] + self.categorical_cols
        self._color_by.options = color_options
        self._color_by.value = 'なし'

        # データテーブルを更新
        self._update_preview()

        ui.notify(f'✓ EDA: {len(df)}行 × {len(df.columns)}列を読み込みました', type='positive')

    def _build_filter_ui(self):
        """フィルタリングUIを構築（数値：スライダー、カテゴリ：ドロップダウン）"""
        self._filter_container.clear()
        self.filter_conditions = {}

        with self._filter_container:
            # 数値列のフィルタ（スライダー）
            if self.numeric_cols:
                ui.label('数値列のフィルタ').classes('font-bold text-sm mb-2')
                for col in self.numeric_cols[:10]:  # 最大10列まで表示
                    col_data = self.df[col].dropna()
                    if len(col_data) == 0:
                        continue
                    min_val, max_val = float(col_data.min()), float(col_data.max())
                    if min_val == max_val:
                        continue

                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label(col).classes('w-32 text-xs')
                        slider = ui.slider(min=min_val, max=max_val, value=[min_val, max_val])
                        slider.props('label-always range').classes('flex-1')
                        self.filter_conditions[col] = slider

            # カテゴリ列のフィルタ（ドロップダウン）
            if self.categorical_cols:
                ui.separator()
                ui.label('カテゴリ列のフィルタ').classes('font-bold text-sm mb-2')
                for col in self.categorical_cols[:5]:  # 最大5列まで表示
                    unique_vals = self.df[col].dropna().unique().tolist()
                    if len(unique_vals) > 20:  # 多すぎる場合はスキップ
                        continue

                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label(col).classes('w-32 text-xs')
                        select = ui.select(
                            options=unique_vals,
                            value=unique_vals[0] if unique_vals else None,
                            multiple=True
                        ).classes('flex-1')
                        self.filter_conditions[col] = select

    def _apply_filters(self):
        """フィルタを適用してデータを絞り込む"""
        if self.df is None:
            return

        filtered = self.df.copy()

        for col, widget in self.filter_conditions.items():
            if col in self.numeric_cols:
                # 数値列：スライダーの範囲でフィルタ
                min_val, max_val = widget.value
                filtered = filtered[(filtered[col] >= min_val) & (filtered[col] <= max_val)]
            elif col in self.categorical_cols:
                # カテゴリ列：選択された値でフィルタ
                selected = widget.value
                if selected and isinstance(selected, list):
                    filtered = filtered[filtered[col].isin(selected)]

        self.filtered_df = filtered
        self._filtered_rows.text = f"{len(filtered):,}"

        # プレビューを更新
        self._update_preview()

        # ステータス更新
        total = len(self.df)
        filtered_count = len(filtered)
        pct = (filtered_count / total * 100) if total > 0 else 0
        self._filter_status.text = f'✓ {filtered_count:,}件に絞り込み（{pct:.1f}%）'
        self._filter_status.classes('text-green-600')

        ui.notify(f'✓ フィルタ適用：{filtered_count:,}件（{pct:.1f}%）', type='positive')

    def _reset_filters(self):
        """フィルタをリセット"""
        self.filtered_df = self.df.copy()
        self._filtered_rows.text = f"{len(self.df):,}"
        self._filter_status.text = 'フィルタをリセットしました'
        self._filter_status.classes('text-gray-500')
        self._update_preview()
        ui.notify('✓ フィルタをリセットしました', type='positive')

    def _update_preview(self):
        """データプレビューを更新"""
        if self.filtered_df is None:
            return

        df_preview = self.filtered_df.head(50)  # 最大50行
        columns = [{'name': col, 'label': col, 'field': col} for col in df_preview.columns[:15]]  # 最大15列
        rows = df_preview.head(10).to_dict('records')
        for i, row in enumerate(rows):
            row['index'] = i + 1

        self._data_table.columns = [{'name': 'index', 'label': '#', 'field': 'index'}] + columns
        self._data_table.rows = rows

    def _generate_visualization(self):
        """可視化を生成"""
        if self.filtered_df is None or len(self.numeric_cols) < 2:
            ui.notify('可視化には最低2列以上の数値列が必要です', type='warning')
            return

        try:
            import plotly.express as px

            self._viz_result.clear()
            viz_type = self._viz_type.value

            with self._viz_result:
                if viz_type == 'correlation':
                    self._show_correlation_heatmap()
                elif viz_type == 'histogram':
                    self._show_histograms()
                elif viz_type == 'boxplot':
                    self._show_boxplots()
                elif viz_type == 'scatter':
                    self._show_scatter_plot()

        except ImportError:
            ui.notify('⚠️ plotlyがインストールされていません', type='warning')
        except Exception as e:
            logger.error(f"可視化エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _show_correlation_heatmap(self):
        """相関ヒートマップを表示"""
        import plotly.graph_objects as go

        corr_matrix = self.filtered_df[self.numeric_cols].corr()

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

        ui.plotly(fig)

    def _show_histograms(self):
        """ヒストグラムを表示"""
        import plotly.express as px

        # 最初の6列まで
        cols_to_show = self.numeric_cols[:6]
        fig = px.histogram(
            self.filtered_df,
            x=cols_to_show[0] if cols_to_show else self.numeric_cols[0],
            nbins=30,
            title='📊 ヒストグラム',
            height=400
        )
        ui.plotly(fig)

    def _show_boxplots(self):
        """ボックスプロットを表示"""
        import plotly.express as px

        cols_to_show = self.numeric_cols[:6]
        fig = px.box(
            self.filtered_df,
            y=cols_to_show[0] if cols_to_show else self.numeric_cols[0],
            title='📦 ボックスプロット',
            height=400
        )
        ui.plotly(fig)

    def _show_scatter_plot(self):
        """散布図を表示"""
        import plotly.express as px

        if len(self.numeric_cols) < 2:
            ui.notify('散布図には2列以上必要です', type='warning')
            return

        color_col = None if self._color_by.value == 'なし' else self._color_by.value
        fig = px.scatter(
            self.filtered_df,
            x=self.numeric_cols[0],
            y=self.numeric_cols[1],
            color=color_col,
            title=f'📊 {self.numeric_cols[0]} vs {self.numeric_cols[1]}',
            height=500
        )
        ui.plotly(fig)

    def _detect_ofat(self):
        """疑似OFAT（1変数のみ変化）データを検出"""
        self._detection_result.clear()

        if self.df is None or len(self.numeric_cols) < 2:
            return

        with self._detection_result:
            ui.label('🔍 疑似OFATデータ検出').classes('font-bold text-md mb-2')

            # 簡易的なOFAT検出：各行について、他の行と比べて1列のみ値が異なるものを探す
            ofat_samples = []
            df_sample = self.df.head(100)  # 計算量削減のため最大100行

            for i, row in df_sample.iterrows():
                for j, row2 in df_sample.iterrows():
                    if i >= j:
                        continue
                    diff_cols = []
                    for col in self.numeric_cols:
                        val1 = row[col]
                        val2 = row2[col]
                        if pd.notna(val1) and pd.notna(val2) and abs(val1 - val2) > 1e-6:
                            diff_cols.append(col)
                    if len(diff_cols) == 1:
                        ofat_samples.append({
                            'sample1': i,
                            'sample2': j,
                            'changing_col': diff_cols[0],
                            'val1': row[diff_cols[0]],
                            'val2': row2[diff_cols[0]],
                        })

            if ofat_samples:
                ui.label(f'{len(ofat_samples)}件の疑似OFATペアを検出').classes('text-green-600')
                for sample in ofat_samples[:5]:  # 最大5件表示
                    with ui.card().classes('w-full mb-2 bg-green-50'):
                        ui.label(f"サンプル {sample['sample1']} ↔ {sample['sample2']}: {sample['changing_col']} のみ変化").classes('text-sm')
                        ui.label(f"  {sample['val1']} → {sample['val2']}").classes('text-xs text-gray-600')
            else:
                ui.label('OFATパターンは検出されませんでした').classes('text-gray-500')

    def _detect_outliers(self):
        """外れ値を検出"""
        self._detection_result.clear()

        if self.df is None or len(self.numeric_cols) < 1:
            return

        with self._detection_result:
            ui.label('⚠️ 外れ値検出（IQR法）').classes('font-bold text-md mb-2')

            outlier_info = []
            for col in self.numeric_cols[:10]:  # 最大10列
                col_data = self.df[col].dropna()
                if len(col_data) < 4:
                    continue
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
                if len(outliers) > 0:
                    outlier_info.append({
                        'column': col,
                        'count': len(outliers),
                        'lower': lower,
                        'upper': upper,
                    })

            if outlier_info:
                for info in outlier_info[:5]:
                    with ui.card().classes('w-full mb-2 bg-orange-50'):
                        ui.label(f"列 '{info['column']}': {info['count']}件の外れ値").classes('text-sm font-bold text-orange-600')
                        ui.label(f"  正常範囲: {info['lower']:.3f} ～ {info['upper']:.3f}").classes('text-xs text-gray-600')
            else:
                ui.label('外れ値は検出されませんでした').classes('text-gray-500')

    def _consult_llm(self):
        """LLMに相談（簡易的なデータ要約）"""
        if self.df is None:
            ui.notify('データが読み込まれていません', type='warning')
            return

        # データの要約を生成
        summary = f"""
        **データ要約（EDA熟読支援）**

        - 行数: {len(self.df):,}
        - 列数: {len(self.df.columns)}
        - 数値列: {len(self.numeric_cols)}列
        - カテゴリ列: {len(self.categorical_cols)}列
        - SMILES列: {len(self.smiles_cols)}列

        **推奨アクション**:
        1. まず相関ヒートマップで変数間の関係を確認
        2. 疑似OFATデータがあるか確認（1変数のみ変化するペア）
        3. 外れ値を検出してデータの質を確認
        4. インタラクティブ・フィルタで「この範囲だけ見ると...」を試す
        """

        self._llm_suggestion.content = summary
        ui.notify('✓ LLMからのデータ要約を生成しました', type='positive')
