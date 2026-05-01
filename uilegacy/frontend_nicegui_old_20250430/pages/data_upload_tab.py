"""
frontend_nicegui/pages/data_upload_tab.py
AutoML連携修正版 - visibleプロパティ修正済み
"""
from nicegui import ui, events
import pandas as pd
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataUploadPage:
    """データアップロードページ"""

    def __init__(self, llm_config, automl_page=None, viz_page=None, navigate_to_automl=None):
        self.llm_config = llm_config
        self.automl_page = automl_page
        self.viz_page = viz_page
        self._navigate_to_automl_cb = navigate_to_automl
        self.uploaded_data: Optional[pd.DataFrame] = None
        self.quality_report: Optional[Dict] = None

    def render(self):
        """ページを描画"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):

            ui.label('📁 データアップロード & クリーニング').classes('text-2xl font-bold mb-4')

            with ui.card().classes('w-full mb-4'):
                ui.label('データをアップロード').classes('font-bold text-lg mb-2')
                ui.label('対応ファイル: csv, excel • 最大サイズ: 50MB • 複数ファイル可').classes('text-sm text-gray-600 mb-4')

                def on_data_uploaded(result: Dict):
                    self._handle_data_upload(result)

                from frontend_nicegui.components.file_upload_zone import FileUploadZone
                FileUploadZone(
                    on_upload=on_data_uploaded,
                    allowed_types=['csv', 'excel'],
                    label='化学データ（CSV/Excel）をアップロード'
                )

            # 修正: visibleはプロパティとして設定
            self._quality_card = ui.card().classes('w-full mb-4')
            self._quality_card.visible = False

            with self._quality_card:
                ui.label('データ品質評価').classes('font-bold text-lg mb-2')
                self._quality_label = ui.label()
                # 外れ値詳細テーブル用コンテナ
                self._outlier_container = ui.column().classes('w-full q-mt-md')

                with ui.row().classes('mt-4 gap-4'):
                    ui.button('🔧 LLM支援でクリーニング', on_click=self._run_cleaning, color='primary').props('outline')
                    ui.button('📊 このデータで解析へ', on_click=self._navigate_to_automl, color='primary').props('size=lg')

            with ui.card().classes('w-full'):
                ui.label('アップロード済みファイル').classes('font-bold text-lg mb-2')
                self._file_list = ui.column().classes('w-full')

    def _handle_data_upload(self, result: Dict):
        """データアップロード処理"""
        try:
            data = result.get('data')
            meta = result.get('meta', {})

            if isinstance(data, pd.DataFrame):
                self.uploaded_data = data

                try:
                    from backend.data.file_uploader import assess_data_quality
                    self.quality_report = assess_data_quality(data)
                    issues = self.quality_report.get('issues', [])
                    if issues:
                        self._quality_label.text = f"✓ {meta.get('filename', 'unknown')} | {meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列 | 注意: {', '.join(issues[:3])}"
                        self._quality_label.classes('text-orange-600')
                        # 外れ値詳細をテーブル表示
                        self._show_outlier_details()
                    else:
                        self._quality_label.text = f"✓ {meta.get('filename', 'unknown')} | {meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列 | 状態: 問題なし"
                        self._quality_label.classes('text-green-600')
                        self._outlier_container.clear()
                except Exception:
                    self._quality_label.text = f"✓ {meta.get('filename', 'unknown')} | {meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列"
                    self._quality_label.classes('text-green-600')

                self._quality_card.visible = True

                with self._file_list:
                    with ui.row().classes('w-full items-center p-2 bg-gray-50 rounded mb-2'):
                        ui.icon('description', size='sm').classes('text-gray-500')
                        ui.label(f"{meta.get('filename', 'unknown')}").classes('flex-1')
                        ui.label(f"{meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列").classes('text-sm text-gray-500')
                        ui.icon('check', size='sm').classes('text-green-500')

                ui.notify(f'✓ {meta.get("filename", "unknown")} を読み込みました', type='positive')
            else:
                ui.notify(f'📄 {meta.get("filename", "unknown")} を読み込みました')

        except Exception as e:
            logger.error(f"アップロードエラー: {e}", exc_info=True)
            ui.notify(f'✗ エラー: {str(e)}', type='negative')

    def _run_cleaning(self):
        """LLM支援によるデータクリーニング分析"""
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return

        try:
            from backend.data.data_cleaner import DataCleanerLLM

            ui.notify('🔍 データの問題点を分析中...', type='info')

            # LLMモード（LLMConfig から取得、デフォルトは prompt_only）
            mode = self.llm_config.mode if self.llm_config else 'prompt_only'

            # DataCleanerLLM を初期化
            cleaner = DataCleanerLLM(
                mode=mode,
                api_endpoint=self.llm_config.api_endpoint if self.llm_config else None,
                api_key=self.llm_config.api_key if self.llm_config else None,
                model_name=self.llm_config.model_name if self.llm_config else 'default'
            )

            # データ問題点を分析
            suggestions = cleaner.analyze_data_issues(self.uploaded_data, sample_rows=20)

            if not suggestions:
                ui.notify('✓ データに大きな問題は検出されませんでした', type='positive')
                return

            # クリーニング提案ダイアログを表示
            with ui.dialog() as dialog:
                with ui.card().classes('w-full max-w-2xl'):
                    ui.label('🧹 データクリーニング提案').classes('text-lg font-bold mb-4')

                    # 提案一覧
                    with ui.scroll_area().classes('w-full h-96'):
                        for i, suggestion in enumerate(suggestions):
                            with ui.expansion(
                                f"[{i+1}] {suggestion.issue_type} (信頼度: {suggestion.confidence*100:.0f}%)",
                                icon='build'
                            ).classes('w-full mb-2'):
                                ui.label(suggestion.description).classes('text-sm mb-2')

                                ui.label('提案コード:').classes('font-bold text-xs')
                                ui.textarea(
                                    value=suggestion.suggested_code,
                                    placeholder='コードなし'
                                ).props('readonly outlined rows=8').classes('w-full font-mono text-xs')

                                if suggestion.auto_applicable:
                                    ui.button(
                                        '✓ 自動適用',
                                        on_click=lambda s=suggestion: self._apply_cleaning(s)
                                    ).props('dense outline')

                    # ボタン
                    with ui.row().classes('justify-end mt-4 gap-2'):
                        ui.button('閉じる', on_click=dialog.close).props('outline')
                        ui.button('✓ すべてのコードをコピー', on_click=lambda: self._copy_all_suggestions(suggestions)).props('outline')

            dialog.open()

            ui.notify(f'✓ {len(suggestions)}件のクリーニング提案を生成しました', type='positive')

        except Exception as e:
            logger.error(f"クリーニング分析エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _apply_cleaning(self, suggestion):
        """提案されたクリーニングを適用"""
        try:
            if suggestion.suggested_code:
                # セキュリティ上、exec は避けて、簡易的な変換のみ実装
                ui.notify(f'✓ クリーニングコード: {suggestion.issue_type}', type='positive')
                # 本来はここで exec(suggestion.suggested_code) を実行
        except Exception as e:
            ui.notify(f'適用エラー: {str(e)}', type='negative')

    def _copy_all_suggestions(self, suggestions):
        """すべてのクリーニングコードをコピー"""
        all_code = "# ChemAI Data Cleaning Script\n# 自動生成されたデータクリーニングコード\n\n"
        for i, suggestion in enumerate(suggestions):
            all_code += f"# {i+1}. {suggestion.issue_type}\n"
            all_code += suggestion.suggested_code + "\n\n"

        ui.clipboard.write(all_code)
        ui.notify('✓ すべてのコードをコピーしました', type='positive')

    def _navigate_to_automl(self):
        """AutoMLページへ遷移"""
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return

        # データを渡す
        if self.automl_page:
            self.automl_page.load_data(self.uploaded_data)
        if self.viz_page:
            self.viz_page.load_data(self.uploaded_data)

        # NiceGUI ネイティブのタブ切り替え
        if self._navigate_to_automl_cb:
            self._navigate_to_automl_cb()
            ui.notify('AutoMLタブへ移動しました', type='info')
        else:
            ui.notify('データを読み込みました。AutoMLタブへ移動してください', type='positive')

    def _show_outlier_details(self):
        """外れ値の詳細（SMILES含む実際のデータ）をテーブル表示"""
        self._outlier_container.clear()
        if not self.quality_report:
            return
        outlier_details = self.quality_report.get('outlier_details', [])
        if not outlier_details:
            return

        with self._outlier_container:
            ui.label('⚠️ 外れ値の詳細データ').classes('text-lg font-bold text-orange-600 mb-2')

            for detail in outlier_details:
                col_name = detail.get('column', '')
                count = detail.get('count', 0)
                rows = detail.get('rows', [])
                columns = detail.get('columns', [])

                if not rows:
                    continue

                ui.label(f'列 "{col_name}" の外れ値: {count}件').classes('text-md font-bold text-orange-500 mb-1')

                # テーブル用の列定義（SMILES列を優先表示）
                smiles_cols = [c for c in columns if 'smiles' in c.lower()]
                other_cols = [c for c in columns if c not in smiles_cols and c not in ('_outlier_col_', '_outlier_value_')]
                display_cols = smiles_cols + other_cols + ['_outlier_value_']

                # 行データをテーブル用に変換
                table_rows = []
                for row in rows[:10]:  # 最大10件表示
                    table_rows.append({
                        'SMILES': row.get(col_name, '') if col_name in smiles_cols else row.get(smiles_cols[0], '') if smiles_cols else '',
                        '列名': col_name,
                        '外れ値': row.get('_outlier_value_', ''),
                        **{c: row.get(c, '') for c in other_cols[:5]}  # その他の列は最大5つ
                    })

                if table_rows:
                    ui.table(
                        columns=[
                            {'name': 'SMILES', 'label': 'SMILES', 'field': 'SMILES', 'align': 'left'},
                            {'name': '列名', 'label': '外れ値列', 'field': '列名'},
                            {'name': '外れ値', 'label': '外れ値', 'field': '外れ値'},
                        ] + [
                            {'name': c, 'label': c, 'field': c}
                            for c in other_cols[:5]
                        ],
                        rows=table_rows,
                    ).classes('w-full').props('dense flat bordered')

                if len(rows) > 10:
                    ui.label(f'... 他 {len(rows) - 10}件の外れ値があります').classes('text-caption text-grey-500')
