"""
frontend_nicegui/pages/settings_page.py
Settings page - LLM設定、ドメイン知識管理、UMA設定
仕様書4.1.1, 9.6, 9.8に基づく
"""
from nicegui import ui
from typing import Optional, Dict, List, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SettingsPage:
    """Settings page - LLM設定、ドメイン知識管理"""

    def __init__(self):
        self.llm_config = None
        self.domain_knowledge: Dict = {}
        self.uma_weight_path: Optional[str] = None

    def render(self):
        """Settingsページを描画"""
        with ui.column().classes('w-full max-w-7xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('⚙️ Settings').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('LLM設定、ドメイン知識管理、UMA設定').classes('text-gray-600')

            # LLM設定 (仕様書9.6)
            with ui.card().classes('w-full mb-4'):
                ui.label('💬 LLM設定 (仕様書9.6)').classes('text-xl font-bold mb-2')

                ui.markdown("""
                **LLMプロバイダー**:
                - ローカルLLM: 量子化モデル（GGUF等）、BONSAI
                - 外部LLM API: OpenAI, Anthropic, Google等
                - モデル選択支援: canirun.ai参照によるハードウェア適合性チェック
                """).classes('text-sm text-gray-500 mb-4')

                # LLM設定ダイアログを開くボタン
                try:
                    from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog
                    dialog = LLMConfigDialog()
                    dialog.create_trigger_button(
                        ui.row(),
                        icon='settings',
                        label='LLM設定を開く'
                    ).props('color=primary')
                except Exception as e:
                    ui.label(f'LLM設定ダイアログの読み込みエラー: {e}').classes('text-red-500')
                    ui.label('LLM設定機能 - 準備中').classes('text-gray-500')

                # モデル選択支援
                with ui.row().classes('w-full gap-2 mt-4'):
                    ui.button('🔍 canirun.aiでモデル適合性チェック', color='primary', size='sm').props('outline')
                    ui.button('📥 量子化モデルをダウンロード', color='primary', size='sm').props('outline')

            # ドメイン知識管理 (仕様書9.8)
            with ui.card().classes('w-full mb-4'):
                ui.label('🧠 ドメイン知識管理 (仕様書9.8)').classes('text-xl font-bold mb-2')

                ui.markdown("""
                **ドメイン知識の種類** (仕様書3.4.2):
                1. **変数の性質に関する知識** - 制御可能変数 vs 成り行き変数
                2. **サンプルに関する知識** - 信頼性の高い/低いサンプル
                3. **目的変数に関する知識** - 物性の物理化学的挙動
                4. **系全体に関する知識** - 材料の種類、既知の制約
                """).classes('text-sm text-gray-500 mb-4')

                # 知識入力方式
                with ui.row().classes('w-full gap-4 mb-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('知識入力タイミング').classes('text-xs text-gray-500')
                        self._knowledge_source = ui.select(
                            options={
                                'interview': 'LLM Interview時（対話形式）',
                                'data_upload': 'Data Upload時（構造化フォーム）',
                                'feature_selection': '特徴量選択時（優先度設定）',
                                'constraints': '制約設定時（ドメイン由来）',
                                'eda': 'EDA時（観察メモ・アノテーション）',
                                'inverse': '逆解析時（探索空間の制約）',
                            },
                            value='interview'
                        ).classes('w-full')

                # 現在のドメイン知識一覧
                ui.label('📋 現在のドメイン知識').classes('font-bold text-lg mb-2')

                self._knowledge_container = ui.column().classes('w-full')

                with ui.row().classes('w-full gap-2'):
                    ui.button('➕ 新しい知識を追加', on_click=self._add_knowledge, color='primary', size='sm')
                    ui.button('📋 知識をJSON出力', on_click=self._export_knowledge, size='sm').props('outline')
                    ui.button('📂 JSONから読み込み', on_click=self._import_knowledge, size='sm').props('outline')

            # UMA設定 (仕様書4.1.1)
            with ui.card().classes('w-full mb-4'):
                ui.label('🤖 UMA設定 (仕様書4.1.1)').classes('text-xl font-bold mb-2')

                ui.markdown("""
                **UMA (Universal Molecular Architecture)**:
                - 汎用分子表現モデル
                - 超軽量モデル: CPUのみでも高速動作（例: uma-sm, uma-xs等）
                - 重量モデル: GPU推奨、高い表現力（例: uma-large等）
                - 重み（ウェイト）の指定方式: ユーザーが任意のフォルダに重みファイルを配置
                """).classes('text-sm text-gray-500 mb-4')

                # UMA重みパス設定
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('UMA重みパス設定').classes('text-xs text-gray-500')
                        self._uma_path = ui.input(
                            label='重みファイルのフォルダパス',
                            placeholder='/path/to/uma/weights'
                        ).classes('w-full')

                    with ui.column().classes('w-32'):
                        ui.button('📂 フォルダ選択', on_click=self._select_uma_folder, size='sm').classes('mt-6')

                # リソースチェック
                with ui.row().classes('w-full gap-4 mt-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('リソースチェック').classes('text-xs text-gray-500')
                        ui.label('CPUのみ・メモリ8GB以下 → 超軽量モデルのみ有効化').classes('text-xs text-gray-500')
                        ui.label('GPUなし・重量モデル指定 → 自動スキップ・警告').classes('text-xs text-gray-500')

                    with ui.column().classes('w-32'):
                        ui.button('🔍 リソースをチェック', on_click=self._check_resources, color='primary', size='sm')

                # システム判定表示
                self._resource_status = ui.label('').classes('text-sm mt-2')

            # 特徴量エンジン設定
            with ui.card().classes('w-full mb-4'):
                ui.label('🔬 特徴量エンジン設定').classes('text-xl font-bold mb-2')

                ui.markdown("""
                **特徴量計算戦略** (仕様書4.2):
                - 事前全計算が前提: ユーザーが「再計算」ボタンを押す仕組みは不要
                - 軽量特徴量の先行計算: RDKit等はデータ読み込み時に即座に計算
                - 重い計算は裏で: xTBやCOSMO-RS等は別スレッドでバックグラウンド実行
                - 計算状況の可視化: プログレスバーとともに計算済み/計算中/未計算を明示
                """).classes('text-sm text-gray-500 mb-4')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('RDKit (2D記述子)').classes('text-sm')
                        ui.switch('有効', value=True).classes('ml-2')
                    with ui.column().classes('flex-1'):
                        ui.label('xTB (GFN2)').classes('text-sm')
                        ui.switch('有効', value=True).classes('ml-2')
                    with ui.column().classes('flex-1'):
                        ui.label('COSMO-RS').classes('text-sm')
                        ui.switch('有効', value=True).classes('ml-2')
                    with ui.column().classes('flex-1'):
                        ui.label('UMA').classes('text-sm')
                        ui.switch('有効（中優先度）', value=False).classes('ml-2')

            # 保存・読み込み
            with ui.card().classes('w-full mb-4'):
                ui.label('💾 設定の保存・読み込み').classes('text-xl font-bold mb-2')

                with ui.row().classes('w-full gap-2'):
                    ui.button('💾 設定を保存', on_click=self._save_settings, color='primary')
                    ui.button('📂 設定を読み込み', on_click=self._load_settings).props('outline')
                    ui.button('🔄 デフォルトに戻す', on_click=self._reset_settings).props('outline')

    def _add_knowledge(self):
        """新しいドメイン知識を追加"""
        with ui.dialog() as dialog:
            with ui.card().classes('w-full max-w-2xl'):
                ui.label('➕ 新しいドメイン知識を追加').classes('text-lg font-bold mb-4')

                # 知識タイプ
                knowledge_type = ui.select(
                    options={
                        'variable': '変数の性質に関する知識',
                        'sample': 'サンプルに関する知識',
                        'target': '目的変数に関する知識',
                        'system': '系全体に関する知識',
                    },
                    label='知識タイプ',
                    value='variable'
                ).classes('w-full mb-4')

                # 知識内容
                content = ui.textarea(
                    label='知識内容',
                    placeholder='例: この系では温度が上がると屈折率は必ず下がる'
                ).classes('w-full mb-4').props('rows=4')

                # 関連変数
                variable = ui.input(
                    label='関連変数（オプション）',
                    placeholder='例: 温度, 屈折率'
                ).classes('w-full mb-4')

                with ui.row().classes('w-full justify-end'):
                    ui.button('キャンセル', on_click=dialog.close).props('flat')
                    ui.button('追加', on_click=lambda: self._save_knowledge(
                        knowledge_type.value,
                        content.value,
                        variable.value,
                        dialog
                    ), color='primary')

        dialog.open()

    def _save_knowledge(self, knowledge_type, content, variable, dialog):
        """ドメイン知識を保存"""
        if not content:
            ui.notify('知識内容を入力してください', type='warning')
            return

        if knowledge_type not in self.domain_knowledge:
            self.domain_knowledge[knowledge_type] = []

        self.domain_knowledge[knowledge_type].append({
            'content': content,
            'variable': variable,
            'timestamp': self._get_timestamp(),
        })

        self._update_knowledge_display()
        dialog.close()
        ui.notify('✓ ドメイン知識を追加しました', type='positive')

    def _update_knowledge_display(self):
        """ドメイン知識の表示を更新"""
        self._knowledge_container.clear()

        with self._knowledge_container:
            if not self.domain_knowledge:
                ui.label('まだ知識が登録されていません').classes('text-gray-500 text-sm')
                return

            for knowledge_type, items in self.domain_knowledge.items():
                if not items:
                    continue

                type_labels = {
                    'variable': '変数の性質',
                    'sample': 'サンプル',
                    'target': '目的変数',
                    'system': '系全体',
                }

                with ui.expansion(f"{type_labels.get(knowledge_type, knowledge_type)} ({len(items)}件)", icon='psychology').classes('w-full mb-2'):
                    for i, item in enumerate(items):
                        with ui.card().classes('w-full mb-2 bg-gray-50'):
                            ui.label(item['content']).classes('text-sm')
                            if item.get('variable'):
                                ui.label(f"関連変数: {item['variable']}").classes('text-xs text-gray-500')
                            ui.label(f"登録: {item.get('timestamp', '')}").classes('text-xs text-gray-400')

    def _export_knowledge(self):
        """ドメイン知識をJSON出力"""
        if not self.domain_knowledge:
            ui.notify('出力する知識がありません', type='warning')
            return

        try:
            config_dir = Path('configs')
            config_dir.mkdir(exist_ok=True)

            timestamp = self._get_timestamp()
            path = config_dir / f'domain_knowledge_{timestamp}.json'

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.domain_knowledge, f, ensure_ascii=False, indent=2)

            ui.notify(f'✓ 知識を保存: {path}', type='positive')
        except Exception as e:
            logger.error(f"知識出力エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _import_knowledge(self):
        """JSONからドメイン知識を読み込み"""
        ui.notify('JSON読み込み（準備中）', type='info')

    def _select_uma_folder(self):
        """UMA重みフォルダを選択"""
        ui.notify('フォルダ選択（準備中）', type='info')

    def _check_resources(self):
        """リソースをチェック"""
        import psutil

        # CPU・メモリ情報を取得
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)

        # GPUチェック（簡易版）
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

        # 判定
        if memory_gb < 8:
            status = '⚠️ メモリ不足（8GB未満）。超軽量モデルのみ有効化します。'
            color = 'text-orange-600'
        elif not gpu_available and memory_gb < 16:
            status = '⚠️ GPUなし・メモリ16GB未満。重量モデルはスキップします。'
            color = 'text-orange-600'
        else:
            status = '✓ リソースは十分です。全モデルを有効化できます。'
            color = 'text-green-600'

        self._resource_status.text = f"CPU: {cpu_count}コア | メモリ: {memory_gb:.1f}GB | GPU: {'あり' if gpu_available else 'なし'} | {status}"
        self._resource_status.classes(f'text-sm {color} mt-2')

        ui.notify('✓ リソースチェック完了', type='positive')

    def _save_settings(self):
        """設定を保存"""
        config = {
            'llm': {
                'mode': getattr(self.llm_config, 'mode', 'prompt_only'),
                'model_name': getattr(self.llm_config, 'model_name', 'default'),
            },
            'domain_knowledge': self.domain_knowledge,
            'uma': {
                'weight_path': self._uma_path.value if hasattr(self, '_uma_path') else None,
            },
            'timestamp': self._get_timestamp(),
        }

        config_dir = Path('configs')
        config_dir.mkdir(exist_ok=True)

        with open(config_dir / 'settings.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        ui.notify('✓ 設定を保存しました: configs/settings.json', type='positive')

    def _load_settings(self):
        """設定を読み込み"""
        try:
            config_path = Path('configs/settings.json')
            if not config_path.exists():
                ui.notify('設定ファイルが見つかりません', type='warning')
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.domain_knowledge = config.get('domain_knowledge', {})
            self._update_knowledge_display()

            ui.notify('✓ 設定を読み込みました', type='positive')
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _reset_settings(self):
        """設定をデフォルトに戻す"""
        self.domain_knowledge = {}
        self._update_knowledge_display()

        if hasattr(self, '_uma_path'):
            self._uma_path.value = ''

        ui.notify('✓ 設定をデフォルトに戻しました', type='positive')

    @staticmethod
    def _get_timestamp():
        """現在のタイムスタンプを取得"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
