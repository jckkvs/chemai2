"""
NiceGUI用 特徴量セット選択コンポーネント
"""
from nicegui import ui
from typing import List, Dict, Optional, Callable

from backend.chem.descriptor_sets import DescriptorSet, DescriptorSetManager
from backend.chem.smiles_feature_calculator import SMILESFeatureCalculator


class FeatureSetSelector:
    """特徴量セットの選択・作成・編集用コンポーネント"""
    
    def __init__(
        self,
        smiles_column: Optional[str] = None,
        on_set_selected: Optional[Callable[[DescriptorSet], None]] = None,
        on_calculate: Optional[Callable[[], None]] = None,
    ):
        self.smiles_column = smiles_column
        self.on_set_selected = on_set_selected
        self.on_calculate = on_calculate
        
        self.manager = DescriptorSetManager.load_from_file()
        self.calculator = SMILESFeatureCalculator()
        
        self.selected_set: Optional[DescriptorSet] = None
        self._container: Optional[ui.element] = None
    
    def render(self, container: ui.element):
        """コンポーネントをレンダリング"""
        self._container = container
        
        with container:
            ui.label('⚗️ SMILES特徴量セット').classes('text-lg font-bold mb-2')
            
            # 既存セットの選択
            with ui.row().classes('w-full items-center gap-2'):
                ui.label('プリセット:').classes('mr-1')
                
                set_options = {
                    s.name: f"{s.name} ({s.summary})" for s in self.manager.list_enabled()
                }
                self._set_select = ui.select(
                    options=set_options,
                    value=None,
                    on_change=self._on_set_change,
                ).classes('flex-grow min-w-48')
                
                ui.button('🔄', on_click=self._trigger_calculate).props('dense unelevated').tooltip('再計算')
            
            # カスタムセット作成
            with ui.expansion('🔧 カスタムセット作成', icon='settings').classes('w-full mt-2 border rounded'):
                self._render_custom_form()
            
            # 選択中のセット詳細
            self._details_card = ui.card().classes('w-full mt-2 hidden')
            with self._details_card:
                self._render_details()
    
    def _on_set_change(self, e):
        """セット選択時のハンドラ"""
        set_name = e.value
        self.selected_set = self.manager.get(set_name)
        
        if self.selected_set:
            self._details_card.classes('w-full mt-2').remove_class('hidden')
            self._render_details()
            
            if self.on_set_selected:
                self.on_set_selected(self.selected_set)
    
    def _render_custom_form(self):
        """カスタムセット作成フォーム"""
        with ui.column().classes('w-full gap-2 p-2'):
            # 基本情報
            with ui.row().classes('w-full items-center gap-4'):
                name_input = ui.input('セット名', placeholder='My custom set').classes('flex-grow')
                # desc_input = ui.input('説明（オプション）').classes('w-64')
            
            # エンジン選択
            ui.label('使用するエンジン:').classes('mt-2 font-medium')
            engine_checks = {}
            
            from backend.chem.descriptor_sets import ENGINE_FLAG_KEYS, ENGINE_LABELS
            
            # Blueprintで指定されたもの + システムにあるもの
            blueprint_flags = {
                'use_mordred': 'Mordred (1800+記述子)',
                'use_xtb': 'xTB (量子化学)',
                'use_cosmo': 'COSMO-RS (溶媒和)',
                'use_unipka': 'UniPKa (pKa/LogD)',
                'use_skfp': 'scikit-FP (多様フィンガープリント)',
                'use_molai': 'MolAI (深層学習埋め込み)',
            }
            
            with ui.grid(columns=2).classes('w-full'):
                for flag in ENGINE_FLAG_KEYS:
                    label = blueprint_flags.get(flag, ENGINE_LABELS.get(flag, flag))
                    engine_checks[flag] = ui.checkbox(label).classes('text-sm')
            
            # MolAI設定
            with ui.row().classes('items-center mt-2'):
                ui.label('MolAI 次元数:').classes('mr-2 text-sm')
                molai_dim = ui.number(value=32, min=8, max=128, step=8).classes('w-24').props('outlined dense')
            
            # 保存ボタン
            def save_set():
                if not name_input.value:
                    ui.notify('セット名を入力してください', type='warning')
                    return
                
                engine_flags_dict = {
                    flag: chk.value for flag, chk in engine_checks.items()
                }
                
                new_set = DescriptorSet(
                    name=name_input.value,
                    engine_flags=engine_flags_dict,
                    molai_n_components=int(molai_dim.value),
                )
                
                self.manager.add(new_set)
                self.manager.save_to_file()
                
                # 選択リストを更新
                self._set_select.options = {
                    s.name: f"{s.name} ({s.summary})" for s in self.manager.list_enabled()
                }
                self._set_select.set_value(new_set.name)
                
                ui.notify(f'セット "{new_set.name}" を保存しました', type='positive')
            
            ui.button('💾 セットを保存', on_click=save_set).classes('mt-2 w-full').props('unelevated color=primary')
    
    def _render_details(self):
        """選択中のセット詳細を表示"""
        if not self.selected_set:
            return
        
        self._details_card.clear()
        with self._details_card:
            with ui.row().classes('w-full justify-between items-center'):
                ui.label(f'📋 {self.selected_set.name}').classes('font-bold text-lg')
                ui.chip('選択中', color='primary').props('dense')
            
            # 使用エンジン
            active = self.selected_set.active_engines
            if active:
                ui.label('構成エンジン:').classes('mt-2 text-sm text-grey-7')
                with ui.row().classes('flex-wrap gap-1'):
                    for engine in active:
                        ui.badge(engine, color='blue-2', text_color='blue-9').props('outline dense')
            
            # 計算ボタン
            ui.button('▶️ 特徴量を計算実行', on_click=self._trigger_calculate).classes('mt-4 w-full').props('unelevated color=positive')
    
    def _trigger_calculate(self):
        """特徴量計算をトリガー"""
        if self.selected_set and self.on_calculate:
            self.on_calculate()
    
    def get_selected_set(self) -> Optional[DescriptorSet]:
        """現在選択中のセットを取得"""
        return self.selected_set
