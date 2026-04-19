"""
NiceGUI用 統合数値データに対する制約設定パネル
"""
from nicegui import ui
from typing import List, Dict, Optional, Callable, Literal

from backend.models.monotonic_constraints import UnifiedConstraintManager


class UnifiedConstraintsPanel:
    """統合データ（数値+SMILES特徴量）に対する制約設定"""
    
    def __init__(
        self,
        feature_columns: List[str],
        feature_set_info: Optional[Dict[str, List[str]]] = None,
        on_constraints_updated: Optional[Callable[[Dict], None]] = None,
    ):
        self.feature_columns = feature_columns
        self.feature_set_info = feature_set_info or {}
        self.on_constraints_updated = on_constraints_updated
        
        self.constraint_manager = UnifiedConstraintManager()
        self._constraints: Dict[str, str] = {}  # {column: direction}
    
    def render(self, container: ui.element):
        """パネルをレンダリング"""
        with container:
            ui.label('⚙️ 単調性・一括制約設定').classes('text-lg font-bold mb-2')
            
            # 特徴量セットでフィルタ
            with ui.row().classes('w-full items-center mb-2 gap-4'):
                with ui.row().classes('items-center'):
                    ui.label('表示フィルタ:').classes('mr-2 text-sm')
                    self._set_filter = ui.select(
                        options={'all': 'すべてを表示'} | {k: k for k in self.feature_set_info.keys()},
                        value='all',
                        on_change=lambda e: self._refresh_table(),
                    ).classes('w-48').props('dense outlined')
            
                # 一括設定
                with ui.row().classes('items-center'):
                    ui.label('選択列を一括設定:').classes('mr-2 text-sm')
                    self._batch_direction = ui.select(
                        options={'none': 'なし', 'positive': '↗ 増加', 'negative': '↘ 減少'},
                        value='none',
                    ).classes('w-32').props('dense outlined')
                    ui.button('適用', on_click=self._batch_apply).props('dense unelevated color=primary').classes('ml-1')
            
            # 制約設定テーブル
            self._table = ui.table(
                columns=[
                    {'name': 'column', 'label': '特徴量名', 'field': 'column', 'sortable': True, 'align': 'left'},
                    {'name': 'set', 'label': '所属セット', 'field': 'set', 'sortable': True, 'align': 'left'},
                    {'name': 'constraint', 'label': '単調性制約', 'field': 'constraint', 'align': 'center'},
                ],
                rows=self._build_rows(),
                row_key='column',
                pagination={'rowsPerPage': 15}
            ).classes('w-full border rounded-lg')
            
            # 制約変更用のセルテンプレート
            self._table.add_slot('body-cell-constraint', '''
                <q-td :props="props">
                    <q-select
                        v-model="props.row.constraint"
                        :options="['none', 'positive', 'negative']"
                        dense
                        borderless
                        emit-value
                        map-options
                        @update:model-value="() => $parent.$emit('update_constraint', props.row)"
                    >
                        <template v-slot:option="opt">
                            <q-item v-bind="opt.itemProps">
                                <q-item-section>
                                    <q-item-label v-if="opt.opt === 'none'">-</q-item-label>
                                    <q-item-label v-if="opt.opt === 'positive'">↗ 増加</q-item-label>
                                    <q-item-label v-if="opt.opt === 'negative'">↘ 減少</q-item-label>
                                </q-item-section>
                            </q-item>
                        </template>
                        <template v-slot:selected>
                            <span v-if="props.row.constraint === 'none'">-</span>
                            <span v-if="props.row.constraint === 'positive'" class="text-green">↗ 増加</span>
                            <span v-if="props.row.constraint === 'negative'" class="text-red">↘ 減少</span>
                        </template>
                    </q-select>
                </q-td>
            ''')
            
            self._table.on('update_constraint', lambda e: self._update(e.args['column'], e.args['constraint']))
            
            # 適用ボタン
            with ui.row().classes('w-full justify-end mt-4'):
                ui.button('✅ 制約を確定・反映', on_click=self._apply).classes('px-6').props('unelevated color=positive')
    
    def _refresh_table(self):
        """テーブルを更新"""
        self._table.rows = self._build_rows()
        self._table.update()
    
    def _build_rows(self) -> List[Dict]:
        """テーブル用の行データを構築"""
        filter_value = self._set_filter.value if hasattr(self, '_set_filter') else 'all'
        
        # フィルタ適用後の列を取得
        if filter_value == 'all':
            target_cols = self.feature_columns
        else:
            target_cols = self.feature_set_info.get(filter_value, [])
        
        rows = []
        for col in target_cols:
            # 特徴量セットの判別
            set_name = '基本数値'
            for sid, cols in self.feature_set_info.items():
                if col in cols:
                    set_name = sid
                    break
            
            rows.append({
                'column': col,
                'set': set_name,
                'constraint': self._constraints.get(col, 'none'),
            })
        return rows
    
    def _update(self, column: str, direction: str):
        """個別の制約を更新"""
        self._constraints[column] = direction
        self.constraint_manager.set_monotonic(column, direction)
    
    def _batch_apply(self):
        """一括設定を適用"""
        direction = self._batch_direction.value
        filter_value = self._set_filter.value
        
        if filter_value == 'all':
            target_cols = self.feature_columns
        else:
            target_cols = self.feature_set_info.get(filter_value, [])
        
        for col in target_cols:
            self._constraints[col] = direction
            self.constraint_manager.set_monotonic(col, direction)
        
        self._refresh_table()
        ui.notify(f'表示中の {len(target_cols)} 列に「{direction}」制約を適用しました')
    
    def _apply(self):
        """制約を確定してコールバック"""
        constraints_dict = self.constraint_manager.get_constraints_for_model(
            self.feature_columns
        )
        
        if self.on_constraints_updated:
            self.on_constraints_updated(constraints_dict)
        
        ui.notify('✅ 単調性制約をモデル設定に保存しました', type='positive')
    
    def get_constraints_dict(self) -> Dict:
        """現在の制約設定を取得"""
        return self.constraint_manager.get_constraints_for_model(self.feature_columns)
