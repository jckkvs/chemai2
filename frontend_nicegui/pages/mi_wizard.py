"""
frontend_nicegui/pages/mi_wizard.py
非コーディング研究者向け解析ウィザード
既存UIと共存：既存タブを維持し、新規タブとして追加
"""
from nicegui import ui, events
import pandas as pd
from typing import Optional, Dict, List
from backend.core.hardware_detector import HardwareDetector
from backend.core.model_selector import ModelSelector, ModelRecommendation
from backend.core.agent_orchestrator import MIAgentOrchestrator, AnalysisRequest


class MIWizard:
    """マテリアルズインフォマティクス解析ウィザード"""
    
    def __init__(self):
        self.hardware = HardwareDetector().get_spec()
        self.selector = ModelSelector()
        self.recommended_model: Optional[ModelRecommendation] = None
        self.current_data: Optional[pd.DataFrame] = None
        self._render()
    
    def _render(self):
        """ウィザード画面を描画"""
        with ui.column().classes('w-full max-w-4xl mx-auto p-4'):
            
            # 1. ハードウェアステータス表示
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('items-center mb-2'):
                    ui.icon('settings_suggest', color='primary', size='md')
                    ui.label('🔧 検出されたハードウェアと推奨設定').classes('font-bold text-lg')
                
                with ui.row().classes('w-full gap-4 p-2 bg-gray-50 rounded'):
                    with ui.column():
                        ui.label('GPU/VRAM').classes('text-xs text-gray-500 uppercase')
                        gpu_names = ', '.join(g['name'] for g in self.hardware.gpus) or 'None'
                        vram_total = sum(g['vram_total_gb'] for g in self.hardware.gpus)
                        ui.label(f"{gpu_names} ({vram_total:.1f}GB)").classes('font-medium')
                    
                    with ui.column():
                        ui.label('System RAM').classes('text-xs text-gray-500 uppercase')
                        ui.label(f"{self.hardware.ram_total_gb:.1f}GB").classes('font-medium')
                    
                    with ui.column():
                        ui.label('Performance Tier').classes('text-xs text-gray-500 uppercase')
                        ui.badge(self.hardware.inference_tier, color='blue').classes('p-1')
                
                # 推奨モデル表示
                self._recommended_model_row = ui.row().classes('w-full items-center mt-4 p-3 bg-blue-50 rounded border border-blue-100')
                self._show_recommended_model()
            
            # 2. 解析ウィザードステップ
            with ui.tabs().classes('w-full') as self._tabs:
                self._step1 = ui.tab('1. データ準備', icon='file_upload')
                self._step2 = ui.tab('2. 解析目的', icon='psychology')
                self._step3 = ui.tab('3. 実行・確認', icon='play_circle')
            
            with ui.tab_panels(self._tabs, value=self._step1).classes('w-full bg-transparent shadow-none') as self._panels:
                
                # Step 1: データ準備
                with ui.tab_panel(self._step1):
                    ui.label('📁 解析データのアップロード').classes('font-bold mb-2')
                    self._render_data_upload()
                
                # Step 2: 解析目的
                with ui.tab_panel(self._step2):
                    ui.label('🎯 解析目的を選択').classes('font-bold mb-2')
                    self._render_goal_selection()
                
                # Step 3: 実行・確認
                with ui.tab_panel(self._step3):
                    ui.label('⚡ 解析実行').classes('font-bold mb-2')
                    self._render_execution_panel()
            
            # ナビゲーションボタン
            with ui.row().classes('w-full justify-between mt-6'):
                self._prev_btn = ui.button('← 前へ', on_click=self._prev_step).props('outline').disable()
                self._next_btn = ui.button('次へ →', on_click=self._next_step).props('color=primary')
    
    def _show_recommended_model(self):
        """推奨モデルを表示"""
        if not self.recommended_model:
            self.recommended_model = self.selector.select_best_model(self.hardware)
        
        self._recommended_model_row.clear()
        with self._recommended_model_row:
            ui.icon('auto_awesome', color='blue').classes('mr-2')
            with ui.column().classes('flex-1'):
                ui.label(f"推奨モデル: {self.recommended_model.model_name} ({self.recommended_model.quantization})").classes('font-bold')
                with ui.row().classes('text-xs text-gray-600 gap-x-4'):
                    ui.label(f"期待速度: {self.recommended_model.expected_tps:.1f} tokens/sec")
                    ui.label(f"コンテキスト: {self.recommended_model.context_max:,} tokens")
                
                if self.recommended_model.confidence < 0.7:
                    ui.label("⚠️ 注意: ハードウェア制限により速度が低下する可能性があります").classes('text-orange-600 text-xs mt-1')
            
            ui.button('モデル変更', on_click=self._show_model_selector).props('flat dense color=blue')
    
    def _render_data_upload(self):
        """データアップロード領域"""
        from frontend_nicegui.components.file_upload_zone import FileUploadZone
        
        def on_data_loaded(result: Dict):
            self.current_data = result.get('data')
            if isinstance(self.current_data, pd.DataFrame):
                ui.notify(f'✓ {result["meta"]["filename"]} を読み込みました', type='positive')
                self._data_preview.visible = True
                self._data_preview.options['rowData'] = self.current_data.head(5).to_dict('records')
                self._data_preview.update()
                # 次のステップを有効化
                self._next_btn.enable()
        
        FileUploadZone(
            on_upload=on_data_loaded,
            allowed_types=['csv', 'excel'],
            label='化学データ（CSV/Excel）をアップロード'
        )
        
        # データプレビュー (AG Grid形式)
        ui.label('プレビュー（先頭5行）').classes('mt-4 text-sm font-bold')
        self._data_preview = ui.aggrid({
            'columnDefs': [],
            'rowData': [],
        }).classes('w-full h-40 mt-2')
        self._data_preview.visible = False
    
    def _render_goal_selection(self):
        """解析目的選択"""
        goal_options = {
            'property_prediction': '物性値の予測（回帰分析）',
            'classification': '材料分類（分類分析）',
            'similarity_search': '類似材料の検索',
            'visualization': 'データの可視化・探索',
            'descriptor_generation': '分子記述子の自動生成'
        }
        
        with ui.card().classes('w-full p-4'):
            self._goal_select = ui.radio(
                options=goal_options,
                value='property_prediction'
            ).props('dense')
            
            ui.button('解析方針を生成', on_click=self._generate_plan, color='primary').classes('mt-4')
        
        self._plan_display = ui.markdown().classes('mt-6 p-4 bg-gray-50 rounded border border-gray-200 w-full')
        self._plan_display.visible = False
    
    def _generate_plan(self):
        """エージェントに解析方針を生成させる"""
        if self.current_data is None:
            ui.notify('先にデータをアップロードしてください', type='warning')
            return
        
        # 実際には MIAgentOrchestrator を使用
        # ここではデモ用に内容を表示
        with ui.notify('AIエージェントが解析方針を検討中...', spinner=True):
            # モック応答
            plan_text = f"""
### 📋 解析プラン: {self._goal_select.value}
1. **データ前処理**: 欠損値を検出し、化学的妥当性に基づき補完します。
2. **記述子生成**: SMILESが含まれる場合、RDKitを用いて物理化学的特徴量を抽出します。
3. **モデル構築**: `RandomForest` または `LightGBM` を用いて、ロバストな予測モデルを構築します。
4. **結果の解釈**: SHAPを用いて、どの部分構造が物性に寄与しているかを可視化します。

### 💻 生成予定のコード
- Python (pandas, scikit-learn, plotly)
"""
            self._plan_display.set_content(plan_text)
            self._plan_display.visible = True
            self._next_btn.enable()
        
        ui.notify('解析方針を生成しました', type='positive')
    
    def _render_execution_panel(self):
        """実行パネル"""
        with ui.card().classes('w-full p-6 text-center border-2 border-dashed border-primary/20'):
            ui.icon('rocket_launch', size='4em', color='primary').classes('opacity-50')
            ui.label('自律解析の準備が整いました').classes('text-xl font-bold mt-2')
            ui.label('「実行」ボタンを押すと、AIがコードを生成し、解析を自動で開始します。').classes('text-gray-600 mb-4')
            
            with ui.row().classes('justify-center gap-4'):
                ui.button('▶ 解析を実行', on_click=self._execute_analysis, color='positive').props('size=lg elevated')
                ui.button('📋 コードを確認', on_click=self._show_code_dialog).props('outline')
        
        self._execution_log = ui.column().classes('w-full mt-6')
        self._result_area = ui.column().classes('w-full mt-4')
    
    def _execute_analysis(self):
        """解析の自律実行"""
        with ui.column().classes('w-full') as log:
            ui.label('⏳ ステップ1: データの検証中...').classes('text-blue-600')
            ui.label('⏳ ステップ2: 記述子の生成中...').classes('text-blue-600')
            ui.label('⏳ ステップ3: モデルの学習中...').classes('text-blue-600')
        
        ui.notify('解析を開始しました（シミュレーション）', type='info')
        # 実際にはバックグラウンドで実行
    
    def _next_step(self):
        """ステップを次に進める"""
        current = self._panels.value
        if current == self._step1:
            self._panels.value = self._step2
            self._prev_btn.enable()
        elif current == self._step2:
            self._panels.value = self._step3
            self._next_btn.disable()
    
    def _prev_step(self):
        """ステップを前に戻す"""
        current = self._panels.value
        if current == self._step2:
            self._panels.value = self._step1
            self._prev_btn.disable()
        elif current == self._step3:
            self._panels.value = self._step2
            self._next_btn.enable()

    def _show_model_selector(self):
        ui.notify('モデル選択ダイアログは現在開発中です', type='info')

    def _show_code_dialog(self):
        with ui.dialog() as dialog, ui.card().classes('w-[600px]'):
            ui.label('生成された解析コード').classes('text-lg font-bold')
            ui.code("""# 自動生成された解析スクリプト (サンプル)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# データ読み込み
df = pd.read_csv('uploaded_data.csv')
# ... 以下解析コード ...
""", language='python').classes('w-full h-80 overflow-auto')
            ui.button('閉じる', on_click=dialog.close).props('flat').classes('ml-auto')
        dialog.open()
