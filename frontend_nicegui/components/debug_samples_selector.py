"""
デバッグ用サンプルデータ選択コンポーネント
"""
import os
import pandas as pd
from nicegui import ui, app
from pathlib import Path

# サンプルデータの定義
DEBUG_SAMPLES = {
    "mixture_smiles_only": {
        "name": "🧪 混合物 SMILES の回帰（WT% のみ）",
        "file": "mixture_smiles_only.csv",
        "desc": "混合物データ（SMILES+WT% のみ）。数値特徴量なしで加重平均変換と回帰のテスト用。",
        "target_col": "Target_BoilingPoint_C",
        "task_type": "regression"
    },
    "mixture_smiles_numeric": {
        "name": "🧪 混合物 SMILES+ 数値データ",
        "file": "mixture_smiles_numeric.csv",
        "desc": "混合物データ（SMILES+WT%+ 温度/圧力/pH/撹拌速度など）。SMILES と数値特徴量の両方を使った回帰タスク用。",
        "target_col": "Target_Yield_pct",
        "task_type": "regression"
    },
    "mixture_regression_debug": {
        "name": "🧪 混合物回帰 (WT% + 数値)",
        "file": "mixture_regression_debug.csv",
        "desc": "混合物データ（SMILES+WT%+MOL%+温度/圧力など）。加重平均変換と回帰のテスト用。",
        "target_col": "Target_Property", # 生成スクリプトに合わせる
        "task_type": "regression"
    },
    "monotonicity_test": {
        "name": "📈 単調性制約テスト",
        "file": "monotonicity_test.csv",
        "desc": "分子量・LogP・TPSA と溶解度の関係。単調増加制約のテスト用。",
        "target_col": "Solubility_mg_L",
        "task_type": "regression"
    },
    "timeseries_leak_test": {
        "name": "⏰ 時系列リーク検出テスト",
        "file": "timeseries_leak_test.csv",
        "desc": "日付・バッチIDを含むデータ。時系列リーク・バッチ効果リークの検出テスト用。",
        "target_col": "Yield_pct",
        "task_type": "regression"
    },
    "xtb_dependency_test": {
        "name": "⚛️ xTB 外部ツール依存テスト",
        "file": "xtb_dependency_test.csv",
        "desc": "小分子のみを含むデータ。GFN2-xTB 計算の正常動作確認用。",
        "target_col": "HOMO_eV",
        "task_type": "regression"
    },
    "classification_balanced": {
        "name": "✅ 分類タスク (バランス済み)",
        "file": "classification_balanced.csv",
        "desc": "Active/Inactive が均等に含まれる分類データ。クラス不均衡対策のテスト用。",
        "target_col": "Activity",
        "task_type": "classification"
    }
}

def create_debug_samples_selector(on_data_loaded=None):
    """
    デバッグ用サンプル選択 UI を作成する
    
    Args:
        on_data_loaded: データ読み込み完了時のコールバック関数 (df, task_type, target_col)
    """
    
    with ui.card().classes('w-full p-4 glass-card'):
        ui.label('🧪 デバッグ用サンプル').classes('text-lg font-bold mb-2')
        
        # サンプル選択ドロップダウン
        sample_options = {k: v["name"] for k, v in DEBUG_SAMPLES.items()}
        sample_select = ui.select(
            options=sample_options,
            label='サンプル選択',
            value=None,
            on_change=lambda e: update_sample_info(e.value)
        ).classes('w-full').props('outlined dense')
        
        # 説明表示エリア
        info_area_container = ui.column().classes('w-full mt-2')
        preview_area_container = ui.column().classes('w-full mt-2')
        
        def update_sample_info(key):
            """サンプル情報とプレビューを更新"""
            info_area_container.clear()
            preview_area_container.clear()
            
            if not key or key not in DEBUG_SAMPLES:
                return
            
            info = DEBUG_SAMPLES[key]
            
            # 説明更新
            with info_area_container:
                ui.markdown(f"**説明:** {info['desc']}").classes('text-body2')
                ui.markdown(f"**目的変数:** `{info['target_col']}` | **タスク:** {info['task_type']}").classes('text-caption text-grey-5')
            
            # プレビュー表示
            # パス計算: chemai2/data/samples/debug/
            file_path = Path(__file__).parent.parent.parent / 'data' / 'samples' / 'debug' / info['file']
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, nrows=5)
                    with preview_area_container:
                        ui.label('📋 データプレビュー (上位 5 行):').classes('font-bold mt-2 text-caption')
                        ui.table.from_pandas(df).classes('w-full text-xs').props('dense flat bordered')
                except Exception as e:
                    with preview_area_container:
                        ui.notify(f'プレビュー読み込みエラー: {e}', color='negative')
            else:
                with preview_area_container:
                    ui.notify(f'ファイルが見つかりません: {info["file"]}', color='warning')
        
        # 読み込みボタン
        def load_sample():
            key = sample_select.value
            if not key or key not in DEBUG_SAMPLES:
                ui.notify('サンプルを選択してください', color='warning')
                return
            
            info = DEBUG_SAMPLES[key]
            file_path = Path(__file__).parent.parent.parent / 'data' / 'samples' / 'debug' / info['file']
            
            if not file_path.exists():
                ui.notify(f'ファイルが見つかりません: {file_path}', color='negative')
                return
            
            try:
                df = pd.read_csv(file_path)
                ui.notify(f'{info["name"]} を読み込みました ({len(df)} 行)', color='positive')
                
                if on_data_loaded:
                    on_data_loaded(df, info['task_type'], info['target_col'], info['file'])
                    
            except Exception as e:
                ui.notify(f'読み込みエラー: {e}', color='negative')
        
        ui.button('🚀 このデータで開始', on_click=load_sample, icon='play_arrow').classes('w-full mt-4').props('unelevated color=primary')

# テスト用スタンドアローン実行
if __name__ in {"__main__", "__mp_main__"}:
    @ui.page('/')
    def test_page():
        def handle_data_loaded(df, task_type, target_col, filename):
            ui.notify(f'データ読込成功: {len(df)}行, タスク={task_type}, 目的変数={target_col}')
            with ui.dialog() as dlg, ui.card():
                ui.label(f'読み込みデータ: {df.shape}')
                ui.table.from_pandas(df.head())
                ui.button('Close', on_click=dlg.close)
            dlg.open()
        
        create_debug_samples_selector(on_data_loaded=handle_data_loaded)
    
    ui.run(title='Debug Samples Test')
