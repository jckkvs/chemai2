"""
デバッグ用サンプルデータ選択コンポーネント

このコンポーネントは、開発やテスト目的で標準的なデータセットを
素早くロードするためのインターフェースを提供します。
"""
import os
import pandas as pd
from nicegui import ui, app
from pathlib import Path

# サンプルデータの定義
DEBUG_SAMPLES = {
    # ==================== 回帰 (SMILES) カテゴリ ====================
    "simple_smiles_regression": {
        "name": "🧪 シンプルSMILES回帰",
        "file": "simple_smiles_regression.csv",
        "desc": "単一化合物のSMILESと物性値（沸点/溶解度など）。基本的な回帰タスク用。",
        "target_col": "BoilingPoint_C",
        "task_type": "regression",
        "category": "regression"
    },
    "simple_smiles_classification": {
        "name": "🏷️ シンプルSMILES分類",
        "file": "simple_smiles_classification.csv",
        "desc": "単一化合物のSMILESと活性/不活性ラベル。基本的な分類タスク用。",
        "target_col": "Activity",
        "task_type": "classification",
        "category": "classification"
    },
    
    "mixture_smiles_only": {
        "name": "🧪 混合物 SMILES (WT% のみ)",
        "file": "mixture_smiles_only.csv",
        "desc": "混合物データ（SMILES+WT% のみ）。数値特徴量なしで加重平均変換と回帰のテスト用。",
        "target_col": "TARGET_BOILINGPOINT_C",
        "task_type": "regression",
        "category": "regression"  # SMILES使用
    },
    "mixture_smiles_numeric": {
        "name": "🧪 混合物 SMILES+数値データ",
        "file": "mixture_smiles_numeric.csv",
        "desc": "混合物データ（SMILES+WT%+温度/圧力/pHなど）。SMILESと数値特徴量の両方を使用。",
        "target_col": "TARGET_YIELD_PCT",
        "task_type": "regression",
        "category": "regression"  # SMILES使用
    },
    "mixture_regression_debug": {
        "name": "🧪 混合物回帰 (WT% + 数値)",
        "file": "mixture_regression_debug.csv",
        "desc": "混合物データ（SMILES+WT%+MOL%+温度/圧力など）。加重平均変換と回帰のテスト用。",
        "target_col": "TARGET_PROPERTY",
        "task_type": "regression",
        "category": "regression"  # SMILES使用
    },
    "monotonicity_test": {
        "name": "📈 単調性制約テスト",
        "file": "monotonicity_test.csv",
        "desc": "分子量・LogP・TPSA と溶解度の関係。単調増加制約のテスト用。",
        "target_col": "SOLUBILITY_MG_L",
        "task_type": "regression",
        "category": "regression"  # SMILES使用
    },
    "xtb_dependency_test": {
        "name": "⚛️ xTB 外部ツール依存テスト",
        "file": "xtb_dependency_test.csv",
        "desc": "小分子のみを含むデータ。GFN2-xTB 計算の正常動作確認用。",
        "target_col": "HOMO_EV",
        "task_type": "regression",
        "category": "regression"  # SMILES使用
    },
    
    # ==================== 分類 (SMILES) カテゴリ ====================
    "classification_balanced": {
        "name": "✅ 分類タスク (バランス済み)",
        "file": "classification_balanced.csv",
        "desc": "Active/Inactive が均等に含まれる分類データ。クラス不均衡対策のテスト用。",
        "target_col": "Activity",
        "task_type": "classification",
        "category": "classification"  # SMILES使用
    },
    
    # ==================== 数値のみ カテゴリ ====================
    "timeseries_leak_test": {
        "name": "⏰ 時系列リーク検出テスト",
        "file": "timeseries_leak_test.csv",
        "desc": "日付・バッチIDを含むデータ。時系列リーク・バッチ効果リークの検出テスト用。",
        "target_col": "YIELD_PCT",
        "task_type": "regression",
        "category": "numeric"  # ← 修正：数値のみに移動
    },
    "numeric_only_regression": {
        "name": "📊 数値データのみ（物性予測）",
        "file": "numeric_only_regression.csv",
        "desc": "分子記述子（分子量、LogP、TPSA、HBA、HBD等）のみを使用した回帰タスク。",
        "target_col": "BoilingPoint_C",
        "task_type": "regression",
        "category": "numeric"  # SMILES不使用
    },
    "numeric_only_solubility": {
        "name": "📊 数値データのみ（溶解度予測）",
        "file": "numeric_only_solubility.csv",
        "desc": "分子記述子と実験条件（温度、pH）から溶解度を予測。",
        "target_col": "Solubility_mg_L",
        "task_type": "regression",
        "category": "numeric"  # SMILES不使用
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

        # カテゴリ別タブ
        with ui.tabs().classes('w-full').props('dense') as sample_tabs:
            tab_regression = ui.tab('stats', label='🧪 回帰 (SMILES)')
            tab_classification = ui.tab('check', label='🏷️ 分類 (SMILES)')
            tab_numeric = ui.tab('pin', label='📊 数値のみ')
        
        with ui.tab_panels(sample_tabs, value='stats').classes('w-full bg-transparent'):
            # 回帰サンプル
            with ui.tab_panel('stats'):
                _render_sample_buttons(
                    category='regression',
                    on_data_loaded=on_data_loaded
                )
            
            # 分類サンプル
            with ui.tab_panel('check'):
                _render_sample_buttons(
                    category='classification',
                    on_data_loaded=on_data_loaded
                )
            
            # 数値のみサンプル
            with ui.tab_panel('pin'):
                _render_sample_buttons(
                    category='numeric',
                    on_data_loaded=on_data_loaded
                )

def _render_sample_buttons(category: str, on_data_loaded=None):
    """指定カテゴリのサンプルをボタン形式で表示"""
    # カテゴリ別にフィルタリング
    filtered_samples = {
        k: v for k, v in DEBUG_SAMPLES.items()
        if v.get('category') == category
    }
    
    if not filtered_samples:
        ui.label('サンプルデータはありません').classes('text-grey')
        return
    
    # ボタンをグリッド配置
    with ui.row().classes('w-full q-gutter-sm flex-wrap'):
        for key, info in filtered_samples.items():
            btn = ui.button(
                info['name'],
                icon='play_arrow',
                on_click=lambda k=key: _load_sample(k, on_data_loaded),
                color='primary'
            ).props('outline unelevated no-caps').classes('q-ma-xs')
            
            # ホバーで説明表示
            btn.props(f'tooltip="{info["desc"]}"')

def _load_sample(key: str, on_data_loaded=None):
    """サンプルデータを読み込む"""
    if key not in DEBUG_SAMPLES:
        ui.notify('無効なサンプルキーです', color='negative')
        return
    
    info = DEBUG_SAMPLES[key]
    file_path = Path(__file__).parent.parent.parent / 'data' / 'samples' / 'debug' / info['file']
    
    if not file_path.exists():
        ui.notify(f'ファイルが見つかりません: {info["file"]}', color='negative')
        return
    
    try:
        df = pd.read_csv(file_path)
        ui.notify(f'{info["name"]} を読み込みました ({len(df)} 行)', color='positive')
        
        if on_data_loaded:
            on_data_loaded(df, info['task_type'], info['target_col'], info['file'])
    except Exception as e:
        ui.notify(f'読み込みエラー: {e}', color='negative')

# テスト用スタンドアローン実行
if __name__ in {"__main__", "__mp_main__"}:
    @ui.page('/')
    def test_page():
        ui.dark_mode(True)
        def handle_data_loaded(df, task_type, target_col, filename):
            ui.notify(f'データ読込成功: {len(df)}行, タスク={task_type}, 目的変数={target_col}')
            with ui.dialog() as dlg, ui.card():
                ui.label(f'読み込みデータ: {df.shape}')
                ui.table.from_pandas(df.head())
                ui.button('Close', on_click=dlg.close)
            dlg.open()
        
        create_debug_samples_selector(on_data_loaded=handle_data_loaded)
    
    ui.run(title='Debug Samples Test')
