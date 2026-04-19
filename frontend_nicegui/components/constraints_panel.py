"""
frontend_nicegui/components/constraints_panel.py

単調性制約およびグループ制約の設定用UI。
数値列（元の表）とSMILES記述子セットの両方に対して一元的に制約を適用できる。
"""
from nicegui import ui
from typing import Dict, Any, List, Optional

def render_constraints_panel(state: dict):
    """
    統合制約設定パネルのレンダリング
    """
    ui.label("⚖️ 統合制約設定 (Monotonicity & Group)").classes("text-xl font-bold q-mb-md")
    
    # 状態の初期化
    if "monotonic_constraints" not in state:
        state["monotonic_constraints"] = {}
    if "constraint_groups" not in state:
        state["constraint_groups"] = {}

    # 全数値特徴量の特定
    df = state.get("df")
    if df is None:
        ui.label("データがロードされていません。").classes("text-grey italic")
        return

    # 数値列の抽出（SMILES特徴量セット由来のものも含む）
    # ここでは DataMerger を使って統合された最新のDFを取得するか、
    # 既存の df と generated_smiles_features をマージして考える
    raw_numeric_cols = df.select_dtypes(include="number").columns.tolist()
    smiles_features = state.get("generated_smiles_features", [])
    
    # すべての数値オプション
    all_numeric_options = sorted(list(set(raw_numeric_cols + smiles_features)))
    
    if not all_numeric_options:
        ui.label("制約を適用可能な数値列が見つかりません。").classes("text-grey italic")
        return

    # 1. セット/グループ単位の一括制約
    with ui.card().classes("w-full q-mb-md"):
        ui.label("⚙️ 一括制約（セット・グループ単位）").classes("text-lg font-semibold q-mb-sm")
        
        with ui.row().classes("w-full items-center q-gutter-md"):
            # 特徴量セットに対する制約
            if "feature_sets" in state and state["feature_sets"]:
                with ui.column().classes("flex-grow"):
                    ui.label("SMILESセット全体に適用:").classes("text-sm text-grey-7")
                    for set_id, s_info in state["feature_sets"].items():
                        set_key = f"__set__:{set_id}"
                        current_val = state["monotonic_constraints"].get(set_key, 0)
                        ui.select(
                            options={1: "単調増加", -1: "単調減少", 0: "制約なし"},
                            label=f"セット: {s_info['name']}",
                            value=current_val,
                            on_change=lambda e, sk=set_key: _update_constraint(state, sk, e.value)
                        ).classes("w-full")

    # 2. 個別特徴量の制約（検索機能付き）
    with ui.card().classes("w-full"):
        ui.label("🔍 特徴量ごとの個別制約").classes("text-lg font-semibold q-mb-sm")
        
        search = ui.input(placeholder="特徴量名で検索...", on_change=lambda e: feature_list.refresh(e.value)).props("clearable icon=search").classes("w-full q-mb-md")
        
        @ui.refreshable
        def feature_list(filter_text: str = ""):
            filtered = [f for f in all_numeric_options if not filter_text or filter_text.lower() in f.lower()]
            
            if not filtered:
                ui.label("条件に一致する特徴量はありません").classes("text-grey italic")
                return
                
            # スクロールエリア
            with ui.scroll_area().classes("h-96 w-full"):
                for feat in filtered:
                    with ui.row().classes("w-full items-center border-b py-2 q-gutter-md"):
                        # 出自アイコン
                        is_smiles = feat in smiles_features
                        icon = "science" if is_smiles else "table_rows"
                        color = "purple" if is_smiles else "blue"
                        ui.icon(icon, color=color).classes("text-xl")
                        
                        ui.label(feat).classes("flex-grow font-mono text-sm overflow-hidden")
                        
                        current_val = state["monotonic_constraints"].get(feat, 0)
                        ui.select(
                            options={1: "+", -1: "-", 0: "×"},
                            value=current_val,
                            on_change=lambda e, f=feat: _update_constraint(state, f, e.value)
                        ).props("dense options-dense").classes("w-16")
        
        feature_list()

def _update_constraint(state: dict, feature: str, value: int):
    if value == 0:
        if feature in state["monotonic_constraints"]:
            del state["monotonic_constraints"][feature]
    else:
        state["monotonic_constraints"][feature] = value
    
    # ユーザーへのフィードバック
    dir_str = "単調増加" if value == 1 else "単調減少" if value == -1 else "解除"
    ui.notify(f"'{feature}' に {dir_str} 制約を設定しました", color="info" if value != 0 else "grey")
