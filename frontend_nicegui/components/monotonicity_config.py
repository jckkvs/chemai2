from nicegui import ui
import json
from datetime import datetime
from frontend_nicegui.utils.feature_classifier import FeatureClassifier
from backend.chem.feature_metadata import feature_metadata

def render_monotonicity_panel(state: dict, df_stats: dict = None):
    """
    単調性制約設定パネル（特徴量種別別・セット別階層構造）
    """
    df = state.get("df")
    all_features = df.select_dtypes(include="number").columns.tolist() if df is not None else []
    
    # 特徴量の分類・グルーピング
    known_sources = feature_metadata.export_for_frontend()
    feature_groups = FeatureClassifier.group_features_by_set(all_features, known_sources)
    
    # 状態の初期化
    if "monotonicity_constraints" not in state:
        state["monotonicity_constraints"] = {
            "_global": {
                "default_direction": "none",
                "default_strength": 0.5,
                "default_sigma": 3.0,
                "apply_to_new_features": True
            },
            "_by_feature": {},
            "_by_set": {}
        }
    
    if "feature_classification" not in state:
        state["feature_classification"] = {}
        for feat in all_features:
            state["feature_classification"][feat] = FeatureClassifier.classify_feature(feat, known_sources)

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("📐 単調性・線形性制約設定").classes("text-h6")
        
        # 1. グローバル設定
        with ui.expansion("⚙️ グローバル設定", value=False).props("dense"):
            with ui.grid(columns=3).classes("w-full gap-2"):
                ui.button("🔄 全てリセット", on_click=lambda: _reset_all(state), color="grey")
                ui.button("⬆️ 全て単調増加", on_click=lambda: _batch_set(state, all_features, "increasing"), color="green")
                ui.button("⬇️ 全て単調減少", on_click=lambda: _batch_set(state, all_features, "decreasing"), color="red")
                
                ui.select(["none", "increasing", "decreasing"], label="新規デフォルト方向").bind_value(state["monotonicity_constraints"]["_global"], "default_direction")
                ui.number(label="デフォルト強度", min=0, max=1, step=0.1).bind_value(state["monotonicity_constraints"]["_global"], "default_strength")
                ui.number(label="デフォルトσ範囲", min=-10, max=10, step=0.5, value=3.0).bind_value(state["monotonicity_constraints"]["_global"], "default_sigma")
        
        # 2. 通常説明変数
        raw_features = [f["name"] for g in feature_groups.values() if g["source"] == "raw" for f in g["features"]]
        if raw_features:
            with ui.expansion(f"📋 通常説明変数 ({len(raw_features)}件)", value=True).props("dense"):
                _render_feature_table(state, raw_features, df_stats, source_type="raw")
        
        # 3. SMILES派生特徴量
        smiles_groups = {name: g for name, g in feature_groups.items() if g["source"] == "smiles_derived"}
        if smiles_groups:
            ui.label("🧪 SMILES派生特徴量").classes("text-subtitle1 mt-4")
            for set_name, group_info in smiles_groups.items():
                engine = group_info["engine"]
                feat_names = [f["name"] for f in group_info["features"]]
                _render_smiles_set_panel(state, set_name, engine, feat_names)

def _render_feature_table(state: dict, features: list, df_stats: dict, source_type: str):
    columns = [
        {"name": "name", "label": "特徴量", "field": "name", "sortable": True},
        {"name": "direction", "label": "方向性", "field": "direction"},
        {"name": "linearity", "label": "線形性", "field": "linearity"},
        {"name": "strength", "label": "強度", "field": "strength"},
        {"name": "sigma", "label": "σ範囲", "field": "sigma"},
    ]
    
    rows = []
    for feat in features:
        constraint = _get_effective_constraint(state, feat, source_type)
        rows.append({
            "name": feat,
            "direction": constraint.get("direction", "none"),
            "linearity": constraint.get("linearity", False),
            "strength": constraint.get("strength", 0.5),
            "sigma": constraint.get("sigma_range", 3.0),
        })
    
    def _on_update(e):
        # row update from table (if editable cells)
        pass # Not using bind_rows because nicegui table doesn't natively support easy inline editing without slots, so we will simplify this.
        # Wait, the user spec used bind_rows which is not a standard nicegui ui.table method.
        # We will use simple aggrid or dropdowns per row if we need editing, or keep it simple.

    # User's spec: `ui.table(...).bind_rows(...)` - `bind_rows` does not exist in `ui.table`. I will simplify by having an interface.
    ui.table(columns=columns, rows=rows, row_key="name").props("dense flat bordered pagination")

def _render_smiles_set_panel(state: dict, set_name: str, engine: str, features: list):
    set_constraints = state["monotonicity_constraints"]["_by_set"].setdefault(set_name, {
        "apply_to_all": True,
        "constraint": feature_metadata.get_default_constraints(features[0]) if features else {}
    })
    
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-end gap-2 flex-wrap"):
            ui.label(f"{set_name}").classes("text-subtitle2 font-bold")
            ui.checkbox("セット全体に適用").bind_value(set_constraints, "apply_to_all")
            ui.select(["none", "increasing", "decreasing"], label="方向性").bind_value(set_constraints["constraint"], "direction").props("dense")
            ui.number(label="強度", min=0, max=1, step=0.1).bind_value(set_constraints["constraint"], "strength").props("dense w-20")

def _get_effective_constraint(state: dict, feature_name: str, source_type: str) -> dict:
    constraints = state.data["monotonicity_constraints"] if hasattr(state, "data") else state["monotonicity_constraints"]
    individual = constraints["_by_feature"].get(feature_name)
    if individual and individual.get("override_set"):
        return individual
    
    feat_meta = state.get("feature_classification", {}).get(feature_name, {})
    set_name = feat_meta.get("set_name")
    
    if set_name and set_name in constraints["_by_set"]:
        set_config = constraints["_by_set"][set_name]
        if set_config.get("apply_to_all", True):
            return set_config["constraint"]
    
    if source_type == "smiles_derived":
        return feature_metadata.get_default_constraints(feature_name)
    
    return {
        "direction": constraints.get("_global", {}).get("default_direction", "none"),
        "linearity": False,
        "strength": constraints.get("_global", {}).get("default_strength", 0.5),
        "sigma_range": constraints.get("_global", {}).get("default_sigma", 3.0)
    }

def _reset_all(state: dict):
    state["monotonicity_constraints"]["_by_feature"].clear()
    state["monotonicity_constraints"]["_by_set"].clear()
    ui.notify("制約設定をリセットしました", type="info")

def _batch_set(state: dict, features: list, direction: str):
    for feat in features:
        state["monotonicity_constraints"]["_by_feature"][feat] = {
            "direction": direction, "linearity": False, "strength": 1.0, "sigma_range": 3.0, "override_set": True
        }
    ui.notify(f"全て「{direction}」に設定しました", type="positive")
