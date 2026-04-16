"""
単調性制約設定パネル（動的UI生成）
"""
from nicegui import ui
from typing import Dict, List, Any, Optional
import pandas as pd


@ui.refreshable
def render_monotonicity_constraint_panel(state: dict):
    """
    単調性制約設定パネル
    自動的に特徴量を検出し、動的にUIを生成
    """
    
    with ui.card().classes("w-full q-pa-md"):
        # ヘッダーセクション
        _render_header(state)
        
        ui.separator().classes("q-my-md")
        
        # 自動提案セクション
        _render_auto_suggestions_section(state)
        
        ui.separator().classes("q-my-md")
        
        # 手動設定セクション（動的生成）
        _render_manual_constraints_section(state)
        
        ui.separator().classes("q-my-md")
        
        # 一括操作セクション
        _render_batch_operations(state)


def _render_header(state: dict):
    """ヘッダーセクション"""
    with ui.row().classes("w-full items-center justify-between"):
        with ui.column():
            ui.label("📊 単調性制約設定").classes("text-xl font-bold")
            ui.label(
                "特徴量と目的変数の間の単調性関係を定義します。\n"
                "化学的知見を手動で反映できます。"
            ).classes("text-sm text-grey-7")
        
        # 自動適用トグル
        with ui.column().classes("items-end"):
            auto_toggle = ui.toggle(
                {True: "有効", False: "無効"},
                value=state.get("auto_apply_constraints", True),
                on_change=lambda e: _on_auto_toggle(e.value, state)
            ).props('color=primary')
            ui.label("化学的知見の自動提案").classes("text-xs text-grey-7")


def _render_auto_suggestions_section(state: dict):
    """自動提案セクション"""
    
    if not state.get("auto_apply_constraints", True):
        return
    
    with ui.expansion("🤖 自動提案された制約", icon="auto_awesome", value=True).classes("w-full"):
        
        # 自動提案の説明
        ui.markdown("""
        以下の化学的知見に基づいて制約を自動提案しています：
        - **分子量 (MW)**: 増加 → 溶解度減少 ↘
        - **LogP**: 増加 → 溶解度減少 ↘  
        - **TPSA/HBA/HBD**: 増加 → 溶解度増加 ↗
        """).classes("text-sm text-grey-7 q-mb-md")
        
        # 自動提案チップを動的に生成
        auto_constraints = _get_auto_constraints(state)
        
        if auto_constraints:
            with ui.row().classes("q-gutter-sm flex-wrap"):
                for feature, constraint in auto_constraints.items():
                    _render_auto_constraint_chip(feature, constraint, state)
            
            # 一括適用ボタン
            ui.button(
                "✨ 自動提案をすべて適用",
                on_click=lambda: _apply_all_auto_constraints(state),
                color="primary"
            ).props("unelevated").classes("q-mt-md")
        else:
            ui.label("データを読み込むと自動提案が表示されます").classes("text-grey")


def _render_manual_constraints_section(state: dict):
    """手動設定セクション（動的UI生成）"""
    
    ui.label("🔧 手動設定").classes("text-lg font-semibold q-mb-sm")
    ui.markdown("""
    各特徴量に対して**手動で単調性制約を設定**できます。
    自動提案を上書きしたり、独自の化学的知見を反映できます。
    """).classes("text-sm text-grey-7 q-mb-md")
    
    # 特徴量を動的に検出
    features = _detect_features(state)
    
    if not features:
        ui.warning("設定可能な特徴量が見つかりません。先にデータを読み込んでください。")
        return
    
    # 動的に制約設定UIを生成
    _render_dynamic_constraint_grid(features, state)


def _render_dynamic_constraint_grid(features: List[str], state: dict):
    """特徴量に応じて動的にグリッドUIを生成"""
    
    # グリッドコンテナ
    with ui.grid(columns=3).classes("w-full q-gutter-md"):
        
        for feature in features:
            _render_single_constraint_card(feature, state)


def _render_single_constraint_card(feature: str, state: dict):
    """単一特徴量の制約設定カードを動的に生成"""
    
    # 現在の設定を取得
    current_constraint = _get_current_constraint(feature, state)
    auto_constraint = _get_auto_constraint_for_feature(feature, state)
    
    with ui.card().classes("w-full"):
        # ヘッダー
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(feature).classes("font-bold text-base")
            
            # 自動提案バッジ
            if auto_constraint != "none":
                ui.chip(
                    text=f"自動: {_constraint_to_label(auto_constraint)}",
                    color="purple",
                    size="sm"
                ).props("outline")
        
        ui.separator().classes("q-my-xs")
        
        # 化学的根拠
        rationale = _get_chemical_rationale(feature)
        if rationale:
            ui.label(rationale).classes("text-xs text-grey-7 q-mb-sm")
        
        # 制約選択ボタン群
        with ui.row().classes("w-full q-gutter-xs justify-center"):
            # 制約なし
            btn_none = ui.button(
                icon="remove",
                text="なし",
                on_click=lambda f=feature: _set_constraint(f, "none", state),
                color="grey"
            ).props("outline dense").classes("flex-grow")
            
            # 単調増加
            btn_inc = ui.button(
                icon="arrow_upward",
                text="増加",
                on_click=lambda f=feature: _set_constraint(f, "increasing", state),
                color="green"
            ).props("outline dense").classes("flex-grow")
            
            # 単調減少
            btn_dec = ui.button(
                icon="arrow_downward",
                text="減少",
                on_click=lambda f=feature: _set_constraint(f, "decreasing", state),
                color="red"
            ).props("outline dense").classes("flex-grow")
            
            # 自動提案
            btn_auto = ui.button(
                icon="auto_awesome",
                text="自動",
                on_click=lambda f=feature: _set_constraint(f, "auto", state),
                color="purple"
            ).props("outline dense").classes("flex-grow")
        
        # 現在の選択をハイライト
        _highlight_current_selection(current_constraint, {
            "none": btn_none,
            "increasing": btn_inc,
            "decreasing": btn_dec,
            "auto": btn_auto
        })
        
        # ステータス表示
        with ui.row().classes("w-full items-center justify-between q-mt-xs"):
            status_text = _get_status_text(feature, current_constraint, auto_constraint, state)
            ui.label(status_text).classes("text-xs").bind_text_from(
                state, 
                "_constraint_status",
                backward=lambda s, f=feature: s.get(f, "未設定")
            )


def _render_batch_operations(state: dict):
    """一括操作セクション"""
    
    with ui.row().classes("w-full q-gutter-sm justify-end"):
        # リセットボタン
        ui.button(
            icon="refresh",
            text="すべてリセット",
            on_click=lambda: _reset_all_constraints(state),
            color="negative"
        ).props("outline")
        
        # 自動提案適用ボタン
        ui.button(
            icon="auto_fix_normal",
            text="自動提案を適用",
            on_click=lambda: _apply_all_auto_constraints(state),
            color="secondary"
        ).props("outline")
        
        # 保存ボタン
        ui.button(
            icon="save",
            text="設定を保存",
            on_click=lambda: _save_constraints(state),
            color="positive"
        ).props("unelevated")


def _render_auto_constraint_chip(feature: str, constraint: str, state: dict):
    """自動提案チップをレンダリング"""
    
    color_map = {
        "increasing": "green",
        "decreasing": "red",
        "none": "grey"
    }
    
    icon_map = {
        "increasing": "arrow_upward",
        "decreasing": "arrow_downward",
        "none": "remove"
    }
    
    label_map = {
        "increasing": "増加 ↗",
        "decreasing": "減少 ↘",
        "none": "制約なし"
    }
    
    color = color_map.get(constraint, "grey")
    icon = icon_map.get(constraint, "remove")
    label = label_map.get(constraint, "unknown")
    
    rationale = _get_chemical_rationale(feature)
    
    chip = ui.chip(
        text=f"{feature}: {label}",
        icon=icon,
        color=color
    ).props(f"outline clickable tooltip=\"{rationale}\"")
    
    # クリックで手動設定にコピー
    chip.on("click", lambda f=feature, c=constraint: _copy_auto_to_manual(f, c, state))


# ============================================================================
# ヘルパー関数
# ============================================================================

def _detect_features(state: dict) -> List[str]:
    """データから特徴量を動的に検出"""
    
    features = []
    
    # 1. column_rolesから説明変数を取得
    if "column_roles" in state:
        for col, role in state["column_roles"].items():
            if role == "feature":
                features.append(col)
    
    # 2. SMILESから生成された特徴量も含める
    if "generated_smiles_features" in state:
        for feat in state["generated_smiles_features"]:
            if feat not in features:
                features.append(feat)
    
    # 3. データフレームから数値列を補完
    if "df" in state and state["df"] is not None:
        df = state["df"]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            if col not in features and col != state.get("target_col"):
                features.append(col)
    
    return sorted(features)


def _get_auto_constraints(state: dict) -> Dict[str, str]:
    """自動提案された制約を取得"""
    
    # 化学的知見に基づくデフォルト提案
    auto_constraints = {}
    
    # 一般的な化学的知見
    chemical_knowledge = {
        "MW": "decreasing",
        "MolecularWeight": "decreasing",
        "LogP": "decreasing",
        "TPSA": "increasing",
        "HBA": "increasing",
        "NumHAcceptors": "increasing",
        "HBD": "increasing",
        "NumHDonors": "increasing",
        "RingCount": "decreasing",
        "NumRings": "decreasing",
        "MolecularComplexity": "decreasing",
        "NumRotatableBonds": "decreasing",
    }
    
    # データの特徴量とマッチング
    features = _detect_features(state)
    for feature in features:
        for known_feat, constraint in chemical_knowledge.items():
            if known_feat.lower() in feature.lower() or feature.lower() in known_feat.lower():
                auto_constraints[feature] = constraint
                break
    
    # stateに保存
    state["auto_monotonicity_constraints"] = auto_constraints
    
    return auto_constraints


def _get_auto_constraint_for_feature(feature: str, state: dict) -> str:
    """特定の特徴量の自動提案を取得"""
    auto_constraints = _get_auto_constraints(state)
    return auto_constraints.get(feature, "none")


def _get_current_constraint(feature: str, state: dict) -> str:
    """現在の手動設定を取得"""
    manual_constraints = state.get("manual_monotonicity_constraints", {})
    return manual_constraints.get(feature, "auto")


def _set_constraint(feature: str, constraint: str, state: dict):
    """制約を設定"""
    
    if "manual_monotonicity_constraints" not in state:
        state["manual_monotonicity_constraints"] = {}
    
    state["manual_monotonicity_constraints"][feature] = constraint
    
    # ステータス更新
    if "_constraint_status" not in state:
        state["_constraint_status"] = {}
    
    constraint_labels = {
        "none": "制約なし",
        "increasing": "単調増加 ↗",
        "decreasing": "単調減少 ↘",
        "auto": "自動提案"
    }
    
    state["_constraint_status"][feature] = constraint_labels.get(constraint, "未設定")
    
    # 通知
    ui.notify(
        f"'{feature}' を {constraint_labels[constraint]} に設定",
        color="positive",
        timeout=1500
    )
    
    # UIを更新
    render_monotonicity_constraint_panel.refresh(state)


def _get_chemical_rationale(feature: str) -> str:
    """化学的根拠を取得"""
    
    rationales = {
        "MW": "分子量↑ → 分子間相互作用↑ → 溶解度↓",
        "MolecularWeight": "分子量↑ → 分子間相互作用↑ → 溶解度↓",
        "LogP": "疎水性↑ → 水への溶解度↓",
        "TPSA": "極性表面積↑ → 水との相互作用↑ → 溶解度↑",
        "HBA": "水素結合受容体↑ → 水素結合↑ → 溶解度↑",
        "NumHAcceptors": "水素結合受容体↑ → 水素結合↑ → 溶解度↑",
        "HBD": "水素結合供与体↑ → 水素結合↑ → 溶解度↑",
        "NumHDonors": "水素結合供与体↑ → 水素結合↑ → 溶解度↑",
        "RingCount": "環数↑ → 剛性↑ → 溶解度↓",
        "NumRings": "環数↑ → 剛性↑ → 溶解度↓",
        "MolecularComplexity": "複雑さ↑ → 溶解度↓",
        "NumRotatableBonds": "柔軟性↑ → 溶解度↓（一般的）",
    }
    
    # 部分一致で検索
    feature_lower = feature.lower()
    for key, rationale in rationales.items():
        if key.lower() in feature_lower or feature_lower in key.lower():
            return rationale
    
    return ""


def _constraint_to_label(constraint: str) -> str:
    """制約値をラベルに変換"""
    labels = {
        "increasing": "増加 ↗",
        "decreasing": "減少 ↘",
        "none": "制約なし",
        "auto": "自動"
    }
    return labels.get(constraint, "不明")


def _highlight_current_selection(current: str, button_map: Dict[str, ui.button]):
    """現在の選択状況に合わせてボタンのスタイルを調整"""
    for val, btn in button_map.items():
        if val == current:
            # 選択中のボタンは塗りつぶし
            btn.props(remove="outline")
            btn.props(add="unelevated")
        else:
            # 非選択はアウトライン
            btn.props(remove="unelevated")
            btn.props(add="outline")


def _get_status_text(feature: str, current: str, auto: str, state: dict) -> str:
    """ステータステキストを取得"""
    
    if current == "auto":
        return f"自動: {_constraint_to_label(auto)}"
    else:
        return f"手動: {_constraint_to_label(current)}"


def _on_auto_toggle(value: bool, state: dict):
    """自動適用トグルのハンドラ"""
    state["auto_apply_constraints"] = value
    ui.notify(
        f"自動提案を{'有効' if value else '無効'}にしました",
        color="info"
    )
    render_monotonicity_constraint_panel.refresh(state)


def _copy_auto_to_manual(feature: str, constraint: str, state: dict):
    """自動提案を手動設定にコピー"""
    _set_constraint(feature, constraint, state)
    ui.notify(f"'{feature}' の制約を手動設定にコピー", color="positive")


def _apply_all_auto_constraints(state: dict):
    """すべての自動提案を適用"""
    
    auto_constraints = _get_auto_constraints(state)
    
    for feature, constraint in auto_constraints.items():
        _set_constraint(feature, constraint, state)
    
    ui.notify(
        f"{len(auto_constraints)}個の制約を適用しました",
        color="positive"
    )


def _reset_all_constraints(state: dict):
    """すべての制約をリセット"""
    
    if "manual_monotonicity_constraints" in state:
        state["manual_monotonicity_constraints"] = {}
    
    if "_constraint_status" in state:
        state["_constraint_status"] = {}
    
    ui.notify("すべての制約をリセットしました", color="info")
    render_monotonicity_constraint_panel.refresh(state)


def _save_constraints(state: dict):
    """制約設定を保存"""
    
    manual = state.get("manual_monotonicity_constraints", {})
    auto = state.get("auto_monotonicity_constraints", {})
    
    # 最終的な制約を確定
    final_constraints = {}
    
    features = _detect_features(state)
    for feature in features:
        manual_val = manual.get(feature, "auto")
        auto_val = auto.get(feature, "none")
        
        # 手動設定が"auto"または未設定の場合は自動提案を使用
        if manual_val == "auto" or manual_val not in ["none", "increasing", "decreasing"]:
            final_constraints[feature] = auto_val
        else:
            final_constraints[feature] = manual_val
    
    # stateに保存
    state["monotonicity_constraints"] = final_constraints
    
    ui.notify(
        f"{len(final_constraints)}個の制約設定を保存しました",
        color="positive",
        timeout=3000
    )
    
    # デバッグ用出力
    print("💾 保存された単調性制約:")
    for feat, const in final_constraints.items():
        print(f"  {feat}: {const}")
