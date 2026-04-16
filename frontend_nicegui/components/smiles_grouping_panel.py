"""
SMILES特徴量グルーピング設定パネル
"""
from nicegui import ui
from typing import Dict, List, Optional


def render_smiles_grouping_panel(state: dict):
    """
    SMILESから生成された特徴量のグルーピング設定
    
    化学的意味に基づいて記述子をグループ化し、グループ単位で標準化・選択を行う
    """
    
    ui.label("📦 SMILES特徴量グルーピング").classes("text-xl font-bold q-mb-md")
    
    ui.markdown("""
    SMILESから生成された分子記述子を、化学的意味に基づいてグループ化します。
    
    **推奨グループ例**:
    - `basic_properties`: MW, LogP, TPSA（基本物性）
    - `hydrogen_bonding`: HBA, HBD（水素結合関連）
    - `ring_systems`: RingCount, NumAromaticRings（環系関連）
    - `complexity`: MolecularComplexity, FractionCSP3（複雑さ関連）
    
    グループ化により、物理的に整合性のある標準化と特徴量選択が可能になります。
    """).classes("text-sm text-grey-7 q-mb-md")
    
    # 生成済み特徴量の確認
    generated_features = state.get("generated_smiles_features", [])
    
    # もし空なら、既存の feature_sets から取得を試みる
    if not generated_features and "feature_sets" in state:
        all_feats = set()
        for sinfo in state["feature_sets"].values():
            all_feats.update(sinfo.get("features", []))
        generated_features = list(all_feats)

    if not generated_features:
        ui.info("まず「特徴量生成」タブでSMILES記述子を生成してください")
        return
    
    # 状態初期化
    if "smiles_feature_groups" not in state:
        state["smiles_feature_groups"] = {}
    if "smiles_group_scale_method" not in state:
        state["smiles_group_scale_method"] = "individual"
    
    # ─────────────────────────────────────────────
    # 推奨テンプレート
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("🎯 推奨テンプレート").classes("text-lg font-semibold q-mb-sm")
        
        with ui.row().classes("w-full q-gutter-sm"):
            ui.button(
                "🧪 基本物性セット",
                on_click=lambda: _apply_basic_properties_template(generated_features, state),
                color="primary"
            ).props("outline")
            
            ui.button(
                "💧 水素結合セット",
                on_click=lambda: _apply_hydrogen_bonding_template(generated_features, state),
                color="secondary"
            ).props("outline")
            
            ui.button(
                "🔗 環系セット",
                on_click=lambda: _apply_ring_systems_template(generated_features, state),
                color="secondary"
            ).props("outline")
            
            ui.button(
                "🧩 全特徴量を個別",
                on_click=lambda: _clear_smiles_groups(state),
                color="grey"
            ).props("outline")
        
        ui.markdown("""
        上記ボタンをクリックすると、化学的に意味のあるグループ定義を自動適用します。
        手動でカスタマイズも可能です。
        """).classes("text-xs text-grey-7 q-mt-sm")
    
    # ─────────────────────────────────────────────
    # 既存グループの表示
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("📋 定義済みグループ").classes("text-lg font-semibold q-mb-sm")
        
        if state["smiles_feature_groups"]:
            with ui.grid(columns=2).classes("w-full q-gutter-md"):
                for group_name, features in state["smiles_feature_groups"].items():
                    with ui.card().classes("w-full bg-purple-50 dark:bg-purple-900/20"):
                        with ui.row().classes("w-full items-center"):
                            ui.label(f"🧪 {group_name}").classes("font-bold flex-grow")
                            ui.chip(f"{len(features)}個", color="purple").props("outline")
                            ui.button(
                                icon="edit",
                                on_click=lambda gn=group_name: _edit_smiles_group_dialog(gn, state, generated_features)
                            ).props("flat dense")
                            ui.button(
                                icon="delete",
                                on_click=lambda gn=group_name: _remove_smiles_group(gn, state)
                            ).props("flat dense color=negative")
                        
                        with ui.row().classes("q-gutter-xs q-ml-md flex-wrap"):
                            for feat in features:
                                ui.chip(feat, color="grey").props("outline size=sm")
        else:
            ui.label("グループが定義されていません。テンプレートを使用するか手動で作成してください。").classes("text-grey")
    
    # ─────────────────────────────────────────────
    # 手動グループ作成
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("➕ 手動でグループ作成").classes("text-lg font-semibold q-mb-sm")
        
        with ui.row().classes("w-full q-gutter-md"):
            group_name = ui.input(
                label="グループ名",
                placeholder="例: basic_properties, ring_systems"
            ).classes("w-64")
            
            feature_select = ui.select(
                options=generated_features,
                label="SMILES特徴量を選択（複数可）",
                multiple=True,
                with_chips=True
            ).classes("flex-grow")
            
            ui.button(
                "グループを追加",
                on_click=lambda: _add_smiles_group(group_name.value, feature_select.value, state),
                color="primary"
            ).props("unelevated")
    
    # ─────────────────────────────────────────────
    # スケーリング設定
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("⚙️ SMILES特徴量の標準化設定").classes("text-lg font-semibold q-mb-sm")
        
        scale_method = ui.select(
            options=[
                ("individual", "個別に標準化（各記述子ごとに独立）"),
                ("global_max", "全体最大標準偏差で統一"),
                ("none", "スケーリングしない")
            ],
            label="グループ外の特徴量の処理",
            value=state.get("smiles_group_scale_method", "individual")
        ).classes("w-full")
        
        def on_method_change(e):
            state["smiles_group_scale_method"] = e.value
        
        scale_method.on("change", on_method_change)
    
    # ─────────────────────────────────────────────
    # GroupLASSO設定
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("🎯 SMILES特徴量のGroupLASSO").classes("text-lg font-semibold q-mb-sm")
        
        use_group_lasso = ui.checkbox(
            "SMILES特徴量でGroupLASSOを使用",
            value=state.get("use_smiles_group_lasso", False)
        ).classes("q-mb-sm")
        
        with ui.row().classes("w-full q-gutter-md").bind_visibility_from(use_group_lasso, 'value'):
            alpha = ui.number(
                label="正則化パラメータ α",
                value=state.get("smiles_group_lasso_alpha", 1.0),
                min=0.001,
                max=1000.0,
                step=0.1
            ).classes("w-48")
            
            def on_alpha_change(e):
                state["smiles_group_lasso_alpha"] = e.value
            
            alpha.on("change", on_alpha_change)
        
        def on_lasso_toggle(e):
            state["use_smiles_group_lasso"] = e.value
        
        use_group_lasso.on("change", on_lasso_toggle)
    
    # ─────────────────────────────────────────────
    # 適用ボタン
    # ─────────────────────────────────────────────
    if state["smiles_feature_groups"]:
        with ui.card().classes("w-full"):
            ui.label("📊 設定サマリー").classes("text-lg font-semibold q-mb-sm")
            
            with ui.grid(columns=3).classes("w-full q-gutter-md"):
                ui.label(f"定義済みグループ: {len(state['smiles_feature_groups'])}個").classes("text-sm")
                ui.label(f"スケーリング: {state.get('smiles_group_scale_method', 'individual')}").classes("text-sm")
                ui.label(f"GroupLASSO: {'有効' if state.get('use_smiles_group_lasso', False) else '無効'}").classes("text-sm")
            
            ui.button(
                "✅ SMILES特徴量設定を保存",
                on_click=lambda: _apply_smiles_group_settings(state),
                color="positive"
            ).props("unelevated").classes("q-mt-md")


# ============================================================
# SMILES特徴量用テンプレート
# ============================================================

def _apply_basic_properties_template(features: List[str], state: dict):
    """基本物性テンプレートを適用"""
    basic_keywords = ["MW", "MolecularWeight", "LogP", "TPSA", "MolLogP", "MolTPSA"]
    basic_features = [f for f in features if any(kw in f for kw in basic_keywords)]
    
    if basic_features:
        if "smiles_feature_groups" not in state:
            state["smiles_feature_groups"] = {}
        state["smiles_feature_groups"]["basic_properties"] = basic_features
        ui.notify(f"基本物性グループを作成: {', '.join(basic_features)}", color="positive")


def _apply_hydrogen_bonding_template(features: List[str], state: dict):
    """水素結合関連テンプレートを適用"""
    hb_keywords = ["HBA", "HBD", "NumHAcceptors", "NumHDonors", "NumHAcceptors", "NumHDonors"]
    hb_features = [f for f in features if any(kw in f for kw in hb_keywords)]
    
    if hb_features:
        if "smiles_feature_groups" not in state:
            state["smiles_feature_groups"] = {}
        state["smiles_feature_groups"]["hydrogen_bonding"] = hb_features
        ui.notify(f"水素結合グループを作成: {', '.join(hb_features)}", color="positive")


def _apply_ring_systems_template(features: List[str], state: dict):
    """環系関連テンプレートを適用"""
    ring_keywords = ["Ring", "Aromatic", "Aliphatic", "Saturated", "Heterocycle"]
    ring_features = [f for f in features if any(kw in f for kw in ring_keywords)]
    
    if ring_features:
        if "smiles_feature_groups" not in state:
            state["smiles_feature_groups"] = {}
        state["smiles_feature_groups"]["ring_systems"] = ring_features
        ui.notify(f"環系グループを作成: {', '.join(ring_features)}", color="positive")


# ============================================================
# SMILES特徴量用ヘルパー関数
# ============================================================

def _add_smiles_group(group_name: str, features: List[str], state: dict):
    """SMILES特徴量グループを追加"""
    if not group_name or not group_name.strip():
        ui.error("グループ名を入力してください")
        return
    
    if not features:
        ui.error("特徴量を1つ以上選択してください")
        return
    
    if "smiles_feature_groups" not in state:
        state["smiles_feature_groups"] = {}
    
    if group_name in state["smiles_feature_groups"]:
        ui.warning(f"グループ '{group_name}' は既に存在します")
        return
    
    state["smiles_feature_groups"][group_name] = features
    ui.notify(f"SMILES特徴量グループ '{group_name}' を作成", color="positive")


def _edit_smiles_group_dialog(group_name: str, state: dict, available_features: List[str]):
    """SMILES特徴量グループ編集ダイアログ"""
    current_features = state.get("smiles_feature_groups", {}).get(group_name, [])
    
    with ui.dialog() as dlg, ui.card().classes("w-96"):
        ui.label(f"✏️ '{group_name}' を編集").classes("text-lg font-bold q-mb-md")
        
        new_name = ui.input(label="グループ名", value=group_name).classes("w-full")
        
        feature_select = ui.select(
            options=available_features,
            label="特徴量を選択",
            multiple=True,
            with_chips=True,
            value=current_features
        ).classes("w-full")
        
        with ui.row().classes("w-full justify-end q-mt-md"):
            ui.button("キャンセル", on_click=dlg.close).props("flat")
            ui.button("削除", on_click=lambda: _remove_smiles_group(group_name, state, dlg), color="negative").props("outline")
            ui.button("保存", on_click=lambda: _update_smiles_group(group_name, new_name.value, feature_select.value, state, dlg), color="primary").props("unelevated")
    
    dlg.open()


def _update_smiles_group(old_name: str, new_name: str, features: List[str], state: dict, dialog=None):
    """SMILES特徴量グループを更新"""
    if not new_name or not features:
        ui.error("グループ名と特徴量を入力してください")
        return
    
    if "smiles_feature_groups" not in state:
        state["smiles_feature_groups"] = {}
    
    if old_name != new_name and new_name in state["smiles_feature_groups"]:
        ui.warning(f"グループ '{new_name}' は既に存在します")
        return
    
    if old_name != new_name:
        del state["smiles_feature_groups"][old_name]
    
    state["smiles_feature_groups"][new_name] = features
    ui.notify(f"グループ '{new_name}' を更新", color="positive")
    
    if dialog:
        dialog.close()


def _remove_smiles_group(group_name: str, state: dict, dialog=None):
    """SMILES特徴量グループを削除"""
    if "smiles_feature_groups" in state and group_name in state["smiles_feature_groups"]:
        del state["smiles_feature_groups"][group_name]
        ui.notify(f"グループ '{group_name}' を削除", color="info")
    
    if dialog:
        dialog.close()


def _clear_smiles_groups(state: dict):
    """SMILES特徴量グループをすべてクリア"""
    state["smiles_feature_groups"] = {}
    ui.notify("SMILES特徴量グループをクリアしました", color="info")


def _apply_smiles_group_settings(state: dict):
    """SMILES特徴量設定を保存"""
    if not state.get("smiles_feature_groups"):
        ui.warning("SMILES特徴量グループが定義されていません")
        return
    
    ui.notify(
        f"SMILES特徴量設定を保存しました（{len(state['smiles_feature_groups'])}グループ）",
        color="positive",
        timeout=3000
    )
