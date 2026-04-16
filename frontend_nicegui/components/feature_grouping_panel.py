"""
特徴量グルーピング設定パネル（表データ用）
"""
from nicegui import ui
from typing import Dict, List, Optional
import pandas as pd


def render_feature_grouping_panel(state: dict):
    """
    表データの特徴量グルーピング設定
    
    同じ物理量の特徴量をグループ化し、グループ単位で標準化・特徴量選択を行う
    """
    
    ui.label("📦 特徴量グルーピング").classes("text-xl font-bold q-mb-md")
    
    ui.markdown("""
    関連する特徴量をグループ化することで、以下の利点があります：
    
    - **標準化の整合性**: 同じ温度センサー群などが同じスケールで扱われる
    - **GroupLASSO対応**: グループ単位で特徴量を選択可能
    - **モデル解釈性向上**: 物理的に意味のある単位で特徴量を管理
    
    **例**: `Temp_A`, `Temp_B`, `Temp_C` → `temperature` グループ
    """).classes("text-sm text-grey-7 q-mb-md")
    
    if "df" not in state or state["df"] is None:
        ui.warning("先にデータを読み込んでください")
        return
    
    df = state["df"]
    column_roles = state.get("column_roles", {})
    
    # 説明変数のみ対象
    feature_cols = [col for col, role in column_roles.items() if role == "feature"]
    
    # もし column_roles が空なら、既存の target_col/exclude_cols から推測（互換性のため）
    if not feature_cols:
        target_col = state.get("target_col")
        exclude_cols = state.get("exclude_cols", [])
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
        
    numeric_cols = [col for col in feature_cols if col in df.select_dtypes(include=['number']).columns]
    
    if not numeric_cols:
        ui.info("数値説明変数が見つかりません")
        return
    
    # 状態初期化
    if "feature_groups" not in state:
        state["feature_groups"] = {}
    if "group_scale_method" not in state:
        state["group_scale_method"] = "individual"
    
    # ─────────────────────────────────────────────
    # 既存グループの表示
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("📋 定義済みグループ").classes("text-lg font-semibold q-mb-sm")
        
        if state["feature_groups"]:
            with ui.grid(columns=2).classes("w-full q-gutter-md"):
                for group_name, features in state["feature_groups"].items():
                    with ui.card().classes("w-full bg-blue-50 dark:bg-blue-900/20"):
                        with ui.row().classes("w-full items-center"):
                            ui.label(f"📁 {group_name}").classes("font-bold flex-grow")
                            ui.chip(f"{len(features)}個", color="blue").props("outline")
                            ui.button(
                                icon="edit",
                                on_click=lambda gn=group_name: _edit_group_dialog(gn, state, numeric_cols)
                            ).props("flat dense")
                            ui.button(
                                icon="delete",
                                on_click=lambda gn=group_name: _remove_group(gn, state)
                            ).props("flat dense color=negative")
                        
                        with ui.row().classes("q-gutter-xs q-ml-md flex-wrap"):
                            for feat in features:
                                ui.chip(feat, color="grey").props("outline size=sm")
        else:
            ui.label("グループが定義されていません").classes("text-grey")
            ui.button(
                "➕ 最初のグループを作成",
                on_click=lambda: _create_group_dialog(state, numeric_cols),
                color="primary"
            ).props("unelevated").classes("q-mt-md")
    
    # ─────────────────────────────────────────────
    # 自動検出ボタン
    # ─────────────────────────────────────────────
    with ui.row().classes("w-full q-gutter-sm q-mb-md"):
        ui.button(
            "🔍 自動検出（温度/圧力/pH/濃度）",
            on_click=lambda: _auto_detect_groups(df, numeric_cols, state),
            color="secondary"
        ).props("outline")
        
        ui.button(
            "📦 全特徴量を1グループ",
            on_click=lambda: _create_single_group(numeric_cols, state),
            color="secondary"
        ).props("outline")
        
        ui.button(
            "🗑️ すべてクリア",
            on_click=lambda: _clear_all_groups(state),
            color="negative"
        ).props("outline")
    
    # ─────────────────────────────────────────────
    # グループ外の特徴量の処理方法
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("⚙️ グループ外の特徴量の処理").classes("text-lg font-semibold q-mb-sm")
        
        scale_method = ui.select(
            options=[
                ("individual", "個別に標準化（各特徴量ごとに独立）"),
                ("global_max", "全体最大標準偏差で統一"),
                ("none", "スケーリングしない（元の値を維持）")
            ],
            label="処理方法",
            value=state.get("group_scale_method", "individual")
        ).classes("w-full")
        
        def on_method_change(e):
            state["group_scale_method"] = e.value
            ui.notify(f"処理方法を '{e.label}' に変更", color="info", timeout=1500)
        
        scale_method.on("change", on_method_change)
        
        ui.markdown("""
        - **個別に標準化**: グループに属さない特徴量は独立に標準化（デフォルト）
        - **全体最大で統一**: 全特徴量の最大標準偏差で一括スケーリング（LASSO/RIDGE推奨）
        - **スケーリングしない**: 元の値を維持（既に正規化済みの場合など）
        """).classes("text-xs text-grey-7")
    
    # ─────────────────────────────────────────────
    # GroupLASSO設定
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("🎯 GroupLASSO設定").classes("text-lg font-semibold q-mb-sm")
        
        use_group_lasso = ui.checkbox(
            "GroupLASSOで特徴量選択を行う",
            value=state.get("use_group_lasso", False)
        ).classes("q-mb-sm")
        
        with ui.row().classes("w-full q-gutter-md").bind_visibility_from(use_group_lasso, 'value'):
            alpha = ui.number(
                label="正則化パラメータ α",
                value=state.get("group_lasso_alpha", 1.0),
                min=0.001,
                max=1000.0,
                step=0.1
            ).classes("w-48")
            
            def on_alpha_change(e):
                state["group_lasso_alpha"] = e.value
            
            alpha.on("change", on_alpha_change)
        
        def on_lasso_toggle(e):
            state["use_group_lasso"] = e.value
            ui.notify(
                f"GroupLASSOを{'有効' if e.value else '無効'}にしました",
                color="info"
            )
        
        use_group_lasso.on("change", on_lasso_toggle)
        
        ui.markdown("""
        **GroupLASSO**: 同一グループの特徴量をまとめて選択/除外する正則化回帰
        - αが大きいほどスパース（選択される特徴量が少ない）
        - 物理的に意味のある単位で特徴量を選択可能
        """).classes("text-xs text-grey-7")
    
    # ─────────────────────────────────────────────
    # プレビューと適用
    # ─────────────────────────────────────────────
    if state["feature_groups"]:
        with ui.card().classes("w-full"):
            ui.label("📊 設定サマリー").classes("text-lg font-semibold q-mb-sm")
            
            with ui.grid(columns=3).classes("w-full q-gutter-md"):
                ui.label(f"定義済みグループ: {len(state['feature_groups'])}個").classes("text-sm")
                ui.label(f"スケーリング方法: {state.get('group_scale_method', 'individual')}").classes("text-sm")
                ui.label(f"GroupLASSO: {'有効' if state.get('use_group_lasso', False) else '無効'}").classes("text-sm")
            
            ui.button(
                "✅ 設定を保存して機械学習タブへ",
                on_click=lambda: _apply_group_settings(state),
                color="positive"
            ).props("unelevated").classes("q-mt-md")


# ============================================================
# 内部ヘルパー関数
# ============================================================

def _create_group_dialog(state: dict, available_features: List[str]):
    """新規グループ作成ダイアログ"""
    
    with ui.dialog() as dlg, ui.card().classes("w-96"):
        ui.label("➕ 新規グループ作成").classes("text-lg font-bold q-mb-md")
        
        group_name = ui.input(
            label="グループ名",
            placeholder="例: temperature, pressure, concentration"
        ).classes("w-full")
        
        feature_select = ui.select(
            options=available_features,
            label="特徴量を選択（複数可）",
            multiple=True,
            with_chips=True
        ).classes("w-full")
        
        with ui.row().classes("w-full justify-end q-mt-md"):
            ui.button("キャンセル", on_click=dlg.close).props("flat")
            ui.button(
                "作成",
                on_click=lambda: _add_group(group_name.value, feature_select.value, state, dlg),
                color="primary"
            ).props("unelevated")
    
    dlg.open()


def _edit_group_dialog(group_name: str, state: dict, available_features: List[str]):
    """既存グループ編集ダイアログ"""
    
    current_features = state.get("feature_groups", {}).get(group_name, [])
    
    with ui.dialog() as dlg, ui.card().classes("w-96"):
        ui.label(f"✏️ '{group_name}' を編集").classes("text-lg font-bold q-mb-md")
        
        new_name = ui.input(
            label="グループ名",
            value=group_name
        ).classes("w-full")
        
        feature_select = ui.select(
            options=available_features,
            label="特徴量を選択（複数可）",
            multiple=True,
            with_chips=True,
            value=current_features
        ).classes("w-full")
        
        with ui.row().classes("w-full justify-end q-mt-md"):
            ui.button("キャンセル", on_click=dlg.close).props("flat")
            ui.button(
                "削除",
                on_click=lambda: _remove_group(group_name, state, dlg),
                color="negative"
            ).props("outline")
            ui.button(
                "保存",
                on_click=lambda: _update_group(group_name, new_name.value, feature_select.value, state, dlg),
                color="primary"
            ).props("unelevated")
    
    dlg.open()


def _add_group(group_name: str, features: List[str], state: dict, dialog=None):
    """グループを追加"""
    if not group_name or not group_name.strip():
        ui.error("グループ名を入力してください")
        return
    
    if not features:
        ui.error("特徴量を1つ以上選択してください")
        return
    
    if "feature_groups" not in state:
        state["feature_groups"] = {}
    
    # 重複チェック
    if group_name in state["feature_groups"]:
        ui.warning(f"グループ '{group_name}' は既に存在します")
        return
    
    state["feature_groups"][group_name] = features
    ui.notify(f"グループ '{group_name}' を作成（{len(features)}個の特徴量）", color="positive")
    
    if dialog:
        dialog.close()


def _update_group(old_name: str, new_name: str, features: List[str], state: dict, dialog=None):
    """グループを更新"""
    if not new_name or not new_name.strip():
        ui.error("グループ名を入力してください")
        return
    
    if not features:
        ui.error("特徴量を1つ以上選択してください")
        return
    
    if "feature_groups" not in state:
        state["feature_groups"] = {}
    
    # 名前変更の場合
    if old_name != new_name and new_name in state["feature_groups"]:
        ui.warning(f"グループ '{new_name}' は既に存在します")
        return
    
    if old_name != new_name:
        del state["feature_groups"][old_name]
    
    state["feature_groups"][new_name] = features
    ui.notify(f"グループ '{new_name}' を更新", color="positive")
    
    if dialog:
        dialog.close()


def _remove_group(group_name: str, state: dict, dialog=None):
    """グループを削除"""
    if "feature_groups" in state and group_name in state["feature_groups"]:
        del state["feature_groups"][group_name]
        ui.notify(f"グループ '{group_name}' を削除", color="info")
    
    if dialog:
        dialog.close()


def _auto_detect_groups(df: pd.DataFrame, numeric_cols: List[str], state: dict):
    """自動検出"""
    from backend.preprocessing.group_scaler import auto_detect_groups
    
    # 説明変数のみで検出
    feature_subset = df[numeric_cols]
    detected = auto_detect_groups(feature_subset, verbose=True)
    
    if not detected:
        ui.info("自動検出可能なグループが見つかりませんでした")
        return
    
    if "feature_groups" not in state:
        state["feature_groups"] = {}
    
    # 既存とマージ
    state["feature_groups"].update(detected)
    
    msg = f"{len(detected)}個のグループを自動検出: {', '.join(detected.keys())}"
    ui.notify(msg, color="positive")


def _create_single_group(features: List[str], state: dict):
    """全特徴量を1つのグループに"""
    if "feature_groups" not in state:
        state["feature_groups"] = {}
    
    state["feature_groups"]["all_features"] = features
    ui.notify(f"全{len(features)}個の特徴量を1つのグループに設定", color="positive")


def _clear_all_groups(state: dict):
    """すべてのグループをクリア"""
    state["feature_groups"] = {}
    ui.notify("すべてのグループ定義をクリアしました", color="info")


def _apply_group_settings(state: dict):
    """設定を保存して適用"""
    from backend.preprocessing.group_scaler import GroupStandardScaler
    
    if not state.get("feature_groups"):
        ui.warning("グループが定義されていません")
        return
    
    # スケーラーを作成（必要時に初期化）
    if "group_scaler" not in state or state.get("_group_scaler_dirty", True):
        scaler = GroupStandardScaler(
            feature_groups=state["feature_groups"],
            default_scale_method=state.get("group_scale_method", "individual")
        )
        state["group_scaler"] = scaler
        state["_group_scaler_dirty"] = False
    
    ui.notify(
        f"グループ標準化設定を保存しました（{len(state['feature_groups'])}グループ）",
        color="positive",
        timeout=3000
    )
    
    # 機械学習タブに移動（オプション）
    # ui.run_javascript("document.querySelector('[aria-label=\"機械学習\"]').click()")
