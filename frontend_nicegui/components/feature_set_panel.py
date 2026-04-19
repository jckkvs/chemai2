"""
frontend_nicegui/components/feature_set_panel.py

SMILES特徴量セット（Feature Set）の管理用UI。
ユーザーが複数の特徴量セット（基本、高度、量子力学など）を定義し、計算を実行できる。
"""
from nicegui import ui
from typing import Dict, Any, List

from backend.chem import ADAPTER_REGISTRY

def render_feature_set_panel(state: dict):
    """
    特徴量セット管理パネルのレンダリング
    """
    ui.label("🧪 SMILES特徴量セット管理").classes("text-xl font-bold q-mb-md")
    
    if "feature_sets" not in state:
        state["feature_sets"] = {}
        # 初期セット
        state["feature_sets"]["set1"] = {
            "name": "基本記述子セット",
            "engines": {"RDKit": {}},
            "status": "idle"
        }

    # 1. セット追加用UI
    with ui.row().classes("w-full items-center q-mb-md"):
        new_set_id = ui.input(label="新しいセットID", placeholder="set2").classes("w-32")
        new_set_name = ui.input(label="セット名", placeholder="量子化学セット").classes("flex-grow")
        ui.button(
            "セットを追加", 
            icon="add",
            on_click=lambda: _add_set(state, new_set_id.value, new_set_name.value)
        ).props("unelevated")

    ui.separator().classes("q-my-md")

    # 2. 既存セットの表示
    if not state["feature_sets"]:
        ui.label("特徴量セットがありません。上のフォームから追加してください。").classes("text-grey italic")
        return

    for set_id, s_info in state["feature_sets"].items():
        with ui.expansion(f"📦 {s_info['name']} ({set_id})", icon="science").classes("w-full border rounded-lg q-mb-sm"):
            with ui.column().classes("w-full p-4"):
                # エンジン選択
                ui.label("構成エンジン:").classes("font-semibold")
                with ui.row().classes("flex-wrap q-gutter-sm"):
                    for engine_name in ADAPTER_REGISTRY.keys():
                        enabled = engine_name in s_info["engines"]
                        ui.checkbox(
                            engine_name, 
                            value=enabled,
                            on_change=lambda e, sid=set_id, en=engine_name: _toggle_engine(state, sid, en, e.value)
                        )
                
                ui.separator().classes("q-my-sm")
                
                # 操作ボタン
                with ui.row().classes("w-full justify-between items-center"):
                    with ui.row().classes("q-gutter-sm"):
                        ui.button(
                            "計算実行", 
                            icon="play_arrow", 
                            color="positive",
                            on_click=lambda sid=set_id: _run_calculation(state, sid)
                        ).props("unelevated")
                        
                        status_label = ui.label(f"ステータス: {s_info.get('status', '未計算')}")
                        if "success_rate" in s_info:
                            ui.label(f"(成功率: {s_info['success_rate']:.1%})").classes("text-sm text-grey-7")
                    
                    ui.button(
                        "セットを削除", 
                        icon="delete", 
                        color="negative",
                        on_click=lambda sid=set_id: _remove_set(state, sid)
                    ).props("flat")

def _add_set(state: dict, set_id: str, name: str):
    if not set_id or not name:
        ui.notify("IDと名前を入力してください", type="warning")
        return
    if set_id in state["feature_sets"]:
        ui.notify("そのIDは既に使用されています", type="negative")
        return
    
    state["feature_sets"][set_id] = {
        "name": name,
        "engines": {"RDKit": {}},
        "status": "idle"
    }
    ui.notify(f"セット '{name}' を追加しました", type="positive")
    ui.update()

def _remove_set(state: dict, set_id: str):
    if set_id in state["feature_sets"]:
        del state["feature_sets"][set_id]
        ui.notify(f"セット '{set_id}' を削除しました")
        ui.update()

def _toggle_engine(state: dict, set_id: str, engine_name: str, enabled: bool):
    s_info = state["feature_sets"].get(set_id)
    if not s_info: return
    
    if enabled:
        s_info["engines"][engine_name] = {}
    else:
        if engine_name in s_info["engines"]:
            del s_info["engines"][engine_name]

async def _run_calculation(state: dict, set_id: str):
    """
    バックエンドの SMILESFeatureSetManager を呼び出して計算を実行する
    """
    from backend.chem.smiles_feature_sets import SMILESFeatureSetManager
    
    s_info = state["feature_sets"].get(set_id)
    if not s_info: return
    
    smiles_col = state.get("smiles_col")
    if not smiles_col:
        ui.notify("SMILES列が選択されていません", type="negative")
        return
    
    df = state.get("df")
    if df is None:
        ui.notify("データがロードされていません", type="negative")
        return
        
    smiles_list = df[smiles_col].dropna().unique().tolist()
    
    ui.notify(f"セット '{set_id}' の計算を開始します...", type="info")
    s_info["status"] = "running"
    ui.update()
    
    try:
        # マネージャーを state から取得または新規作成
        if "_feature_set_manager" not in state:
            state["_feature_set_manager"] = SMILESFeatureSetManager()
        manager = state["_feature_set_manager"]

        # マネージャーにセット情報を同期（UI上の変更を反映）
        fset = manager.create_set(set_id, s_info["name"])
        fset.enabled_engines = s_info["engines"]
        
        # 計算実行
        result_df = await manager.calculate_set(set_id, smiles_list)
        
        if result_df is not None:
            s_info["status"] = "finished"
            s_info["success_rate"] = manager.get_set(set_id).success_rate
            s_info["features"] = result_df.columns.tolist()
            
            # --- 🚀 統合 precalc_df へのマージ ---
            if "precalc_df" not in state or state["precalc_df"] is None:
                # インデックスを元データの物と合わせる必要がある
                # result_df は smiles_list 順。元データで SMILES が最初に出現する行に合わせるか、
                # 単純に全行分を再構築するか。
                # DataMerger を使って全行分に拡張したものを生成するのが最も安全。
                from backend.data.data_merger import DataMerger
                merger = DataMerger(df)
                merged_res = merger.merge(smiles_col=smiles_col, feature_set_configs=[{
                    "id": set_id, "name": s_info["name"], "engines": s_info["engines"]
                }])
                state["precalc_df"] = merged_res.df
            else:
                # 既存の precalc_df に新しいセットをマージ
                # すでに存在する列は上書きするかスキップ
                new_cols = [c for c in result_df.columns if c not in state["precalc_df"].columns]
                if new_cols:
                    from backend.data.data_merger import DataMerger
                    merger = DataMerger(df)
                    # 全行対応の DataFrame を生成
                    merged_res = merger.merge(smiles_col=smiles_col, feature_set_configs=[{
                        "id": set_id, "name": s_info["name"], "engines": s_info["engines"]
                    }])
                    # 既に precalc_df がある場合は join
                    state["precalc_df"] = state["precalc_df"].combine_first(merged_res.df)
            
            # グローバル状態に列名をキャッシュ（グルーピング用）
            all_feats = set(state.get("generated_smiles_features", []))
            all_feats.update(result_df.columns.tolist())
            state["generated_smiles_features"] = sorted(list(all_feats))
            
            ui.notify(f"計算完了: {len(result_df.columns)}個の記述子を生成しました", type="positive")
            
            # 再描画をトリガー（DataTab 全体の再構築が必要な場合がある）
            state["_data_version"] = state.get("_data_version", 0) + 1
        else:
            s_info["status"] = "failed"
            ui.notify("計算に失敗しました。エンジン設定を確認してください。", type="negative")
    except Exception as e:
        logger.error(f"Feature set calculation error: {e}")
        s_info["status"] = "error"
        ui.notify(f"エラーが発生しました: {str(e)}", type="negative")
    
    ui.update()
