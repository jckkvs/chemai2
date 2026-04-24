"""
frontend_nicegui/components/data_tab.py
データ設定タブ：データ読込・列の役割設定・SMILES特徴量・EDA・パイプライン設計
全機能をサブタブで構造化。Progressive Disclosure で初心者/上級者を両立。
"""
from __future__ import annotations  # 修正: future → __future__
import io
import asyncio
import inspect
import importlib
import logging
from typing import Any
import numpy as np
import pandas as pd
from nicegui import ui
from frontend_nicegui.components.feature_comparison_dashboard import render_feature_comparison_dashboard
from frontend_nicegui.components.debug_samples_selector import create_debug_samples_selector

logger = logging.getLogger(__name__)  # 修正: name → __name__

# ─── サンプルSMILES ─────────────────────────────────
SAMPLE_SMILES = [
    "C", "CC", "CCC", "CCO", "CCN", "c1ccccc1", "c1ccccc1O",
    "CC(=O)O", "CC(C)C", "C1CCCCC1", "c1ccncc1", "c1ncncn1", "C1COCCO1",
    "CC(=O)OC", "CCOC", "CCOCC", "CC(O)CC", "c1ccc(Cl)cc1",
    "CC(=O)N", "CCCCCO", "c1ccc(F)cc1", "CC(C)=O", "OCCO",
    "CC(=O)CC", "CCCCO",
]

# ── 全エンジン定義 ──
_ALL_ENGINES: list[tuple[str, str, str, dict]] = [
    ("RDKit", "backend.chem.rdkit_adapter", "RDKitAdapter", {"compute_fp": False}),
    ("Mordred", "backend.chem.mordred_adapter", "MordredAdapter", {"selected_only": True}),
    ("GroupContrib", "backend.chem.group_contrib_adapter", "GroupContribAdapter", {}),
    ("DescriptaStorus", "backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter", {}),
    ("MolAI", "backend.chem.molai_adapter", "MolAIAdapter", {"n_components": 6}),
    ("scikit-FP", "backend.chem.skfp_adapter", "SkfpAdapter", {"fp_types": ["ECFP", "MACCS"]}),
    ("UMA", "backend.chem.uma_adapter", "UMAAdapter", {}),
    ("Mol2Vec", "backend.chem.mol2vec_adapter", "Mol2VecAdapter", {}),
    ("PaDEL", "backend.chem.padel_adapter", "PaDELAdapter", {}),
    ("Molfeat", "backend.chem.molfeat_adapter", "MolfeatAdapter", {}),
    ("XTB", "backend.chem.xtb_adapter", "XTBAdapter", {}),
    ("UniPKa", "backend.chem.unipka_adapter", "UniPkaAdapter", {}),
    ("COSMO-RS", "backend.chem.cosmo_adapter", "CosmoAdapter", {}),
    ("Chemprop", "backend.chem.chemprop_adapter", "ChempropAdapter", {}),
]

def render_data_tab(state: dict[str, Any]) -> None:
    """データ設定タブ全体を描画する。"""
    with ui.tabs().classes("full-width").props("dense active-color=cyan indicator-color=cyan") as sub_tabs:
        tab_load = ui.tab("load", label="📂 データ読込", icon="upload_file")
        tab_cols = ui.tab("columns", label="🏷️ 列の役割・単調性", icon="settings")
        tab_constraints = ui.tab("constraints", label="📐 制約設定", icon="rule")
        tab_smiles = ui.tab("smiles", label="⚗️ SMILES特徴量", icon="science")
        tab_mixture = ui.tab("mixture", label="🧪 混合物設定", icon="blender")
        tab_eda = ui.tab("eda", label="📊 EDA", icon="analytics")

    # ── @ui.refreshable を使った各タブの描画関数定義 ──
    @ui.refreshable
    def _tab_load_view():
        _render_data_load(state)

    @ui.refreshable
    def _tab_columns_view():
        from frontend_nicegui.components.column_role_panel import render_column_role_panel
        render_column_role_panel(state)

    @ui.refreshable
    def _tab_constraints_view():
        from frontend_nicegui.components.constraint_panel import render_constraint_panel
        render_constraint_panel(state)

    @ui.refreshable
    def _tab_eda_view():
        _render_eda(state)

    @ui.refreshable
    def _tab_pipeline_view():
        _render_pipeline(state)

    _refreshable_views = {
        "load": _tab_load_view,
        "columns": _tab_columns_view,
        "eda": _tab_eda_view,
    }

    # ── タブパネルを描画（全パネル即時描画）──
    with ui.tab_panels(sub_tabs, value=tab_load).classes("full-width"):
        with ui.tab_panel(tab_load):
            _tab_load_view()

        with ui.tab_panel(tab_cols):
            _tab_columns_view()

        with ui.tab_panel(tab_constraints):
            _tab_constraints_view()

        with ui.tab_panel(tab_smiles):
            _smiles_container = ui.column().classes("full-width")

            def _rebuild_smiles():
                """SMILESタブの内容をクリアして再描画する。"""
                _smiles_container.clear()
                with _smiles_container:
                    try:
                        from frontend_nicegui.components.smiles_feature_panel import render_smiles_feature_panel
                        render_smiles_feature_panel(state)
                    except Exception as _e:
                        logger.error(f"[DataTab] SMILES tab render error: {_e}", exc_info=True)
                        ui.label(f"⚠️ 表示エラー: {_e}").classes("text-red q-pa-md")

            _rebuild_smiles()

        with ui.tab_panel(tab_mixture):
            from frontend_nicegui.components.mixture_input_panel import render_mixture_panel
            render_mixture_panel(state)

        with ui.tab_panel(tab_eda):
            _tab_eda_view()

    # ── stateに再描画ヘルパーを登録 ──
    def _refresh_tabs_fn():
        """全サブタブを再描画する（loadタブ除く）。"""
        try:
            _rebuild_smiles()
            logger.debug("[DataTab] rebuilt smiles tab via container")
        except Exception as exc:
            logger.warning(f"[DataTab] smiles container rebuild failed: {exc}")
        
        for key, view_fn in _refreshable_views.items():
            if key != "load":
                try:
                    view_fn.refresh()
                    logger.debug(f"[DataTab] refreshed tab {key!r}")
                except Exception as exc:
                    logger.warning(f"[DataTab] refresh failed for {key!r}: {exc}")
        
        for refresh_key in ("_refresh_eda_main", "_refresh_inverse", "_refresh_results", "_refresh_doe"):
            fn = state.get(refresh_key)
            if callable(fn):
                try:
                    fn()
                    logger.debug(f"[DataTab] called {refresh_key}")
                except Exception as exc:
                    logger.warning(f"[DataTab] {refresh_key} failed: {exc}")

    state["_refresh_tabs"] = _refresh_tabs_fn

# =================================================================
# サブタブ1: データ読込
# =================================================================
def _render_data_load(state: dict) -> None:
    """ファイルアップロード + サンプル + ベンチマークのデータ読込UI"""
    df_existing = state.get("df")
    fn_existing = state.get("filename", "")
    
    if df_existing is not None and not df_existing.empty:
        status_text = f"✅ {fn_existing} ({len(df_existing)}行 × {len(df_existing.columns)}列)"
        upload_status = ui.label(status_text).classes("text-green q-mt-sm")
    else:
        upload_status = ui.label(" ").classes("text-grey-5 q-mt-sm")
    
    preview_container = ui.column().classes("full-width q-mt-md")

    if df_existing is not None and not df_existing.empty:
        _show_preview(df_existing, preview_container)

    async def handle_upload(e):
        """ファイルアップロードハンドラ - NiceGUI 3.x 完全対応版"""
        try:
            logger.info("=== ファイルアップロード開始 ===")
            
            filename = getattr(e, 'name', None)
            if not filename:
                filename = getattr(e, 'filename', 'uploaded_file.csv')
            
            content = None
            
            if hasattr(e, 'content'):
                c = e.content
                if inspect.iscoroutine(c):
                    logger.warning("e.content is coroutine, attempting to read differently")
                    c = None
                if c is not None and isinstance(c, (bytes, str)):
                    content = c
                    logger.info("✓ e.content (bytes/str) を使用")
            
            if content is None and hasattr(e, 'file'):
                f = e.file
                if isinstance(f, bytes):
                    content = f
                    logger.info("✓ e.file (bytes) を使用")
                elif hasattr(f, 'read'):
                    read_result = f.read()
                    if inspect.iscoroutine(read_result):
                        logger.warning("f.read() returned coroutine - this is unexpected")
                        content = str(f)
                    else:
                        content = read_result
                        logger.info("✓ e.file.read() を使用")
                else:
                    content = f
                    logger.info("✓ e.file (fallback) を使用")
            
            if content is None and isinstance(e, bytes):
                content = e
                logger.info("✓ Event 自体が bytes")
            
            if content is None:
                logger.error(f"✗ ファイルコンテンツを取得できませんでした。Event type: {type(e)}")
                ui.notify('✗ ファイルの読み取りに失敗しました', type='negative')
                return
            
            logger.info(f"ファイル名: {filename}, コンテンツ型: {type(content)}, サイズ: {len(content) if isinstance(content, (bytes, str)) else 'N/A'}")
            
            try:
                if filename.endswith('.csv'):
                    if isinstance(content, bytes):
                        df_loaded = pd.read_csv(io.BytesIO(content), float_precision='high')
                    else:
                        df_loaded = pd.read_csv(io.StringIO(content), float_precision='high')
                elif filename.endswith(('.xlsx', '.xls')):
                    if isinstance(content, bytes):
                        df_loaded = pd.read_excel(io.BytesIO(content))
                    else:
                        ui.notify('Excel ファイルは bytes 形式が必要です', type='warning')
                        return
                else:
                    upload_status.text = "❌ CSV/Excelファイルのみ対応"
                    ui.notify('サポートされていないファイル形式です', type='warning')
                    return
            except Exception as parse_err:
                logger.error(f"CSV/Excel パースエラー: {parse_err}", exc_info=True)
                ui.notify(f'ファイル形式エラー: {str(parse_err)[:100]}', type='negative')
                return

            for col in df_loaded.select_dtypes(include=['float16', 'float32', 'int8', 'int16', 'int32', 'int64']).columns:
                df_loaded[col] = df_loaded[col].astype('float64')
            
            state["df"] = df_loaded
            state["filename"] = filename
            state["automl_result"] = None
            state["pipeline_result"] = None
            state["precalc_done"] = False
            
            from nicegui import app
            try:
                csv_buffer = io.StringIO()
                df_loaded.to_csv(csv_buffer, index=False)
                app.storage.user['current_df_csv'] = csv_buffer.getvalue()
                app.storage.user['current_df_columns'] = list(df_loaded.columns)
                app.storage.user['current_df_shape'] = df_loaded.shape
            except Exception as storage_err:
                logger.warning(f"app.storage への保存に失敗: {storage_err}")
            
            app.storage.user['data_loaded'] = True
            app.storage.user['data_filename'] = filename
            app.storage.user['data_timestamp'] = pd.Timestamp.now().isoformat()
            
            logger.info(f"✅ DataFrame読み込み完了: {df_loaded.shape[0]}行 × {df_loaded.shape[1]}列")
            
            upload_status.text = f"✅ {filename} 読み込み完了 ({len(df_loaded)}行 × {len(df_loaded.columns)}列)"
            upload_status.classes(remove="text-red", add="text-green")
            _show_preview(df_loaded, preview_container)
            _update_metrics(state, metrics_row)
            
            _auto_detect_columns(state)
            
            refresh = state.get("_refresh_tabs")
            if refresh:
                try:
                    refresh()
                except Exception as refresh_err:
                    logger.warning(f"タブ再描画エラー: {refresh_err}")
            
            try:
                from frontend_nicegui.main import render_llm_analysis_report
                asyncio.create_task(render_llm_analysis_report(df_loaded, metadata={"source": "upload", "filename": filename}))
            except ImportError:
                pass
            
            ui.notify(f'✅ {filename} を読み込みました ({df_loaded.shape[0]}行)', type='positive')
            
        except AttributeError as ae:
            logger.error(f"AttributeError: {ae}", exc_info=True)
            upload_status.text = f"❌ エラー: ファイル属性の取得に失敗しました"
            upload_status.classes(remove="text-green", add="text-red")
            ui.notify(f'✗ {str(ae)}', type='negative')
            
        except pd.errors.EmptyDataError:
            logger.error("Empty CSV file")
            upload_status.text = "❌ エラー: ファイルが空です"
            upload_status.classes(remove="text-green", add="text-red")
            ui.notify('✗ ファイルが空です', type='negative')
            
        except Exception as ex:
            logger.error(f"❌ 予期せぬエラー: {ex}", exc_info=True)
            upload_status.text = f"❌ {type(ex).__name__}: {str(ex)}"
            upload_status.classes(remove="text-green", add="text-red")
            ui.notify(f'✗ {str(ex)}', type='negative')

    ui.upload(
        on_upload=handle_upload,
        label="CSV / Excel をドラッグ&ドロップ",
        auto_upload=True,
    ).props('accept=".csv,.xlsx,.xls" color="purple"').classes("full-width")

    metrics_row = ui.row().classes("q-gutter-md q-mt-md full-width")
    _update_metrics(state, metrics_row)

    with ui.expansion("🧪 デバッグ用サンプルデータ", icon="science").classes("full-width q-mt-md").props("default-opened"):
        ui.label("開発・検証用のサンプルデータを選択してロードできます。").classes("text-caption text-grey-6 q-mb-md")
        
        def handle_debug_data_loaded(df, task_type, target_col, filename):
            state["df"] = df
            state["filename"] = filename
            state["automl_result"] = None
            state["pipeline_result"] = None
            state["precalc_done"] = False
            state["precalc_df"] = None
            state["_chem_adapters"] = None
            state["_applied_recommendation"] = None
            
            try:
                from frontend_nicegui.main import set_loaded_data
                set_loaded_data(df)
            except ImportError:
                pass

            _auto_detect_columns(state)
            
            state["task_type"] = task_type
            state["target_col"] = target_col
            
            upload_status.text = f"✅ {filename} 読み込み完了 ({len(df)}行)"
            _show_preview(df, preview_container)
            _update_metrics(state, metrics_row)
            
            refresh = state.get("_refresh_tabs")
            if refresh:
                refresh()

            try:
                from frontend_nicegui.main import render_llm_analysis_report
                asyncio.create_task(render_llm_analysis_report(df, metadata={"source": "debug_sample", "filename": filename}))
            except ImportError:
                pass

        create_debug_samples_selector(on_data_loaded=handle_debug_data_loaded)

        ui.separator()
        ui.label("公開ベンチマーク").classes("text-subtitle2 q-mt-sm")
        ui.label("ケモインフォマティクスで使われる標準データセット").classes("text-caption text-grey-6")

        with ui.row().classes("q-gutter-sm"):
            for name, desc, target in [
                ("esol", "ESOL 水溶解度 (1,128件)", "measured log solubility in mols per litre"),
                ("freesolv", "FreeSolv 水和自由エネ (642件)", "expt"),
                ("lipophilicity", "Lipophilicity 脂溶性 (4,200件)", "exp"),
            ]:
                def _load_bench(bname=name, btarget=target):
                    try:
                        from backend.data.benchmark_datasets import load_benchmark
                        df_bench = load_benchmark(bname)
                        state["df"] = df_bench
                        state["filename"] = f"benchmark_{bname}.csv"
                        state["automl_result"] = None
                        state["pipeline_result"] = None
                        state["precalc_done"] = False
                        state["precalc_df"] = None
                        state["_chem_adapters"] = None
                        state["_applied_recommendation"] = None
                        
                        try:
                            from frontend_nicegui.main import set_loaded_data
                            set_loaded_data(df_bench)
                        except ImportError:
                            pass

                        _auto_detect_columns(state)
                        state["target_col"] = btarget
                        upload_status.text = f"✅ {bname} ロード完了 ({len(df_bench)}行)"
                        upload_status.classes(remove="text-red", add="text-green")
                        _show_preview(df_bench, preview_container)
                        _update_metrics(state, metrics_row)
                        refresh = state.get("_refresh_tabs")

                        if refresh:
                            refresh()

                        try:
                            from frontend_nicegui.main import render_llm_analysis_report
                            asyncio.create_task(render_llm_analysis_report(df_bench, metadata={"source": "benchmark", "dataset": bname}))
                        except ImportError:
                            pass

                        ui.notify(f"✅ {bname} をロードしました", type="positive")
                    except Exception as ex:
                        ui.notify(f"エラー: {ex}", type="negative")

                ui.button(f"📥 {desc}", on_click=_load_bench).props("outline color=orange size=sm").tooltip(f"目的変数: {target}")

# =================================================================
# サブタブ2: 列の役割設定
# =================================================================
pass

def _on_target_change(val: str, state: dict) -> None:
    state["target_col"] = val
    state["precalc_done"] = False
    if state["df"] is not None and val in state["df"].columns:
        if pd.api.types.is_float_dtype(state["df"][val]):
            state["task_type"] = "regression"
        else:
            state["task_type"] = "classification"

# =================================================================
# サブタブ3: SMILES特徴量
# =================================================================
pass

def _show_descriptor_summary(state: dict, container) -> None:
    """記述子計算結果のサマリーを表示"""
    container.clear()
    precalc = state.get("precalc_df")
    if precalc is None:
        return
    with container:
        n = len(precalc.columns)
        calc_summary = state.get("calc_summary", {})

        with ui.row().classes("q-gutter-md"):
            with ui.card().classes("glass-card q-pa-md"):
                ui.label(str(n)).classes("text-h4 text-bold hero-gradient")
                ui.label("総記述子数").classes("text-caption text-grey-5")
            ok_count = len(calc_summary)
            with ui.card().classes("glass-card q-pa-md"):
                ui.label(str(ok_count)).classes("text-h4 text-bold hero-gradient")
                ui.label("成功エンジン").classes("text-caption text-grey-5")

        ui.separator()
        ui.label("エンジン別結果").classes("text-subtitle2 q-mt-md")
        rows = []
        for eng, cnt in calc_summary.items():
            rows.append({"エンジン": eng, "記述子数": cnt, "状態": "✅ 成功"})
        if rows:
            ui.table(
                columns=[
                    {"name": "エンジン", "label": "エンジン", "field": "エンジン", "align": "left"},
                    {"name": "記述子数", "label": "記述子数", "field": "記述子数"},
                    {"name": "状態", "label": "状態", "field": "状態", "align": "left"},
                ],
                rows=rows,
            ).classes("full-width").props("dense flat bordered")

# =================================================================
# サブタブ4: EDA
# =================================================================
def _render_eda(state: dict) -> None:
    """特徴量の探索的データ分析 — 統合EDAパネルに委譲。"""
    if state["df"] is None:
        with ui.column().classes("q-pa-md items-center full-width"):
            ui.label("⚠️ まずデータを読み込んでください").classes("text-amber text-h6")
            ui.label("EDAを実行するにはデータのロードが必要です。").classes("text-grey text-sm")
            ui.button("🔍 データ読み込み状態を確認 (Debug)", on_click=lambda: ui.navigate.to("/debug_data")).props("outline color=amber")
        return
    from frontend_nicegui.components.eda_panel import render_eda_panel
    render_eda_panel(state)

# =================================================================
# サブタブ5: パイプライン設計
# =================================================================
def _render_pipeline(state: dict) -> None:
    """CV設定・前処理・特徴選択・モデル選択・単調制約"""
    if state["df"] is None:
        with ui.column().classes("q-pa-md items-center full-width"):
            ui.label("⚠️ まずデータを読み込んでください").classes("text-amber text-h6")
            ui.label("ファイルを選択してもこのメッセージが出る場合は、以下のデバッグページで状態を確認してください。").classes("text-grey text-sm")
            ui.button("🔍 データ読み込み状態を確認 (Debug)", on_click=lambda: ui.navigate.to("/debug_data")).props("outline color=amber")
        return

    df = state["df"]
    target_col = state.get("target_col", "")
    task = state.get("task_type", "regression")
    if task == "auto":
        task = "regression" if (target_col and pd.api.types.is_float_dtype(df[target_col])) else "classification"

    with ui.expansion("💾 設定プリセット（保存/読込）", icon="bookmark").classes("full-width q-mb-md"):
        from backend.preset_manager import save_preset as _save_preset
        from backend.preset_manager import load_preset as _load_preset
        from backend.preset_manager import list_presets as _list_presets
        from backend.preset_manager import delete_preset as _delete_preset

        preset_list_container = ui.column().classes("full-width")

        def _refresh_preset_list():
            preset_list_container.clear()
            presets = _list_presets()
            with preset_list_container:
                if not presets:
                    ui.label("保存済みプリセットはありません").classes("text-caption text-grey q-pa-sm")
                else:
                    for p in presets:
                        with ui.card().classes("full-width q-pa-xs q-mb-xs glass-card"):
                            with ui.row().classes("items-center full-width justify-between"):
                                with ui.column().classes("q-gutter-none"):
                                    ui.label(p["name"]).classes("text-subtitle2 text-bold")
                                    desc = p.get("description", "")
                                    if desc:
                                        ui.label(desc).classes("text-caption text-grey").style("font-size: 0.7rem;")
                                    ui.label(f"{p['n_settings']}個の設定 | {p.get('created_at', '')[:10]}").classes(
                                        "text-caption text-grey"
                                    ).style("font-size:0.82rem;")
                                with ui.row().classes("q-gutter-xs"):
                                    pname = p["name"]

                                    def _do_load(name=pname):
                                        try:
                                            meta = _load_preset(name, state)
                                            ui.notify(f"✅ プリセット '{name}' を読み込みました ({len(meta['keys_loaded'])}件)", type="positive")
                                        except Exception as ex:
                                            ui.notify(f"エラー: {ex}", type="negative")

                                    def _do_delete(name=pname):
                                        _delete_preset(name)
                                        ui.notify(f"🗑️ '{name}' を削除しました", type="info")
                                        _refresh_preset_list()

                                    ui.button("📥", on_click=_do_load).props("flat dense size=xs color=cyan").tooltip("読込")
                                    ui.button("🗑️", on_click=_do_delete).props("flat dense size=xs color=red").tooltip("削除")

        _refresh_preset_list()

        ui.separator()
        ui.label("新規プリセット保存").classes("text-subtitle2 q-mt-sm")
        with ui.row().classes("items-end q-gutter-sm full-width"):
            preset_name_input = ui.input("プリセット名", placeholder="例: ADMET予測用").classes("col-4")
            preset_desc_input = ui.input("説明（任意）", placeholder="例: 単調性制約あり").classes("col-4")

            def _do_save():
                name = preset_name_input.value
                if not name:
                    ui.notify("プリセット名を入力してください", type="warning")
                    return
                try:
                    _save_preset(name, state, description=preset_desc_input.value or "")
                    ui.notify(f"✅ '{name}' を保存しました", type="positive")
                    preset_name_input.value = ""
                    preset_desc_input.value = ""
                    _refresh_preset_list()
                except Exception as ex:
                    ui.notify(f"保存エラー: {ex}", type="negative")

            ui.button("💾 保存", on_click=_do_save).props("outline color=cyan size=sm no-caps")

    with ui.expansion("📤 設定エクスポート / インポート（YAML）", icon="import_export").classes("full-width q-mb-sm"):
        from backend.preset_manager import export_config_yaml, import_config_yaml

        ui.label("📤 エクスポート（コピーして共有）").classes("text-subtitle2")
        export_area = ui.textarea("YAML設定", value="").classes("full-width").props("outlined readonly rows=4")

        def _do_export():
            yaml_text = export_config_yaml(state)
            export_area.value = yaml_text
            ui.notify("✅ 設定をエクスポートしました — テキストをコピーしてください", type="positive")

        ui.button("📤 エクスポート", on_click=_do_export).props("outline color=teal size=sm no-caps")

        ui.separator().classes("q-my-sm")

        ui.label("📥 インポート（YAMLを貼り付け）").classes("text-subtitle2")
        import_area = ui.textarea("YAML設定を貼り付け", value="").classes("full-width").props("outlined rows=4")

        def _do_import():
            text = import_area.value.strip()
            if not text:
                ui.notify("YAMLテキストを貼り付けてください", type="warning")
                return
            try:
                count = import_config_yaml(text, state)
                ui.notify(f"✅ {count}件の設定をインポートしました", type="positive")
                import_area.value = ""
            except Exception as ex:
                ui.notify(f"インポートエラー: {ex}", type="negative")

        ui.button("📥 インポート", on_click=_do_import).props("outline color=amber size=sm no-caps")

    with ui.expansion("📜 解析履歴", icon="history").classes("full-width q-mb-md"):
        from backend.preset_manager import list_history

        history = list_history(limit=10)
        if not history:
            ui.label("解析履歴はまだありません").classes("text-caption text-grey q-pa-sm")
        else:
            rows = []
            for h in history:
                rows.append({
                    "日時": h.get("timestamp", "")[:16].replace("T", "  "),
                    "ファイル": h.get("filename", ""),
                    "最良モデル": h.get("best_model", ""),
                    "スコア": f"{h.get('best_score', 0):.4f}",
                    "時間": f"{h.get('elapsed_seconds', 0):.1f}秒",
                })
            columns = [
                {"name": c, "label": c, "field": c, "align": "left", "sortable": True}
                for c in ["日時", "ファイル", "最良モデル", "スコア", "時間"]
            ]
            ui.table(columns=columns, rows=rows).classes("full-width").props("dense flat bordered")

    from frontend_nicegui.components.cv_config_ui import render_cv_config
    render_cv_config(state)

    ui.separator().classes("q-my-sm")

    with ui.expansion("🔧 前処理設定（スケーリング・欠損値・変換）", icon="transform").classes("full-width"):
        ui.label("列の型ごとに異なる前処理を適用します。デフォルト設定で問題なく動作します。").classes("text-caption text-grey q-mb-sm")

        with ui.card().classes("glass-card q-pa-sm full-width q-mb-sm"):
            ui.label("🔢 数値列").classes("text-subtitle2")
            with ui.row().classes("q-gutter-sm items-end"):
                ui.select(
                    options={
                        "standard": "StandardScaler (平均0, 分散1)",
                        "robust": "RobustScaler (外れ値に頑健)",
                        "minmax": "MinMaxScaler (0-1正規化)",
                        "maxabs": "MaxAbsScaler",
                        "none": "なし",
                    },
                    label="スケーラー",
                    value=state.get("num_scaler", "standard"),
                    on_change=lambda e: state.update({"num_scaler": e.value}),
                ).classes("w-56")

                ui.select(
                    options={
                        "median": "中央値で補完",
                        "mean": "平均値で補完",
                        "knn": "KNN Imputer",
                        "iterative": "IterativeImputer (MICE)",
                        "drop": "欠損行を削除",
                    },
                    label="欠損値処理",
                    value=state.get("num_imputer", "median"),
                    on_change=lambda e: state.update({"num_imputer": e.value}),
                ).classes("w-48")

                ui.select(
                    options={
                        "none": "なし",
                        "boxcox": "Box-Cox変換",
                        "yeojohnson": "Yeo-Johnson変換",
                        "quantile_uniform": "QuantileTransformer (uniform)",
                        "quantile_normal": "QuantileTransformer (normal)",
                        "log1p": "log(1+x)変換",
                    },
                    label="非線形変換",
                    value=state.get("num_transform", "none"),
                    on_change=lambda e: state.update({"num_transform": e.value}),
                ).classes("w-56")

        with ui.card().classes("glass-card q-pa-sm full-width q-mb-sm"):
            ui.label("🔤 カテゴリ列").classes("text-subtitle2")
            with ui.row().classes("q-gutter-sm items-end"):
                ui.select(
                    options={
                        "onehot": "OneHotEncoding",
                        "ordinal": "OrdinalEncoding",
                        "target": "TargetEncoding",
                        "binary": "BinaryEncoding",
                    },
                    label="エンコーディング",
                    value=state.get("cat_encoder", "onehot"),
                    on_change=lambda e: state.update({"cat_encoder": e.value}),
                ).classes("w-48")

                ui.select(
                    options={
                        "most_frequent": "最頻値で補完",
                        "constant": "定数 ('missing')",
                        "drop": "欠損行を削除",
                    },
                    label="欠損値処理",
                    value=state.get("cat_imputer", "most_frequent"),
                    on_change=lambda e: state.update({"cat_imputer": e.value}),
                ).classes("w-48")

    with ui.expansion("🎯 特徴量生成・選択", icon="filter_alt").classes("full-width"):
        ui.label("生成").classes("text-subtitle2")
        with ui.row().classes("q-gutter-sm"):
            ui.checkbox(
                "PolynomialFeatures（交互作用項）",
                value=state.get("do_polynomial", False),
                on_change=lambda e: state.update({"do_polynomial": e.value}),
            ).tooltip("二次の交互作用項を自動生成します。列数が大幅に増加するため注意。")

            if state.get("do_polynomial"):
                ui.number(
                    label="次数", value=state.get("poly_degree", 2),
                    min=2, max=3, step=1,
                    on_change=lambda e: state.update({"poly_degree": int(e.value)}),
                ).classes("w-20")

                ui.checkbox(
                    "interaction_only",
                    value=state.get("poly_interaction_only", True),
                    on_change=lambda e: state.update({"poly_interaction_only": e.value}),
                ).tooltip("True: 交互作用のみ（x1*x2）、False: 二乗項も含む（x1^2, x1*x2）")

        ui.separator().classes("q-my-xs")

        ui.label("選択").classes("text-subtitle2")
        _selector_label = "回帰" if task == "regression" else "分類"
        ui.select(
            options={
                "none": "選択しない（全特徴量を使用）",
                "variance": "VarianceThreshold (分散閾値)",
                "selectkbest_f": f"SelectKBest (F-test, {_selector_label})",
                "selectkbest_mi": f"SelectKBest (Mutual Info, {_selector_label})",
                "select_from_model_lasso": "SelectFromModel (Lasso / L1)",
                "select_from_model_rf": "SelectFromModel (RandomForest)",
                "rfe": "RFE (再帰的特徴量削除)",
                "boruta": "Boruta (全関連特徴量選択)",
            },
            label="特徴量選択手法",
            value=state.get("feature_selector", "none"),
            on_change=lambda e: state.update({"feature_selector": e.value}),
        ).classes("full-width").tooltip(
            "SelectFromModelやBorutaは内部でモデルを使用。タスク（回帰/分類）に自動適応。"
        )

        if state.get("feature_selector", "none") not in ("none", "variance"):
            ui.number(
                label="選択する特徴量数 (k)",
                value=state.get("n_features_to_select", 20),
                min=1, max=500, step=1,
                on_change=lambda e: state.update({"n_features_to_select": int(e.value)}),
            ).classes("w-40")

    ui.separator()
    ui.label("🤖 使用するモデル").classes("text-subtitle1 q-mt-md")

    try:
        from backend.models.factory import list_models, get_default_automl_models

        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)

        if "selected_models" not in state or not state["selected_models"]:
            state["selected_models"] = defaults

        with ui.row().classes("q-gutter-sm q-mb-sm"):
            def _select_all():
                state["selected_models"] = [m["key"] for m in available]
                ui.notify(f"全{len(available)}モデルを選択", type="info")
            def _select_defaults():
                state["selected_models"] = defaults
                ui.notify(f"デフォルト{len(defaults)}モデルを選択", type="info")
            def _select_fast():
                fast_keys = [m["key"] for m in available
                             if any(t in m.get("tags", []) for t in ["linear", "tree"])]
                state["selected_models"] = fast_keys[:8]
                ui.notify(f"高速{len(fast_keys[:8])}モデルを選択", type="info")

            ui.button("デフォルト", on_click=_select_defaults).props("outline size=sm no-caps color=cyan")
            ui.button("高速モデルのみ", on_click=_select_fast).props("outline size=sm no-caps color=teal")
            ui.button("全モデル", on_click=_select_all).props("flat size=sm no-caps color=grey")
            n_sel = len(state.get("selected_models", []))
            ui.badge(f"{n_sel}選択中", color="cyan").props("outline")

        categories: dict[str, list] = {"線形系": [], "カーネル系": [], "決定木系": [], "その他": []}
        for m in available:
            k = m["key"].lower() + m["name"].lower()
            if any(x in k for x in ["linear", "ridge", "lasso", "elastic", "logistic", "ard", "huber", "pls", "bayesian"]):
                cat = "線形系"
            elif any(x in k for x in ["svr", "svc", "support", "rbf", "kernel", "gaussian"]):
                cat = "カーネル系"
            elif any(x in k for x in ["tree", "forest", "boost", "gbm", "gradient", "rgf", "figs", "rule"]):
                cat = "決定木系"
            else:
                cat = "その他"
            categories[cat].append(m)

        with ui.tabs().classes("full-width").props("dense") as model_tabs:
            tabs = {}
            for cat_name in categories:
                if categories[cat_name]:
                    tabs[cat_name] = ui.tab(cat_name)

        with ui.tab_panels(model_tabs).classes("full-width"):
            for cat_name, models in categories.items():
                if not models:
                    continue
                with ui.tab_panel(tabs[cat_name]):
                    with ui.row().classes("q-gutter-sm flex-wrap"):
                        for m in models:
                            is_checked = m["key"] in state.get("selected_models", [])
                            cb = ui.checkbox(m["name"], value=is_checked).tooltip(
                                f"タグ: {', '.join(m.get('tags', []))}"
                            )
                            cb.on_value_change(
                                lambda e, key=m["key"]: _toggle_model(state, key, e.value)
                            )

        _render_model_auto_params(state, available)

    except Exception as ex:
        ui.label(f"モデル一覧取得エラー: {ex}").classes("text-red")

    _render_monotonic_constraints(state, df, target_col)

    ui.separator()
    with ui.expansion("🔬 その他の詳細設定", icon="tune").classes("full-width q-mt-md"):
        with ui.row().classes("q-gutter-md"):
            ui.checkbox("EDA実行", value=state.get("do_eda", True)).on_value_change(
                lambda e: state.update({"do_eda": e.value})
            )
            ui.checkbox("前処理実行", value=state.get("do_prep", True)).on_value_change(
                lambda e: state.update({"do_prep": e.value})
            )
            ui.checkbox("評価実行", value=state.get("do_eval", True)).on_value_change(
                lambda e: state.update({"do_eval": e.value})
            )
            ui.checkbox("PCA実行", value=state.get("do_pca", True)).on_value_change(
                lambda e: state.update({"do_pca": e.value})
            )
            ui.checkbox("SHAP解析", value=state.get("do_shap", True)).on_value_change(
                lambda e: state.update({"do_shap": e.value})
            )

def _render_monotonic_constraints(state: dict, df: pd.DataFrame, target_col: str) -> None:
    """説明変数ごとの単調制約UI — ダイアログベース。"""
    from frontend_nicegui.components.dialog_manager import (
        create_settings_dialog,
        render_settings_summary,
    )
    numeric_cols = [c for c in df.select_dtypes(include='number').columns
                    if c != target_col and c not in state.get("exclude_cols", [])]

    if not numeric_cols:
        return

    if "monotonic_constraints" not in state:
        state["monotonic_constraints"] = {}

    constraints = state["monotonic_constraints"]

    n_inc = sum(1 for v in constraints.values() if v == 1)
    n_dec = sum(1 for v in constraints.values() if v == -1)
    n_total = n_inc + n_dec

    summary = [f"対象列: {len(numeric_cols)}個"]
    if n_total > 0:
        summary.append(f"↗ 増加: {n_inc}件, ↘ 減少: {n_dec}件")
        examples = [(c, v) for c, v in constraints.items() if v != 0][:3]
        for c, v in examples:
            sym = "↗" if v == 1 else "↘"
            summary.append(f"  {sym} {c}")
    else:
        summary.append("制約なし（デフォルト）")

    def _build_content():
        ui.label("⚠️ 上級者向け機能: ドメイン知識に基づき設定してください。").classes("text-caption text-amber q-mb-sm")
        ui.label("各説明変数の目的変数に対する単調増加/減少の制約を設定。XGBoost, LightGBM, monotonic kernel等で利用されます。").classes("text-caption text-grey q-mb-sm")

        with ui.row().classes("q-gutter-sm q-mb-sm"):
            ui.button("全て制約なし", on_click=lambda: (constraints.clear(), ui.notify("全制約をリセット", type="info"))).props("flat dense no-caps size=sm color=grey")

        for col in numeric_cols:
            current = constraints.get(col, 0)
            with ui.row().classes("items-center q-gutter-xs full-width q-mb-xs"):
                ui.label(col).classes("text-body2").style("width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;")
                ui.radio({0: "制約なし", 1: "↗ 単調増加", -1: "↘ 単調減少"}, value=current, on_change=lambda e, c=col: constraints.update({c: e.value})).props("dense inline")

    def _open_dialog():
        dlg = create_settings_dialog(title="📐 単調性制約設定", icon="trending_up", width="85vw", max_width="800px", content_builder=_build_content, state=state, snapshot_keys=["monotonic_constraints"])
        dlg.open()

    render_settings_summary(icon="trending_up", title="単調性制約", summary_lines=summary, button_label="⚙️ 制約設定", on_click=_open_dialog, badge_text=f"{n_total}件設定" if n_total > 0 else "なし", badge_color="amber" if n_total > 0 else "grey")

def _toggle_model(state: dict, key: str, checked: bool) -> None:
    """モデルの選択/解除をstateに反映"""
    selected = state.get("selected_models", [])
    if checked and key not in selected:
        selected.append(key)
    elif not checked and key in selected:
        selected.remove(key)
    state["selected_models"] = selected

def _render_model_auto_params(state: dict, available_models: list) -> None:
    """選択されたモデルごとにパラメータ自動UIを生成する。"""
    selected = state.get("selected_models", [])
    if not selected:
        return

    model_classes = {}
    for m in available_models:
        if m["key"] in selected and "class" in m:
            model_classes[m["key"]] = (m["name"], m["class"])

    if not model_classes:
        return

    ui.separator()
    with ui.expansion(f"⚙️ 選択モデルのパラメータ設定 ({len(model_classes)}モデル)", icon="tune").classes("full-width q-mt-md"):
        ui.label("各モデルの引数を自動検出して表示しています。デフォルト値のまま変更しなければ標準設定で実行されます。").classes("text-caption text-grey-6 q-mb-md")

        if "model_params" not in state:
            state["model_params"] = {}

        for model_key, (model_name, model_cls) in model_classes.items():
            with ui.expansion(f"🔹 {model_name} ({model_cls.__name__})", icon="settings").classes("full-width q-mb-xs"):
                try:
                    from frontend_nicegui.components.auto_params_ui import render_param_editor
                    from backend.ui.param_schema import introspect_params
                    specs = introspect_params(model_cls)
                    if specs:
                        existing = state["model_params"].get(model_key, {})
                        values = render_param_editor(specs, title=model_name, values=existing)
                        state["model_params"][model_key] = values
                    else:
                        ui.label("ℹ️ パラメータなし").classes("text-grey-6")
                except Exception as ex:
                    ui.label(f"⚠️ パラメータ取得エラー: {ex}").classes("text-amber")

def _render_adapter_auto_params(state: dict) -> None:
    """各SMILES記述子エンジンのパラメータ自動UIを生成する。"""
    if "adapter_params" not in state:
        state["adapter_params"] = {}

    ui.label("各エンジンの引数を自動検出して表示しています。変更しなければデフォルト設定で計算されます。").classes("text-caption text-grey-6 q-mb-md")

    for ename, emod, ecls, ekwargs in _ALL_ENGINES:
        try:
            mod = importlib.import_module(emod)
            adapter_cls = getattr(mod, ecls)

            from backend.ui.param_schema import introspect_params
            specs = introspect_params(adapter_cls)

            if not specs:
                continue

            with ui.expansion(f"🔹 {ename} ({len(specs)}パラメータ)", icon="settings").classes("full-width q-mb-xs"):
                try:
                    from frontend_nicegui.components.auto_params_ui import render_param_editor
                    existing = state["adapter_params"].get(ename, {})
                    values = render_param_editor(specs, title=ename, values=existing, compact=True)
                    state["adapter_params"][ename] = values
                except Exception as ex:
                    ui.label(f"⚠️ {ex}").classes("text-amber")

        except Exception:
            pass

# =================================================================
# ユーティリティ関数
# =================================================================
def _auto_detect_columns(state: dict) -> None:
    """目的変数・SMILES列を自動検出してstateに設定"""
    df = state["df"]
    if df is None:
        return
    
    state["target_col"] = df.columns[-1]
    state["smiles_col"] = ""
    
    try:
        from backend.data.type_detector import TypeDetector
        detector = TypeDetector()
        dr = detector.detect(df)
        if dr.smiles_columns:
            state["smiles_col"] = dr.smiles_columns[0]
        else:
            for col in df.columns:
                if col.lower() == "smiles":
                    state["smiles_col"] = col
                    break
    except Exception:
        for col in df.columns:
            if col.lower() == "smiles":
                state["smiles_col"] = col
                break

    target = state["target_col"]
    if pd.api.types.is_float_dtype(df[target]):
        state["task_type"] = "regression"
    else:
        state["task_type"] = "classification"

    smart_fn = state.get("_apply_smart_defaults")
    if callable(smart_fn):
        try:
            smart_fn()
        except Exception:
            pass
        
    try:
        from frontend_nicegui.utils.feature_classifier import FeatureClassifier
        from backend.models.monotonic_constraints import ConstraintRangeCalculator
        from backend.chem.feature_metadata import feature_metadata
        
        known_sources = feature_metadata.export_for_frontend()
        feature_cols = [c for c in df.columns if c not in {state["target_col"], state["smiles_col"]}]
        
        state["feature_stats"] = ConstraintRangeCalculator.compute_feature_stats(df, feature_cols)
        
        state["feature_classification"] = {}
        for feat in feature_cols:
            state["feature_classification"][feat] = FeatureClassifier.classify_feature(feat, known_sources)
        
        if "monotonicity_constraints" in state:
            state["monotonicity_constraints"]["_by_feature"].clear()
            state["monotonicity_constraints"]["_by_set"].clear()
            
    except Exception as e:
        logger.warning(f"特徴量メタデータの登録に失敗しました: {e}")

    state.pop("dim_red_results", None)
    if "data" in getattr(state, "__dict__", {}):
        pass
    try:
        if hasattr(state, "data"):
            state.data.pop("dim_red_results", None)
            state.data["dim_red_computing"] = False
        else:
            state["dim_red_computing"] = False
    except Exception:
        pass

def _show_preview(df: pd.DataFrame, container) -> None:
    """DataFrameのプレビューをテーブルとして表示"""
    container.clear()
    with container:
        preview = df.head(8)
        columns = [{"name": col, "label": col, "field": col, "align": "left", "sortable": True} for col in preview.columns]
        rows = []
        for _, row in preview.iterrows():
            row_dict = {}
            for col in preview.columns:
                v = row[col]
                if pd.isna(v):
                    row_dict[col] = "—"
                elif isinstance(v, float):
                    row_dict[col] = f"{v:.4f}"
                else:
                    row_dict[col] = str(v)
            rows.append(row_dict)
        ui.table(columns=columns, rows=rows).classes("full-width").props("dense flat bordered")

def _update_metrics(state: dict, container) -> None:
    """メトリクスカードの更新"""
    container.clear()
    df = state.get("df")
    if df is None:
        return
    with container:
        for val, lbl, icon_name in [
            (f"{df.shape[0]:,}", "行数", "table_rows"),
            (str(df.shape[1]), "列数", "view_column"),
            (f"{df.isna().mean().mean():.1%}", "欠損率", "warning"),
            (str(df.select_dtypes(include='number').shape[1]), "数値列", "numbers"),
        ]:
            with ui.card().classes("glass-card q-pa-sm"):
                ui.icon(icon_name, color="cyan", size="xs")
                ui.label(val).classes("text-h6 text-bold hero-gradient")
                ui.label(lbl).classes("text-caption text-grey-5")
