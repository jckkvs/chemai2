"""
frontend_nicegui/components/data_tab.py

データ設定タブ：データ読込・列の役割設定・SMILES特徴量・EDA・パイプライン設計
全機能をサブタブで構造化。Progressive Disclosure で初心者/上級者を両立。
"""
from __future__ import annotations

import io
import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)

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
    ("RDKit",           "backend.chem.rdkit_adapter",           "RDKitAdapter",           {"compute_fp": False}),
    ("Mordred",         "backend.chem.mordred_adapter",         "MordredAdapter",         {"selected_only": True}),
    ("GroupContrib",    "backend.chem.group_contrib_adapter",   "GroupContribAdapter",     {}),
    ("DescriptaStorus", "backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter",  {}),
    ("MolAI",           "backend.chem.molai_adapter",           "MolAIAdapter",           {"n_components": 6}),
    ("scikit-FP",       "backend.chem.skfp_adapter",            "SkfpAdapter",            {"fp_types": ["ECFP", "MACCS"]}),
    ("UMA",             "backend.chem.uma_adapter",             "UMAAdapter",             {}),
    ("Mol2Vec",         "backend.chem.mol2vec_adapter",         "Mol2VecAdapter",         {}),
    ("PaDEL",           "backend.chem.padel_adapter",           "PaDELAdapter",           {}),
    ("Molfeat",         "backend.chem.molfeat_adapter",         "MolfeatAdapter",         {}),
    ("XTB",             "backend.chem.xtb_adapter",             "XTBAdapter",             {}),
    ("UniPKa",          "backend.chem.unipka_adapter",          "UniPkaAdapter",          {}),
    ("COSMO-RS",        "backend.chem.cosmo_adapter",           "CosmoAdapter",           {}),
    ("Chemprop",        "backend.chem.chemprop_adapter",        "ChempropAdapter",        {}),
]


def render_data_tab(state: dict[str, Any]) -> None:
    """データ設定タブ全体を描画する。"""

    with ui.tabs().classes("full-width").props("dense active-color=cyan indicator-color=cyan") as sub_tabs:
        tab_load = ui.tab("load", label="📂 データ読込", icon="upload_file")
        tab_cols = ui.tab("columns", label="🏷️ 列の役割", icon="settings")
        tab_smiles = ui.tab("smiles", label="⚗️ SMILES特徴量", icon="science")
        tab_eda = ui.tab("eda", label="📊 EDA", icon="analytics")
        tab_pipeline = ui.tab("pipeline", label="⚙️ パイプライン", icon="tune")

    # ── 各タブ内のコンテナ（遅延レンダリング用） ──
    containers: dict[str, ui.column] = {}
    rendered: dict[str, bool] = {}

    with ui.tab_panels(sub_tabs, value=tab_load).classes("full-width") as panels:

        with ui.tab_panel(tab_load):
            containers["load"] = ui.column().classes("full-width")
            rendered["load"] = False

        with ui.tab_panel(tab_cols):
            containers["columns"] = ui.column().classes("full-width")
            rendered["columns"] = False

        with ui.tab_panel(tab_smiles):
            containers["smiles"] = ui.column().classes("full-width")
            rendered["smiles"] = False

        with ui.tab_panel(tab_eda):
            containers["eda"] = ui.column().classes("full-width")
            rendered["eda"] = False

        with ui.tab_panel(tab_pipeline):
            containers["pipeline"] = ui.column().classes("full-width")
            rendered["pipeline"] = False

    # ── レンダラーマップ ──
    _renderers = {
        "load":     _render_data_load,
        "columns":  _render_column_roles,
        "smiles":   _render_smiles_features,
        "eda":      _render_eda,
        "pipeline": _render_pipeline,
    }

    def _render_tab(tab_key: str, force: bool = False) -> None:
        """指定タブのコンテナ内容を（再）構築する。"""
        if tab_key not in containers:
            return
        # 初回 or 強制リフレッシュ
        if force or not rendered.get(tab_key):
            c = containers[tab_key]
            c.clear()
            with c:
                _renderers[tab_key](state)
            rendered[tab_key] = True

    # ── 初回: データ読込タブだけ即座にレンダリング ──
    _render_tab("load")

    # ── タブ切り替え時: 選択されたタブを（再）レンダリング ──
    def _on_tab_change(e) -> None:
        tab_key = e.value if isinstance(e.value, str) else getattr(e.value, "value", str(e.value))
        _render_tab(tab_key, force=True)

    sub_tabs.on_value_change(_on_tab_change)

    # ── stateに再描画ヘルパーを登録（データ読込からタブ更新を要求可能にする） ──
    state["_refresh_tabs"] = lambda: [rendered.update({k: False}) for k in rendered]



# ================================================================
# サブタブ1: データ読込
# ================================================================
def _render_data_load(state: dict) -> None:
    """ファイルアップロード + サンプル + ベンチマークのデータ読込UI"""

    upload_status = ui.label("").classes("text-grey-5 q-mt-sm")
    preview_container = ui.column().classes("full-width q-mt-md")

    async def handle_upload(e):
        content = e.content.read()
        name = e.name
        try:
            if name.endswith(".csv"):
                state["df"] = pd.read_csv(io.BytesIO(content))
            elif name.endswith((".xlsx", ".xls")):
                state["df"] = pd.read_excel(io.BytesIO(content))
            else:
                upload_status.text = "❌ CSV/Excelファイルのみ対応"
                return
            state["filename"] = name
            state["automl_result"] = None
            state["pipeline_result"] = None
            _auto_detect_columns(state)
            df = state["df"]
            upload_status.text = f"✅ {name} 読み込み完了 ({len(df)}行 × {len(df.columns)}列)"
            upload_status.classes(remove="text-red", add="text-green")
            _show_preview(df, preview_container)
            _update_metrics(state, metrics_row)
            ui.notify(f"✅ {name} を読み込みました", type="positive")
        except Exception as ex:
            upload_status.text = f"❌ エラー: {ex}"
            upload_status.classes(remove="text-green", add="text-red")

    ui.upload(
        on_upload=handle_upload,
        label="CSV / Excel をドラッグ&ドロップ",
        auto_upload=True,
    ).props('accept=".csv,.xlsx,.xls" color="purple"').classes("full-width")

    # メトリクスカード行
    metrics_row = ui.row().classes("q-gutter-md q-mt-md full-width")
    _update_metrics(state, metrics_row)

    # ── サンプルデータ（折りたたみ） ──
    with ui.expansion("🧪 サンプルデータ / ベンチマーク", icon="science").classes("full-width q-mt-md"):

        ui.label("デバッグ用サンプル").classes("text-subtitle2 q-mt-sm")
        with ui.row().classes("q-gutter-sm"):

            def _load_sample_regression():
                np.random.seed(42)
                n = 25
                state["df"] = pd.DataFrame({
                    "SMILES": np.random.choice(SAMPLE_SMILES, n),
                    "solubility_logS": np.random.randn(n) * 2 - 2,
                })
                state["filename"] = "sample_regression.csv"
                state["automl_result"] = None
                state["pipeline_result"] = None
                state["precalc_done"] = False
                _auto_detect_columns(state)
                state["task_type"] = "regression"
                upload_status.text = f"✅ 回帰サンプル ({n}行)"
                upload_status.classes(remove="text-red", add="text-green")
                _show_preview(state["df"], preview_container)
                _update_metrics(state, metrics_row)
                ui.notify("回帰サンプルデータを読み込みました", type="positive")

            def _load_sample_classification():
                np.random.seed(42)
                n = 25
                state["df"] = pd.DataFrame({
                    "SMILES": np.random.choice(SAMPLE_SMILES, n),
                    "is_toxic": np.random.randint(0, 2, n),
                })
                state["filename"] = "sample_classification.csv"
                state["automl_result"] = None
                state["pipeline_result"] = None
                state["precalc_done"] = False
                _auto_detect_columns(state)
                state["task_type"] = "classification"
                upload_status.text = f"✅ 分類サンプル ({n}行)"
                upload_status.classes(remove="text-red", add="text-green")
                _show_preview(state["df"], preview_container)
                _update_metrics(state, metrics_row)
                ui.notify("分類サンプルデータを読み込みました", type="positive")

            def _load_sample_numeric():
                np.random.seed(42)
                n = 30
                state["df"] = pd.DataFrame({
                    "temperature": np.random.uniform(20, 80, n),
                    "pressure": np.random.exponential(5, n),
                    "catalyst": np.random.choice(["A型", "B型", "C型"], n),
                    "time_h": np.random.uniform(1, 24, n),
                    "yield": np.random.randn(n) * 10 + 75,
                })
                state["filename"] = "sample_numeric.csv"
                state["automl_result"] = None
                state["pipeline_result"] = None
                state["precalc_done"] = False
                _auto_detect_columns(state)
                state["smiles_col"] = ""
                state["task_type"] = "regression"
                upload_status.text = f"✅ 数値サンプル ({n}行)"
                upload_status.classes(remove="text-red", add="text-green")
                _show_preview(state["df"], preview_container)
                _update_metrics(state, metrics_row)
                ui.notify("数値サンプルデータを読み込みました", type="positive")

            ui.button("🧪 回帰 (SMILES)", on_click=_load_sample_regression).props("outline color=purple size=sm")
            ui.button("🏷️ 分類 (SMILES)", on_click=_load_sample_classification).props("outline color=blue size=sm")
            ui.button("📊 数値のみ", on_click=_load_sample_numeric).props("outline color=teal size=sm")

        # ── ベンチマークデータ ──
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
                        _auto_detect_columns(state)
                        state["target_col"] = btarget
                        upload_status.text = f"✅ {bname} ロード完了 ({len(df_bench)}行)"
                        upload_status.classes(remove="text-red", add="text-green")
                        _show_preview(df_bench, preview_container)
                        _update_metrics(state, metrics_row)
                        ui.notify(f"✅ {bname} をロードしました", type="positive")
                    except Exception as ex:
                        ui.notify(f"エラー: {ex}", type="negative")

                ui.button(
                    f"📥 {desc}", on_click=_load_bench
                ).props("outline color=orange size=sm").tooltip(f"目的変数: {target}")


# ================================================================
# サブタブ2: 列の役割設定
# ================================================================
def _render_column_roles(state: dict) -> None:
    """目的変数・SMILES列・除外列などの設定UI"""

    def _build_ui():
        container.clear()
        with container:
            if state["df"] is None:
                ui.label("⚠️ まず「📂 データ読込」タブでデータを読み込んでください").classes("text-amber q-pa-md")
                return

            df = state["df"]
            all_cols = list(df.columns)

            with ui.row().classes("full-width q-gutter-lg"):
                # ── 左列: 必須設定 ──
                with ui.column().classes("col-6"):
                    ui.label("必須設定").classes("text-subtitle1 text-bold")

                    # 目的変数
                    cur_target = state.get("target_col") or all_cols[-1]
                    ui.select(
                        options=all_cols,
                        label="🎯 目的変数（予測したい列）",
                        value=cur_target if cur_target in all_cols else all_cols[-1],
                        on_change=lambda e: _on_target_change(e.value, state),
                    ).classes("full-width q-mb-md").tooltip("機械学習で予測する列を選択してください")

                    # タスク種別
                    _auto_task = "regression" if cur_target in all_cols and pd.api.types.is_float_dtype(df[cur_target]) else "classification"
                    _task_label = "📈 回帰（数値予測）" if _auto_task == "regression" else "🏷️ 分類（カテゴリ予測）"
                    ui.label(f"タスク種別: {_task_label}（自動判定）").classes("text-grey-5 q-mb-sm")
                    ui.select(
                        options={"auto": "自動判定", "regression": "回帰", "classification": "分類"},
                        label="タスクタイプ",
                        value=state.get("task_type", "auto"),
                        on_change=lambda e: state.update({"task_type": e.value}),
                    ).classes("full-width q-mb-md")

                    # SMILES列
                    smiles_options = ["（なし）"] + all_cols
                    cur_smiles = state.get("smiles_col", "")
                    smiles_val = cur_smiles if cur_smiles in all_cols else "（なし）"
                    ui.select(
                        options=smiles_options,
                        label="🧬 SMILES列（化合物構造 / 任意）",
                        value=smiles_val,
                        on_change=lambda e: state.update({
                            "smiles_col": "" if e.value == "（なし）" else e.value,
                            "precalc_done": False,
                        }),
                    ).classes("full-width q-mb-md").props("clearable").tooltip(
                        "SMILES形式の化合物構造が含まれる列。指定すると14エンジンで記述子を自動計算します。"
                    )

                # ── 右列: オプション設定（折りたたみ） ──
                with ui.column().classes("col-5"):
                    with ui.expansion("🔧 詳細な列役割設定（上級者）", icon="settings").classes("full-width"):
                        excl_opts = [c for c in all_cols
                                     if c != state.get("target_col") and c != state.get("smiles_col")]

                        # 除外列
                        ui.select(
                            options=excl_opts,
                            label="🚫 除外する列",
                            value=state.get("exclude_cols", []),
                            on_change=lambda e: state.update({"exclude_cols": e.value or []}),
                        ).classes("full-width q-mb-sm").props("multiple clearable use-chips").tooltip(
                            "目的変数・SMILES以外で解析に使わない列（ID列・メモ列等）"
                        )

                        # グループ列
                        group_opts = ["（なし）"] + excl_opts
                        ui.select(
                            options=group_opts,
                            label="👥 グループ列（GroupKFold用）",
                            value=state.get("group_col", "（なし）"),
                            on_change=lambda e: state.update({
                                "group_col": None if e.value == "（なし）" else e.value
                            }),
                        ).classes("full-width q-mb-sm").tooltip(
                            "GroupKFold等でリーク防止に使うグループID列（例:バッチID）"
                        )

                        # 時系列列
                        ui.select(
                            options=group_opts,
                            label="📅 時系列列（任意）",
                            value=state.get("time_col", "（なし）"),
                            on_change=lambda e: state.update({
                                "time_col": None if e.value == "（なし）" else e.value
                            }),
                        ).classes("full-width q-mb-sm").tooltip(
                            "時間的順序を持つ列。TimeSeriesSplit等に使用します。"
                        )

                        # Sample weight列
                        ui.select(
                            options=group_opts,
                            label="⚖️ Sample weight列（任意）",
                            value=state.get("weight_col", "（なし）"),
                            on_change=lambda e: state.update({
                                "weight_col": None if e.value == "（なし）" else e.value
                            }),
                        ).classes("full-width q-mb-sm").tooltip(
                            "各サンプルの重みを示す列。信頼度の高いサンプルを重視する場合に指定。"
                        )

            # ── 列役割サマリー ──
            ui.separator()
            ui.label("📋 列役割サマリー").classes("text-subtitle2 q-mt-md")
            _render_column_summary(state)

    container = ui.column().classes("full-width")
    _build_ui()


def _on_target_change(val: str, state: dict) -> None:
    state["target_col"] = val
    state["precalc_done"] = False
    # タスク自動判定
    if state["df"] is not None and val in state["df"].columns:
        if pd.api.types.is_float_dtype(state["df"][val]):
            state["task_type"] = "regression"
        else:
            state["task_type"] = "classification"


def _render_column_summary(state: dict) -> None:
    """列役割のサマリーテーブルを表示"""
    if state["df"] is None:
        return

    rows = []
    for c in state["df"].columns:
        if c == state.get("target_col"):
            role = "🎯 目的変数"
        elif c == state.get("smiles_col"):
            role = "🧬 SMILES"
        elif c in state.get("exclude_cols", []):
            role = "🚫 除外"
        elif c == state.get("group_col"):
            role = "👥 グループ"
        elif c == state.get("time_col"):
            role = "📅 時系列"
        elif c == state.get("weight_col"):
            role = "⚖️ weight"
        else:
            role = "✅ 説明変数"
        rows.append({"列名": c, "役割": role, "型": str(state["df"][c].dtype)})

    columns = [
        {"name": "列名", "label": "列名", "field": "列名", "align": "left", "sortable": True},
        {"name": "役割", "label": "役割", "field": "役割", "align": "left"},
        {"name": "型", "label": "データ型", "field": "型"},
    ]
    ui.table(columns=columns, rows=rows).classes("full-width").props("dense flat bordered")


# ================================================================
# サブタブ3: SMILES特徴量
# ================================================================
def _render_smiles_features(state: dict) -> None:
    """SMILES記述子の計算と選択UI"""

    if state["df"] is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    if not state.get("smiles_col"):
        with ui.card().classes("glass-card q-pa-lg"):
            ui.icon("info", color="cyan", size="md")
            ui.label("SMILES列が設定されていません").classes("text-subtitle1")
            ui.label("「🏷️ 列の役割」タブでSMILES列を指定してください。\n"
                     "SMILES列がない場合、このステップはスキップできます。").classes("text-grey-5")
        return

    desc_status = ui.label("").classes("q-mt-md")
    desc_progress = ui.linear_progress(value=0, show_value=False).classes("q-mt-sm")
    desc_progress.visible = False
    results_container = ui.column().classes("full-width q-mt-md")

    # ── エンジンパラメータ自動UI（上級者用） ──
    with ui.expansion("🔧 エンジン別パラメータ設定（上級者）", icon="tune").classes("full-width q-mt-sm"):
        _render_adapter_auto_params(state)

    # 計算済みならサマリを表示
    if state.get("precalc_done") and state.get("precalc_df") is not None:
        precalc = state["precalc_df"]
        desc_status.text = f"✅ {len(precalc.columns)}個の記述子が計算済みです"
        _show_descriptor_summary(state, results_container)

    async def calc_descriptors():
        smiles_list = state["df"][state["smiles_col"]].dropna().tolist()
        n = len(smiles_list)
        if n == 0:
            ui.notify("有効なSMILESが0件です", type="warning")
            return

        desc_status.text = "⏳ 計算中..."
        desc_progress.visible = True
        desc_progress.value = 0.05

        df_result = pd.DataFrame(index=range(n))
        calc_summary = {}
        total = len(_ALL_ENGINES)

        # ユーザーが設定したエンジンパラメータを取得
        adapter_params = state.get("adapter_params", {})

        for i, (ename, emod, ecls, ekwargs) in enumerate(_ALL_ENGINES):
            desc_status.text = f"⏳ {ename} を計算中... ({i+1}/{total})"
            desc_progress.value = (i + 1) / total
            try:
                mod = importlib.import_module(emod)
                adapter_cls = getattr(mod, ecls)
                # ユーザー設定パラメータをマージ
                user_params = adapter_params.get(ename, {})
                merged_kwargs = {**ekwargs}
                if user_params:
                    from backend.ui.param_schema import introspect_params, apply_params
                    specs = introspect_params(adapter_cls)
                    applied = apply_params(specs, user_params)
                    merged_kwargs.update(applied)
                adapter = adapter_cls(**merged_kwargs)
                if not adapter.is_available():
                    continue
                eres = adapter.compute(smiles_list)
                if hasattr(eres, 'descriptors') and eres.descriptors is not None:
                    edf = eres.descriptors
                    new_cols = [c for c in edf.columns if c not in df_result.columns]
                    if new_cols:
                        edf_new = edf[new_cols].copy()
                        edf_new.index = range(n)
                        df_result = pd.concat([df_result, edf_new], axis=1)
                        calc_summary[ename] = len(new_cols)
            except Exception as ex:
                logger.warning(f"{ename} スキップ: {ex}")

        # クリーンアップ
        df_result = df_result.loc[:, ~df_result.columns.duplicated()]
        df_result = df_result.apply(pd.to_numeric, errors="coerce").convert_dtypes()
        df_result = df_result.dropna(axis=1, how="all")

        state["precalc_df"] = df_result
        state["precalc_done"] = True
        state["selected_descriptors"] = list(df_result.columns)
        state["calc_summary"] = calc_summary

        n_desc = len(df_result.columns)
        desc_status.text = f"✅ {n_desc}個の記述子を計算完了"
        desc_progress.visible = False
        ui.notify(f"✅ {n_desc}個の記述子を計算完了", type="positive")
        _show_descriptor_summary(state, results_container)

    # 計算ボタン
    if not state.get("precalc_done"):
        ui.button(
            "⚗️ 全エンジンで記述子を計算", on_click=calc_descriptors
        ).props("color=purple size=lg icon=science").classes("q-mt-md")
    else:
        with ui.row().classes("q-gutter-sm"):
            ui.button(
                "🔄 再計算", on_click=calc_descriptors
            ).props("outline color=purple size=sm")


def _show_descriptor_summary(state: dict, container) -> None:
    """記述子計算結果のサマリーを表示"""
    container.clear()
    precalc = state.get("precalc_df")
    if precalc is None:
        return

    with container:
        n = len(precalc.columns)
        calc_summary = state.get("calc_summary", {})

        # メトリクスカード
        with ui.row().classes("q-gutter-md"):
            with ui.card().classes("glass-card q-pa-md"):
                ui.label(str(n)).classes("text-h4 text-bold hero-gradient")
                ui.label("総記述子数").classes("text-caption text-grey-5")
            ok_count = len(calc_summary)
            with ui.card().classes("glass-card q-pa-md"):
                ui.label(str(ok_count)).classes("text-h4 text-bold hero-gradient")
                ui.label("成功エンジン").classes("text-caption text-grey-5")

        # エンジン別結果テーブル
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


# ================================================================
# サブタブ4: EDA
# ================================================================
def _render_eda(state: dict) -> None:
    """特徴量の探索的データ分析"""

    if state["df"] is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    df = state["df"]

    # ── 基本統計 ──
    ui.label("📊 データ概要").classes("text-subtitle1")
    with ui.row().classes("q-gutter-md full-width"):
        for val, lbl in [
            (f"{df.shape[0]:,}", "行数"),
            (str(df.shape[1]), "列数"),
            (f"{df.isna().mean().mean():.1%}", "欠損率"),
            (str(df.select_dtypes(include='number').shape[1]), "数値列数"),
        ]:
            with ui.card().classes("glass-card q-pa-sm"):
                ui.label(val).classes("text-h5 text-bold hero-gradient")
                ui.label(lbl).classes("text-caption text-grey-5")

    # ── 統計量テーブル ──
    ui.separator()
    ui.label("📈 統計量サマリー").classes("text-subtitle2 q-mt-md")
    stat_rows = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_total = len(s)
        missing_rate = n_missing / n_total * 100 if n_total > 0 else 0.0
        cardinality = int(s.nunique(dropna=True))
        row = {
            "列名": col,
            "ユニーク数": cardinality,
            "欠損数": n_missing,
            "欠損率(%)": f"{missing_rate:.1f}",
        }
        numeric_s = pd.to_numeric(s, errors="coerce")
        if numeric_s.notna().any():
            row["最小値"] = f"{numeric_s.min():.4g}"
            row["最大値"] = f"{numeric_s.max():.4g}"
            row["平均値"] = f"{numeric_s.mean():.4g}"
        else:
            row["最小値"] = row["最大値"] = row["平均値"] = "—"
        stat_rows.append(row)

    stat_columns = [
        {"name": c, "label": c, "field": c, "align": "left" if c == "列名" else "center", "sortable": True}
        for c in ["列名", "ユニーク数", "欠損数", "欠損率(%)", "最小値", "最大値", "平均値"]
    ]
    ui.table(columns=stat_columns, rows=stat_rows).classes("full-width").props("dense flat bordered")

    # ── データプレビュー ──
    ui.separator()
    ui.label("🔍 データプレビュー（先頭8行）").classes("text-subtitle2 q-mt-md")
    _show_preview(df, ui.column().classes("full-width"))

    # ── データ型判定 ──
    ui.separator()
    with ui.expansion("🔍 変数型の自動判定結果", icon="search").classes("full-width"):
        try:
            from backend.data.type_detector import TypeDetector
            detector = TypeDetector()
            dr = detector.detect(df)
            type_rows = []
            for c in df.columns:
                t = "❓ 不明"
                if c == state.get("target_col"):
                    t = "🎯 目的変数"
                elif c in getattr(dr, "numeric_columns", []):
                    t = "🔢 数値"
                elif c in getattr(dr, "categorical_columns", []):
                    t = "🔤 カテゴリ"
                elif c in getattr(dr, "binary_columns", []):
                    t = "0️⃣1️⃣ 2値"
                elif c in getattr(dr, "smiles_columns", []):
                    t = "🧬 SMILES"
                elif c in getattr(dr, "ignored_columns", []):
                    t = "❌ 除外（一意・定数等）"
                type_rows.append({"列名": c, "判定型": t})
            ui.table(
                columns=[
                    {"name": "列名", "label": "列名", "field": "列名", "align": "left"},
                    {"name": "判定型", "label": "判定型", "field": "判定型", "align": "left"},
                ],
                rows=type_rows,
            ).classes("full-width").props("dense flat bordered")
        except Exception as ex:
            ui.label(f"型判定エラー: {ex}").classes("text-red")


# ================================================================
# サブタブ5: パイプライン設計
# ================================================================
def _render_pipeline(state: dict) -> None:
    """CV設定・モデル選択"""

    if state["df"] is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    # ── CV設定 ──
    ui.label("⚙️ 交差検証設定").classes("text-subtitle1")
    with ui.row().classes("q-gutter-md"):
        cv_folds = ui.number(
            label="CV分割数", value=state.get("cv_folds", 5),
            min=2, max=20, step=1,
            on_change=lambda e: state.update({"cv_folds": int(e.value)}),
        ).classes("col-2").tooltip("交差検証のFold分割数。通常は3〜10。")

        timeout = ui.number(
            label="タイムアウト(秒)", value=state.get("timeout", 300),
            min=30, max=3600, step=30,
            on_change=lambda e: state.update({"timeout": int(e.value)}),
        ).classes("col-2").tooltip("全体の制限時間。超過するとモデルをスキップします。")

        ui.select(
            options=["auto", "standard", "robust", "minmax", "none"],
            label="スケーラー",
            value=state.get("scaler", "auto"),
            on_change=lambda e: state.update({"scaler": e.value}),
        ).classes("col-2").tooltip("特徴量の正規化手法を選択します。")

    # ── モデル選択 ──
    ui.separator()
    ui.label("🤖 使用するモデル").classes("text-subtitle1 q-mt-md")

    try:
        from backend.models.factory import list_models, get_default_automl_models

        task = state.get("task_type", "regression")
        if task == "auto":
            target = state.get("target_col")
            task = "regression" if (target and state["df"] is not None
                                    and pd.api.types.is_float_dtype(state["df"][target])) else "classification"

        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)

        if "selected_models" not in state or not state["selected_models"]:
            state["selected_models"] = defaults

        # カテゴリ分け
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

        # 選択数表示
        n_sel = len(state.get("selected_models", []))
        ui.label(f"📋 {n_sel}モデル選択中").classes("text-caption text-grey-5 q-mt-sm")

        # ── 選択モデルのパラメータ自動UI ──
        _render_model_auto_params(state, available)

    except Exception as ex:
        ui.label(f"モデル一覧取得エラー: {ex}").classes("text-red")

    # ── パイプライン詳細設定（上級者用折りたたみ） ──
    ui.separator()
    with ui.expansion("🔬 パイプライン詳細設定（上級者）", icon="tune").classes("full-width q-mt-md"):
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


def _toggle_model(state: dict, key: str, checked: bool) -> None:
    """モデルの選択/解除をstateに反映"""
    selected = state.get("selected_models", [])
    if checked and key not in selected:
        selected.append(key)
    elif not checked and key in selected:
        selected.remove(key)
    state["selected_models"] = selected


def _render_model_auto_params(state: dict, available_models: list) -> None:
    """
    選択されたモデルごとにパラメータ自動UIを生成する。

    introspect_params() でクラスの __init__ パラメータを自動検出し、
    auto_params_ui.render_param_editor() でUIウィジェットを自動描画する。
    新モデル追加時にUIコード変更は不要。
    """
    selected = state.get("selected_models", [])
    if not selected:
        return

    # モデルclass辞書を構築
    model_classes = {}
    for m in available_models:
        if m["key"] in selected and "class" in m:
            model_classes[m["key"]] = (m["name"], m["class"])

    if not model_classes:
        return

    ui.separator()
    with ui.expansion(
        f"⚙️ 選択モデルのパラメータ設定 ({len(model_classes)}モデル)",
        icon="tune",
    ).classes("full-width q-mt-md"):
        ui.label(
            "各モデルの引数を自動検出して表示しています。"
            "デフォルト値のまま変更しなければ標準設定で実行されます。"
        ).classes("text-caption text-grey-6 q-mb-md")

        if "model_params" not in state:
            state["model_params"] = {}

        for model_key, (model_name, model_cls) in model_classes.items():
            with ui.expansion(
                f"🔹 {model_name} ({model_cls.__name__})",
                icon="settings",
            ).classes("full-width q-mb-xs"):
                try:
                    from frontend_nicegui.components.auto_params_ui import render_param_editor
                    from backend.ui.param_schema import introspect_params
                    specs = introspect_params(model_cls)
                    if specs:
                        existing = state["model_params"].get(model_key, {})
                        values = render_param_editor(
                            specs,
                            title=model_name,
                            values=existing,
                        )
                        state["model_params"][model_key] = values
                    else:
                        ui.label("ℹ️ パラメータなし").classes("text-grey-6")
                except Exception as ex:
                    ui.label(f"⚠️ パラメータ取得エラー: {ex}").classes("text-amber")


def _render_adapter_auto_params(state: dict) -> None:
    """
    各SMILES記述子エンジンのパラメータ自動UIを生成する。

    introspect_params() でアダプタクラスの __init__ パラメータを自動検出。
    パラメータがある場合のみUIを表示する。
    """
    if "adapter_params" not in state:
        state["adapter_params"] = {}

    ui.label(
        "各エンジンの引数を自動検出して表示しています。"
        "変更しなければデフォルト設定で計算されます。"
    ).classes("text-caption text-grey-6 q-mb-md")

    for ename, emod, ecls, ekwargs in _ALL_ENGINES:
        try:
            mod = importlib.import_module(emod)
            adapter_cls = getattr(mod, ecls)

            from backend.ui.param_schema import introspect_params
            specs = introspect_params(adapter_cls)

            if not specs:
                continue  # パラメータなしのエンジンはスキップ

            with ui.expansion(
                f"🔹 {ename} ({len(specs)}パラメータ)",
                icon="settings",
            ).classes("full-width q-mb-xs"):
                try:
                    from frontend_nicegui.components.auto_params_ui import render_param_editor
                    existing = state["adapter_params"].get(ename, {})
                    values = render_param_editor(
                        specs,
                        title=ename,
                        values=existing,
                        compact=True,
                    )
                    state["adapter_params"][ename] = values
                except Exception as ex:
                    ui.label(f"⚠️ {ex}").classes("text-amber")

        except Exception:
            pass  # インポート不可のエンジンはスキップ


# ================================================================
# ユーティリティ関数
# ================================================================

def _auto_detect_columns(state: dict) -> None:
    """目的変数・SMILES列を自動検出してstateに設定"""
    df = state["df"]
    if df is None:
        return

    # 目的変数: 最後の列
    state["target_col"] = df.columns[-1]

    # SMILES列: "smiles" という名前の列を探す
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

    # タスク自動判定
    target = state["target_col"]
    if pd.api.types.is_float_dtype(df[target]):
        state["task_type"] = "regression"
    else:
        state["task_type"] = "classification"


def _show_preview(df: pd.DataFrame, container) -> None:
    """DataFrameのプレビューをテーブルとして表示"""
    container.clear()
    with container:
        preview = df.head(8)
        columns = [
            {"name": col, "label": col, "field": col, "align": "left", "sortable": True}
            for col in preview.columns
        ]
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
