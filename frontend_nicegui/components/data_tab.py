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

    is_advanced = state.get("user_mode", "beginner") == "advanced"

    with ui.tabs().classes("full-width").props("dense active-color=cyan indicator-color=cyan") as sub_tabs:
        tab_load = ui.tab("load", label="📂 データ読込", icon="upload_file")
        tab_cols = ui.tab("columns", label="🏷️ 列の役割", icon="settings")
        tab_smiles = ui.tab("smiles", label="⚗️ SMILES特徴量", icon="science")
        if not is_advanced:
            tab_smiles.set_visibility(False)
        tab_eda = ui.tab("eda", label="📊 EDA", icon="analytics")
        if not is_advanced:
            tab_eda.set_visibility(False)
        tab_pipeline = ui.tab("pipeline", label="⚙️ パイプライン", icon="tune")
        if not is_advanced:
            tab_pipeline.set_visibility(False)

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
                    with ui.expansion("🔧 詳細な列役割設定", icon="settings").classes("full-width"):
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
    """列役割のカード形式ビジュアルサマリー + 欠損値ヒートマップ。"""
    if state["df"] is None:
        return

    df = state["df"]

    # 役割→色・アイコン・ラベルのマップ
    ROLE_MAP = {
        "target":  {"icon": "🎯", "label": "目的変数", "color": "rgba(0,212,255,0.25)",  "border": "#00d4ff"},
        "smiles":  {"icon": "🧬", "label": "SMILES",   "color": "rgba(123,47,247,0.25)", "border": "#7b2ff7"},
        "exclude": {"icon": "🚫", "label": "除外",     "color": "rgba(180,60,60,0.15)",  "border": "#b43c3c"},
        "group":   {"icon": "👥", "label": "グループ", "color": "rgba(100,200,200,0.15)", "border": "#64c8c8"},
        "time":    {"icon": "📅", "label": "時系列",   "color": "rgba(200,180,100,0.15)", "border": "#c8b464"},
        "weight":  {"icon": "⚖️", "label": "weight",   "color": "rgba(160,160,200,0.15)", "border": "#a0a0c8"},
        "feature": {"icon": "✅", "label": "説明変数", "color": "rgba(74,222,128,0.12)",  "border": "rgba(74,222,128,0.3)"},
    }

    def _get_role(col: str) -> str:
        if col == state.get("target_col"):
            return "target"
        elif col == state.get("smiles_col"):
            return "smiles"
        elif col in state.get("exclude_cols", []):
            return "exclude"
        elif col == state.get("group_col"):
            return "group"
        elif col == state.get("time_col"):
            return "time"
        elif col == state.get("weight_col"):
            return "weight"
        return "feature"

    # ── 欠損ヒートマップ概要 ──
    total_na = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    overall_rate = total_na / total_cells * 100 if total_cells > 0 else 0

    if total_na > 0:
        na_color = "red" if overall_rate > 20 else ("amber" if overall_rate > 5 else "green")
        with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
            f"border: 1px solid rgba(251,191,36,0.3); border-radius: 8px;"
            f"background: rgba(50,40,0,0.15);"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("warning", color=na_color)
                ui.label(f"欠損値: {total_na:,}セル ({overall_rate:.1f}%)").classes(f"text-body2 text-{na_color}")
                ui.label(f"— {df.isna().any(axis=1).sum():,}行に欠損あり").classes("text-caption text-grey")

    # ── カード形式グリッド ──
    with ui.row().classes("full-width q-gutter-xs").style("flex-wrap: wrap;"):
        for col in df.columns:
            role_key = _get_role(col)
            role = ROLE_MAP[role_key]
            col_data = df[col]
            na_count = int(col_data.isna().sum())
            na_pct = na_count / len(col_data) * 100 if len(col_data) > 0 else 0
            n_unique = int(col_data.nunique(dropna=True))
            dtype_str = str(col_data.dtype)

            # 欠損バーの色
            bar_color = "#4ade80" if na_pct == 0 else ("#fbbf24" if na_pct < 20 else "#f87171")
            bar_width = max(2, min(100, 100 - na_pct))  # 充填率

            with ui.card().classes("q-pa-xs cursor-pointer").style(
                f"min-width: 130px; max-width: 180px; flex: 1 1 130px;"
                f"border: 1px solid {role['border']}; border-radius: 8px;"
                f"background: {role['color']};"
            ).tooltip(
                f"列名: {col}\n型: {dtype_str}\n欠損: {na_count}/{len(col_data)} ({na_pct:.1f}%)\n"
                f"ユニーク値: {n_unique}"
            ):
                # 役割アイコン + 列名
                with ui.row().classes("items-center q-gutter-none").style("overflow: hidden;"):
                    ui.label(role["icon"]).style("font-size: 0.8rem;")
                    ui.label(col).classes("text-caption text-bold").style(
                        "overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 120px;"
                    )

                # データ型 + ユニーク数
                ui.label(f"{dtype_str} | {n_unique}種").classes("text-caption text-grey").style("font-size: 0.65rem;")

                # 欠損率バー
                ui.html(
                    f'<div style="width:100%;height:3px;border-radius:2px;background:rgba(255,255,255,0.1);margin-top:2px;">'
                    f'<div style="width:{bar_width}%;height:100%;border-radius:2px;background:{bar_color};"></div>'
                    f'</div>'
                )


# ================================================================
# サブタブ3: SMILES特徴量
# ================================================================
def _render_smiles_features(state: dict) -> None:
    """SMILES記述子プラグイン管理UI"""

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

    # ── プラグイン管理UI（動的生成） ──
    from frontend_nicegui.components.descriptor_plugins_ui import render_descriptor_plugins
    render_descriptor_plugins(state)

    # ── 計算ステータス表示 ──
    if state.get("precalc_done") and state.get("precalc_df") is not None:
        precalc = state["precalc_df"]
        ui.label(f"✅ {len(precalc.columns)}個の記述子が計算済みです").classes("q-mt-md text-positive")
        results_container = ui.column().classes("full-width q-mt-sm")
        _show_descriptor_summary(state, results_container)
    else:
        ui.label("⏳ SMILES検出後、記述子は自動計算されます").classes("q-mt-md text-grey-5")


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
    """特徴量の探索的データ分析（SMILES記述子含む）"""

    if state["df"] is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    df = state["df"]
    precalc_df = state.get("precalc_df")
    target_col = state.get("target_col", "")

    # ── 基本統計 ──
    ui.label("📊 データ概要").classes("text-subtitle1")
    n_desc = precalc_df.shape[1] if precalc_df is not None else 0
    with ui.row().classes("q-gutter-md full-width"):
        for val, lbl in [
            (f"{df.shape[0]:,}", "行数"),
            (str(df.shape[1]), "元の列数"),
            (str(n_desc), "SMILES記述子"),
            (f"{df.isna().mean().mean():.1%}", "欠損率"),
            (str(df.select_dtypes(include='number').shape[1]), "数値列数"),
        ]:
            with ui.card().classes("glass-card q-pa-sm"):
                ui.label(val).classes("text-h5 text-bold hero-gradient")
                ui.label(lbl).classes("text-caption text-grey-5")

    # ── 統計量テーブル（元データ） ──
    ui.separator()
    ui.label("📈 統計量サマリー（元データ）").classes("text-subtitle2 q-mt-md")
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

    # ══════════════════════════════════════════════════════
    # SMILES記述子のEDA（precalc_dfが存在する場合）
    # ══════════════════════════════════════════════════════
    if precalc_df is not None and not precalc_df.empty:
        ui.separator()
        ui.label(f"⚗️ SMILES記述子 EDA（{precalc_df.shape[1]}記述子）").classes("text-subtitle1 q-mt-md")

        # ── 目的変数との相関係数 ──
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            target_s = df[target_col].iloc[:len(precalc_df)].reset_index(drop=True)
            try:
                corr = precalc_df.iloc[:len(target_s)].corrwith(target_s, method="pearson").abs().dropna()
                if not corr.empty:
                    top_n = 20
                    top_corr = corr.sort_values(ascending=False).head(top_n)

                    ui.label(f"📊 目的変数 ({target_col}) との相関 TOP {min(top_n, len(top_corr))}").classes(
                        "text-subtitle2 q-mt-sm"
                    )

                    # バーチャートで表示
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_bar(
                            x=list(top_corr.values),
                            y=list(top_corr.index),
                            orientation="h",
                            marker_color=["#00d4ff" if v > 0.3 else "#fbbf24" if v > 0.1 else "#555577"
                                          for v in top_corr.values],
                        )
                        fig.update_layout(
                            xaxis_title="|r| (ピアソン相関係数の絶対値)",
                            yaxis=dict(autorange="reversed"),
                            height=max(250, len(top_corr) * 22),
                            margin=dict(l=10, r=10, t=10, b=30),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#ccc", size=10),
                        )
                        ui.plotly(fig).classes("full-width")
                    except ImportError:
                        # Plotlyなしの場合はテーブル表示
                        corr_rows = [{"記述子": name, "|r|": f"{val:.4f}"}
                                     for name, val in top_corr.items()]
                        corr_cols = [
                            {"name": "記述子", "label": "記述子", "field": "記述子", "align": "left"},
                            {"name": "|r|", "label": "|r|", "field": "|r|", "align": "center"},
                        ]
                        ui.table(columns=corr_cols, rows=corr_rows).classes("full-width").props("dense flat bordered")
            except Exception:
                pass

        # ── 記述子の統計量テーブル ──
        with ui.expansion(
            f"📈 記述子の統計量（{precalc_df.shape[1]}列）", icon="analytics",
        ).classes("full-width q-mt-sm"):
            desc_stats = precalc_df.describe().T
            desc_rows = []
            for col_name in desc_stats.index:
                row = {"列名": col_name}
                for stat_name in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                    if stat_name in desc_stats.columns:
                        v = desc_stats.loc[col_name, stat_name]
                        row[stat_name] = f"{v:.4g}" if pd.notna(v) else "—"
                    else:
                        row[stat_name] = "—"
                desc_rows.append(row)

            desc_columns = [
                {"name": c, "label": c, "field": c,
                 "align": "left" if c == "列名" else "center", "sortable": True}
                for c in ["列名", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ]
            ui.table(
                columns=desc_columns, rows=desc_rows,
                pagination={"rowsPerPage": 20},
            ).classes("full-width").props("dense flat bordered")

        # ── 欠損・定数列の警告 ──
        const_cols = [c for c in precalc_df.columns if precalc_df[c].nunique() <= 1]
        high_na_cols = [c for c in precalc_df.columns if precalc_df[c].isna().mean() > 0.5]
        if const_cols or high_na_cols:
            with ui.card().classes("full-width q-pa-sm q-mt-sm").style(
                "border: 1px solid rgba(251,191,36,0.4); background: rgba(50,40,0,0.2); border-radius: 8px;"
            ):
                if const_cols:
                    ui.label(f"⚠️ 定数列 ({len(const_cols)}列): {', '.join(const_cols[:10])}{'...' if len(const_cols) > 10 else ''}").classes("text-caption text-amber")
                if high_na_cols:
                    ui.label(f"⚠️ 高欠損列 >50% ({len(high_na_cols)}列): {', '.join(high_na_cols[:10])}{'...' if len(high_na_cols) > 10 else ''}").classes("text-caption text-amber")

    # ── 🧹 データクリーニングアクション ──
    ui.separator()
    ui.label("🧹 データクリーニングアクション").classes("text-subtitle1 q-mt-md")
    ui.label(
        "EDAの結果に基づいて、不要な列や欠損行をワンクリックで処理できます。"
    ).classes("text-caption text-grey q-mb-sm")

    with ui.row().classes("q-gutter-sm full-width flex-wrap"):

        # ── 定数列を除外 ──
        const_all = [c for c in df.columns if df[c].nunique() <= 1 and c != state.get("target_col")]
        if precalc_df is not None:
            const_desc = [c for c in precalc_df.columns if precalc_df[c].nunique() <= 1]
        else:
            const_desc = []
        n_const = len(const_all) + len(const_desc)

        def _remove_const():
            excluded = list(state.get("exclude_cols", []))
            added = [c for c in const_all if c not in excluded]
            excluded.extend(added)
            state["exclude_cols"] = excluded
            # precalc_dfからも定数列を除外
            if precalc_df is not None and const_desc:
                state["precalc_df"] = precalc_df.drop(columns=const_desc, errors="ignore")
            ui.notify(f"✅ 定数列 {n_const}列を除外しました", type="positive")

        btn_const = ui.button(
            f"🗑️ 定数列を除外 ({n_const}列)",
            on_click=_remove_const,
        ).props("outline color=amber size=sm no-caps")
        if n_const == 0:
            btn_const.disable()

        # ── 高欠損列を除外 ──
        high_na_thresh = 0.5
        high_na_all = [c for c in df.columns
                       if df[c].isna().mean() > high_na_thresh
                       and c != state.get("target_col")]
        n_high_na = len(high_na_all)

        def _remove_high_na():
            excluded = list(state.get("exclude_cols", []))
            added = [c for c in high_na_all if c not in excluded]
            excluded.extend(added)
            state["exclude_cols"] = excluded
            ui.notify(f"✅ 高欠損列(>50%) {len(added)}列を除外しました", type="positive")

        btn_na = ui.button(
            f"🗑️ 高欠損列を除外 ({n_high_na}列, >50%)",
            on_click=_remove_high_na,
        ).props("outline color=amber size=sm no-caps")
        if n_high_na == 0:
            btn_na.disable()

        # ── 欠損行を削除 ──
        n_na_rows = int(df.isna().any(axis=1).sum())

        def _drop_na_rows():
            state["df"] = df.dropna().reset_index(drop=True)
            dropped = n_na_rows
            ui.notify(f"✅ 欠損行 {dropped}行を削除 → {len(state['df'])}行", type="positive")

        btn_drop = ui.button(
            f"🗑️ 欠損行を削除 ({n_na_rows}行)",
            on_click=_drop_na_rows,
        ).props("outline color=red size=sm no-caps")
        if n_na_rows == 0:
            btn_drop.disable()

    # ── 列を手動で除外（セレクタ） ──
    with ui.expansion("🔧 列を手動で除外", icon="remove_circle").classes("full-width q-mt-sm"):
        exclude_opts = [c for c in df.columns
                        if c != state.get("target_col")
                        and c != state.get("smiles_col")]
        current_excluded = state.get("exclude_cols", [])

        def _on_exclude_change(e):
            state["exclude_cols"] = e.value or []
            ui.notify(f"除外列を更新: {len(e.value or [])}列", type="info")

        ui.select(
            options=exclude_opts,
            label="除外する列を選択",
            value=current_excluded,
            on_change=_on_exclude_change,
        ).classes("full-width").props("multiple clearable use-chips outlined dense")

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
    """CV設定・前処理・特徴選択・モデル選択・単調制約"""

    if state["df"] is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    df = state["df"]
    target_col = state.get("target_col", "")
    task = state.get("task_type", "regression")
    if task == "auto":
        task = "regression" if (target_col and pd.api.types.is_float_dtype(df[target_col])) else "classification"

    # ────────────────────────────────────────────
    # 1. 交差検証設定
    # ────────────────────────────────────────────
    ui.label("🔄 交差検証（CV）設定").classes("text-subtitle1")

    with ui.row().classes("q-gutter-md items-end full-width"):
        # CV方式
        try:
            from backend.models.cv_manager import _CV_REGISTRY
            cv_options = {k: v["name"] for k, v in _CV_REGISTRY.items()}
        except ImportError:
            cv_options = {
                "kfold": "K-Fold", "stratified_kfold": "Stratified K-Fold",
                "repeated_kfold": "Repeated K-Fold", "logo": "Leave-One-Group-Out",
                "group_kfold": "GroupKFold", "loo": "Leave-One-Out",
                "timeseries": "TimeSeriesSplit",
            }

        cv_select = ui.select(
            options=cv_options,
            label="CV方式",
            value=state.get("cv_key", "auto"),
            on_change=lambda e: state.update({"cv_key": e.value}),
        ).classes("w-48").tooltip(
            "auto: タスクに応じて自動選択（回帰→KFold、分類→StratifiedKFold）\n"
            "グループ系: グループ列の設定が必要"
        )

        # N分割
        ui.number(
            label="分割数", value=state.get("cv_folds", 5),
            min=2, max=20, step=1,
            on_change=lambda e: state.update({"cv_folds": int(e.value)}),
        ).classes("w-24").tooltip("K-Fold等の分割数。通常3〜10。")

        # タイムアウト
        ui.number(
            label="タイムアウト(秒)", value=state.get("timeout", 300),
            min=30, max=3600, step=30,
            on_change=lambda e: state.update({"timeout": int(e.value)}),
        ).classes("w-28").tooltip("全体の制限時間。超過するとモデルをスキップ。")

    ui.separator().classes("q-my-sm")

    # ────────────────────────────────────────────
    # 2. 前処理設定（ColumnTransformer相当）
    # ────────────────────────────────────────────
    with ui.expansion(
        "🔧 前処理設定（スケーリング・欠損値・変換）", icon="transform",
    ).classes("full-width"):
        ui.label(
            "列の型ごとに異なる前処理を適用します。デフォルト設定で問題なく動作します。"
        ).classes("text-caption text-grey q-mb-sm")

        # 数値列の前処理
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

        # カテゴリ列の前処理
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

    # ────────────────────────────────────────────
    # 3. 特徴量生成・選択
    # ────────────────────────────────────────────
    with ui.expansion("🎯 特徴量生成・選択", icon="filter_alt").classes("full-width"):
        # 特徴量生成
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

        # 特徴量選択
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

    # ────────────────────────────────────────────
    # 4. モデル選択
    # ────────────────────────────────────────────
    ui.separator()
    ui.label("🤖 使用するモデル").classes("text-subtitle1 q-mt-md")

    try:
        from backend.models.factory import list_models, get_default_automl_models

        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)

        if "selected_models" not in state or not state["selected_models"]:
            state["selected_models"] = defaults

        # クイック選択ボタン
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

        # ── 選択モデルのパラメータ自動UI ──
        _render_model_auto_params(state, available)

    except Exception as ex:
        ui.label(f"モデル一覧取得エラー: {ex}").classes("text-red")

    # ────────────────────────────────────────────
    # 5. 単調制約（説明変数ごと）
    # ────────────────────────────────────────────
    _render_monotonic_constraints(state, df, target_col)

    # ────────────────────────────────────────────
    # 6. 詳細設定（上級者用折りたたみ）
    # ────────────────────────────────────────────
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
    """説明変数ごとの単調制約UI。"""
    numeric_cols = [c for c in df.select_dtypes(include='number').columns
                    if c != target_col and c not in state.get("exclude_cols", [])]

    if not numeric_cols:
        return

    with ui.expansion(
        f"📐 単調制約（{len(numeric_cols)}数値列）", icon="trending_up",
    ).classes("full-width q-mt-sm"):
        ui.label(
            "各説明変数の目的変数に対する単調増加/減少の制約を設定できます。"
            "XGBoost, LightGBM, monotonic kernel等で利用されます。デフォルトは制約なし。"
        ).classes("text-caption text-grey q-mb-sm")

        if "monotonic_constraints" not in state:
            state["monotonic_constraints"] = {}

        constraints = state["monotonic_constraints"]

        # テーブル形式で表示
        for col in numeric_cols:
            current = constraints.get(col, 0)
            with ui.row().classes("items-center q-gutter-xs full-width q-mb-xs"):
                ui.label(col).classes("text-body2").style("width: 200px; overflow: hidden; text-overflow: ellipsis;")
                sel = ui.select(
                    options={0: "制約なし", 1: "↗ 単調増加", -1: "↘ 単調減少"},
                    value=current,
                    on_change=lambda e, c=col: constraints.update({c: e.value}),
                ).props("dense outlined").classes("w-36")
                sel.tooltip(f"{col}: 0=制約なし, 1=単調増加, -1=単調減少")


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

    # スマートデフォルト適用
    smart_fn = state.get("_apply_smart_defaults")
    if callable(smart_fn):
        try:
            smart_fn()
        except Exception:
            pass


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
