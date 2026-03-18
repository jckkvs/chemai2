"""
ChemAI ML Studio - NiceGUI Edition
===================================
Pure Python UI using NiceGUI framework.
Shares the same backend as the Streamlit and Django editions.

Usage:
    python main.py
    → http://localhost:8080
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# backendへのパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from nicegui import ui, app

# ─────────────────────────────────────────────
# ダークテーマCSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0d0d1a;
    --bg-secondary: #1a1a2e;
    --bg-card: rgba(255, 255, 255, 0.05);
    --border: rgba(255, 255, 255, 0.12);
    --text-primary: #e0e0f0;
    --text-secondary: #a0a0c0;
    --accent-blue: #00d4ff;
    --accent-purple: #7b2ff7;
    --accent-green: #4ade80;
}

body {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary), #16213e) !important;
}

.nicegui-content { max-width: 1400px; margin: 0 auto; }

.hero-gradient {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), #ff6b9d);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glass-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}
"""

# サンプルSMILES
SAMPLE_SMILES = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCCO", "CC=O",
    "c1ccc(O)cc1", "CC(=O)OC", "CCOC", "CCN",
    "CC(C)(C)O", "c1ccc(N)cc1", "OC(=O)c1ccccc1", "CCOCC",
    "CC(O)CC", "c1ccc(Cl)cc1", "CC(=O)N", "CCCCCO",
    "c1ccc(F)cc1", "CC(C)=O", "OCCO", "c1ccncc1",
    "CC(=O)CC", "CCCCO", "c1ccc(C)cc1",
]


# ─────────────────────────────────────────────
# メインページ
# ─────────────────────────────────────────────
@ui.page("/")
def main_page():
    # --- ページスコープの状態（storage不使用） ---
    state = {
        "df": None,
        "filename": "",
        "target_col": "",
        "smiles_col": "",
        "task_type": "regression",
        "precalc_df": None,
        "selected_descriptors": [],
    }

    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

    # ── ヘッダー ──
    with ui.header().classes("items-center justify-between"):
        ui.label("⚗️ ChemAI ML Studio").classes("text-h5 text-bold hero-gradient")
        ui.label("NiceGUI Edition").classes("text-caption text-grey-6")

    # ── サイドバー ──
    with ui.left_drawer(value=True).classes("bg-dark"):
        ui.label("⚗️ ChemAI").classes("text-h6 q-mb-md hero-gradient")

        with ui.column().classes("q-gutter-sm"):
            ui.button("📁 データ読込", on_click=lambda: stepper.set_value("data")).props(
                "flat color=white align=left"
            ).classes("full-width")
            ui.button("🎯 列の設定", on_click=lambda: stepper.set_value("columns")).props(
                "flat color=white align=left"
            ).classes("full-width")
            ui.button("⚗️ SMILES特徴量", on_click=lambda: stepper.set_value("smiles")).props(
                "flat color=white align=left"
            ).classes("full-width")
            ui.button("🚀 解析実行", on_click=lambda: stepper.set_value("analysis")).props(
                "flat color=white align=left"
            ).classes("full-width")
            ui.button("📊 結果", on_click=lambda: stepper.set_value("results")).props(
                "flat color=white align=left"
            ).classes("full-width")

        ui.separator()
        ui.link("❓ ヘルプ", "/help").classes("text-white q-mt-md")
        ui.space()
        ui.label("NiceGUI Edition").classes("text-caption text-grey-7")

    # ── メインコンテンツ ──
    with ui.stepper().classes("full-width").props("vertical") as stepper:

        # ═══════════ Step 1: データ読込 ═══════════
        with ui.step("data", title="📁 データ読込"):
            ui.label("CSVファイルをアップロード、またはサンプルデータを使用").classes("text-grey-5")

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
                    df = state["df"]
                    upload_status.text = f"✅ {name} 読み込み完了 ({len(df)}行 × {len(df.columns)}列)"
                    upload_status.classes(remove="text-red", add="text-green")
                    _show_preview(df, preview_container)
                    _update_column_selects()
                    ui.notify(f"✅ {name} を読み込みました", type="positive")
                except Exception as ex:
                    upload_status.text = f"❌ エラー: {ex}"
                    upload_status.classes(remove="text-green", add="text-red")

            ui.upload(
                on_upload=handle_upload,
                label="CSV / Excel をドラッグ&ドロップ",
                auto_upload=True,
            ).props('accept=".csv,.xlsx,.xls" color="purple"').classes("full-width")

            ui.separator()
            ui.label("🧪 サンプルデータ").classes("text-subtitle1 q-mt-md")

            def load_sample_regression():
                np.random.seed(42)
                state["df"] = pd.DataFrame({
                    "SMILES": SAMPLE_SMILES,
                    "target_value": np.random.randn(25) * 2 + 5,
                })
                state["filename"] = "sample_regression.csv"
                state["target_col"] = "target_value"
                state["smiles_col"] = "SMILES"
                upload_status.text = "✅ サンプルデータ読み込み完了 (25行 × 2列)"
                upload_status.classes(remove="text-red", add="text-green")
                _show_preview(state["df"], preview_container)
                _update_column_selects()
                ui.notify("サンプルデータを読み込みました", type="positive")

            def load_sample_numeric():
                np.random.seed(42)
                n = 30
                state["df"] = pd.DataFrame({
                    "feature1": np.random.randn(n),
                    "feature2": np.random.randn(n) * 2,
                    "feature3": np.random.uniform(0, 10, n),
                    "target_value": np.random.randn(n) * 2 + 5,
                })
                state["filename"] = "sample_numeric.csv"
                state["target_col"] = "target_value"
                state["smiles_col"] = ""
                upload_status.text = f"✅ 数値サンプル読み込み完了 ({n}行 × 4列)"
                upload_status.classes(remove="text-red", add="text-green")
                _show_preview(state["df"], preview_container)
                _update_column_selects()
                ui.notify("数値サンプルを読み込みました", type="positive")

            with ui.row().classes("q-gutter-sm"):
                ui.button("🧪 回帰サンプル (SMILES)", on_click=load_sample_regression).props("color=purple outline")
                ui.button("📊 回帰サンプル (数値)", on_click=load_sample_numeric).props("color=blue outline")

            with ui.stepper_navigation():
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ═══════════ Step 2: 列の設定 ═══════════
        with ui.step("columns", title="🎯 列の設定"):
            ui.label("目的変数とSMILES列を選択してください").classes("text-grey-5")

            target_select = ui.select(
                options=[],
                label="目的変数",
                on_change=lambda e: state.update({"target_col": e.value or ""}),
            ).classes("full-width q-mb-md")

            smiles_select = ui.select(
                options=[],
                label="SMILES列（任意・なしでもOK）",
                on_change=lambda e: state.update({"smiles_col": e.value or ""}),
            ).classes("full-width q-mb-md").props("clearable")

            task_select = ui.select(
                options={"regression": "回帰", "classification": "分類"},
                label="タスクタイプ",
                value="regression",
                on_change=lambda e: state.update({"task_type": e.value}),
            ).classes("full-width q-mb-md")

            column_info = ui.label("").classes("text-grey-5 q-mt-sm")

            def _update_column_selects():
                """データ読み込み後にセレクターのオプションを更新"""
                if state["df"] is not None:
                    cols = list(state["df"].columns)
                    target_select.options = cols
                    target_select.update()
                    smiles_select.options = cols
                    smiles_select.update()
                    if state["target_col"] and state["target_col"] in cols:
                        target_select.value = state["target_col"]
                    if state["smiles_col"] and state["smiles_col"] in cols:
                        smiles_select.value = state["smiles_col"]
                    column_info.text = f"📋 {len(cols)}列: {', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}"

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ═══════════ Step 3: SMILES特徴量 ═══════════
        with ui.step("smiles", title="⚗️ SMILES特徴量設計"):
            ui.label("14エンジンで分子記述子を自動計算").classes("text-grey-5")

            desc_status = ui.label("").classes("q-mt-md")
            desc_progress = ui.linear_progress(value=0, show_value=False).classes("q-mt-sm")
            desc_progress.visible = False
            desc_results_container = ui.column().classes("full-width q-mt-md")

            async def calc_descriptors():
                if state["df"] is None:
                    ui.notify("まずデータを読み込んでください", type="warning")
                    return
                if not state["smiles_col"]:
                    ui.notify("SMILES列を設定してください。SMILES列がない場合はこのステップをスキップできます。", type="warning")
                    return

                desc_status.text = "⏳ 計算中..."
                desc_progress.visible = True
                desc_progress.value = 0.1

                try:
                    from backend.chem import get_available_adapters
                    available = get_available_adapters()
                    smiles_list = state["df"][state["smiles_col"]].tolist()

                    all_dfs = []
                    engine_results = {}
                    total = len(available)

                    for i, (name, adapter_cls) in enumerate(available.items()):
                        desc_status.text = f"⏳ {name} を計算中... ({i+1}/{total})"
                        desc_progress.value = (i + 1) / total
                        try:
                            adapter = adapter_cls()
                            desc_df = adapter.compute(smiles_list)
                            if desc_df is not None and len(desc_df.columns) > 0:
                                all_dfs.append(desc_df)
                                engine_results[name] = len(desc_df.columns)
                        except Exception as e:
                            engine_results[name] = f"エラー: {e}"

                    if all_dfs:
                        combined = pd.concat(all_dfs, axis=1).dropna(axis=1, how="all")
                        state["precalc_df"] = combined
                        state["selected_descriptors"] = list(combined.columns)
                        n = len(combined.columns)
                        desc_status.text = f"✅ {n}個の記述子を計算完了"
                        desc_progress.value = 1.0
                        ui.notify(f"✅ {n}個の記述子を計算完了", type="positive")

                        # 結果表示
                        desc_results_container.clear()
                        with desc_results_container:
                            with ui.row().classes("q-gutter-md"):
                                with ui.card().classes("glass-card"):
                                    ui.label(str(n)).classes("text-h4 text-bold hero-gradient")
                                    ui.label("記述子数").classes("text-caption text-grey-5")
                                with ui.card().classes("glass-card"):
                                    ok_count = sum(1 for v in engine_results.values() if isinstance(v, int))
                                    ui.label(str(ok_count)).classes("text-h4 text-bold hero-gradient")
                                    ui.label("成功エンジン").classes("text-caption text-grey-5")

                            ui.separator()
                            ui.label("エンジン別結果").classes("text-subtitle1 q-mt-md")
                            rows = []
                            for eng, val in engine_results.items():
                                if isinstance(val, int):
                                    rows.append({"エンジン": eng, "記述子数": val, "状態": "✅ 成功"})
                                else:
                                    rows.append({"エンジン": eng, "記述子数": 0, "状態": f"❌ {val}"})
                            ui.table(
                                columns=[
                                    {"name": "エンジン", "label": "エンジン", "field": "エンジン", "align": "left"},
                                    {"name": "記述子数", "label": "記述子数", "field": "記述子数"},
                                    {"name": "状態", "label": "状態", "field": "状態", "align": "left"},
                                ],
                                rows=rows,
                            ).classes("full-width")
                    else:
                        desc_status.text = "❌ 計算可能な記述子がありませんでした"
                        ui.notify("記述子を計算できませんでした", type="negative")

                except Exception as e:
                    desc_status.text = f"❌ エラー: {e}"
                    ui.notify(f"計算エラー: {e}", type="negative")
                finally:
                    desc_progress.visible = False

            ui.button(
                "⚗️ 全エンジンで記述子を計算", on_click=calc_descriptors
            ).props("color=purple size=lg").classes("q-mt-md")

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ═══════════ Step 4: 解析実行 ═══════════
        with ui.step("analysis", title="🚀 解析実行"):
            ui.label("モデル設定と解析実行").classes("text-grey-5")

            with ui.card().classes("glass-card q-mt-md"):
                ui.label("⏳ 解析実行機能は次のフェーズで実装予定です").classes("text-amber")
                ui.label("AutoML設定、CV分割、モデル選択、ハイパーパラメータチューニングが利用できるようになります。").classes(
                    "text-grey-5 q-mt-sm"
                )

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ═══════════ Step 5: 結果 ═══════════
        with ui.step("results", title="📊 結果"):
            ui.label("解析結果とモデル解釈").classes("text-grey-5")

            with ui.card().classes("glass-card q-mt-md"):
                ui.label("⏳ 結果表示機能は次のフェーズで実装予定です").classes("text-amber")
                ui.label("メトリクス表示、Plotlyチャート、SHAP解釈性が利用できるようになります。").classes(
                    "text-grey-5 q-mt-sm"
                )

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")


def _show_preview(df: pd.DataFrame, container):
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
        ui.label(f"({len(df)}行 × {len(df.columns)}列)").classes("text-caption text-grey-5 q-mt-xs")


# ─────────────────────────────────────────────
# ヘルプページ
# ─────────────────────────────────────────────
@ui.page("/help")
def help_page():
    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

    with ui.header().classes("items-center"):
        ui.link("← 戻る", "/").classes("text-white q-mr-md")
        ui.label("❓ ヘルプ - ChemAI ML Studio").classes("text-h6")

    with ui.column().classes("q-pa-lg q-gutter-md").style("max-width:900px;margin:0 auto;"):
        ui.label("ChemAI ML Studio").classes("text-h4 hero-gradient")
        ui.markdown("""
## 使い方

1. **📁 データ読込**: CSV/Excelファイルをアップロード
2. **🎯 列の設定**: 目的変数とSMILES列を選択
3. **⚗️ SMILES特徴量**: 14エンジンで分子記述子を自動計算
4. **🚀 解析実行**: モデル設定・学習・評価
5. **📊 結果**: 予測精度・SHAP解釈・可視化

## 対応エンジン

| エンジン | 記述子数 | 特徴 |
|---|---|---|
| RDKit | 200+ | 標準的な分子記述子 |
| Mordred | 73 | 1800+から厳選 |
| scikit-FP | 10+ FP | ECFP, MACCS等 |
| Mol2Vec | 300 | Word2Vec埋め込み |
| GroupContrib | 9 | Joback基団寄与法 |
| MolAI (UMA) | PCA可変 | Meta Universal Model |
| DescriptaStorus | 200+ | Merck高速記述子 |
| PaDEL | 1800+ | Java記述子(オプション) |
| Molfeat | 多数 | Datamol統合FP |
| Gasteiger | 4 | 部分電荷統計量 |

## 3つのフロントエンド

| 版 | コマンド | ポート |
|---|---|---|
| Streamlit | `streamlit run frontend_streamlit/app.py` | 8501 |
| Django | `python frontend_django/manage.py runserver` | 8000 |
| NiceGUI | `python frontend_nicegui/main.py` | 8080 |
""")


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="ChemAI ML Studio",
        dark=True,
        port=8080,
        reload=True,
        storage_secret="chemai-nicegui-secret",
    )
