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
# アプリ状態管理
# ─────────────────────────────────────────────
class AppState:
    """セッション単位の状態を管理"""
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.filename: str = ""
        self.target_col: str = ""
        self.smiles_col: str = ""
        self.task_type: str = "regression"
        self.precalc_df: pd.DataFrame | None = None
        self.selected_descriptors: list[str] = []
        self.analysis_result = None


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
    -webkit-text-fill-color: transparent;
}

.glass-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

.q-table__card { background: transparent !important; }
.q-table thead th { color: var(--text-secondary) !important; }
"""


# ─────────────────────────────────────────────
# メインページ
# ─────────────────────────────────────────────
@ui.page("/")
def main_page():
    state: AppState = app.storage.user.get("state", None)
    if state is None:
        state = AppState()
        app.storage.user["state"] = state

    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

    # ── ヘッダー ──
    with ui.header().classes("items-center justify-between"):
        ui.label("⚗️ ChemAI ML Studio").classes("text-h5 text-bold hero-gradient")
        ui.label("NiceGUI Edition").classes("text-caption text-grey-6")

    # ── サイドバー ──
    with ui.left_drawer(value=True).classes("bg-dark"):
        ui.label("⚗️ ChemAI").classes("text-h6 q-mb-md hero-gradient")

        with ui.column().classes("q-gutter-sm"):
            ui.button("📁 データ読込", on_click=lambda: stepper.set_value("data")).props("flat color=white align=left").classes("full-width")
            ui.button("🎯 列の設定", on_click=lambda: stepper.set_value("columns")).props("flat color=white align=left").classes("full-width")
            ui.button("⚗️ SMILES特徴量", on_click=lambda: stepper.set_value("smiles")).props("flat color=white align=left").classes("full-width")
            ui.button("🚀 解析実行", on_click=lambda: stepper.set_value("analysis")).props("flat color=white align=left").classes("full-width")
            ui.button("📊 結果", on_click=lambda: stepper.set_value("results")).props("flat color=white align=left").classes("full-width")

        ui.separator()
        ui.link("❓ ヘルプ", "/help").classes("text-white q-mt-md")
        ui.space()
        ui.label("NiceGUI Edition").classes("text-caption text-grey-7")

    # ── メインコンテンツ ──
    with ui.stepper().classes("full-width").props("vertical") as stepper:

        # ──── Step 1: データ読込 ────
        with ui.step("data", title="📁 データ読込"):
            ui.label("CSVファイルをアップロード、またはサンプルデータを使用").classes("text-grey-5")

            async def handle_upload(e):
                content = e.content.read()
                name = e.name
                if name.endswith(".csv"):
                    state.df = pd.read_csv(io.BytesIO(content))
                elif name.endswith((".xlsx", ".xls")):
                    state.df = pd.read_excel(io.BytesIO(content))
                state.filename = name
                upload_status.text = f"✅ {name} 読み込み完了 ({len(state.df)}行 × {len(state.df.columns)}列)"
                upload_status.classes(remove="text-red", add="text-green")
                data_table.update()

            ui.upload(
                on_upload=handle_upload,
                label="CSV / Excel をドラッグ&ドロップ",
                auto_upload=True,
            ).props('accept=".csv,.xlsx,.xls" color="purple"').classes("full-width")

            upload_status = ui.label("").classes("text-grey-5 q-mt-sm")

            # サンプルデータ
            ui.separator()
            ui.label("🧪 サンプルデータ").classes("text-subtitle1 q-mt-md")

            def load_sample_regression():
                np.random.seed(42)
                smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCCO", "CC=O",
                          "c1ccc(O)cc1", "CC(=O)OC", "CCOC", "CCN",
                          "CC(C)(C)O", "c1ccc(N)cc1", "OC(=O)c1ccccc1", "CCOCC",
                          "CC(O)CC", "c1ccc(Cl)cc1", "CC(=O)N", "CCCCCO",
                          "c1ccc(F)cc1", "CC(C)=O", "OCCO", "c1ccncc1",
                          "CC(=O)CC", "CCCCO", "c1ccc(C)cc1"]
                state.df = pd.DataFrame({
                    "SMILES": smiles,
                    "target_value": np.random.randn(25) * 2 + 5,
                })
                state.filename = "sample_regression.csv"
                state.target_col = "target_value"
                state.smiles_col = "SMILES"
                upload_status.text = f"✅ サンプルデータ読み込み完了 (25行 × 2列)"
                upload_status.classes(remove="text-red", add="text-green")
                ui.notify("サンプルデータを読み込みました", type="positive")

            with ui.row().classes("q-gutter-sm"):
                ui.button("🧪 回帰サンプル (SMILES)", on_click=load_sample_regression).props("color=purple outline")
                ui.button("📊 回帰サンプル (数値)", on_click=lambda: ui.notify("準備中")).props("color=blue outline")

            with ui.stepper_navigation():
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ──── Step 2: 列の設定 ────
        with ui.step("columns", title="🎯 列の設定"):
            ui.label("目的変数とSMILES列を選択してください").classes("text-grey-5")

            def get_columns():
                return list(state.df.columns) if state.df is not None else []

            target_select = ui.select(
                options=get_columns,
                label="目的変数",
                value=state.target_col,
                on_change=lambda e: setattr(state, "target_col", e.value),
            ).classes("full-width q-mb-md")

            smiles_select = ui.select(
                options=lambda: [""] + get_columns(),
                label="SMILES列（任意）",
                value=state.smiles_col,
                on_change=lambda e: setattr(state, "smiles_col", e.value or ""),
            ).classes("full-width q-mb-md")

            task_select = ui.select(
                options={"regression": "回帰", "classification": "分類"},
                label="タスクタイプ",
                value=state.task_type,
                on_change=lambda e: setattr(state, "task_type", e.value),
            ).classes("full-width q-mb-md")

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ──── Step 3: SMILES特徴量 ────
        with ui.step("smiles", title="⚗️ SMILES特徴量設計"):
            ui.label("14エンジンで分子記述子を自動計算").classes("text-grey-5")

            desc_status = ui.label("").classes("q-mt-md")

            async def calc_descriptors():
                if state.df is None or not state.smiles_col:
                    ui.notify("データとSMILES列を先に設定してください", type="warning")
                    return

                desc_status.text = "⏳ 計算中..."
                ui.notify("記述子計算を開始しました", type="info")

                try:
                    from backend.chem import get_available_adapters
                    available = get_available_adapters()
                    smiles_list = state.df[state.smiles_col].tolist()

                    all_dfs = []
                    for name, adapter_cls in available.items():
                        try:
                            adapter = adapter_cls()
                            desc_df = adapter.compute(smiles_list)
                            if desc_df is not None and len(desc_df.columns) > 0:
                                all_dfs.append(desc_df)
                        except Exception:
                            pass

                    if all_dfs:
                        state.precalc_df = pd.concat(all_dfs, axis=1).dropna(axis=1, how="all")
                        state.selected_descriptors = list(state.precalc_df.columns)
                        n = len(state.precalc_df.columns)
                        desc_status.text = f"✅ {n}個の記述子を計算完了（{len(available)}エンジン使用）"
                        ui.notify(f"✅ {n}個の記述子を計算完了", type="positive")
                    else:
                        desc_status.text = "❌ 計算可能な記述子がありませんでした"
                except Exception as e:
                    desc_status.text = f"❌ エラー: {e}"
                    ui.notify(f"計算エラー: {e}", type="negative")

            ui.button("⚗️ 全エンジンで記述子を計算", on_click=calc_descriptors).props("color=purple size=lg").classes("q-mt-md")

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ──── Step 4: 解析実行 ────
        with ui.step("analysis", title="🚀 解析実行"):
            ui.label("モデル設定と解析実行").classes("text-grey-5")
            ui.label("⏳ 解析実行機能は近日実装予定").classes("text-amber q-mt-lg")

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")
                ui.button("次へ →", on_click=stepper.next).props("color=primary")

        # ──── Step 5: 結果 ────
        with ui.step("results", title="📊 結果"):
            ui.label("解析結果とモデル解釈").classes("text-grey-5")
            ui.label("⏳ 結果表示機能は近日実装予定").classes("text-amber q-mt-lg")

            with ui.stepper_navigation():
                ui.button("← 戻る", on_click=stepper.previous).props("flat")


# ─────────────────────────────────────────────
# ヘルプページ
# ─────────────────────────────────────────────
@ui.page("/help")
def help_page():
    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

    with ui.header().classes("items-center"):
        ui.link("← 戻る", "/").classes("text-white q-mr-md")
        ui.label("❓ ヘルプ - ChemAI ML Studio").classes("text-h6")

    with ui.column().classes("q-pa-lg q-gutter-md"):
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

## 3つのフロントエンド

- **Streamlit版**: `streamlit run frontend_streamlit/app.py`
- **Django版**: `python frontend_django/manage.py runserver`
- **NiceGUI版**: `python frontend_nicegui/main.py`
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
