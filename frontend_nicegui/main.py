"""
ChemAI ML Studio - NiceGUI Edition
===================================
Pure Python UI using NiceGUI framework.
ステッパーUI → タブベース + サイドバーのレイアウト。
初心者はワンクリック解析、上級者は詳細設定パネルで両立。

Usage:
    python frontend_nicegui/main.py
    → http://localhost:8080
"""
from __future__ import annotations

import sys
from pathlib import Path

# backendへのパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import pandas as pd
from nicegui import ui, app

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# プレミアム ダークテーマ CSS
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
    --accent-amber: #fbbf24;
}

body {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary), #16213e) !important;
}

.nicegui-content { max-width: 1600px; margin: 0 auto; }

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

/* Primary ボタン: グラデーション */
.btn-primary {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}

/* サイドバー ステップインジケーター */
.step-indicator {
    display: flex;
    align-items: center;
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 8px;
    transition: background 0.2s;
}
.step-indicator:hover {
    background: rgba(255,255,255,0.05);
}
.step-done { color: var(--accent-green); }
.step-pending { color: #555577; }

/* メインタブのアンダーライン */
.q-tabs__content { border-bottom: 1px solid var(--border); }

/* ダークスクロールバー */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }

/* 展開パネルのスタイル */
.q-expansion-item { border-radius: 8px !important; margin-bottom: 4px; }
"""


# ─────────────────────────────────────────────
# メインページ
# ─────────────────────────────────────────────
@ui.page("/")
def main_page():

    # ── ページスコープの共有ステート ──
    # 空の状態で開始。ユーザーがデータを読み込むまで待機。
    state = {
        # データ（未読み込み）
        "df": None,
        "filename": None,
        # 列役割（未設定）
        "target_col": None,
        "smiles_col": None,
        "task_type": "regression",
        "exclude_cols": [],
        "group_col": None,
        "time_col": None,
        "weight_col": None,
        # SMILES記述子
        "precalc_df": None,
        "precalc_done": False,
        "selected_descriptors": [],
        "calc_summary": {},
        "_applied_recommendation": None,
        # パイプライン: CV
        "cv_key": "auto",
        "cv_folds": 5,
        "timeout": 300,
        # パイプライン: 前処理
        "num_scaler": "standard",
        "num_imputer": "median",
        "num_transform": "none",
        "cat_encoder": "onehot",
        "cat_imputer": "most_frequent",
        # パイプライン: 特徴量生成・選択
        "do_polynomial": False,
        "feature_selector": "none",
        # パイプライン: モデル
        "selected_models": [],
        "model_params": {},
        "monotonic_constraints": {},
        # パイプライン: フラグ
        "do_eda": True,
        "do_prep": True,
        "do_eval": True,
        "do_pca": True,
        "do_shap": True,
        # 結果
        "automl_result": None,
        "pipeline_result": None,
    }

    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

    # ═══════════════════════════════════════════════════════════
    # ヘッダー
    # ═══════════════════════════════════════════════════════════
    with ui.header().classes("items-center justify-between q-px-lg"):
        with ui.row().classes("items-center q-gutter-sm"):
            ui.label("⚗️").classes("text-h5")
            ui.label("ChemAI ML Studio").classes("text-h5 text-bold hero-gradient")
            ui.badge("NiceGUI", color="purple").props("floating")

        # ── ワンクリック解析ボタン（ヘッダー常設） ──
        analysis_status_container = ui.column().classes("full-width")

        async def _run_analysis():
            if state["df"] is None:
                ui.notify("📂 まずデータを読み込んでください", type="warning")
                return
            if not state["target_col"]:
                ui.notify("🎯 目的変数を設定してください", type="warning")
                return

            # ボタン無効化（二重実行防止）
            run_btn.disable()
            run_btn.text = "⏳ 解析中..."
            try:
                from frontend_nicegui.components.analysis_runner import run_analysis
                await run_analysis(
                    state,
                    analysis_status_container,
                    on_complete=lambda: main_tabs.set_value("results"),
                )
            finally:
                run_btn.enable()
                run_btn.text = "🚀 解析開始"

        run_btn = ui.button(
            "🚀 解析開始", on_click=_run_analysis,
        ).classes("btn-primary").props("size=md icon=rocket_launch no-caps")
        run_btn.tooltip("EDA → 前処理 → AutoML → 評価 → SHAP まで自動実行")

    # ═══════════════════════════════════════════════════════════
    # サイドバー — ステップインジケーター + ジャンプ
    # ═══════════════════════════════════════════════════════════
    with ui.left_drawer(value=True).classes("bg-dark q-pa-md").props("width=220"):
        ui.label("⚗️ ChemAI").classes("text-h6 q-mb-sm hero-gradient")
        ui.separator()

        # ステップインジケーター
        step_container = ui.column().classes("full-width q-mt-sm")

        def _update_sidebar():
            step_container.clear()
            has_data = state["df"] is not None
            has_target = bool(state.get("target_col"))
            has_smiles = bool(state.get("smiles_col"))
            has_result = state.get("automl_result") is not None

            steps = [
                ("📂 データ読込", has_data),
                ("🎯 目的変数設定", has_target),
                ("🧬 SMILES検出", has_smiles),
                ("🚀 解析完了", has_result),
            ]
            with step_container:
                for label, done in steps:
                    icon = "✅" if done else "⬜"
                    color = "step-done" if done else "step-pending"
                    ui.html(
                        f'<div class="step-indicator">'
                        f'<span class="{color}" style="font-size:0.85rem;">'
                        f'{icon} {label}</span></div>'
                    )

                # データサマリー
                if has_data:
                    df = state["df"]
                    ui.separator()
                    ui.label(state.get("filename", "")).classes("text-caption text-grey-6")
                    ui.label(f"{df.shape[0]:,}行 × {df.shape[1]}列").classes("text-caption text-grey-6")

                if has_result:
                    ar = state["automl_result"]
                    ui.separator()
                    ui.label(f"🏆 {ar.best_model_key}").classes("text-caption text-cyan")
                    ui.label(f"スコア: {ar.best_score:.4f}").classes("text-caption text-grey-6")

        _update_sidebar()
        # タイマーで定期更新
        ui.timer(2.0, _update_sidebar)

        # ジャンプボタン
        ui.separator()
        ui.button(
            "📂 データ設定", on_click=lambda: main_tabs.set_value("data")
        ).props("flat color=white align=left size=sm no-caps").classes("full-width")
        ui.button(
            "📊 結果確認", on_click=lambda: main_tabs.set_value("results")
        ).props("flat color=white align=left size=sm no-caps").classes("full-width")

        ui.space()
        ui.separator()
        ui.link("❓ ヘルプ", "/help").classes("text-white")
        ui.label("v2.0 — NiceGUI Edition").classes("text-caption text-grey-7 q-mt-sm")

    # ═══════════════════════════════════════════════════════════
    # メインコンテンツ — 2タブ構造
    # ═══════════════════════════════════════════════════════════

    # 解析状態表示エリア（タブの上）
    with analysis_status_container:
        pass  # analysis_runnerが動的に書き込む

    with ui.tabs().classes("full-width q-mt-sm").props(
        "active-color=cyan indicator-color=cyan align=left"
    ) as main_tabs:
        data_tab = ui.tab("data", label="📂 データ設定", icon="settings")
        results_tab = ui.tab("results", label="📊 結果確認", icon="analytics")

    with ui.tab_panels(main_tabs, value=data_tab).classes("full-width"):

        # ── データ設定タブ ──
        with ui.tab_panel(data_tab):
            from frontend_nicegui.components.data_tab import render_data_tab
            render_data_tab(state)

        # ── 結果確認タブ ──
        with ui.tab_panel(results_tab):
            from frontend_nicegui.components.results_tab import render_results_tab
            render_results_tab(state)

    # ── SMILES列がある場合、特徴量計算をバックグラウンドで自動実行 ──
    # precalc_done=False の間だけ発火する定期ポーリング型。
    # SMILES列変更時に precalc_done=False にリセットすれば再計算がトリガーされる。
    _computing = {"active": False}  # 二重実行防止フラグ

    async def _auto_compute_descriptors():
        if _computing["active"]:
            return  # 既に計算中
        if state["df"] is None or not state.get("smiles_col"):
            return
        if state.get("precalc_done"):
            return  # 計算済み

        smiles_col = state["smiles_col"]
        if smiles_col not in state["df"].columns:
            return

        _computing["active"] = True
        try:
            from nicegui import run
            from backend.chem.descriptors import compute_all_descriptors
            smiles_list = state["df"][smiles_col].dropna().tolist()
            if not smiles_list:
                state["precalc_done"] = True
                return
            ui.notify("⚗️ 全エンジンで記述子を自動計算中...", type="info", timeout=3000)
            df_desc = await run.io_bound(compute_all_descriptors, smiles_list)
            state["precalc_df"] = df_desc
            state["precalc_done"] = True
            ui.notify(
                f"✅ 記述子計算完了: {df_desc.shape[1]}個",
                type="positive", timeout=5000,
            )
            # 目的変数名から推薦記述子セットを自動適用
            _auto_apply_recommendation(state)
        except Exception as e:
            logger.warning(f"自動記述子計算エラー: {e}")
            ui.notify(f"⚠️ 記述子計算エラー: {e}", type="warning", timeout=5000)
            state["precalc_done"] = True  # エラー時も無限ループ防止
        finally:
            _computing["active"] = False

    def _auto_apply_recommendation(state: dict):
        """目的変数名から推薦記述子セットを自動適用する。"""
        target_col = state.get("target_col", "")
        if not target_col or state.get("_applied_recommendation"):
            return
        try:
            from backend.chem.recommender import get_target_recommendation_by_name
            rec = get_target_recommendation_by_name(target_col)
            if rec:
                state["selected_descriptors"] = [d.name for d in rec.descriptors]
                state["_applied_recommendation"] = rec
                ui.notify(
                    f"📌 推薦適用: {rec.target_name} ({len(rec.descriptors)}記述子)",
                    type="info", timeout=5000,
                )
        except ImportError:
            pass

    # 5秒ごとにチェック。precalc_done=Falseなら計算実行、Trueなら何もしない。
    ui.timer(5.0, _auto_compute_descriptors)


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

### 初心者向け（最短2クリック）
1. **📂 データ読込**: CSV/Excelをアップロード（またはサンプル/ベンチマークを選択）
2. **🚀 解析開始**: ヘッダーの「解析開始」ボタンを押す → 自動でEDA・AutoML・評価・SHAP
3. **📊 結果確認**: 自動的に結果タブに切り替わります

### 上級者向け（詳細設定）
- **🏷️ 列の役割**: 目的変数・SMILES列の手動変更、除外列・グループ列・時系列列の設定
- **⚗️ SMILES特徴量**: 14エンジンの記述子を個別に選択
- **📊 EDA**: データ品質チェック・統計量サマリー
- **⚙️ パイプライン**: CV分割数、使用モデル、スケーラー、単調性制約

## UI設計思想

| 原則 | 説明 |
|---|---|
| **Progressive Disclosure** | 初心者は自動設定で即実行。上級者は折りたたみ展開で詳細設定 |
| **ワンクリック解析** | データ読込 → 解析開始 = 最短2クリック |
| **Smart Defaults** | 目的変数・タスク種別・SMILES列を自動判定 |
| **ボタン階層** | 塗り=必須操作、Outline=オプション、Flat=詳細用 |

## 対応記述子エンジン (14種)

| エンジン | 特徴 |
|---|---|
| RDKit | 標準分子記述子 200+ |
| Mordred | 1800+から厳選 73 |
| GroupContrib | Joback基団寄与法 9 |
| DescriptaStorus | Merck高速記述子 200+ |
| MolAI | Meta Universal Model (PCA) |
| scikit-FP | ECFP, MACCS等フィンガープリント |
| UMA | Universal Molecular Adapter |
| Mol2Vec | Word2Vec分子埋め込み 300 |
| PaDEL | Java記述子 1800+ |
| Molfeat | Datamol統合FP |
| XTB | 半経験的量子化学 |
| UniPKa | pKa推定 |
| COSMO-RS | 溶媒和特性 |
| Chemprop | GNNベース記述子 |

## 3つのフロントエンド

| 版 | コマンド | ポート |
|---|---|---|
| **NiceGUI** | `python frontend_nicegui/main.py` | **8085** |
| Streamlit | `streamlit run frontend_streamlit/app.py` | 8501 |
| Django | `python frontend_django/manage.py runserver` | 8000 |
""")


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="ChemAI ML Studio",
        dark=True,
        port=8085,
        reload=False,
        storage_secret="chemai-v3-clean",
    )

