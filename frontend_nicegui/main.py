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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+JP:wght@300;400;500;700&display=swap');

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

    /* F-06: フォント設定 — 科学ツールとしての信頼感 */
    --font-sans: 'Inter', 'Noto Sans JP', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;

    /* F-05: セマンティックカラー — 状態別 */
    --status-success: #4ade80;
    --status-warning: #fbbf24;
    --status-error: #f87171;
    --status-info: #60a5fa;
    --status-cancel: #fb923c;

    /* F-05: タブカテゴリ別カラー */
    --tab-data: #60a5fa;
    --tab-eda: #34d399;
    --tab-pipeline: #a78bfa;
    --tab-results: #fbbf24;
    --tab-inverse: #f472b6;
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

/* F-02: フォントサイズ14px下限 — 漢字視認性保証 */
.text-caption { font-size: 0.82rem !important; }  /* 13px → 下限保証 */
.q-field__label { font-size: 0.88rem !important; }

/* F-08: 解析開始ボタン — パルスアニメーション */
@keyframes pulse-glow {
    0% { box-shadow: 0 0 5px rgba(0,212,255,0.4); }
    50% { box-shadow: 0 0 20px rgba(0,212,255,0.7), 0 0 40px rgba(123,47,247,0.3); }
    100% { box-shadow: 0 0 5px rgba(0,212,255,0.4); }
}
.btn-run-analysis {
    animation: pulse-glow 2s ease-in-out infinite !important;
    font-size: 1.1rem !important;
    padding: 10px 28px !important;
    border-radius: 12px !important;
}
.btn-run-analysis:hover {
    animation: none !important;
    transform: scale(1.05) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.5) !important;
}

/* F-04: ローディングスピナー */
@keyframes spin { to { transform: rotate(360deg); } }
.loading-spinner {
    display: inline-block;
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.2);
    border-top-color: var(--accent-blue);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 6px;
}

/* F-07: タブ別テーマカラー（F-05セマンティック変数使用） */
.tab-data .q-tab--active { color: var(--tab-data) !important; }
.tab-eda .q-tab--active { color: var(--tab-eda) !important; }
.tab-pipeline .q-tab--active { color: var(--tab-pipeline) !important; }
.tab-results .q-tab--active { color: var(--tab-results) !important; }
.tab-inverse .q-tab--active { color: var(--tab-inverse) !important; }

/* F-15: サイドバー解析ステータスバー */
.sidebar-status-bar {
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 6px 0;
}
.sidebar-status-bar.running {
    border-color: rgba(0, 212, 255, 0.4);
    background: rgba(0, 212, 255, 0.1);
    animation: status-pulse 2s ease-in-out infinite;
}
.sidebar-status-bar.cancelled {
    border-color: rgba(251, 146, 60, 0.4);
    background: rgba(251, 146, 60, 0.08);
}
.sidebar-status-bar.done {
    border-color: rgba(74, 222, 128, 0.4);
    background: rgba(74, 222, 128, 0.08);
}
@keyframes status-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* F-03: WCAG 2.1 色覚対応 — コントラスト比4.5:1以上 */
/* 色覚多様性(CVD)対応: 赤/緑の区別に依存しないよう形状・アイコンで補完 */
.color-safe-success { color: var(--status-success); }
.color-safe-success::before { content: "✅ "; }
.color-safe-warning { color: var(--status-warning); }
.color-safe-warning::before { content: "⚠️ "; }
.color-safe-error { color: var(--status-error); }
.color-safe-error::before { content: "❌ "; }
.color-safe-info { color: var(--status-info); }
.color-safe-info::before { content: "ℹ️ "; }

/* F-03: 高コントラストモード（OSの設定連携） */
@media (prefers-contrast: high) {
    :root {
        --bg-card: rgba(255, 255, 255, 0.12);
        --border: rgba(255, 255, 255, 0.3);
        --text-primary: #ffffff;
    }
    .glass-card { border-width: 2px !important; }
}

/* F-06: フォントファミリーの統一 */
body, .q-page, .q-drawer, .q-dialog {
    font-family: var(--font-sans) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
code, pre, .text-monospace, .q-field__native {
    font-family: var(--font-mono) !important;
}
/* 日本語テキストの最小フォントサイズ保証 (F-02: 文字の大きさ) */
.text-caption { font-size: max(11px, 0.75rem) !important; }
.text-body2 { font-size: max(13px, 0.85rem) !important; }

/* F-14: Material Icon統一ルール */
/* アイコンサイズの一貫性: ボタン=20px, ラベル=16px, タイトル=24px */
.q-btn .q-icon { font-size: 20px !important; }
.text-caption .q-icon { font-size: 16px !important; }
.text-h5 .q-icon, .text-h6 .q-icon { font-size: 24px !important; }

/* ── F-22: 桜井メソッド UI拡張 (ワクワク感とフィードバック) ── */
@keyframes slide-up-fade {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}
.animate-slide-up {
    animation: slide-up-fade 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
}
.delay-100 { animation-delay: 0.1s; }
.delay-200 { animation-delay: 0.2s; }
.delay-300 { animation-delay: 0.3s; }
.delay-400 { animation-delay: 0.4s; }
.delay-500 { animation-delay: 0.5s; }

.hover-bounce {
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
}
.hover-bounce:hover {
    transform: scale(1.03) translateY(-4px) !important;
    box-shadow: 0 12px 35px rgba(0, 212, 255, 0.25) !important;
    z-index: 10;
}

@keyframes success-glow {
    0% { box-shadow: 0 0 10px rgba(74, 222, 128, 0.2); border-color: rgba(74, 222, 128, 0.3); }
    50% { box-shadow: 0 0 25px rgba(74, 222, 128, 0.6), inset 0 0 10px rgba(74, 222, 128, 0.1); border-color: rgba(74, 222, 128, 0.7); }
    100% { box-shadow: 0 0 10px rgba(74, 222, 128, 0.2); border-color: rgba(74, 222, 128, 0.3); }
}
.best-model-glow {
    animation: success-glow 3s infinite;
    background: linear-gradient(135deg, rgba(74, 222, 128, 0.08), rgba(0, 212, 255, 0.05)) !important;
}

@keyframes shake-warning {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-4px); }
    20%, 40%, 60%, 80% { transform: translateX(4px); }
}
.animate-shake {
    animation: shake-warning 0.6s cubic-bezier(.36,.07,.19,.97) both;
}
"""


# ─────────────────────────────────────────────
# プリフライトチェック（解析前の自動検証）
# ─────────────────────────────────────────────
def _preflight_check(state: dict) -> list[str]:
    """解析前にデータの不備を自動検出し、問題リストを返す。

    Returns:
        問題メッセージのリスト（空なら問題なし）
    """
    issues: list[str] = []
    df = state.get("df")
    target_col = state.get("target_col")

    if df is None:
        issues.append("📂 まずデータを読み込んでください")
        return issues

    if not target_col:
        issues.append("🎯 目的変数を設定してください")
        return issues

    # 目的変数の存在確認
    if target_col not in df.columns:
        issues.append(f"🎯 目的変数 '{target_col}' がデータに存在しません")
        return issues

    # 目的変数の欠損チェック
    na_ratio = df[target_col].isna().mean()
    if na_ratio > 0.5:
        issues.append(f"⚠️ 目的変数の欠損率が {na_ratio:.0%} と高すぎます")
    elif na_ratio > 0:
        issues.append(f"ℹ️ 目的変数に {na_ratio:.1%} の欠損があります（欠損行は自動除外されます）")

    # サンプル数チェック
    n_valid = df[target_col].notna().sum()
    if n_valid < 10:
        issues.append(f"⚠️ 有効サンプル数が {n_valid}件と少なすぎます（最低10件必要）")

    # 定数目的変数チェック
    if df[target_col].nunique() <= 1:
        issues.append("⚠️ 目的変数の値がすべて同じです")

    # 特徴量が0列
    exclude_cols = set(state.get("exclude_cols", []))
    exclude_cols.add(target_col)
    smiles_col = state.get("smiles_col")
    if smiles_col:
        exclude_cols.add(smiles_col)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    precalc_df = state.get("precalc_df")
    n_total_features = len(feature_cols) + (precalc_df.shape[1] if precalc_df is not None else 0)
    if n_total_features == 0:
        issues.append("⚠️ 特徴量が0列です（除外列の設定を見直してください）")

    return issues


# ─────────────────────────────────────────────
# メインページ
# ─────────────────────────────────────────────
@ui.page("/")
def main_page():

    # ── ページスコープの共有ステート ──
    # 空の状態で開始。ユーザーがデータを読み込むまで待機。
    state = {
        # UIモード
        "user_mode": "beginner",  # beginner / advanced
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



        with ui.button(icon="help_outline").props("flat round size=sm color=grey").tooltip(
            "ショートカット: Ctrl+Enter=解析開始 | ?=ヘルプ"
        ):
            with ui.menu().props("anchor='bottom right' self='top right'"):
                with ui.card().classes("q-pa-sm").style("min-width: 280px;"):
                    ui.label("⌨️ キーボードショートカット").classes("text-subtitle2 text-bold")
                    for key, desc in [
                        ("Ctrl + Enter", "解析開始"),
                        ("Ctrl + 1", "データ設定タブ"),
                        ("Ctrl + 2", "結果確認タブ"),
                        ("Ctrl + 3", "逆解析タブ"),
                    ]:
                        with ui.row().classes("items-center q-gutter-xs"):
                            ui.badge(key, color="grey-8").props("dense")
                            ui.label(desc).classes("text-caption")

        # ── ワンクリック解析ボタン（ヘッダー常設） ──
        analysis_status_container = ui.column().classes("full-width")

        async def _run_analysis():
            # ── プリフライトチェック ──
            issues = _preflight_check(state)
            if issues:
                for issue in issues:
                    ui.notify(issue, type="warning", timeout=5000)
                return

            # ボタン無効化（二重実行防止） + F-04 ローディングUI
            run_btn.disable()
            run_btn.text = "⏳ 解析中..."
            run_btn._classes = [c for c in run_btn._classes if c != "btn-run-analysis"]
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
                run_btn.classes("btn-run-analysis")

        # F-08: 解析開始ボタン — 大きく+パルスアニメーション+改善ツールチップ
        run_btn = ui.button(
            "🚀 解析開始", on_click=_run_analysis,
        ).classes("btn-primary btn-run-analysis").props(
            "size=lg icon=rocket_launch no-caps unelevated"
        )
        run_btn.tooltip(
            "ワンクリックで全自動ML: データ前処理 → 特徴選択 → "
            "複数モデル比較 → 最良モデル評価 → SHAP解析まで一括実行"
        )
        # _run_analysis を state に格納 → descriptor_plugins_ui から呼べるようにする
        state["_run_analysis"] = _run_analysis

    # F-11: キーボードショートカット登録
    ui.keyboard(
        on_key=lambda e: (
            _run_analysis() if e.key == 'Enter' and e.modifiers.ctrl and not e.action.repeat else
            main_tabs.set_value('data') if e.key == '1' and e.modifiers.ctrl else
            main_tabs.set_value('results') if e.key == '2' and e.modifiers.ctrl else
            main_tabs.set_value('inverse') if e.key == '3' and e.modifiers.ctrl else
            None
        ),
    )

    # ═════════════════════════════════════════════════
    # スマートデフォルト（データ特性に基づく自動設定）
    # ═════════════════════════════════════════════════
    def _apply_smart_defaults():
        """st = state のデータ特性に基づく自動設定。"""
        df = state.get("df")
        target_col = state.get("target_col")
        if df is None or not target_col:
            return

        n = len(df)
        # CV分割数: データサイズに応じて調整
        state["cv_folds"] = 5 if n > 200 else (3 if n > 50 else 2)

        # モデル選択: データサイズで絞る
        if n > 1000:
            state["selected_models"] = []  # 全モデル
        elif n > 200:
            state.setdefault("selected_models", [])
        else:
            # 少量データ: 軽量モデルのみ
            from backend.models.factory import get_default_automl_models
            task_type = state.get("task_type", "regression")
            if task_type == "auto":
                task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
            defaults = get_default_automl_models(task=task_type)
            fast_models = [m for m in defaults if "Ridge" in m or "RF" in m or "LGBM" in m or "Lasso" in m]
            state["selected_models"] = fast_models or defaults[:3]

        # スケーラー: 目的変数の分布で判断
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            target_s = df[target_col].dropna()
            if len(target_s) > 10:
                skewness = abs(target_s.skew())
                state["num_scaler"] = "standard" if skewness < 1.5 else "robust"

        # タイムアウト: データサイズに応じて
        state["timeout"] = 600 if n > 2000 else (300 if n > 500 else 120)

    # stateにスマートデフォルト関数を登録
    state["_apply_smart_defaults"] = _apply_smart_defaults

    # ═══════════════════════════════════════════════════════════
    # サイドバー — ステップインジケーター + ジャンプ + 次のステップ
    # ═══════════════════════════════════════════════════════════
    with ui.left_drawer(value=True).classes("bg-dark q-pa-md").props("width=240"):
        ui.label("⚗️ ChemAI").classes("text-h6 q-mb-sm hero-gradient")
        ui.separator()

        # ステップインジケーター
        step_container = ui.column().classes("full-width q-mt-sm")

        def _update_sidebar():
            step_container.clear()
            has_data = state["df"] is not None
            has_target = bool(state.get("target_col"))
            has_smiles = bool(state.get("smiles_col"))
            has_desc = state.get("precalc_done", False)
            has_result = state.get("automl_result") is not None

            # F-15: 解析リアルタイムステータスバー
            from frontend_nicegui.components.analysis_runner import (
                _analysis_running as is_running,
                _cancel_requested as is_cancelled,
            )
            with step_container:
                if is_running:
                    css_class = "sidebar-status-bar cancelled" if is_cancelled else "sidebar-status-bar running"
                    icon = "🛑" if is_cancelled else "⏳"
                    label = "中断処理中..." if is_cancelled else "解析実行中..."
                    ui.html(
                        f'<div class="{css_class}">'\
                        f'<span style="font-size:0.85rem;">{icon} {label}</span></div>'
                    )

            steps = [
                ("📂 データ読込", has_data),
                ("🎯 目的変数設定", has_target),
                ("🧬 SMILES検出", has_smiles),
                ("⚗️ 記述子計算", has_desc),
                ("🚀 解析完了", has_result),
            ]
            with step_container:
                # ── ステップ表示 ──
                for i, (label, done) in enumerate(steps):
                    icon = "✅" if done else "⬜"
                    color = "step-done" if done else "step-pending"
                    # 接続線（最後以外）
                    line_html = ""
                    if i < len(steps) - 1:
                        line_color = "rgba(74,222,128,0.3)" if done else "rgba(255,255,255,0.05)"
                        line_html = f'<div style="border-left:2px solid {line_color};height:8px;margin-left:10px;"></div>'
                    ui.html(
                        f'<div class="step-indicator">'
                        f'<span class="{color}" style="font-size:0.85rem;">'
                        f'{icon} {label}</span></div>{line_html}'
                    )

                # ── 次のステップヒント ──
                ui.separator().classes("q-my-xs")
                if not has_data:
                    ui.html(
                        '<div style="background:rgba(0,212,255,0.08);border-radius:8px;padding:8px;margin:4px 0;">'
                        '<span style="color:#00d4ff;font-size:0.8rem;">'
                        '👉 次: CSV/Excelをアップロード</span></div>'
                    )
                elif not has_target:
                    ui.html(
                        '<div style="background:rgba(0,212,255,0.08);border-radius:8px;padding:8px;margin:4px 0;">'
                        '<span style="color:#00d4ff;font-size:0.8rem;">'
                        '👉 次: 目的変数を設定</span></div>'
                    )
                elif not has_result:
                    ui.html(
                        '<div style="background:rgba(0,212,255,0.08);border-radius:8px;padding:8px;margin:4px 0;">'
                        '<span style="color:#00d4ff;font-size:0.8rem;">'
                        '👉 次: 🚀 解析開始ボタンで開始</span></div>'
                    )
                else:
                    ui.html(
                        '<div style="background:rgba(74,222,128,0.08);border-radius:8px;padding:8px;margin:4px 0;">'
                        '<span style="color:#4ade80;font-size:0.8rem;">'
                        '✨ 解析完了！結果タブで確認</span></div>'
                    )

                # ── データサマリー ──
                if has_data:
                    df = state["df"]
                    ui.separator().classes("q-my-xs")
                    ui.label(state.get("filename", "")).classes("text-caption text-grey-6")
                    ui.label(f"{df.shape[0]:,}行 × {df.shape[1]}列").classes("text-caption text-grey-6")
                    # ミニダッシュボード
                    na_pct = df.isna().mean().mean() * 100
                    n_numeric = df.select_dtypes(include='number').shape[1]
                    na_color = "text-green" if na_pct < 1 else ("text-amber" if na_pct < 10 else "text-red")
                    ui.label(f"欠損: {na_pct:.1f}% | 数値列: {n_numeric}").classes(f"text-caption {na_color}")
                    if state.get("target_col") and state["target_col"] in df.columns:
                        tc = df[state["target_col"]]
                        ui.label(f"目的変数: {tc.nunique()}種, 欠損{tc.isna().sum()}").classes("text-caption text-grey-7")

                if has_result:
                    ar = state["automl_result"]
                    ui.separator().classes("q-my-xs")
                    ui.label(f"🏆 {ar.best_model_key}").classes("text-caption text-cyan")
                    ui.label(f"スコア: {ar.best_score:.4f}").classes("text-caption text-grey-6")

                    # ── 結果後アクション提案 ──
                    ui.separator().classes("q-my-xs")
                    ui.label("💡 次のアクション").classes("text-caption text-grey-5 q-mb-xs")
                    ui.button(
                        "📝 レポート生成",
                        on_click=lambda: main_tabs.set_value("results"),
                    ).props("flat dense color=teal size=xs no-caps").classes("full-width")
                    ui.button(
                        "🔮 バッチ予測",
                        on_click=lambda: main_tabs.set_value("results"),
                    ).props("flat dense color=purple size=xs no-caps").classes("full-width")

        _update_sidebar()
        # タイマーで定期更新
        ui.timer(2.0, _update_sidebar)

        # ジャンプボタン
        ui.separator()
        ui.button(
            "📂 データ設定", on_click=lambda: main_tabs.set_value("data")
        ).props("flat color=white align=left size=sm no-caps").classes("full-width")
        ui.button(
            "🔬 EDA", on_click=lambda: main_tabs.set_value("eda")
        ).props("flat color=white align=left size=sm no-caps").classes("full-width")
        ui.button(
            "⚙️ 設定", on_click=lambda: main_tabs.set_value("pipeline")
        ).props("flat color=white align=left size=sm no-caps").classes("full-width")
        ui.button(
            "📊 結果確認", on_click=lambda: main_tabs.set_value("results")
        ).props("flat color=white align=left size=sm no-caps").classes("full-width")
        ui.button(
            "🔮 逆解析", on_click=lambda: main_tabs.set_value("inverse")
        ).props("flat color=white align=left size=sm no-caps").classes("full-width")

        ui.space()
        ui.separator()
        ui.link("❓ ヘルプ", "/help").classes("text-white")
        ui.link("📚 記述子辞書", "/help/descriptors").classes("text-white text-caption")
        ui.label("v2.2 — NiceGUI Edition").classes("text-caption text-grey-7 q-mt-sm")

        # ── 環境情報 ──
        import sys as _sys
        py_ver = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"
        ui.label(f"Python {py_ver}").classes("text-caption text-grey-8")

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
        eda_tab = ui.tab("eda", label="🔬 EDA", icon="query_stats")
        pipeline_tab = ui.tab("pipeline", label="⚙️ 設定", icon="tune")
        results_tab = ui.tab("results", label="📊 結果確認", icon="analytics")
        inverse_tab = ui.tab("inverse", label="🔮 逆解析", icon="find_replace")
        doe_tab = ui.tab("doe", label="🧪 実験計画", icon="science")

    with ui.tab_panels(main_tabs, value=data_tab).classes("full-width"):

        # ── データ設定タブ ──
        with ui.tab_panel(data_tab):
            from frontend_nicegui.components.data_tab import render_data_tab
            render_data_tab(state)

        # ── EDAタブ（コンテナ方式: データ読み込み後に再描画）──
        with ui.tab_panel(eda_tab):
            _eda_container = ui.column().classes("full-width")
            def _build_eda():
                _eda_container.clear()
                with _eda_container:
                    from frontend_nicegui.components.eda_panel import render_eda_panel
                    render_eda_panel(state)
            _build_eda()
            state["_refresh_eda_main"] = _build_eda

        # ── パイプライン設定タブ ──
        with ui.tab_panel(pipeline_tab):
            from frontend_nicegui.components.leakage_check_ui import render_leakage_check_panel
            render_leakage_check_panel(state)
            ui.separator().classes("q-my-sm")
            from frontend_nicegui.components.cv_config_ui import render_cv_config
            render_cv_config(state)
            ui.separator().classes("q-my-sm")
            from frontend_nicegui.components.pipeline_config_ui import render_pipeline_config
            render_pipeline_config(state)

        # ── 結果確認タブ（コンテナ方式）──
        with ui.tab_panel(results_tab):
            _results_container = ui.column().classes("full-width")
            def _build_results():
                _results_container.clear()
                with _results_container:
                    from frontend_nicegui.components.results_tab import render_results_tab
                    render_results_tab(state)
            _build_results()
            state["_refresh_results"] = _build_results

        # ── 逆解析タブ（コンテナ方式: データ読み込み後に再描画）──
        with ui.tab_panel(inverse_tab):
            _inverse_container = ui.column().classes("full-width")
            def _build_inverse():
                _inverse_container.clear()
                with _inverse_container:
                    from frontend_nicegui.components.inverse_analysis_tab import render_inverse_analysis_tab
                    render_inverse_analysis_tab(state)
            _build_inverse()
            state["_refresh_inverse"] = _build_inverse

        # ── 実験計画タブ（コンテナ方式）──
        with ui.tab_panel(doe_tab):
            _doe_container = ui.column().classes("full-width")
            def _build_doe():
                _doe_container.clear()
                with _doe_container:
                    from frontend_nicegui.components.doe_tab import render_doe_tab
                    render_doe_tab(state)
            _build_doe()
            state["_refresh_doe"] = _build_doe

    # ── SMILES列がある場合、特徴量計算をバックグラウンドで自動実行 ──
    # precalc_done=False の間だけ発火する定期ポーリング型。
    # SMILES列変更時に precalc_done=False にリセットすれば再計算がトリガーされる。
    #
    # ⚠️ Connection Lost 防止の設計:
    #   compute_all_descriptors を単一の run.io_bound で呼ぶと、重い計算（Mordred等）で
    #   WebSocket ハートビートが長時間止まり Connection Lost が発生する。
    #   対策: エンジンごとに run.io_bound を分割し、各呼び出し間でイベントループに制御を返す。
    _computing = {"active": False}  # 二重実行防止フラグ

    # エンジン定義: 全14エンジン
    # ※ is_available()=False のエンジンは _compute_one_engine が自動スキップするため安全。
    # ※ エンジンを増減してはならない。利用可否に関わらず全エンジンを常に試みること。
    _ENGINE_STEPS = [
        {
            "label": "RDKit（基本物理化学記述子 + フィンガープリント）",
            "adapter_cls": ("backend.chem.rdkit_adapter", "RDKitAdapter"),
            "kwargs": {},
        },
        {
            "label": "基団寄与法（Joback法）",
            "adapter_cls": ("backend.chem.group_contrib_adapter", "GroupContribAdapter"),
            "kwargs": {},
        },
        {
            "label": "Mordred（包括的2D/3D記述子 全計算）",
            "adapter_cls": ("backend.chem.mordred_adapter", "MordredAdapter"),
            "kwargs": {},
        },
        {
            "label": "scikit-fingerprints（ECFP・MACCS等）",
            "adapter_cls": ("backend.chem.skfp_adapter", "SkfpAdapter"),
            "kwargs": {},
        },
        {
            "label": "DescriptaStorus（Merck高速記述子）",
            "adapter_cls": ("backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter"),
            "kwargs": {},
        },
        {
            "label": "Molfeat（統合フィンガープリント）",
            "adapter_cls": ("backend.chem.molfeat_adapter", "MolfeatAdapter"),
            "kwargs": {},
        },
        {
            "label": "Mol2Vec（分子埋め込み）",
            "adapter_cls": ("backend.chem.mol2vec_adapter", "Mol2VecAdapter"),
            "kwargs": {},
        },
        {
            "label": "PaDEL（包括的記述子）",
            "adapter_cls": ("backend.chem.padel_adapter", "PaDELAdapter"),
            "kwargs": {},
        },
        {
            "label": "MolAI（CNN潜在ベクトル+PCA）",
            "adapter_cls": ("backend.chem.molai_adapter", "MolAIAdapter"),
            "kwargs": {"n_components": 6},
        },
        {
            "label": "XTB（GFN2-xTB 量子化学計算）",
            "adapter_cls": ("backend.chem.xtb_adapter", "XTBAdapter"),
            "kwargs": {},
        },
        {
            "label": "UniPKa（pKa/LogD予測）",
            "adapter_cls": ("backend.chem.unipka_adapter", "UniPkaAdapter"),
            "kwargs": {},
        },
        {
            "label": "COSMO-RS（溶媒和自由エネルギー）",
            "adapter_cls": ("backend.chem.cosmo_adapter", "CosmoAdapter"),
            "kwargs": {},
        },
        {
            "label": "UMA（Meta FAIR 量子化学）",
            "adapter_cls": ("backend.chem.uma_adapter", "UMAAdapter"),
            "kwargs": {},
        },
        {
            "label": "Chemprop（D-MPNN グラフニューラルネット）",
            "adapter_cls": ("backend.chem.chemprop_adapter", "ChempropAdapter"),
            "kwargs": {},
        },
    ]

    def _compute_one_engine(module_path: str, class_name: str, smiles_list: list, kwargs: dict):
        """1エンジンの記述子を計算する（run.io_bound で呼ぶ純粋関数）。"""
        import importlib
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            adapter = cls(**kwargs)
            if not adapter.is_available():
                return None, 0
            result = adapter.compute(smiles_list)
            df = result.descriptors
            if df is None or df.empty:
                return None, 0
            return df, df.shape[1]
        except Exception as exc:
            logger.debug(f"エンジン {class_name} スキップ: {exc}")
            return None, 0

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
            import pandas as pd

            smiles_list = state["df"][smiles_col].dropna().tolist()
            if not smiles_list:
                state["precalc_done"] = True
                return

            n_mols = len(smiles_list)
            _calc_notif = ui.notify(
                f"⚗️ SMILES特徴量を計算中（{n_mols}件）",
                type="info", timeout=0,  # 手動でdismissするまで表示
            )

            # ── エンジンごとにチャンク実行 (Connection Lost 防止) ──
            # 各 run.io_bound の間でイベントループに制御が戻り、WebSocketハートビートが維持される。
            collected_dfs: list[pd.DataFrame] = []
            n_ok = 0

            for step_info in _ENGINE_STEPS:
                label = step_info["label"]
                module_path, class_name = step_info["adapter_cls"]
                kwargs = step_info["kwargs"]

                logger.debug(f"[AutoCalc] {label} 計算中...")
                try:
                    df_eng, n_cols = await run.io_bound(
                        _compute_one_engine,
                        module_path, class_name, smiles_list, kwargs,
                    )
                    if df_eng is not None and not df_eng.empty:
                        collected_dfs.append(df_eng.reset_index(drop=True))
                        n_ok += 1
                        logger.debug(f"[AutoCalc] {label}: {n_cols}個")
                except Exception as exc:
                    logger.debug(f"[AutoCalc] {label}: スキップ ({exc})")
                # ← ここでイベントループに制御が戻る（await の効果）

            # 全て結合
            if collected_dfs:
                df_desc = pd.concat(collected_dfs, axis=1)
                df_desc = df_desc.loc[:, ~df_desc.columns.duplicated()]
                df_desc = df_desc.apply(pd.to_numeric, errors="coerce")
            else:
                df_desc = pd.DataFrame(index=range(n_mols))

            state["precalc_df"] = df_desc
            state["precalc_done"] = True
            n_desc = df_desc.shape[1]

            # 計算中通知を閉じ、完了通知を表示
            try:
                _calc_notif.dismiss()
            except Exception:
                pass
            ui.notify(
                f"✅ {n_desc}個の記述子を計算しました",
                type="positive", timeout=5000,
            )

            # 目的変数名から推薦記述子セットを自動適用
            _auto_apply_recommendation(state)

            # UIの再描画をトリガー
            refresh_fn = state.get("_refresh_tabs")
            if refresh_fn is not None:
                try:
                    refresh_fn()
                except Exception as exc:
                    logger.warning(f"[AutoCalc] UI更新失敗: {exc}")

        except Exception as e:
            logger.warning(f"[AutoCalc] 特徴量計算エラー: {e}")
            ui.notify(
                "特徴量の計算中にエラーが発生しました",
                type="warning", timeout=5000,
            )
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
                    f"📌 {rec.target_name}: {len(rec.descriptors)}記述子を推込",
                    type="info", timeout=4000,
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

### 初心者向け（最短3クリック）
1. **📂 データ読込**: CSV/Excelをアップロード（またはサンプル/ベンチマークを選択）
2. **🎯 目的変数設定**: 「列の役割」タブで予測したい列（目的変数）を選択
3. **🚀 解析開始**: ヘッダーの「解析開始」ボタンを押す → 自動でEDA・AutoML・評価・SHAP
4. **📊 結果確認**: 自動的に結果タブに切り替わります
5. **🔮 逆解析** *(任意)*: 順解析の結果を使って、目標物性を持つ条件を逆探索

### 上級者向け（詳細設定）
- **🏷️ 列の役割**: 目的変数・SMILES列の手動変更、除外列・グループ列・時系列列の設定
- **⚗️ SMILES特徴量**: 14エンジンの記述子を個別に選択（サブカテゴリ分類付き）
- **📊 EDA**: データ品質チェック・統計量サマリー
- **⚙️ 設定**: CV分割数、使用モデル、スケーラー、単調性制約
- **🔮 逆解析**: ランダムサンプリング / グリッドサーチ / ベイズ最適化 / 遺伝的アルゴリズム / MOLAI逆変換

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
# ヘルプページ: 推奨記述子データベース一覧
# ─────────────────────────────────────────────
@ui.page("/help/descriptors")
def help_descriptors_page():
    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")
    with ui.header().classes("items-center q-px-lg"):
        with ui.row().classes("items-center q-gutter-sm"):
            ui.label("⚗️").classes("text-h5")
            ui.label("ChemAI ML Studio").classes("text-h5 text-bold hero-gradient")
            ui.badge("ヘルプ", color="amber").props("floating")
        ui.button("← メインへ戻る", on_click=lambda: ui.navigate.to("/")).props(
            "flat no-caps color=cyan"
        )

    with ui.column().classes("full-width q-pa-lg"):
        from frontend_nicegui.components.descriptor_help_page import render_descriptor_help
        render_descriptor_help()


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
        reconnect_timeout=120,  # 記述子計算などの重い処理中の再接続タイムアウトを120秒に延長
    )

