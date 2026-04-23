"""
ChemAI Nexus - NiceGUI Edition
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

from nicegui import ui, app
from backend.utils.compatibility import CompatibilityManager

# 起動時互換性チェック（既存機能に影響を与えない非侵襲的実装）
_compat = CompatibilityManager()
_compat.suppress_runtime_warnings()
_env_check = _compat.check_environment()
if _env_check["recommendations"]:
    for rec in _env_check["recommendations"]:
        logging.getLogger(__name__).warning(f"[環境推奨] {rec}")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# アプリ起動時: 設定を環境変数に適用
# ─────────────────────────────────────────────
try:
    from backend.config.settings_manager import SettingsManager as _SM
    _SM.get_instance().apply_to_environment()
except Exception as _e:
    logger.warning("設定の自動適用に失敗しました: %s", _e)

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
.text-caption { font-size: max(13px, 0.82rem) !important; }
.text-body2  { font-size: max(14px, 0.88rem) !important; }
.text-body1  { font-size: max(15px, 0.94rem) !important; }

/* ── フォントサイズ全体的な引き上げ（ユーザー要望） ── */
body, .nicegui-content, .q-page {
    font-size: 16px !important;
}
.q-item__label, .q-field__label, .q-field__native {
    font-size: 15px !important;
}
.q-btn:not(.q-btn--size-xs):not(.q-btn--size-sm) {
    font-size: 15px !important;
}
.q-table tbody td, .q-table thead th {
    font-size: 14px !important;
}
.q-tab__label {
    font-size: 14px !important;
    font-weight: 600;
}
/* サイドバーステップラベル */
.q-drawer .q-item__label {
    font-size: 15px !important;
}
/* キャプション下限 13px で日本語可読性保証 */
.text-caption, .text-overline {
    font-size: 13px !important;
}

/* F-14: Material Icon統一ルール */
/* アイコンサイズの一貫性: ボタン=20px, ラベル=16px, タイトル=24px */
.q-btn .q-icon { font-size: 20px !important; }
.text-caption .q-icon { font-size: 16px !important; }
.text-h5 .q-icon, .text-h6 .q-icon { font-size: 24px !important; }

    /* ── F-22: UI拡張 (ワクワク感とフィードバック) ── */
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

    /* ── モダンなヘッダーデザイン ── */
    .app-title {
        background: linear-gradient(to right, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* サイドバーのグラデーション */
    .nicegui-drawer {
        background: linear-gradient(180deg, #1a1a2e 0%, #0d0d1a 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    .q-header {
        background: linear-gradient(to right, #4f46e5, #9333ea, #db2777) !important; /* Indigo via Purple to Pink */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
    if n_valid < 2:
        issues.append(f"❌ 有効サンプル数が {n_valid}件と少なすぎます（最低2件必要）")

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
async def main_page():
    # ── 初回起動時: モデル存在確認リダイレクト ──
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "llm_analyzer.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_id = cfg.get("model_id", "jckkvs/bonsai-8b-1.58bit")
        local_dir = os.path.join(os.path.dirname(__file__), "..", "models", os.path.basename(model_id))
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            ui.navigate.to("/setup_model")
            return
    except Exception:
        pass

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
        # 単調性制約とメタデータ (Item 13)
        "monotonicity_constraints": {
            "_global": {
                "default_direction": "none",
                "default_strength": 0.5,
                "default_sigma": 3.0,
                "apply_to_new_features": True
            },
            "_by_feature": {},
            "_by_set": {}
        },
        "feature_classification": {},
        "feature_stats": {},
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
        # メトリック評価エンジン
        "metric_evaluator": None,
        "metric_cache": {},
        "available_categories": [],
    }

    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

    # ═══════════════════════════════════════════════════════════
    # ヘッダー完全削除 — 解析ロジックのみ定義
    # ═══════════════════════════════════════════════════════════
    # analysis_status_container はサイドバー内で定義（下記）

    # ── 解析実行ロジック（ヘッダー削除後もサイドバーから呼び出す）──
    def _open_settings():
        from frontend_nicegui.pages.settings_page import open_settings_dialog
        open_settings_dialog()

    async def _run_analysis():
        # ── プリフライトチェック ──
        issues = _preflight_check(state)
        if issues:
            for issue in issues:
                ui.notify(issue, type="warning", timeout=5000)
            return

        # ボタン無効化（二重実行防止）
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

    # state に格納 → descriptor_plugins_ui から呼べるようにする
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
    # サイドバー (Left Drawer) — ナビゲーション & ステータス
    # ═══════════════════════════════════════════════════════════
    with ui.left_drawer(value=True).classes("bg-dark q-pa-none").props("width=260 elevated") as drawer:
        with ui.column().classes("full-height no-wrap"):
            # --- ヘッダーエリア ---
            with ui.column().classes("full-width q-pa-md bg-secondary"):
                with ui.row().classes("items-center justify-between full-width"):
                    ui.label("⚗️ ChemAI Nexus").classes("text-h6 hero-gradient font-bold")
                    ui.button(icon="menu_open", on_click=drawer.toggle).props("flat round color=grey")
                
                # 🚀 解析開始ボタン
                run_btn = ui.button(
                    "🚀 解析開始", on_click=_run_analysis,
                ).classes("start-button full-width q-mt-sm").props(
                    "size=md icon=rocket_launch no-caps unelevated"
                )
                
                # 解析進捗
                analysis_status_container = ui.column().classes("full-width q-mt-sm")
                with analysis_status_container:
                    pass

            ui.separator()

            # --- メインナビゲーション ---
            with ui.column().classes("full-width q-pa-sm q-gutter-y-xs"):
                ui.label("📊 メインワークフロー").classes("text-caption text-grey-5 q-ml-sm q-mt-sm")
                
                def nav_item(icon: str, label: str, tab_id: str):
                    return ui.button(label, icon=icon, on_click=lambda: main_tabs.set_value(tab_id)) \
                        .props("flat align=left no-caps").classes("full-width text-white hover-bounce")

                nav_item("folder", "データ管理", "data")
                nav_item("analytics", "EDA・可視化", "eda")
                nav_item("psychology", "機械学習", "ml")
                nav_item("assignment", "結果・レポート", "results")

            ui.separator().classes("q-mx-md")

            # --- 専門解析ツール ---
            with ui.column().classes("full-width q-pa-sm"):
                with ui.expansion("🔬 専門解析", icon="science").classes("full-width text-white").props("dense"):
                    nav_item("find_replace", "逆解析・最適", "inverse").classes("q-pl-lg")
                    nav_item("biotech", "実験計画 (DoE)", "doe").classes("q-pl-lg")
                    nav_item("cloud_download", "外部モデル連携", "models").classes("q-pl-lg")

                with ui.expansion("⚛️ 量子化学", icon="hub").classes("full-width text-white").props("dense"):
                    nav_item("speed", "計算管理", "computation").classes("q-pl-lg")
                    nav_item("science", "量子特徴量", "quantum").classes("q-pl-lg")

            ui.separator().classes("q-mx-md")

            # --- システム・設定 ---
            with ui.column().classes("full-width q-pa-sm"):
                with ui.expansion("⚙️ システム", icon="settings").classes("full-width text-white").props("dense"):
                    ui.button("アプリ設定", icon="settings", on_click=_open_settings).props("flat align=left no-caps").classes("full-width q-pl-lg")
                    nav_item("delete_sweep", "キャッシュ管理", "data").classes("q-pl-lg") # 仮

            ui.element('div').classes('flex-grow')
            
            # --- フッターエリア (ステップ進捗) ---
            with ui.column().classes("full-width q-pa-md bg-black-10"):
                step_container = ui.column().classes("full-width")
                
                def _update_sidebar():
                    step_container.clear()
                    has_data = state["df"] is not None
                    has_target = bool(state.get("target_col"))
                    has_smiles = bool(state.get("smiles_col"))
                    has_desc = state.get("precalc_done", False)
                    has_result = state.get("automl_result") is not None
                    
                    from frontend_nicegui.components.analysis_runner import _analysis_running as is_running
                    with step_container:
                        if is_running:
                            ui.html('<div class="sidebar-status-bar running"><span style="font-size:0.8rem;">⏳ 解析実行中...</span></div>')
                        
                        steps = [
                            ("受入", has_data),
                            ("SMI", has_smiles),
                            ("計算", has_desc),
                            ("完了", has_result),
                        ]
                        with ui.row().classes("full-width justify-around q-mb-sm"):
                            for label, done in steps:
                                color = "var(--accent-green)" if done else "#555"
                                icon = "●" if done else "○"
                                ui.label(f"{icon}{label}").style(f"color: {color}; font-size: 0.65rem; font-weight: bold;").tooltip(label)
                        
                        ui.label(f"v2.5 — Premium Edition").classes("text-caption text-grey-8 text-center full-width")

                _update_sidebar()
                ui.timer(5.0, _update_sidebar)

    # ── プレミアム CSS スタイル定義 (シンプル & プロフェッショナル) ──
    ui.add_css('''
        .nicegui-header {
            background-color: #1e293b !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        .nicegui-drawer {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        .app-title {
            color: #ffffff;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        
        .start-button {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            transition: all 0.2s ease !important;
        }
        
        .start-button:hover {
            background-color: rgba(255, 255, 255, 0.12) !important;
            transform: translateY(-1px);
        }
    ''')

    # ── プレミアム ヘッダー (トップレベルに配置必須) ──
    with ui.header().classes('bg-slate-800 shadow-sm'):
        with ui.row().classes('w-full items-center justify-between q-pa-md'):
            # ロゴ・タイトル
            with ui.row().classes('items-center gap-3 cursor-pointer').on('click', lambda: main_tabs.set_value('data')):
                ui.icon('science', size='32px', color='white').classes('opacity-80')
                ui.label('ChemAI Nexus').classes('text-2xl font-bold text-white app-title')
            
            # 右側アクション
            with ui.row().classes('items-center gap-4'):
                ui.button(icon="menu", on_click=drawer.toggle).props("flat round dense color=white")
                ui.button(icon="help_outline", on_click=lambda: ui.navigate.to("/help")).props("flat round color=white")
                ui.avatar(icon="person", color="slate-700")

    # ═══════════════════════════════════════════════════════════
    # メインコンテンツ — 統合ワークフロー
    # ═══════════════════════════════════════════════════════════
    with ui.column().classes("full-width items-stretch"):
    
        # ── メインタブ (4つに削減) ──
        with ui.tabs().classes("full-width q-px-md").props("active-color=cyan indicator-color=cyan align=left") as main_tabs:
            tab_data = ui.tab("data", label="📁 データ管理", icon="folder")
            tab_eda = ui.tab("eda", label="📊 EDA・可視化", icon="analytics")
            tab_ml = ui.tab("ml", label="🤖 機械学習", icon="psychology")
            tab_results = ui.tab("results", label="📑 結果・レポート", icon="assignment")
            
            # 非表示タブ (サイドバーからの遷移用)
            tab_inverse = ui.tab("inverse").classes("hidden")
            tab_doe = ui.tab("doe").classes("hidden")
            tab_models = ui.tab("models").classes("hidden")
            tab_computation = ui.tab("computation").classes("hidden")
            tab_quantum = ui.tab("quantum").classes("hidden")

        with ui.tab_panels(main_tabs, value="data").classes("full-width q-pa-md bg-transparent") as panels:
            
            # 1. 📁 データ管理
            with ui.tab_panel("data"):
                from frontend_nicegui.components.data_tab import render_data_tab
                render_data_tab(state)

            # 2. 📊 EDA・可視化
            with ui.tab_panel("eda"):
                _eda_container = ui.column().classes("full-width")
                def _build_eda():
                    _eda_container.clear()
                    with _eda_container:
                        from frontend_nicegui.components.eda_panel import render_eda_panel
                        render_eda_panel(state)
                _build_eda()
                state["_refresh_eda_main"] = _build_eda

            # 3. 🤖 機械学習
            with ui.tab_panel("ml"):
                from frontend_nicegui.components.ml_workflow import render_ml_workflow
                render_ml_workflow(state)

            # 4. 📑 結果・レポート
            with ui.tab_panel("results"):
                from frontend_nicegui.components.results_view_container import render_results_view_container
                render_results_view_container(state)

            # --- 専門ツールパネル (Hidden Tabs) ---
            with ui.tab_panel("inverse"):
                from frontend_nicegui.components.inverse_tab import render_inverse_panel
                render_inverse_panel(state)

            with ui.tab_panel("doe"):
                from frontend_nicegui.components.doe_tab import render_doe_tab
                render_doe_tab(state)

            with ui.tab_panel("models"):
                from frontend_nicegui.pages.model_manager import render_model_manager
                render_model_manager()

            with ui.tab_panel("computation"):
                from frontend_nicegui.components.computation_progress import render_computation_progress
                render_computation_progress(state)

            with ui.tab_panel("quantum"):
                from frontend_nicegui.components.quantum_feature_explorer import render_quantum_feature_explorer
                render_quantum_feature_explorer(state)

    # ── タブ遷移コールバック登録 ──
    state["_switch_to_inverse"] = lambda: main_tabs.set_value("inverse")
    state["_switch_to_results"] = lambda: main_tabs.set_value("results")
    state["_switch_to_data_smiles"] = lambda: main_tabs.set_value("data")

    _REBUILD_MAP = {
        "eda":        "_refresh_eda_main",
        "results":    "_refresh_results", # results_view_container 内の refresh が必要かも
        "inverse":    "_refresh_inverse",
        "doe":        "_refresh_doe",
        "computation": "_refresh_computation",
        "quantum":     "_refresh_quantum",
    }

    def _on_tab_change(e):
        tab_val = getattr(e, "value", None) or str(e)
        key = _REBUILD_MAP.get(str(tab_val))
        if key and key in state:
            state[key]()
        
    main_tabs.on_value_change(_on_tab_change)

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
            # timeout=0は永久表示→dismiss()が効かないケースがあるため
            # 十分長いtimeout (10分) を設定し、完了前に消えないようにする
            _calc_notif = ui.notify(
                f"⚗️ SMILES特徴量を計算中（{n_mols}件）",
                type="info", timeout=600000,  # 10分（計算完了時にdismiss試行）
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
                try:
                    _calc_notif.close()
                except Exception:
                    # NiceGUIのバージョンによりdismiss/closeが使えない場合
                    # JavaScriptで全通知をクリア
                    try:
                        ui.run_javascript("document.querySelectorAll('.q-notification').forEach(n => n.remove())")
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
        ui.label("❓ ヘルプ - ChemAI Nexus").classes("text-h6")

    with ui.column().classes("q-pa-lg q-gutter-md").style("max-width:900px;margin:0 auto;"):
        ui.label("ChemAI Nexus").classes("text-h4 hero-gradient")
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
            ui.label("ChemAI Nexus").classes("text-h5 text-bold hero-gradient")
            ui.badge("ヘルプ", color="amber").props("floating")
        ui.button("← メインへ戻る", on_click=lambda: ui.navigate.to("/")).props(
            "flat no-caps color=cyan"
        )

    with ui.column().classes("full-width q-pa-lg"):
        from frontend_nicegui.components.descriptor_help_page import render_descriptor_help
        render_descriptor_help()

# Plot viewer moved to plot_utils
# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────
if __name__ in {"__main__", "__mp_main__"}:
    from backend.utils.config import IS_WINDOWS
    ui.run(
        title="ChemAI Nexus",
        dark=True,
        port=8085,
        reload=False,
        workers=1,  # Windowsでの安定性のためにワーカーを1に制限
        storage_secret="chemai-v3-clean",
        reconnect_timeout=120,
    )


# ─────────────────────────────────────────────
# モデルセットアップ・初期化シーケンス (Item 20-2: 自動ダウンロード)
# ─────────────────────────────────────────────
from backend.services.model_manager import ModelManager

@ui.page('/setup_model', title='モデル初期化')
async def model_setup_page():
    """初回起動時のモデルダウンロード専用ページ（既存機能を破壊せず分離）"""
    with ui.card().classes('w-full max-w-2xl mx-auto mt-20 p-8 glass-card animate-slide-up'):
        ui.label('📦 AIモデルの初期化中...').classes('text-2xl font-bold text-white mb-4 hero-gradient')
        progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full mb-4').props('stripe animate color=cyan')
        status_label = ui.label('接続確認中...').classes('text-gray-300 text-lg')
        
        log_area = ui.scroll_area().classes('w-full h-32 bg-black/30 rounded p-2 mt-4 text-xs text-green-400 font-mono')
        def log(msg):
            with log_area:
                ui.label(f"> {msg}")
            log_area.scroll_to(percent=1.0)

        async def run_download():
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "llm_analyzer.yaml")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                cfg = {"model_id": "jckkvs/bonsai-8b-1.58bit"}
                
            model_id = cfg.get("model_id", "jckkvs/bonsai-8b-1.58bit")
            local_dir = os.path.join(os.path.dirname(__file__), "..", "models", os.path.basename(model_id))
            manager = ModelManager(model_id, local_dir)
            
            log(f"ターゲットモデル: {model_id}")
            log(f"保存先: {local_dir}")
            
            status_label.text = f"モデル確認中..."
            await asyncio.sleep(1.0)
            
            if manager._check_local_exists():
                log("✅ ローカルにモデルを確認しました。")
                status_label.text = "✅ インストール済みです。"
                progress_bar.value = 1.0
                await asyncio.sleep(1.0)
                ui.navigate.to("/")
            else:
                log("⬇️ ローカルにモデルが見つかりません。ダウンロードを開始します...")
                status_label.text = "⬇️ ダウンロード中... (初回のみ1〜5分程度かかります)"
                progress_bar.value = 0.3
                
                loop = asyncio.get_event_loop()
                try:
                    # ディスク容量チェック（簡易）
                    import shutil
                    total, used, free = shutil.disk_usage("/")
                    free_gb = free // (2**30)
                    log(f"空き容量確認: {free_gb} GB")
                    if free_gb < 10:
                        log("⚠️ 警告: 空き容量が10GB未満です。ダウンロードに失敗する可能性があります。")
                        ui.notify("空き容量が不足している可能性があります(推奨10GB以上)", type='warning')

                    # ダウンロード実行
                    await loop.run_in_executor(None, lambda: manager.ensure_downloaded())
                    
                    log("✅ ダウンロード完了")
                    progress_bar.value = 1.0
                    status_label.text = "✅ 初期化完了。メイン画面に移動します。"
                    await asyncio.sleep(2.0)
                    ui.navigate.to("/")
                except Exception as e:
                    log(f"❌ エラー発生: {str(e)}")
                    status_label.text = "❌ 初期化に失敗しました。"
                    status_label.classes(replace='text-red-400')
                    progress_bar.props('color=red')
                    
                    with ui.column().classes('mt-4 q-gutter-sm'):
                        ui.label('対処方法:').classes('text-white font-bold')
                        ui.markdown(f"1. ネットワーク接続を確認してください。\n2. 手動で `models/{os.path.basename(model_id)}` にモデルファイルを配置してください。").classes('text-gray-400 text-sm')
                        ui.button('再試行', icon='refresh', on_click=lambda: ui.navigate.to('/setup_model')).props('outline color=cyan')
                        ui.button('メイン画面へ強行移動', icon='arrow_forward', on_click=lambda: ui.navigate.to('/')).props('flat color=grey')

    ui.timer(0.5, run_download, once=True)

# ─────────────────────────────────────────────
# データ読み込みデバッグ・検証統合 (Item 21: データ認識不具合対応)
# ─────────────────────────────────────────────
from datetime import datetime
from backend.utils.data_validator import DataValidator

# デバッグ用グローバル状態監視（既存のデータ管理を破壊せず追加）
_data_debug_info = {
    "last_loaded_df": None,
    "load_timestamp": None,
    "load_status": "not_loaded",
    "validation_message": None,
    "details": None
}

def get_current_data_status():
    """現在のデータ状態を取得（既存関数との互換性維持）"""
    global _data_debug_info
    return _data_debug_info.get('last_loaded_df')

def set_loaded_data(df):
    """データ読み込み状態を更新（既存の読み込み処理から呼び出し可能）"""
    global _data_debug_info
    
    is_valid, msg, details = DataValidator.validate_dataframe(df)
    
    _data_debug_info.update({
        "last_loaded_df": df,
        "load_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "load_status": "loaded" if is_valid else "failed",
        "validation_message": msg,
        "details": details
    })
    logger.info(f"データ状態更新: {_data_debug_info['load_status']} - {msg}")

@ui.page('/debug_data', title='データ読み込みデバッグ')
async def debug_data_page():
    """データ読み込み状態を詳細に表示するデバッグページ"""
    with ui.card().classes('w-full max-w-4xl mx-auto mt-10 p-6 glass-card animate-slide-up'):
        ui.label('🔍 データ読み込みデバッグ情報').classes('text-2xl font-bold text-white mb-4 hero-gradient')
        
        with ui.row().classes('items-center gap-4 mb-6'):
            status_color = 'green' if _data_debug_info['load_status'] == 'loaded' else 'red'
            ui.badge(f"状態: {_data_debug_info['load_status']}", color=status_color).classes('text-lg p-2')
            ui.label(f"最終更新: {_data_debug_info['load_timestamp'] or 'なし'}").classes('text-gray-400')

        if _data_debug_info.get('validation_message'):
            with ui.card().classes('bg-black/20 p-3 mb-4 border border-white/10'):
                ui.label('検証メッセージ:').classes('text-gray-400 text-xs')
                ui.label(_data_debug_info['validation_message']).classes('text-white')
        
        if _data_debug_info.get('details'):
            with ui.expansion('📊 データ構造詳細', icon='analytics').classes('text-white'):
                ui.json(_data_debug_info['details']).classes('text-xs')
        
        if _data_debug_info.get('last_loaded_df') is not None:
            df = _data_debug_info['last_loaded_df']
            with ui.expansion('📋 データプレビュー（先頭5行）', icon='table_chart').classes('text-white'):
                ui.table.from_pandas(df.head()).classes('text-xs bg-white/5')
        
        # 診断レポート
        report = DataValidator.generate_diagnostic_report(_data_debug_info.get('last_loaded_df'))
        with ui.expansion('📜 診断レポート全文', icon='assignment').classes('text-white'):
            ui.markdown(f"```text\n{report}\n```").classes('text-xs font-mono bg-black/40 p-2 rounded')
            
        ui.button('メイン画面に戻る', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).classes('mt-6').props('outline color=cyan')

# ─────────────────────────────────────────────
# LLM 解析エンジン統合 (Item 20: 拡張型LLM解析)
# ─────────────────────────────────────────────
import asyncio
import yaml
from backend.services.llm_data_analyzer import LLMDataAnalyzer

async def render_llm_analysis_report(df, metadata=None):
    """LLM解析結果を非同期で取得し、既存UIに並列表示"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "llm_analyzer.yaml")
    analyzer = LLMDataAnalyzer(config_path=config_path)
    
    # 既存の analysis_status_container はサイドバー等に配置されていることを想定
    # ここでは独立した通知とカード表示を行う
    
    with ui.card().classes('w-full mt-4 bg-blue-50 border-l-4 border-blue-500 p-4 shadow-sm animate-slide-up'):
        ui.label('🤖 AI 解析方針レポート生成中...').classes('text-lg font-bold text-blue-800')
        spinner = ui.spinner(size='lg', color='primary')
        
        try:
            result = await analyzer.analyze(df, metadata)
            spinner.delete()
            
            if 'error' in result:
                ui.notification(result.get('error', 'Unknown error'), type='warning', position='top')
                ui.label(f'⚠️ {result.get("error")}').classes('text-red-600')
                return
                
            ui.label('🤖 AI 解析方針レポート').classes('text-lg font-bold text-blue-800 mb-3')
            
            sections = [
                ('📊 データ概要', result.get('data_overview', '情報なし')),
                ('🛠 前処理推奨', result.get('preprocessing', '情報なし')),
                ('🧬 特徴量エンジニアリング', result.get('feature_engineering', '情報なし')),
                ('🤖 モデル候補', '\n'.join([f'- {m}' for m in result.get('model_candidates', [])]) if isinstance(result.get('model_candidates'), list) else result.get('model_candidates', '情報なし')),
                ('📈 検証戦略', result.get('validation_strategy', '情報なし')),
                ('🔍 解釈性計画', result.get('interpretation_plan', '情報なし')),
            ]
            if result.get('cautions'):
                sections.append(('⚠️ 注意点', result.get('cautions')))
                
            for title, content in sections:
                with ui.expansion(title, icon='chevron_right').props('group=llm_analysis').open():
                    ui.markdown(str(content)).classes('text-gray-800')
        except Exception as e:
            spinner.delete()
            ui.label(f'❌ 解析失敗: {e}').classes('text-red-600')

# 全文表示要件のため、以下に反映後の main.py 全文（末尾追記分含む）を提示します。
