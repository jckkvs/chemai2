# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/pipeline_config_ui.py

Pipeline 全設定UI — NiceGUI版
7ステップのタブ形式で全パイプラインを設定。
複数選択 = 全組み合わせを評価 / 未選択 = 適切なデフォルトを自動適用。

Streamlit版のpipeline_config_ui.pyと機能等価。
"""
from __future__ import annotations

from typing import Any
from nicegui import ui


# ═══════════════════════════════════════════════════════════
# ヘルパー
# ═══════════════════════════════════════════════════════════

def _section(icon: str, title: str, desc: str = "") -> None:
    """セクションヘッダー。"""
    ui.label(f"{icon} {title}").classes("text-subtitle1 text-bold q-mt-sm")
    if desc:
        ui.label(desc).classes("text-caption text-grey q-mb-xs").style("font-size:0.75rem;")


def _glass_card():
    """ガラスカードのコンテキストマネージャー。"""
    return ui.card().classes("full-width q-pa-sm q-mb-xs").style(
        "border:1px solid rgba(0,188,212,0.2); border-radius:8px;"
        "background:rgba(0,20,40,0.25);"
    )


# ═══════════════════════════════════════════════════════════
# Tab 0: Excluder
# ═══════════════════════════════════════════════════════════

def _tab_excluder(state: dict) -> None:
    _section("🚫", "Excluder（解析除外列）",
             "解析に使わない列を選択。目的変数・SMILES列は自動除外済み。")
    df = state.get("df")
    if df is None:
        ui.label("データ未読み込み").classes("text-caption text-grey")
        return
    target_col = state.get("target_col", "")
    smiles_col = state.get("smiles_col", "")
    skip = {c for c in (target_col, smiles_col) if c}
    opts = [c for c in df.columns if c not in skip]
    if not opts:
        ui.label("除外できる列がありません").classes("text-caption text-grey")
        return
    prev = state.get("exclude_cols", [])

    def _on_change(e):
        state["exclude_cols"] = list(e.value)
    ui.select(
        opts, multiple=True, value=[c for c in prev if c in opts],
        label="除外列を選択",
        on_change=_on_change,
    ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 1: 数値前処理
# ═══════════════════════════════════════════════════════════

_NUM_IMPUTERS = [
    ("mean", "Mean（平均）"),
    ("median", "Median（中央値）"),
    ("knn", "KNN Imputer"),
    ("iterative", "Iterative Imputer"),
    ("constant", "Constant（固定値）"),
]

_NUM_SCALERS = [
    ("standard", "StandardScaler"),
    ("minmax", "MinMaxScaler"),
    ("robust", "RobustScaler"),
    ("maxabs", "MaxAbsScaler"),
    ("power_yj", "PowerTransformer [YJ]"),
    ("power_bc", "PowerTransformer [BC]"),
    ("quantile_normal", "QuantileTransformer→正規"),
    ("quantile_uniform", "QuantileTransformer→一様"),
    ("none", "スケーリングなし"),
]


def _tab_numeric(state: dict) -> None:
    _section("🔢", "数値列前処理（Imputer × Scaler）",
             "選択した Imputer と Scaler の全組み合わせを評価します。")

    # ── Imputer ──
    ui.label("📊 Imputer（欠損補間）").classes("text-body2 text-bold q-mt-sm")
    imp_keys = state.get("_pg_num_imputers", ["mean"])
    imp_options = [k for k, _ in _NUM_IMPUTERS]
    imp_labels = {k: v for k, v in _NUM_IMPUTERS}

    def _on_imp(e):
        state["_pg_num_imputers"] = list(e.value)
    ui.select(
        imp_options, multiple=True, value=imp_keys,
        label="Imputer（複数選択可）",
        on_change=_on_imp,
    ).props("use-chips outlined dense").classes("full-width")

    ui.separator().classes("q-my-xs")

    # ── Scaler ──
    ui.label("📏 Scaler").classes("text-body2 text-bold")
    scl_keys = state.get("_pg_num_scalers", ["standard"])
    scl_options = [k for k, _ in _NUM_SCALERS]

    def _on_scl(e):
        state["_pg_num_scalers"] = list(e.value)
    ui.select(
        scl_options, multiple=True, value=scl_keys,
        label="Scaler（複数選択可）",
        on_change=_on_scl,
    ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 2: カテゴリ前処理
# ═══════════════════════════════════════════════════════════

_CAT_IMPUTERS = [
    ("most_frequent", "Most Frequent（最頻値）"),
    ("constant", "Constant（指定文字列）"),
    ("knn", "KNN Imputer"),
]

_LOW_ENCODERS = [
    ("onehot", "OneHotEncoder"),
    ("ordinal", "OrdinalEncoder"),
    ("target", "TargetEncoder"),
    ("binary", "BinaryEncoder"),
]

_HIGH_ENCODERS = [
    ("ordinal", "OrdinalEncoder"),
    ("target", "TargetEncoder"),
    ("hashing", "HashingEncoder"),
    ("binary", "BinaryEncoder"),
    ("leaveoneout", "LeaveOneOut"),
]


def _tab_categorical(state: dict) -> None:
    _section("🏷️", "カテゴリ列前処理（Imputer × Encoder）",
             "低カーディナリティと高カーディナリティで別設定可能。")

    # Imputer
    ui.label("🔤 Categorical Imputer").classes("text-body2 text-bold q-mt-sm")
    ci_keys = state.get("_pg_cat_imputers", ["most_frequent"])
    ci_options = [k for k, _ in _CAT_IMPUTERS]

    def _on_ci(e):
        state["_pg_cat_imputers"] = list(e.value)
    ui.select(ci_options, multiple=True, value=ci_keys,
              label="Imputer", on_change=_on_ci,
              ).props("use-chips outlined dense").classes("full-width")

    ui.separator().classes("q-my-xs")

    # Low / High Encoder
    with ui.row().classes("full-width q-gutter-sm"):
        with ui.column().classes("col"):
            ui.label("🔻 低カーディナリティ").classes("text-body2 text-bold")
            le_keys = state.get("_pg_low_encoders", ["onehot"])
            le_options = [k for k, _ in _LOW_ENCODERS]

            def _on_le(e):
                state["_pg_low_encoders"] = list(e.value)
            ui.select(le_options, multiple=True, value=le_keys,
                      label="Encoder", on_change=_on_le,
                      ).props("use-chips outlined dense").classes("full-width")

        with ui.column().classes("col"):
            ui.label("🔺 高カーディナリティ").classes("text-body2 text-bold")
            he_keys = state.get("_pg_high_encoders", ["ordinal"])
            he_options = [k for k, _ in _HIGH_ENCODERS]

            def _on_he(e):
                state["_pg_high_encoders"] = list(e.value)
            ui.select(he_options, multiple=True, value=he_keys,
                      label="Encoder", on_change=_on_he,
                      ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 3: バイナリ前処理
# ═══════════════════════════════════════════════════════════

def _tab_binary(state: dict) -> None:
    _section("⚡", "バイナリ列前処理",
             "0/1, True/False などの2値列の処理設定。")
    bi_imps = state.get("_pg_bin_imputers", ["most_frequent"])

    def _on_bi(e):
        state["_pg_bin_imputers"] = list(e.value)
    ui.select(
        ["most_frequent", "constant", "knn"], multiple=True, value=bi_imps,
        label="Imputer", on_change=_on_bi,
    ).props("use-chips outlined dense").classes("full-width")

    be_keys = state.get("_pg_bin_encoders", ["ordinal"])

    def _on_be(e):
        state["_pg_bin_encoders"] = list(e.value)
    ui.select(
        ["ordinal", "passthrough"], multiple=True, value=be_keys,
        label="Encoder", on_change=_on_be,
    ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 4: 特徴生成
# ═══════════════════════════════════════════════════════════

def _tab_engineer(state: dict) -> None:
    _section("🔧", "Feature Engineering",
             "複数選択で全パターンを評価。未選択 → none。")
    eng_keys = state.get("_pg_engineer", ["none"])

    def _on_eng(e):
        state["_pg_engineer"] = list(e.value)
    ui.select(
        ["none", "polynomial", "interaction_only"],
        multiple=True, value=eng_keys,
        label="生成手法", on_change=_on_eng,
    ).props("use-chips outlined dense").classes("full-width")

    # Polynomial パラメータ
    if "polynomial" in state.get("_pg_engineer", []):
        with _glass_card():
            ui.label("PolynomialFeatures 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "degree", value=state.get("_pg_poly_degree", 2),
                    min=2, max=5, step=1,
                    on_change=lambda e: state.update({"_pg_poly_degree": int(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.checkbox(
                    "interaction_only",
                    value=state.get("_pg_poly_ia", False),
                    on_change=lambda e: state.update({"_pg_poly_ia": e.value}),
                )


# ═══════════════════════════════════════════════════════════
# Tab 5: 特徴選択
# ═══════════════════════════════════════════════════════════

_SELECTORS = [
    ("none", "なし（全特徴量使用）"),
    ("lasso", "Lasso（線形ペナルティ）"),
    ("rfr", "RF重要度（SelectFromModel）"),
    ("select_kbest", "SelectKBest"),
    ("select_percentile", "SelectPercentile"),
    ("boruta", "Boruta"),
]


def _tab_selector(state: dict) -> None:
    _section("🎯", "Feature Selector",
             "複数選択で全組み合わせを評価。未選択 → none。")
    sel_keys = state.get("_pg_selectors", ["none"])
    sel_options = [k for k, _ in _SELECTORS]

    def _on_sel(e):
        state["_pg_selectors"] = list(e.value)
    ui.select(
        sel_options, multiple=True, value=sel_keys,
        label="特徴選択手法", on_change=_on_sel,
    ).props("use-chips outlined dense").classes("full-width")

    # Lasso パラメータ
    if "lasso" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("Lasso 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "alpha", value=state.get("_pg_lasso_alpha", 0.01),
                    min=1e-6, max=10.0, step=0.001, format="%.6f",
                    on_change=lambda e: state.update({"_pg_lasso_alpha": float(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.number(
                    "max_iter", value=state.get("_pg_lasso_mi", 1000),
                    min=100, max=10000, step=100,
                    on_change=lambda e: state.update({"_pg_lasso_mi": int(e.value)}),
                ).props("outlined dense").classes("col-4")

    # SelectKBest パラメータ
    if "select_kbest" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("SelectKBest 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "k", value=state.get("_pg_kbest_k", 10),
                    min=1, max=500, step=1,
                    on_change=lambda e: state.update({"_pg_kbest_k": int(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.select(
                    ["f_regression", "mutual_info_regression", "r_regression",
                     "f_classif", "mutual_info_classif"],
                    value=state.get("_pg_kbest_sf", "f_regression"),
                    label="score_func",
                    on_change=lambda e: state.update({"_pg_kbest_sf": e.value}),
                ).props("outlined dense").classes("col-6")

    # Boruta パラメータ
    if "boruta" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("Boruta 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "n_estimators", value=state.get("_pg_boruta_n", 100),
                    min=10, max=500, step=10,
                    on_change=lambda e: state.update({"_pg_boruta_n": int(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.number(
                    "max_iter", value=state.get("_pg_boruta_mi", 100),
                    min=10, max=500, step=10,
                    on_change=lambda e: state.update({"_pg_boruta_mi": int(e.value)}),
                ).props("outlined dense").classes("col-4")


# ═══════════════════════════════════════════════════════════
# Tab 6: 推定器
# ═══════════════════════════════════════════════════════════

def _tab_estimator(state: dict) -> None:
    task = state.get("task_type", "regression")
    is_reg = task == "regression"
    _section("🤖", f"Estimator（{'回帰' if is_reg else '分類'}）",
             "使用するモデルを選択。factory.py から自動検出。")

    try:
        from backend.models.factory import list_models, get_default_automl_models
        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)
    except Exception as ex:
        ui.label(f"⚠️ モデル一覧取得エラー: {ex}").classes("text-caption text-red")
        return

    # カテゴリ分類
    categories: dict[str, list] = {
        "📐 線形系": [],
        "🌲 決定木/アンサンブル": [],
        "⚙️ カーネル/その他": [],
    }
    for m in available:
        k = (m["key"] + m["name"]).lower()
        if any(x in k for x in ["linear", "ridge", "lasso", "elastic", "logistic", "ard", "huber", "pls", "bayesian"]):
            categories["📐 線形系"].append(m)
        elif any(x in k for x in ["tree", "forest", "boost", "gbm", "gradient", "rgf", "figs", "rule", "hist", "catboost"]):
            categories["🌲 決定木/アンサンブル"].append(m)
        else:
            categories["⚙️ カーネル/その他"].append(m)

    selected_models = state.get("selected_models", [])
    if not selected_models:
        selected_models = list(defaults)

    # 一括操作
    with ui.row().classes("q-gutter-xs q-mb-sm"):
        def _select_all():
            all_keys = [m["key"] for m in available]
            state["selected_models"] = all_keys
        def _select_defaults():
            state["selected_models"] = list(defaults)
        def _select_none():
            state["selected_models"] = []
        ui.button("全選択", on_click=_select_all).props("flat dense size=xs color=cyan no-caps")
        ui.button("推奨のみ", on_click=_select_defaults).props("flat dense size=xs color=teal no-caps")
        ui.button("全解除", on_click=_select_none).props("flat dense size=xs color=grey no-caps")

    # カテゴリごとに展開パネル
    for cat_name, models in categories.items():
        if not models:
            continue
        n_selected = sum(1 for m in models if m["key"] in selected_models)
        with ui.expansion(
            f"{cat_name} ({n_selected}/{len(models)})",
        ).classes("full-width q-mb-xs").props("dense"):
            for m in models:
                mkey = m["key"]
                is_checked = mkey in selected_models

                def _toggle(e, key=mkey):
                    sm = state.get("selected_models", list(defaults))
                    if e.value:
                        if key not in sm:
                            sm.append(key)
                    else:
                        sm = [k for k in sm if k != key]
                    state["selected_models"] = sm

                with ui.row().classes("items-center q-gutter-xs"):
                    ui.checkbox(
                        m["name"], value=is_checked,
                        on_change=_toggle,
                    )
                    if mkey in defaults:
                        ui.badge("推奨", color="teal").props("dense").style("font-size:0.6rem;")


# ═══════════════════════════════════════════════════════════
# 組み合わせ数サマリー
# ═══════════════════════════════════════════════════════════

def _render_combo_summary(state: dict) -> None:
    """リアルタイムで「何通りのパイプラインか」を表示。"""
    n_imp = max(1, len(state.get("_pg_num_imputers", ["mean"])))
    n_scl = max(1, len(state.get("_pg_num_scalers", ["standard"])))
    n_ci = max(1, len(state.get("_pg_cat_imputers", ["most_frequent"])))
    n_le = max(1, len(state.get("_pg_low_encoders", ["onehot"])))
    n_bi = max(1, len(state.get("_pg_bin_imputers", ["most_frequent"])))
    n_eng = max(1, len(state.get("_pg_engineer", ["none"])))
    n_sel = max(1, len(state.get("_pg_selectors", ["none"])))
    n_est = max(1, len(state.get("selected_models", [])))
    n_total = n_imp * n_scl * n_ci * n_le * n_bi * n_eng * n_sel * n_est

    status_color = "teal" if n_total <= 50 else ("amber" if n_total <= 200 else "red")
    status_text = "✅ 適切" if n_total <= 50 else ("⚠️ やや多い" if n_total <= 200 else "🔴 多すぎ")

    with ui.card().classes("full-width q-pa-sm").style(
        f"border:2px solid var(--q-{status_color});border-radius:8px;"
        "background:rgba(0,20,40,0.3);"
    ):
        with ui.row().classes("items-center justify-between full-width"):
            ui.label(f"🔢 評価パイプライン数: {n_total:,} 通り").classes("text-body1 text-bold")
            ui.badge(status_text, color=status_color)
        ui.label(
            f"imp×{n_imp} · scl×{n_scl} · cat×{n_ci} · enc×{n_le} "
            f"· bin×{n_bi} · eng×{n_eng} · sel×{n_sel} · est×{n_est}"
        ).classes("text-caption text-grey").style("font-size:0.7rem;")


# ═══════════════════════════════════════════════════════════
# メインエントリーポイント
# ═══════════════════════════════════════════════════════════

def render_pipeline_config(state: dict) -> None:
    """パイプライン全設定UIをレンダリングする。

    7ステップのタブ形式で全パイプラインを設定。
    全組み合わせ数をリアルタイム表示。
    """
    with ui.card().classes("full-width q-pa-md").style(
        "border:1px solid rgba(0,188,212,0.3);border-radius:12px;"
        "background:rgba(0,20,40,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("tune", color="cyan").classes("text-h5")
            ui.label("Pipeline 全設定（STEP 0〜6）").classes("text-h6")

        ui.label(
            "ステップをタブで切替え → 各ステップでアルゴリズムを選択。"
            "複数選択した場合は全組み合わせを自動評価。"
        ).classes("text-caption text-grey q-mb-sm")

        # ── 7ステップタブ ──
        with ui.tabs().classes("full-width").props(
            "dense no-caps active-color=cyan indicator-color=cyan"
        ) as pg_tabs:
            ui.tab("pg_excl", label="🚫 除外")
            ui.tab("pg_num", label="🔢 数値")
            ui.tab("pg_cat", label="🏷️ カテゴリ")
            ui.tab("pg_bin", label="⚡ バイナリ")
            ui.tab("pg_eng", label="🔧 特徴生成")
            ui.tab("pg_sel", label="🎯 特徴選択")
            ui.tab("pg_est", label="🤖 推定器")

        with ui.tab_panels(pg_tabs, value="pg_excl").classes("full-width"):
            with ui.tab_panel("pg_excl"):
                _tab_excluder(state)
            with ui.tab_panel("pg_num"):
                _tab_numeric(state)
            with ui.tab_panel("pg_cat"):
                _tab_categorical(state)
            with ui.tab_panel("pg_bin"):
                _tab_binary(state)
            with ui.tab_panel("pg_eng"):
                _tab_engineer(state)
            with ui.tab_panel("pg_sel"):
                _tab_selector(state)
            with ui.tab_panel("pg_est"):
                _tab_estimator(state)

        # ── 組み合わせ数サマリー ──
        ui.separator().classes("q-my-sm")
        _render_combo_summary(state)
