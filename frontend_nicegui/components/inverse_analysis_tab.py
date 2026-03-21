"""
frontend_nicegui/components/inverse_analysis_tab.py

逆解析（Inverse Analysis）タブ — NiceGUI版

設計思想:
  - 順解析で得た学習済みモデルを使い、目的変数の目標値から
    最適な説明変数値を逆推定する
  - パターン1: 順解析完了後に逆解析
  - パターン2: パイプライン設定時に同時設定（将来対応）
  - 最適化手法はプラガブル（ランダム/グリッド/ベイズ/GA/MOLAI/将来追加）
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 最適化手法の定義
# ═══════════════════════════════════════════════════════════
OPTIMIZATION_METHODS = [
    {
        "key": "random",
        "label": "🎲 ランダムサンプリング",
        "desc": "説明変数の制約範囲内からランダムに候補を生成。高速だが網羅性は低い。",
        "speed": "⚡高速",
        "params": [
            {"name": "n_samples", "label": "サンプル数", "type": "int", "default": 1000, "min": 100, "max": 100000},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
        ],
    },
    {
        "key": "grid",
        "label": "📐 グリッドサーチ",
        "desc": "各説明変数を等間隔に分割し全組み合わせを評価。低次元向き（3~5変数以下推奨）。",
        "speed": "🟡中速",
        "params": [
            {"name": "n_points", "label": "変数あたりの分割数", "type": "int", "default": 10, "min": 3, "max": 50},
        ],
    },
    {
        "key": "bayesian",
        "label": "🧠 ベイズ最適化",
        "desc": "ガウス過程回帰で探索を効率化。高次元でも効果的。",
        "speed": "🟡中速",
        "params": [
            {"name": "n_trials", "label": "試行回数", "type": "int", "default": 100, "min": 10, "max": 10000},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
            {"name": "acq_func", "label": "獲得関数", "type": "select",
             "options": {"EI": "Expected Improvement", "PI": "Probability of Improvement", "UCB": "Upper Confidence Bound"},
             "default": "EI"},
        ],
    },
    {
        "key": "ga",
        "label": "🧬 遺伝的アルゴリズム",
        "desc": "進化的手法で最適解を探索。多目的最適化にも対応可能。",
        "speed": "🔴低速",
        "params": [
            {"name": "pop_size", "label": "個体数", "type": "int", "default": 50, "min": 10, "max": 500},
            {"name": "n_generations", "label": "世代数", "type": "int", "default": 100, "min": 10, "max": 1000},
            {"name": "mutation_rate", "label": "突然変異率", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01},
            {"name": "crossover_rate", "label": "交叉率", "type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
        ],
    },
    {
        "key": "molai",
        "label": "🧪 MOLAI 逆変換（SMILES専用）",
        "desc": "MolAI潜在空間でベイズ最適化を行い、目標物性を持つ分子構造を生成。",
        "speed": "🔴低速",
        "params": [
            {"name": "n_trials", "label": "試行回数", "type": "int", "default": 50, "min": 10, "max": 1000},
            {"name": "latent_dim", "label": "潜在次元", "type": "int", "default": 6, "min": 2, "max": 64},
        ],
    },
]


# ═══════════════════════════════════════════════════════════
# メインレンダリング
# ═══════════════════════════════════════════════════════════
def render_inverse_analysis_tab(state: dict[str, Any]) -> None:
    """逆解析タブの全UIを描画する。"""

    has_result = state.get("automl_result") is not None
    has_data = state.get("df") is not None
    target_col = state.get("target_col", "")

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm full-width q-mb-md"):
        ui.icon("find_replace", color="purple").classes("text-h4")
        ui.label("逆解析（Inverse Analysis）").classes("text-h5")
        if has_result:
            ui.badge("順解析完了 ✅", color="green").props("outline")
        else:
            ui.badge("順解析未完了", color="amber").props("outline")

    # ── 前提条件チェック ──
    if not has_result:
        _render_prerequisite_notice(state)
        return

    # ── 逆解析設定UI ──
    # 逆解析用のサブステート
    if "_inv" not in state:
        state["_inv"] = {
            "target_mode": "range",       # range / maximize / minimize
            "target_min": None,
            "target_max": None,
            "constraints": {},            # {col: {min, max, fixed, fixed_val}}
            "method": "random",
            "method_params": {},
            "results": None,
        }
    inv = state["_inv"]

    # ── ワークフロー表示 ──
    with ui.row().classes("items-center q-gutter-sm q-mb-md"):
        ui.badge("1", color="green").props("rounded")
        ui.label("順解析完了").classes("text-body2 text-green")
        ui.icon("check", color="green")
        ui.icon("arrow_forward", color="grey")
        ui.badge("2", color="cyan").props("rounded")
        ui.label("逆解析設定").classes("text-body2 text-cyan text-bold")
        ui.icon("arrow_forward", color="grey")
        ui.badge("3", color="grey").props("rounded outline")
        ui.label("実行・結果").classes("text-body2 text-grey")

    # ── 使用モデル選択 ──
    _render_model_selector(state, inv)

    # ── 目的変数の目標設定 ──
    _render_target_settings(state, inv)

    # ── 説明変数の制約設定 ──
    _render_constraint_settings(state, inv)

    # ── 最適化手法選択 ──
    _render_method_selector(inv)

    # ── 実行ボタン ──
    _render_execute_section(state, inv)

    # ── 結果表示 ──
    if inv.get("results") is not None:
        _render_results(state, inv)


# ═══════════════════════════════════════════════════════════
# 前提条件未達成メッセージ
# ═══════════════════════════════════════════════════════════
def _render_prerequisite_notice(state: dict) -> None:
    """順解析が完了していない場合のガイド表示。"""
    with ui.card().classes("full-width q-pa-lg").style(
        "border: 2px dashed rgba(251,191,36,0.5); border-radius: 12px;"
        "background: rgba(50,40,0,0.2);"
    ):
        with ui.column().classes("items-center full-width q-gutter-md"):
            ui.icon("science", color="amber").classes("text-h2")
            ui.label("まず順解析を実行してください").classes("text-h6 text-amber")
            ui.label(
                "逆解析には学習済みモデルが必要です。\n"
                "「データ設定」タブでデータを読み込み、「解析開始」ボタンで順解析を実行すると、\n"
                "そのモデルを使って最適な説明変数値を探索できます。"
            ).classes("text-body2 text-grey text-center").style("white-space: pre-line;")

            # ── ワークフローガイド ──
            with ui.row().classes("items-center q-gutter-sm"):
                ui.badge("1", color="amber").props("rounded")
                ui.label("データ読込").classes("text-body2 text-amber")
                ui.icon("arrow_forward", color="grey")
                ui.badge("2", color="amber").props("rounded outline")
                ui.label("目的変数設定").classes("text-body2 text-amber")
                ui.icon("arrow_forward", color="grey")
                ui.badge("3", color="amber").props("rounded outline")
                ui.label("順解析実行").classes("text-body2 text-amber")
                ui.icon("arrow_forward", color="grey")
                ui.badge("4", color="grey").props("rounded outline")
                ui.label("逆解析").classes("text-body2 text-grey")


# ═══════════════════════════════════════════════════════════
# 使用モデル選択
# ═══════════════════════════════════════════════════════════
def _render_model_selector(state: dict, inv: dict) -> None:
    """逆解析に使用するモデルの選択。"""
    ar = state.get("automl_result")
    if ar is None:
        return

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(0,188,212,0.3); border-radius: 10px;"
        "background: rgba(0,20,40,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("model_training", color="cyan").classes("text-h6")
            ui.label("使用モデル").classes("text-subtitle1 text-bold")

        # ベストモデル表示
        best_key = ar.best_model_key if hasattr(ar, "best_model_key") else "不明"
        best_score = ar.best_score if hasattr(ar, "best_score") else 0.0

        with ui.row().classes("items-center q-gutter-sm"):
            ui.chip(f"🏆 {best_key}", color="cyan").props("outline")
            ui.label(f"スコア: {best_score:.4f}").classes("text-caption text-grey")
            ui.badge("自動選択", color="green").props("outline dense")

        # 他のモデルがある場合は選択可能に
        if hasattr(ar, "all_scores") and ar.all_scores:
            model_options = {k: f"{k} (Score: {v:.4f})" for k, v in ar.all_scores.items()}
            inv.setdefault("selected_model", best_key)

            ui.select(
                model_options,
                value=inv["selected_model"],
                label="使用するモデルを変更",
                on_change=lambda e: inv.update({"selected_model": e.value}),
            ).props("dense outlined").classes("full-width q-mt-sm")


# ═══════════════════════════════════════════════════════════
# 目的変数の目標設定
# ═══════════════════════════════════════════════════════════
def _render_target_settings(state: dict, inv: dict) -> None:
    """目的変数の目標値/範囲を設定する。"""
    target_col = state.get("target_col", "不明")
    df = state.get("df")

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(123,47,247,0.3); border-radius: 10px;"
        "background: rgba(30,10,50,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("target", color="purple").classes("text-h6")
            ui.label("目的変数の目標設定").classes("text-subtitle1 text-bold")
            ui.badge(target_col, color="purple").props("outline")

        # 現在のデータ範囲を表示
        if df is not None and target_col in df.columns:
            col_data = df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(col_data):
                with ui.row().classes("q-gutter-md text-caption text-grey q-mb-sm"):
                    ui.label(f"データ範囲: {col_data.min():.4g} ～ {col_data.max():.4g}")
                    ui.label(f"平均: {col_data.mean():.4g}")
                    ui.label(f"標準偏差: {col_data.std():.4g}")

        # 目標タイプ選択
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            target_mode = ui.toggle(
                {"range": "📏 範囲指定", "maximize": "📈 最大化", "minimize": "📉 最小化"},
                value=inv.get("target_mode", "range"),
                on_change=lambda e: inv.update({"target_mode": e.value}),
            ).props("no-caps dense color=purple")

        # 範囲指定の場合のみ入力欄を表示
        current_mode = inv.get("target_mode", "range")
        if current_mode == "range":
            with ui.row().classes("q-gutter-md full-width"):
                target_min = ui.number(
                    "目標 最小値",
                    value=inv.get("target_min"),
                    on_change=lambda e: inv.update({"target_min": e.value}),
                ).props("dense outlined").classes("w-48")

                target_max = ui.number(
                    "目標 最大値",
                    value=inv.get("target_max"),
                    on_change=lambda e: inv.update({"target_max": e.value}),
                ).props("dense outlined").classes("w-48")
        elif current_mode == "maximize":
            ui.label("目的変数を最大化するよう説明変数を探索します").classes("text-body2 text-grey")
        elif current_mode == "minimize":
            ui.label("目的変数を最小化するよう説明変数を探索します").classes("text-body2 text-grey")


# ═══════════════════════════════════════════════════════════
# 説明変数の制約設定
# ═══════════════════════════════════════════════════════════
def _render_constraint_settings(state: dict, inv: dict) -> None:
    """各説明変数の探索範囲・固定値・ON/OFFを設定するテーブル。"""
    df = state.get("df")
    target_col = state.get("target_col", "")
    precalc_df = state.get("precalc_df")

    if df is None:
        return

    # 説明変数列を特定（目的変数・SMILES列・除外列以外）
    exclude = set(state.get("exclude_cols", []))
    smiles_col = state.get("smiles_col", "")
    feature_cols = [
        c for c in df.columns
        if c != target_col and c != smiles_col and c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # SMILES記述子も候補に追加
    if precalc_df is not None:
        selected_descs = state.get("selected_descriptors", [])
        if selected_descs:
            feature_cols = list(set(feature_cols + selected_descs))

    if not feature_cols:
        ui.label("数値型の説明変数がありません").classes("text-amber")
        return

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(74,222,128,0.3); border-radius: 10px;"
        "background: rgba(10,40,20,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("tune", color="green").classes("text-h6")
            ui.label("説明変数の制約范囲").classes("text-subtitle1 text-bold")
            ui.badge(f"{len(feature_cols)}変数", color="green").props("outline")

        ui.label(
            "各説明変数の探索範囲を設定します。固定するとその値に固定されます。"
        ).classes("text-caption text-grey q-mb-sm")

        # ── 一括操作 ──
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            def _auto_range():
                """データの最小最大を自動設定"""
                for col in feature_cols:
                    source = precalc_df if precalc_df is not None and col in precalc_df.columns else df
                    if col in source.columns:
                        col_data = source[col].dropna()
                        if len(col_data) > 0:
                            inv["constraints"][col] = {
                                "min": float(col_data.min()),
                                "max": float(col_data.max()),
                                "fixed": False,
                                "fixed_val": float(col_data.median()),
                                "active": True,
                            }
                ui.notify(f"✅ {len(feature_cols)}変数のデータ範囲を自動設定", type="positive")

            ui.button("📊 データ範囲を自動設定", on_click=_auto_range).props(
                "outline size=sm no-caps color=green"
            )

            def _expand_range():
                """範囲を±20%拡張"""
                for col, c in inv.get("constraints", {}).items():
                    span = (c.get("max", 0) - c.get("min", 0)) * 0.2
                    c["min"] = c.get("min", 0) - span
                    c["max"] = c.get("max", 0) + span
                ui.notify("範囲を±20%拡張しました", type="info")

            ui.button("↔️ 範囲を±20%拡張", on_click=_expand_range).props(
                "flat size=sm no-caps color=grey"
            )

        # 制約テーブル（最初の20変数を表示、残りは折りたたみ）
        display_cols = feature_cols[:20]
        remaining = feature_cols[20:]

        _render_constraint_table(display_cols, df, precalc_df, inv)

        if remaining:
            with ui.expansion(
                f"📋 残り{len(remaining)}変数を表示", icon="expand_more",
            ).classes("full-width"):
                _render_constraint_table(remaining, df, precalc_df, inv)


def _render_constraint_table(
    cols: list[str], df: pd.DataFrame, precalc_df: pd.DataFrame | None, inv: dict,
) -> None:
    """制約設定テーブルのレンダリング。"""
    for col in cols:
        # データソースからデフォルト値を取得
        source = precalc_df if precalc_df is not None and col in precalc_df.columns else df
        if col not in source.columns:
            continue

        col_data = source[col].dropna()
        if len(col_data) == 0:
            continue

        data_min = float(col_data.min())
        data_max = float(col_data.max())
        data_median = float(col_data.median())

        # 既存の制約またはデフォルト
        constraint = inv.setdefault("constraints", {}).setdefault(col, {
            "min": data_min,
            "max": data_max,
            "fixed": False,
            "fixed_val": data_median,
            "active": True,
        })

        with ui.row().classes(
            "items-center full-width q-py-xs"
        ).style("border-bottom: 1px solid rgba(255,255,255,0.05);"):
            # 変数名
            ui.label(col).classes("text-body2").style("min-width: 160px; max-width: 200px;")

            # 固定チェック
            fixed_cb = ui.checkbox(
                "固定",
                value=constraint.get("fixed", False),
                on_change=lambda e, c=constraint: c.update({"fixed": e.value}),
            ).props("dense").classes("q-mr-sm")

            if not constraint.get("fixed", False):
                # 最小値
                ui.number(
                    "Min",
                    value=constraint.get("min", data_min),
                    on_change=lambda e, c=constraint: c.update({"min": e.value}),
                ).props("dense outlined").style("width: 100px;")

                # 最大値
                ui.number(
                    "Max",
                    value=constraint.get("max", data_max),
                    on_change=lambda e, c=constraint: c.update({"max": e.value}),
                ).props("dense outlined").style("width: 100px;")

                # データ範囲参考
                ui.label(f"(データ: {data_min:.3g}~{data_max:.3g})").classes(
                    "text-caption text-grey"
                )
            else:
                # 固定値
                ui.number(
                    "固定値",
                    value=constraint.get("fixed_val", data_median),
                    on_change=lambda e, c=constraint: c.update({"fixed_val": e.value}),
                ).props("dense outlined").style("width: 120px;")
                ui.label(f"(中央値: {data_median:.3g})").classes("text-caption text-grey")


# ═══════════════════════════════════════════════════════════
# 最適化手法選択
# ═══════════════════════════════════════════════════════════
def _render_method_selector(inv: dict) -> None:
    """最適化手法の選択とパラメータ設定。"""
    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(251,191,36,0.3); border-radius: 10px;"
        "background: rgba(40,30,0,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("psychology", color="amber").classes("text-h6")
            ui.label("最適化手法").classes("text-subtitle1 text-bold")

        # 手法カード
        for method in OPTIMIZATION_METHODS:
            is_selected = inv.get("method") == method["key"]
            border_color = "rgba(251,191,36,0.6)" if is_selected else "rgba(255,255,255,0.08)"
            bg_color = "rgba(50,40,0,0.4)" if is_selected else "rgba(30,30,30,0.2)"

            def _select_method(m=method):
                inv["method"] = m["key"]
                # デフォルトパラメータを設定
                inv["method_params"] = {
                    p["name"]: p["default"] for p in m["params"]
                }

            with ui.card().classes("full-width q-pa-sm q-mb-xs cursor-pointer").style(
                f"border: 1px solid {border_color}; border-radius: 8px;"
                f"background: {bg_color};"
            ).on("click", _select_method):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.radio(
                        {method["key"]: ""},
                        value=inv.get("method"),
                    ).props("dense").on_value_change(lambda e, m=method: inv.update({"method": m["key"]}))

                    with ui.column().classes("q-gutter-none"):
                        with ui.row().classes("items-center q-gutter-xs"):
                            ui.label(method["label"]).classes(
                                "text-body1 text-bold" if is_selected else "text-body1"
                            )
                            ui.badge(method["speed"]).props("outline dense")
                        ui.label(method["desc"]).classes("text-caption text-grey")

            # 選択中の手法のパラメータ設定
            if is_selected and method["params"]:
                with ui.expansion(
                    "⚙️ パラメータ設定", icon="settings",
                ).classes("full-width q-mb-xs").props("dense"):
                    _render_method_params(method, inv)


def _render_method_params(method: dict, inv: dict) -> None:
    """手法固有のパラメータ設定UI。"""
    if "method_params" not in inv:
        inv["method_params"] = {}

    for param in method["params"]:
        name = param["name"]
        label = param["label"]
        p_type = param["type"]
        default = param["default"]

        current = inv["method_params"].get(name, default)

        if p_type == "int":
            ui.number(
                label,
                value=current,
                min=param.get("min", 0),
                max=param.get("max", 99999),
                step=1,
                on_change=lambda e, n=name: inv["method_params"].update({n: int(e.value) if e.value else default}),
            ).props("dense outlined").classes("w-48")
        elif p_type == "float":
            ui.number(
                label,
                value=current,
                min=param.get("min", 0),
                max=param.get("max", 1),
                step=param.get("step", 0.01),
                on_change=lambda e, n=name: inv["method_params"].update({n: float(e.value) if e.value else default}),
            ).props("dense outlined").classes("w-48")
        elif p_type == "select":
            ui.select(
                param["options"],
                value=current,
                label=label,
                on_change=lambda e, n=name: inv["method_params"].update({n: e.value}),
            ).props("dense outlined").classes("w-48")


# ═══════════════════════════════════════════════════════════
# 実行セクション
# ═══════════════════════════════════════════════════════════
def _render_execute_section(state: dict, inv: dict) -> None:
    """逆解析の実行ボタンと進捗表示。"""
    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 2px solid rgba(0,212,255,0.5); border-radius: 12px;"
        "background: linear-gradient(135deg, rgba(0,40,80,0.6), rgba(0,20,60,0.4));"
    ):
        # 設定サマリー
        method_name = next(
            (m["label"] for m in OPTIMIZATION_METHODS if m["key"] == inv.get("method")),
            "未選択"
        )
        target_mode = {"range": "範囲指定", "maximize": "最大化", "minimize": "最小化"}.get(
            inv.get("target_mode", "range"), "不明"
        )
        n_active = sum(
            1 for c in inv.get("constraints", {}).values()
            if c.get("active", True) and not c.get("fixed", False)
        )
        n_fixed = sum(
            1 for c in inv.get("constraints", {}).values()
            if c.get("fixed", False)
        )

        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("summarize", color="cyan").classes("text-h6")
            ui.label("実行サマリー").classes("text-subtitle1 text-bold")

        with ui.row().classes("q-gutter-md text-body2"):
            ui.chip(f"手法: {method_name}").props("outline dense")
            ui.chip(f"目標: {target_mode}").props("outline dense")
            ui.chip(f"探索変数: {n_active}個").props("outline dense color=green")
            if n_fixed > 0:
                ui.chip(f"固定変数: {n_fixed}個").props("outline dense color=amber")

        ui.separator().classes("q-my-sm")

        # 進捗表示エリア
        progress_container = ui.column().classes("full-width q-mb-sm")
        progress_container.set_visibility(False)

        with progress_container:
            inv_progress = ui.linear_progress(
                value=0, show_value=False, color="purple",
            ).props("rounded instant-feedback stripe").style("height: 8px;")
            inv_progress_label = ui.label("準備中...").classes("text-caption text-purple")

        # 実行ボタン
        async def _run_inverse():
            inv_btn.disable()
            inv_btn.text = "⏳ 逆解析実行中..."
            progress_container.set_visibility(True)
            inv_progress.value = 0.1
            inv_progress_label.text = "逆解析を実行中..."

            try:
                # TODO: 実際の逆解析ロジックを実装
                # 現在はダミー結果を生成して動作確認
                import asyncio
                await asyncio.sleep(1)  # 模擬計算

                inv_progress.value = 0.5
                inv_progress_label.text = "候補を評価中..."
                await asyncio.sleep(0.5)

                # ダミー結果生成（将来は実際の最適化結果に置換）
                active_cols = [
                    col for col, c in inv.get("constraints", {}).items()
                    if c.get("active", True) and not c.get("fixed", False)
                ]
                n_candidates = min(20, inv.get("method_params", {}).get("n_samples", 10))

                results = []
                for i in range(n_candidates):
                    row = {"rank": i + 1}
                    for col in active_cols[:10]:
                        c = inv["constraints"][col]
                        cmin = c.get("min", 0)
                        cmax = c.get("max", 1)
                        row[col] = round(np.random.uniform(cmin, cmax), 4)
                    row["predicted"] = round(np.random.uniform(
                        inv.get("target_min", 0) or 0,
                        inv.get("target_max", 10) or 10,
                    ), 4)
                    results.append(row)

                inv["results"] = pd.DataFrame(results)
                inv_progress.value = 1.0
                inv_progress_label.text = f"✅ {n_candidates}件の候補を発見"
                ui.notify(
                    f"逆解析完了: {n_candidates}件の候補",
                    type="positive", timeout=5000,
                )
            except Exception as e:
                inv_progress_label.text = f"エラー: {e}"
                ui.notify(f"逆解析エラー: {e}", type="warning")
            finally:
                inv_btn.enable()
                inv_btn.text = "🔮 逆解析を実行"

        with ui.row().classes("items-center q-gutter-sm"):
            inv_btn = ui.button(
                "🔮 逆解析を実行",
                on_click=_run_inverse,
            ).props("unelevated size=lg no-caps color=purple").classes("text-bold").style(
                "font-size: 1.05rem;"
            )
            inv_btn.tooltip("設定した条件で逆解析を実行します")

            ui.label(
                "※ 現在はプロトタイプ版です。実際の最適化ロジックは後日実装されます。"
            ).classes("text-caption text-grey")


# ═══════════════════════════════════════════════════════════
# 結果表示
# ═══════════════════════════════════════════════════════════
def _render_results(state: dict, inv: dict) -> None:
    """逆解析結果の表示。"""
    results_df = inv.get("results")
    if results_df is None or results_df.empty:
        return

    with ui.card().classes("full-width q-pa-md q-mt-md").style(
        "border: 1px solid rgba(74,222,128,0.4); border-radius: 12px;"
        "background: rgba(10,40,20,0.3);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("emoji_events", color="green").classes("text-h5")
            ui.label("逆解析結果").classes("text-h6 text-bold")
            ui.badge(f"{len(results_df)}件の候補", color="green").props("outline")

        target_col = state.get("target_col", "predicted")

        # テーブル表示
        columns = [
            {"name": c, "label": c, "field": c, "sortable": True}
            for c in results_df.columns
        ]
        rows = results_df.to_dict("records")

        ui.table(
            columns=columns,
            rows=rows,
            row_key="rank",
            pagination={"rowsPerPage": 10},
        ).classes("full-width").props("dense flat bordered")

        # ダウンロードボタン
        with ui.row().classes("q-gutter-sm q-mt-sm"):
            def _download_csv():
                csv_data = results_df.to_csv(index=False)
                ui.download(csv_data.encode("utf-8"), "inverse_results.csv")

            ui.button(
                "📥 CSVダウンロード",
                on_click=_download_csv,
            ).props("outline size=sm no-caps color=green")

            ui.button(
                "📋 クリップボードにコピー",
                on_click=lambda: ui.notify("クリップボードにコピーしました", type="info"),
            ).props("flat size=sm no-caps color=grey")
