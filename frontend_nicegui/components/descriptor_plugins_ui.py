"""
frontend_nicegui/components/descriptor_plugins_ui.py

SMILES記述子パネル — NiceGUI版

設計思想:
  - ユーザーは「選択」するだけ。計算は自動（バックグラウンド）
  - 記述子テーブルに必要な情報をすべて表示（意味・相関・カーディナリティ・ソース）
  - 目的変数に応じた推薦記述子をデフォルト適用
  - ボタン優先度: 塗り=必須操作、outline=オプション、flat=上級者用
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

from frontend_nicegui.components.descriptor_selector_dialog import (
    open_descriptor_detail_dialog,
    render_selected_descriptors_panel,
    render_descriptor_sets_panel,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# エンジン定義（全エンジンをフラットに定義）
# ═══════════════════════════════════════════════════════════
_ENGINE_INFO: list[dict[str, Any]] = [
    {"cls": "RDKitAdapter", "label": "RDKit 基本記述子", "category": "物理化学",
     "dims": "~200", "speed": "⚡高速", "desc": "MW, LogP, TPSA, HBA/HBD等"},
    {"cls": "GroupContribAdapter", "label": "基団寄与法", "category": "物理化学",
     "dims": "~15", "speed": "⚡高速", "desc": "Crippen LogP, MR分解"},
    {"cls": "SkfpAdapter", "label": "scikit-fingerprints", "category": "フィンガープリント",
     "dims": "~2200", "speed": "⚡高速", "desc": "ECFP, MACCS, Avalon等30+種"},
    {"cls": "MolfeatAdapter", "label": "Molfeat", "category": "フィンガープリント",
     "dims": "可変", "speed": "⚡高速", "desc": "統合FPフレームワーク"},
    {"cls": "MordredAdapter", "label": "Mordred", "category": "包括的QSPR",
     "dims": "~1800", "speed": "🟡中速", "desc": "2D/3D記述子を網羅的に計算"},
    {"cls": "DescriptaStorusAdapter", "label": "DescriptaStorus", "category": "包括的QSPR",
     "dims": "~200", "speed": "⚡高速", "desc": "Merck開発の高速記述子"},
    {"cls": "PaDELAdapter", "label": "PaDEL", "category": "包括的QSPR",
     "dims": "~1800", "speed": "🟡中速", "desc": "PaDEL-Descriptor互換"},
    {"cls": "MolAIAdapter", "label": "MolAI (CNN+PCA)", "category": "深層学習",
     "dims": "指定可", "speed": "🟡中速", "desc": "CNN潜在ベクトル→PCA次元圧縮"},
    {"cls": "Mol2VecAdapter", "label": "Mol2Vec", "category": "深層学習",
     "dims": "300", "speed": "🟡中速", "desc": "Word2Vec分散表現"},
    {"cls": "ChempropAdapter", "label": "Chemprop (D-MPNN)", "category": "深層学習",
     "dims": "可変", "speed": "🔴低速", "desc": "Directed Message Passing GNN"},
    {"cls": "XTBAdapter", "label": "xTB (GFN2-xTB)", "category": "量子化学",
     "dims": "~20", "speed": "🔴低速", "desc": "HOMO, LUMO, 双極子, 分極率"},
    {"cls": "CosmoAdapter", "label": "COSMO-RS", "category": "量子化学",
     "dims": "~10", "speed": "🔴低速", "desc": "溶媒和自由エネルギー, σプロファイル"},
    {"cls": "UniPkaAdapter", "label": "UniPKa", "category": "量子化学",
     "dims": "~5", "speed": "🟡中速", "desc": "酸解離定数pKa予測"},
    {"cls": "UMAAdapter", "label": "UMA (Meta FAIR)", "category": "量子化学",
     "dims": "~7", "speed": "🔴低速", "desc": "DFTレベル分子物性"},
]


# ═══════════════════════════════════════════════════════════
# アダプタ可用性チェック
# ═══════════════════════════════════════════════════════════
def _load_adapters() -> dict:
    """各アダプタをbackend.chemから読み込み、可用性を返す。"""
    adapters: dict = {}
    try:
        mod = importlib.import_module("backend.chem")
    except ImportError:
        return adapters
    for eng in _ENGINE_INFO:
        cls_name = eng["cls"]
        try:
            cls = getattr(mod, cls_name)
            adapters[cls_name] = cls()
        except Exception:
            adapters[cls_name] = None
    return adapters


def _is_available(adapters: dict, cls_name: str) -> bool:
    adp = adapters.get(cls_name)
    if adp is None:
        return False
    try:
        return adp.is_available()
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# メインUI
# ═══════════════════════════════════════════════════════════
def render_descriptor_plugins(state: dict[str, Any]) -> None:
    """SMILES記述子パネルの完全なUI。"""

    # アダプタ読み込み（キャッシュ）
    if "_chem_adapters" not in state:
        state["_chem_adapters"] = _load_adapters()
    adapters = state["_chem_adapters"]

    n_ok = sum(1 for eng in _ENGINE_INFO if _is_available(adapters, eng["cls"]))
    n_all = len(_ENGINE_INFO)
    has_precalc = state.get("precalc_done") and state.get("precalc_df") is not None
    n_desc = state["precalc_df"].shape[1] if has_precalc else 0

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm full-width"):
        ui.icon("science").classes("text-cyan text-h5")
        ui.label("SMILES 記述子").classes("text-h6")
        ui.badge(f"{n_ok}/{n_all} エンジン利用可", color="cyan")
        if has_precalc:
            ui.badge(f"✅ {n_desc}個計算済み", color="green")
        else:
            ui.badge("⏳ 計算待ち", color="amber")

    # ─────────────────────────────────────────────────────
    # セクション1: 計算状態 + 推薦記述子
    # ─────────────────────────────────────────────────────
    if not has_precalc:
        with ui.card().classes("full-width q-pa-md q-mb-sm").style(
            "border: 1px solid rgba(251,191,36,0.4); background: rgba(50,40,0,0.3); border-radius: 10px;"
        ):
            ui.label(
                "📋 SMILES記述子はデータ読み込み時に自動計算されます。"
                "計算が完了すると、ここに記述子一覧が表示されます。"
            ).classes("text-body2 text-amber")

            # 手動計算トリガー（自動計算が失敗した場合のフォールバック）
            async def _manual_compute():
                if state.get("df") is None or not state.get("smiles_col"):
                    ui.notify("📂 SMILES列を含むデータを読み込んでください", type="warning")
                    return
                compute_btn.disable()
                compute_btn.text = "⏳ 計算中..."
                try:
                    from nicegui import run
                    from backend.chem.smiles_transformer import precalculate_all_descriptors
                    smiles_list = state["df"][state["smiles_col"]].dropna().tolist()
                    target_name = state.get("target_col", "")

                    # 全利用可能エンジンをON
                    engine_flags = {}
                    for eng in _ENGINE_INFO:
                        key = f"use_{eng['cls'].replace('Adapter', '').lower()}"
                        if _is_available(adapters, eng["cls"]):
                            engine_flags[key] = True

                    df_desc, molai_var = await run.io_bound(
                        precalculate_all_descriptors,
                        smiles_list, target_name, engine_flags,
                    )
                    state["precalc_df"] = df_desc
                    state["precalc_done"] = True
                    if molai_var:
                        state["molai_explained_variance"] = molai_var
                    ui.notify(f"✅ {df_desc.shape[1]}個の記述子を計算完了", type="positive")
                except Exception as e:
                    ui.notify(f"⚠️ 記述子計算エラー: {e}", type="warning")
                finally:
                    compute_btn.enable()
                    compute_btn.text = "🔄 手動で記述子を計算"

            compute_btn = ui.button(
                "🔄 手動で記述子を計算", on_click=_manual_compute,
            ).props("outline size=sm no-caps color=amber")
            compute_btn.tooltip("通常は自動計算されますが、失敗した場合にこのボタンで再実行できます")

    # ─────────────────────────────────────────────────────
    # セクション2: 推薦記述子（目的変数ベース）
    # ─────────────────────────────────────────────────────
    _render_target_recommendations(state, adapters)

    # ─────────────────────────────────────────────────────
    # セクション2.5: 記述子セット管理（複数パターン同時試行）
    # ─────────────────────────────────────────────────────
    render_descriptor_sets_panel(state)

    # ─────────────────────────────────────────────────────
    # セクション3: 計算済み記述子テーブル（メインコンテンツ）
    # ─────────────────────────────────────────────────────
    if has_precalc:
        _render_descriptor_table(state)

    # ─────────────────────────────────────────────────────
    # セクション3.5: 選択済み記述子の一覧確認 + 個別ON/OFF
    # ─────────────────────────────────────────────────────
    if has_precalc:
        render_selected_descriptors_panel(state)

    # ─────────────────────────────────────────────────────
    # セクション4: エンジン詳細（上級者向け折りたたみ）
    # ─────────────────────────────────────────────────────
    with ui.expansion(
        f"⚙️ 計算エンジン詳細（{n_ok}/{n_all} 利用可能）", icon="tune",
    ).classes("full-width q-mt-sm"):
        _render_engine_details(adapters, state)

        # アダプタパラメータの動的UI（上級者向け）
        ui.separator().classes("q-my-sm")
        _render_adapter_params(adapters, state)

    # ─────────────────────────────────────────────────────
    # セクション5: MolAI PCA / 電荷設定 / カスタムプラグイン
    # ─────────────────────────────────────────────────────
    molai_key = "use_molai"
    if state.get(molai_key, False) and _is_available(adapters, "MolAIAdapter"):
        with ui.expansion("📊 MolAI PCA 設定", icon="settings").classes("full-width"):
            _render_molai_pca(state)

    _has_qchem = any(
        state.get(f"use_{e.replace('Adapter','').lower()}", False)
        for e in ["XTBAdapter", "CosmoAdapter", "UniPkaAdapter", "UMAAdapter"]
    )
    if _has_qchem:
        with ui.expansion("🔴 分子電荷・スピン設定（上級者）", icon="bolt").classes("full-width"):
            _render_charge_settings(state)

    with ui.expansion("📁 カスタムプラグイン管理", icon="folder_open").classes("full-width"):
        _render_custom_plugins(state)


# ═══════════════════════════════════════════════════════════
# 記述子テーブル（メイン表示）
# ═══════════════════════════════════════════════════════════
def _render_descriptor_table(state: dict) -> None:
    """計算済み記述子の一覧テーブル。相関・意味・カーディナリティを表示。"""
    precalc_df = state.get("precalc_df")
    if precalc_df is None:
        return

    target_col = state.get("target_col")
    df = state.get("df")
    all_descs = list(precalc_df.columns)
    n_total = len(all_descs)

    # ── 相関係数の計算 ──
    corr_dict: dict[str, float] = {}
    if target_col and df is not None and target_col in df.columns:
        try:
            target_s = df[target_col]
            if pd.api.types.is_numeric_dtype(target_s):
                aligned = target_s.iloc[:len(precalc_df)]
                corr_dict = precalc_df.iloc[:len(aligned)].corrwith(
                    aligned.reset_index(drop=True), method="pearson"
                ).abs().dropna().to_dict()
        except Exception:
            pass

    # ── 推薦記述子の名前セット ──
    rec_names = set()
    applied_rec = state.get("_applied_recommendation")
    if applied_rec:
        rec_names = {d.name for d in applied_rec.descriptors}

    # ── 記述子メタ情報（recommender.pyから取得）──
    desc_meta: dict[str, dict] = {}
    try:
        from backend.chem.recommender import get_all_target_recommendations
        for rec in get_all_target_recommendations():
            for d in rec.descriptors:
                if d.name not in desc_meta:
                    desc_meta[d.name] = {
                        "meaning": d.meaning,
                        "category": d.category,
                        "library": d.library,
                    }
    except ImportError:
        pass

    # ── 記述子ごとのカーディナリティ ──
    cardinality: dict[str, int] = {}
    try:
        cardinality = {col: int(precalc_df[col].nunique()) for col in all_descs}
    except Exception:
        pass

    # ── 選択状態 ──
    selected = set(state.get("selected_descriptors", all_descs))

    # ── ヘッダー行 ──
    with ui.card().classes("full-width q-pa-md q-mb-sm glass-card"):
        with ui.row().classes("items-center q-gutter-sm full-width"):
            ui.icon("table_chart").classes("text-cyan text-h6")
            ui.label(f"記述子一覧: {n_total}個計算済み").classes("text-subtitle1")
            if corr_dict:
                ui.badge(f"|r|計算済み ({len(corr_dict)}列)", color="teal").props("outline")

        ui.separator().classes("q-my-xs")

        # ── 選択操作ボタン ──
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            if applied_rec:
                ui.chip(
                    f"📌 適用中: {applied_rec.target_name}",
                    icon="auto_awesome", color="cyan",
                ).props("outline dense")

            if corr_dict:
                sorted_descs = sorted(all_descs, key=lambda d: corr_dict.get(d, 0), reverse=True)

                def _select_top(n: int, descs=sorted_descs):
                    state["selected_descriptors"] = descs[:n]
                    ui.notify(f"✅ 相関上位{n}件を選択", type="positive")

                ui.button("上位10件", on_click=lambda: _select_top(10)).props(
                    "outline size=sm no-caps color=cyan"
                )
                ui.button("上位30件", on_click=lambda: _select_top(30)).props(
                    "outline size=sm no-caps color=cyan"
                )
                ui.button("上位50件", on_click=lambda: _select_top(50)).props(
                    "outline size=sm no-caps color=grey"
                )

            ui.button(
                "全選択", on_click=lambda: state.update({"selected_descriptors": all_descs})
            ).props("flat size=sm no-caps color=grey")
            ui.button(
                "全解除", on_click=lambda: state.update({"selected_descriptors": []})
            ).props("flat size=sm no-caps color=grey")

        # ── テーブル ──
        sorted_list = sorted(
            all_descs,
            key=lambda d: corr_dict.get(d, 0),
            reverse=True,
        ) if corr_dict else all_descs

        rows = []
        for d in sorted_list:
            meta = desc_meta.get(d, {})
            r_val = corr_dict.get(d)
            rows.append({
                "name": d,
                "selected": "✅" if d in selected else "",
                "corr": f"{r_val:.3f}" if r_val is not None else "—",
                "corr_raw": r_val or 0,
                "cardinality": cardinality.get(d, 0),
                "meaning": meta.get("meaning", ""),
                "category": meta.get("category", ""),
                "library": meta.get("library", ""),
                "recommended": "⭐" if d in rec_names else "",
            })

        columns = [
            {"name": "selected", "label": "✓", "field": "selected", "sortable": True, "align": "center"},
            {"name": "recommended", "label": "⭐", "field": "recommended", "sortable": True, "align": "center"},
            {"name": "name", "label": "記述子名", "field": "name", "sortable": True},
            {"name": "corr", "label": "|r|相関", "field": "corr", "sortable": True},
            {"name": "cardinality", "label": "種類数", "field": "cardinality", "sortable": True},
            {"name": "library", "label": "ソース", "field": "library", "sortable": True},
            {"name": "meaning", "label": "物理化学的意味", "field": "meaning"},
            {"name": "category", "label": "分類", "field": "category", "sortable": True},
        ]

        # 最大50行を表示（ページネーション）
        table = ui.table(
            columns=columns,
            rows=rows,
            row_key="name",
            selection="multiple",
            pagination={"rowsPerPage": 30, "sortBy": "corr", "descending": True},
        ).classes("full-width").props("dense flat bordered")

        # 選択状態をテーブルに反映
        table.selected = [r for r in rows if r["name"] in selected]

        def _on_selection_change(e):
            sel_names = [r["name"] for r in e.selection]
            state["selected_descriptors"] = sel_names

        table.on_select(_on_selection_change)


# ═══════════════════════════════════════════════════════════
# エンジン詳細（上級者向け折りたたみ）
# ═══════════════════════════════════════════════════════════
def _render_engine_details(adapters: dict, state: dict) -> None:
    """エンジンの有効/無効状態を表示（情報提供のみ、ON/OFF操作は行わない）。"""
    # カテゴリごとにグルーピング
    categories: dict[str, list[dict]] = {}
    for eng in _ENGINE_INFO:
        cat = eng.get("category", "その他")
        categories.setdefault(cat, []).append(eng)

    for cat_name, engines in categories.items():
        ui.label(f"■ {cat_name}").classes("text-subtitle2 q-mt-sm q-mb-xs")
        with ui.row().classes("q-gutter-sm full-width"):
            for eng in engines:
                cls_name = eng["cls"]
                avail = _is_available(adapters, cls_name)
                border = "rgba(0,212,255,0.4)" if avail else "rgba(255,255,255,0.08)"
                bg = "rgba(0,30,50,0.4)" if avail else "rgba(30,30,30,0.3)"

                with ui.card().classes("q-pa-sm").style(
                    f"border: 1px solid {border}; border-radius: 8px; background: {bg};"
                    "min-width: 200px; max-width: 300px;"
                ):
                    with ui.row().classes("items-center no-wrap q-gutter-xs"):
                        if avail:
                            ui.icon("check_circle", color="green").classes("text-body1")
                        else:
                            ui.icon("cancel", color="grey").classes("text-body1")
                        ui.label(eng["label"]).classes("text-body2 text-bold")

                    ui.label(
                        f"{eng['speed']} | {eng['dims']}次元 | {eng['desc']}"
                    ).classes("text-caption text-grey").style("font-size: 0.7rem;")

                    # 詳細選択ボタン（カタログが存在するエンジンのみ）
                    _engine_catalog_map = {
                        "RDKitAdapter": "RDKit",
                        "XTBAdapter": "XTB",
                        "GroupContribAdapter": "原子団寄与法",
                        "SkfpAdapter": "scikit-FP",
                        "MordredAdapter": "Mordred",
                        "MolfeatAdapter": "Molfeat",
                        "CosmoAdapter": "COSMO-RS",
                    }
                    catalog_name = _engine_catalog_map.get(cls_name)
                    if catalog_name and avail:
                        ui.button(
                            "詳細選択",
                            on_click=lambda cn=catalog_name: open_descriptor_detail_dialog(
                                cn, state
                            ),
                        ).props("flat dense size=xs no-caps color=cyan").tooltip(
                            "個別の記述子を選択・解除"
                        )


# ═══════════════════════════════════════════════════════════
# 目的変数ベース推薦記述子
# ═══════════════════════════════════════════════════════════
def _render_target_recommendations(state: dict, adapters: dict) -> None:
    """recommender.pyの推薦記述子セットを表示＆ワンクリック適用。"""
    try:
        from backend.chem.recommender import (
            get_all_target_recommendations,
            get_target_categories,
            get_targets_by_category,
            get_target_recommendation_by_name,
        )
    except ImportError:
        return

    all_recs = get_all_target_recommendations()
    if not all_recs:
        return

    target_col = state.get("target_col", "")

    # ── 自動推薦検出 ──
    auto_rec = get_target_recommendation_by_name(target_col) if target_col else None

    if auto_rec:
        with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
            "border: 2px solid rgba(0,212,255,0.5); background: rgba(0,30,60,0.5); border-radius: 10px;"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("auto_awesome", color="amber").classes("text-h5")
                ui.label(f"推薦: {auto_rec.target_name}").classes("text-body1 text-bold")
                applied = state.get("_applied_recommendation")
                if applied and applied.target_name == auto_rec.target_name:
                    ui.chip("適用済み", icon="check", color="green").props("outline dense")
                else:
                    ui.button(
                        "この推薦セットを適用",
                        on_click=lambda rec=auto_rec: _apply_recommendation(rec, state),
                    ).props("size=sm color=cyan no-caps unelevated")

            ui.label(auto_rec.summary).classes("text-caption text-grey q-mt-xs")

            # 含まれる記述子を簡潔に表示
            with ui.row().classes("q-gutter-xs q-mt-xs"):
                for d in auto_rec.descriptors[:6]:
                    lib_color = {
                        "RDKit": "green", "XTB": "orange", "COSMO-RS": "purple",
                        "GroupContribution": "teal", "Uni-pKa": "pink",
                    }.get(d.library, "grey")
                    ui.chip(f"{d.name}", color=lib_color).props("dense outline").classes("text-xs")
                if len(auto_rec.descriptors) > 6:
                    ui.chip(f"+{len(auto_rec.descriptors) - 6}", color="grey").props("dense outline").classes("text-xs")

    # ── 全推薦一覧（折りたたみ） ──
    with ui.expansion(
        f"🔬 全{len(all_recs)}種の目的変数対応推薦セットを見る",
        icon="recommend",
    ).classes("full-width q-mb-sm"):
        ui.label(
            "予測したい物性を選ぶと、学術論文の知見に基づく最適な記述子セットが自動適用されます。"
        ).classes("text-caption text-grey q-mb-sm")

        categories = get_target_categories()
        for cat_name in categories:
            cat_recs = get_targets_by_category(cat_name)
            with ui.expansion(f"📂 {cat_name} ({len(cat_recs)}種)", icon="folder").classes(
                "full-width q-mb-xs"
            ).props("dense"):
                for rec in cat_recs:
                    with ui.row().classes("items-center q-gutter-xs full-width q-mb-xs"):
                        ui.button(
                            rec.target_name,
                            on_click=lambda r=rec: _apply_recommendation(r, state),
                        ).props("outline size=sm no-caps color=cyan").classes("text-xs")
                        libs = sorted(set(d.library for d in rec.descriptors))
                        for lib in libs[:3]:
                            color = {"RDKit": "green", "XTB": "orange",
                                     "COSMO-RS": "purple", "GroupContribution": "teal",
                                     "Uni-pKa": "pink"}.get(lib, "grey")
                            ui.badge(lib, color=color).props("dense outline").classes("text-xs")

    # ── 適用済み推薦の詳細 ──
    applied_rec = state.get("_applied_recommendation")
    if applied_rec:
        with ui.expansion(
            f"📋 適用中: {applied_rec.target_name}", icon="checklist",
        ).classes("full-width").props("default-opened"):
            ui.label(applied_rec.summary).classes("text-caption text-grey q-mb-sm")
            rows = [
                {"name": d.name, "library": d.library, "meaning": d.meaning,
                 "category": d.category, "source": d.source}
                for d in applied_rec.descriptors
            ]
            columns = [
                {"name": "name", "label": "記述子", "field": "name", "sortable": True},
                {"name": "library", "label": "ソース", "field": "library", "sortable": True},
                {"name": "meaning", "label": "物理化学的意味", "field": "meaning"},
                {"name": "category", "label": "分類", "field": "category", "sortable": True},
                {"name": "source", "label": "根拠", "field": "source"},
            ]
            ui.table(columns=columns, rows=rows, row_key="name").classes(
                "full-width"
            ).props("dense flat bordered")


def _apply_recommendation(rec, state: dict) -> None:
    """推薦記述子セットを適用（エンジンは全OFFにしない）。"""
    # 推薦記述子名をstateに保存
    state["selected_descriptors"] = [d.name for d in rec.descriptors]
    state["_applied_recommendation"] = rec

    ui.notify(
        f"✅ 「{rec.target_name}」の推薦記述子セット適用 ({len(rec.descriptors)}個)",
        type="positive", timeout=5000,
    )


# ═══════════════════════════════════════════════════════════
# MolAI PCA 設定
# ═══════════════════════════════════════════════════════════
def _render_molai_pca(state: dict) -> None:
    """MolAI PCA次元数設定。"""
    n_samples = len(state["df"]) if state.get("df") is not None else 25
    auto_n = min(max(n_samples // 2, 5), 128)
    current_n = state.get("molai_n_components", auto_n)

    ui.label(f"サンプル数: {n_samples} → 推奨PCA次元: {auto_n}").classes("text-caption text-grey")

    slider = ui.slider(
        min=1, max=min(256, n_samples), value=current_n, step=1,
    ).props("label-always").classes("full-width")

    def _update_pca(e):
        state["molai_n_components"] = int(e.value)
    slider.on_value_change(_update_pca)

    mev = state.get("molai_explained_variance")
    if mev and mev.get("ratio"):
        _render_pca_chart(mev)


def _render_pca_chart(mev: dict) -> None:
    """MolAI PCA累積寄与率のグラフ。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        ui.label("Plotly未インストール — グラフ表示不可").classes("text-grey")
        return

    evr = mev["ratio"]
    evc = mev["cumulative"]
    pcs = [f"PC{i+1}" for i in range(len(evr))]

    fig = go.Figure()
    fig.add_bar(x=pcs, y=[v * 100 for v in evr], name="寄与率 (%)", marker_color="#4c9be8")
    fig.add_scatter(
        x=pcs, y=[v * 100 for v in evc], name="累積寄与率 (%)",
        mode="lines+markers", yaxis="y2",
        line=dict(color="#f4a261", width=2), marker=dict(size=5),
    )
    fig.update_layout(
        yaxis=dict(title="寄与率 (%)", range=[0, max(v * 100 for v in evr) * 1.15]),
        yaxis2=dict(title="累積寄与率 (%)", overlaying="y", side="right", range=[0, 105], showgrid=False),
        legend=dict(orientation="h", y=1.15),
        height=250, margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
    )
    ui.plotly(fig).classes("full-width")


# ═══════════════════════════════════════════════════════════
# 電荷・スピン設定
# ═══════════════════════════════════════════════════════════
def _render_charge_settings(state: dict) -> None:
    """形式電荷・スピン多重度・プロトン化モード。"""
    ui.label(
        "量子化学計算（XTB/COSMO等）の電荷・スピン設定。通常は変更不要。"
    ).classes("text-caption text-grey q-mb-sm")

    with ui.row().classes("q-gutter-md full-width"):
        charge = ui.number(
            "形式電荷", value=state.get("formal_charge", 0),
            min=-3, max=3, step=1,
        ).props("dense outlined").classes("w-32")

        def _update_charge(e):
            state["formal_charge"] = int(e.value) if e.value is not None else 0
        charge.on_value_change(_update_charge)

        spin = ui.select(
            {1: "1 (閉殻)", 2: "2 (ラジカル)", 3: "3 (三重項)"},
            value=state.get("spin_multiplicity", 1),
            label="スピン多重度",
        ).props("dense outlined").classes("w-40")

        def _update_spin(e):
            state["spin_multiplicity"] = e.value
        spin.on_value_change(_update_spin)

    with ui.row().classes("q-gutter-sm q-mt-sm full-width items-center"):
        prot_mode = ui.select(
            {"as_is": "SMILESのまま", "auto_ph": "pH指定で自動プロトン化", "neutral": "全て中性化"},
            value=state.get("protonate_mode", "as_is"),
            label="プロトン化モード",
        ).props("dense outlined").classes("w-48")

        def _update_prot(e):
            state["protonate_mode"] = e.value
        prot_mode.on_value_change(_update_prot)

        if state.get("protonate_mode") == "auto_ph":
            ph = ui.number(
                "pH", value=state.get("target_ph", 7.4), min=0, max=14, step=0.1,
            ).props("dense outlined").classes("w-24")

            def _update_ph(e):
                state["target_ph"] = float(e.value) if e.value is not None else 7.4
            ph.on_value_change(_update_ph)

    pcm = ui.select(
        {"gasteiger": "Gasteiger", "xtb_mulliken": "XTB Mulliken", "none": "なし"},
        value=state.get("partial_charge_model", "gasteiger"),
        label="部分電荷モデル",
    ).props("dense outlined").classes("w-40 q-mt-sm")

    def _update_pcm(e):
        state["partial_charge_model"] = e.value
    pcm.on_value_change(_update_pcm)


# ═══════════════════════════════════════════════════════════
# カスタムプラグイン管理
# ═══════════════════════════════════════════════════════════
def _render_custom_plugins(state: dict) -> None:
    """カスタムプラグインのアップロード/削除/テンプレート/既存アダプタからの作成UI。"""
    from backend.chem.descriptors import get_custom_dir, invalidate_cache

    ui.label(
        "custom/ ディレクトリに .py ファイルを追加して独自記述子を作成できます。"
        "既存の記述子アダプタをベースにして変更を加える方法が最も効率的です。"
    ).classes("text-grey text-caption q-mb-sm")

    custom_dir = get_custom_dir()
    custom_files = [f for f in custom_dir.glob("*.py") if not f.name.startswith("_")]

    # ── 既存カスタムプラグインの一覧 ──
    if custom_files:
        ui.label(f"📁 登録済みカスタムプラグイン ({len(custom_files)})").classes("text-subtitle2")
        with ui.column().classes("q-gutter-xs"):
            for cf in custom_files:
                with ui.row().classes("items-center q-gutter-xs"):
                    ui.icon("description", color="amber").classes("text-body1")
                    ui.label(cf.name).classes("text-body2")
                    # ソースコード閲覧ボタン
                    ui.button(
                        icon="visibility",
                        on_click=lambda _, f=cf: _show_source_dialog(f),
                    ).props("flat dense color=cyan size=sm").tooltip("ソースコードを確認")
                    ui.button(
                        icon="delete",
                        on_click=lambda _, f=cf: _delete_custom(f),
                    ).props("flat dense color=red size=sm")
    else:
        ui.label("カスタムプラグインはまだありません").classes("text-grey text-caption")

    ui.separator()

    # ── アップロード ──
    async def _on_upload(e):
        content = e.content.read()
        filename = e.name
        if not filename.endswith(".py"):
            ui.notify("⚠️ .py ファイルのみアップロードできます", type="warning")
            return
        dest = custom_dir / filename
        dest.write_bytes(content)
        invalidate_cache()
        ui.notify(f"✅ {filename} をアップロードしました", type="positive")

    ui.upload(
        label="カスタム記述子 .py をアップロード",
        on_upload=_on_upload,
        auto_upload=True,
    ).props("accept='.py' flat dense").classes("full-width")

    # ── テンプレート ──
    with ui.row().classes("q-gutter-sm q-mt-sm"):
        ui.label("テンプレート:").classes("text-caption text-grey")
        for tpl_name in ["_template_simple.py", "_template_with_config.py"]:
            tpl_path = custom_dir / tpl_name
            if tpl_path.exists():
                ui.button(
                    tpl_name.replace("_template_", "").replace(".py", ""),
                    on_click=lambda _, p=tpl_path: ui.download(str(p)),
                ).props("flat dense size=sm color=cyan no-caps")

    # ── 既存記述子アダプタからコピーして作成 ──
    ui.separator()
    ui.label("📋 既存記述子からカスタムプラグインを作成").classes("text-subtitle2 q-mt-sm")
    ui.label(
        "既存のSMILES記述子アダプタのソースコードをコピーし、"
        "カスタムプラグインとして保存してから編集できます。"
    ).classes("text-grey text-caption q-mb-sm")

    # 既存アダプタのソースファイルを検出
    adapter_sources = _detect_adapter_sources()
    if adapter_sources:
        with ui.row().classes("q-gutter-sm flex-wrap"):
            for adapter_name, source_path in adapter_sources.items():
                ui.button(
                    f"📄 {adapter_name}",
                    on_click=lambda _, n=adapter_name, p=source_path: _copy_adapter_to_custom(
                        n, p, custom_dir,
                    ),
                ).props("outline dense size=sm no-caps color=teal").tooltip(
                    f"{source_path.name} をカスタムプラグインとしてコピー"
                )
    else:
        ui.label("利用可能なアダプタが見つかりません").classes("text-grey text-caption")


def _detect_adapter_sources() -> dict[str, Path]:
    """backend/chem/以下のSMILES記述子アダプタのソースファイルを検出。"""
    import inspect
    adapters = {}
    adapter_dir = Path(__file__).resolve().parent.parent.parent / "backend" / "chem"
    if not adapter_dir.exists():
        return adapters

    # 既知のアダプタファイル
    known = [
        ("RDKit", "rdkit_adapter.py"),
        ("Mordred", "mordred_adapter.py"),
        ("XTB", "xtb_adapter.py"),
        ("COSMO-RS", "cosmo_adapter.py"),
        ("Uni-pKa", "unipka_adapter.py"),
        ("UMA", "uma_adapter.py"),
        ("scikit-FP", "scikitfp_adapter.py"),
        ("MolAI", "molai_adapter.py"),
    ]
    for name, filename in known:
        filepath = adapter_dir / filename
        if filepath.exists():
            adapters[name] = filepath

    return adapters


def _copy_adapter_to_custom(adapter_name: str, source_path: Path, custom_dir: Path) -> None:
    """既存アダプタのソースをカスタムプラグインとしてコピー。"""
    from backend.chem.descriptors import invalidate_cache

    dest_name = f"custom_{source_path.stem}.py"
    dest = custom_dir / dest_name

    if dest.exists():
        ui.notify(f"⚠️ {dest_name} は既に存在します。名前を変えてください。", type="warning")
        return

    try:
        # ヘッダーコメントを追加
        original_source = source_path.read_text(encoding="utf-8")
        header = (
            f'"""\n'
            f"カスタムプラグイン: {adapter_name}アダプタから作成\n"
            f"元ファイル: {source_path.name}\n"
            f"\n"
            f"このファイルを自由に編集して、独自の記述子計算ロジックを実装してください。\n"
            f"クラス名やcompute()メソッドのシグネチャを変更すると読み込まれなくなるため注意。\n"
            f'"""\n\n'
        )
        dest.write_text(header + original_source, encoding="utf-8")
        invalidate_cache()
        ui.notify(f"✅ {dest_name} を作成しました。編集してカスタマイズしてください。", type="positive")
    except Exception as e:
        ui.notify(f"⚠️ コピーエラー: {e}", type="warning")


def _show_source_dialog(filepath: Path) -> None:
    """ソースコードを表示するダイアログ。"""
    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        source = "(読み込めません)"

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("width: 80vw; max-width: 900px;"):
        ui.label(f"📄 {filepath.name}").classes("text-h6")
        # NiceGUIのcode要素でソースコード表示
        ui.code(source, language="python").classes("full-width").style(
            "max-height: 60vh; overflow-y: auto; font-size: 0.8rem;"
        )
        ui.button("閉じる", on_click=dlg.close).props("outline color=cyan")
    dlg.open()


def _delete_custom(filepath: Path) -> None:
    """カスタムプラグインを削除。"""
    try:
        filepath.unlink()
        from backend.chem.descriptors import invalidate_cache
        invalidate_cache()
        ui.notify(f"🗑️ {filepath.name} を削除しました", type="info")
    except Exception as e:
        ui.notify(f"⚠️ 削除エラー: {e}", type="warning")


# ═══════════════════════════════════════════════════════════
# アダプタパラメータ動的UI
# ═══════════════════════════════════════════════════════════
def _render_adapter_params(adapters: list[tuple], state: dict) -> None:
    """
    各SMILESアダプタのパラメータを introspect_params で自動検出し、
    動的UIを生成する。パラメータが0個のエンジンはスキップ。
    """
    if "adapter_params" not in state:
        state["adapter_params"] = {}

    ui.label(
        "各エンジンの引数を自動検出して表示しています。"
        "変更しなければデフォルト設定で計算されます。"
    ).classes("text-caption text-grey q-mb-sm")

    any_shown = False
    for name, mod_path, cls_name, _kwargs in adapters:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            adapter_cls = getattr(mod, cls_name)

            from backend.ui.param_schema import introspect_params
            specs = introspect_params(adapter_cls)
            if not specs:
                continue  # パラメータなしのエンジンはスキップ

            any_shown = True
            with ui.expansion(
                f"🔹 {name} ({len(specs)}パラメータ)", icon="settings",
            ).classes("full-width q-mb-xs"):
                try:
                    from frontend_nicegui.components.auto_params_ui import render_param_editor
                    existing = state["adapter_params"].get(name, {})
                    values = render_param_editor(
                        specs, title=name, values=existing, compact=True,
                    )
                    state["adapter_params"][name] = values
                except Exception as ex:
                    ui.label(f"⚠️ {ex}").classes("text-amber")
        except Exception:
            pass  # インポート不可のエンジンはスキップ

    if not any_shown:
        ui.label("設定可能なパラメータを持つエンジンはありません").classes("text-grey text-caption")

