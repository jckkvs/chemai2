"""
frontend_nicegui/components/descriptor_selector_dialog.py

記述子の個別選択ダイアログ + 選択済み一覧 + 複数セット管理。

設計:
  1) 各エンジンの「詳細選択」ボタン → ダイアログ（カテゴリ別展開+チェックボックス）
  2) 選択済み記述子の一覧表示 + 個別ON/OFF
  3) 複数の記述子セット(パターン)を定義し、同時に解析を試行

UIが縦長にならないよう、ダイアログ方式を採用。
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from nicegui import ui

from frontend_nicegui.components.descriptor_catalog import (
    ENGINE_CATALOG_MAP,
    SUPPORTED_ENGINES,
    get_catalog,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. 記述子個別選択ダイアログ
# ═══════════════════════════════════════════════════════════

def open_descriptor_detail_dialog(
    engine_name: str,
    state: dict,
) -> None:
    """
    指定エンジンの記述子をカテゴリ別に展開し、
    各記述子をチェックボックスで個別に選択/解除できるダイアログを開く。

    選択結果は state["selected_descriptors"][engine_name] に list[str] として格納。
    """
    catalog = get_catalog(engine_name)
    if not catalog:
        ui.notify(f"⚠️ {engine_name}のカタログは未定義です", type="warning")
        return

    # 選択状態の初期化
    if "selected_descriptors" not in state:
        state["selected_descriptors"] = {}
    if engine_name not in state["selected_descriptors"]:
        # デフォルト: 全記述子をON
        all_names = []
        for descs in catalog.values():
            for d in descs:
                if not d["name"].startswith("_"):
                    all_names.append(d["name"])
        state["selected_descriptors"][engine_name] = list(all_names)

    selected = set(state["selected_descriptors"][engine_name])
    all_desc_names = []
    for descs in catalog.values():
        for d in descs:
            if not d["name"].startswith("_"):
                all_desc_names.append(d["name"])

    # チェックボックスの参照を保持（全選択/全解除で使う）
    checkbox_refs: list[tuple[str, Any]] = []

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style(
        "width: 85vw; max-width: 1100px; max-height: 85vh;"
    ):
        # ── ヘッダー ──
        with ui.row().classes("items-center justify-between full-width q-mb-sm"):
            ui.label(f"🔬 {engine_name} 記述子選択").classes("text-h6")
            n_sel = len([n for n in all_desc_names if n in selected])
            count_lbl = ui.label(f"{n_sel}/{len(all_desc_names)} 選択中").classes(
                "text-caption text-cyan"
            )

        def _update_count():
            n = len([n for n in all_desc_names if n in selected])
            count_lbl.text = f"{n}/{len(all_desc_names)} 選択中"

        # ── クイックボタン ──
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            def _select_all():
                for n in all_desc_names:
                    selected.add(n)
                for name, cb in checkbox_refs:
                    cb.value = True
                _update_count()

            def _deselect_all():
                for n in all_desc_names:
                    selected.discard(n)
                for name, cb in checkbox_refs:
                    cb.value = False
                _update_count()

            ui.button("全選択", on_click=_select_all).props(
                "outline size=sm no-caps color=cyan"
            )
            ui.button("全解除", on_click=_deselect_all).props(
                "outline size=sm no-caps color=grey"
            )

        ui.separator()

        # ── カテゴリ別記述子一覧 ──
        with ui.scroll_area().style("max-height: 55vh;"):
            for cat_name, descs in catalog.items():
                cat_actual = [d for d in descs if not d["name"].startswith("_")]
                cat_group = [d for d in descs if d["name"].startswith("_")]
                n_cat_sel = len([d for d in cat_actual if d["name"] in selected])

                with ui.expansion(
                    f"{cat_name}  ({n_cat_sel}/{len(cat_actual)})",
                    icon="folder",
                ).classes("full-width q-mb-xs"):
                    # カテゴリ内全選択/解除
                    cat_names_list = [d["name"] for d in cat_actual]
                    with ui.row().classes("q-gutter-xs q-mb-xs"):
                        def _cat_on(ns=cat_names_list):
                            for n in ns:
                                selected.add(n)
                            for name, cb in checkbox_refs:
                                if name in ns:
                                    cb.value = True
                            _update_count()

                        def _cat_off(ns=cat_names_list):
                            for n in ns:
                                selected.discard(n)
                            for name, cb in checkbox_refs:
                                if name in ns:
                                    cb.value = False
                            _update_count()

                        ui.button("全選択", on_click=_cat_on).props(
                            "flat dense size=xs no-caps color=cyan"
                        )
                        ui.button("全解除", on_click=_cat_off).props(
                            "flat dense size=xs no-caps color=grey"
                        )

                    # グループアイテム（FPなど）
                    for g in cat_group:
                        b = g.get("bits", "")
                        bits_str = f" ({b}bit)" if b else ""
                        ui.label(f"  📦 {g.get('short', g['name'])}{bits_str}").classes(
                            "text-caption text-grey"
                        )

                    # 個別記述子チェックボックス
                    for desc in cat_actual:
                        dname = desc["name"]
                        short = desc.get("short", "")
                        with ui.row().classes("items-center q-gutter-xs").style(
                            "min-height: 28px;"
                        ):
                            cb = ui.checkbox(
                                dname,
                                value=(dname in selected),
                            ).props("dense").style("min-width: 200px;")

                            def _on_change(e, n=dname):
                                if e.value:
                                    selected.add(n)
                                else:
                                    selected.discard(n)
                                _update_count()

                            cb.on_value_change(_on_change)
                            checkbox_refs.append((dname, cb))

                            if short:
                                ui.label(short).classes(
                                    "text-caption text-grey"
                                ).style("font-size: 0.72rem;")

        ui.separator()

        # ── フッター ──
        with ui.row().classes("justify-end q-gutter-sm"):
            def _apply():
                state["selected_descriptors"][engine_name] = [
                    n for n in all_desc_names if n in selected
                ]
                n = len(state["selected_descriptors"][engine_name])
                ui.notify(f"✅ {engine_name}: {n}記述子を選択", type="positive")
                dlg.close()

            ui.button("キャンセル", on_click=dlg.close).props("flat no-caps color=grey")
            ui.button("適用", on_click=_apply).props("no-caps color=cyan")

    dlg.open()


# ═══════════════════════════════════════════════════════════
# 2. 選択済み記述子の一覧表示 + 個別ON/OFF
# ═══════════════════════════════════════════════════════════

def render_selected_descriptors_panel(state: dict) -> None:
    """
    現在選択されている記述子の一覧を表示し、
    各記述子の使用/不使用をチェックボックスで個別に切り替えられるUI。
    ダイアログ形式で表示する（画面が縦長にならないように）。
    """
    precalc_df = state.get("precalc_df")
    if precalc_df is None:
        return

    all_cols = list(precalc_df.columns)
    # 現在の使用記述子リスト（未設定なら全部ON）
    if "active_descriptors" not in state:
        state["active_descriptors"] = list(all_cols)

    n_active = len(state["active_descriptors"])
    n_total = len(all_cols)

    def _open_active_dialog():
        active_set = set(state["active_descriptors"])

        # カタログから説明を引く
        desc_shorts: dict[str, str] = {}
        for engine_name in SUPPORTED_ENGINES:
            catalog = get_catalog(engine_name)
            if catalog:
                for cat_descs in catalog.values():
                    for d in cat_descs:
                        desc_shorts[d["name"]] = d.get("short", "")

        cb_refs: list[tuple[str, Any]] = []

        with ui.dialog() as dlg2, ui.card().classes("q-pa-md").style(
            "width: 80vw; max-width: 1000px; max-height: 85vh;"
        ):
            with ui.row().classes("items-center justify-between full-width q-mb-sm"):
                ui.label("📋 使用する記述子を選択").classes("text-h6")
                cnt_lbl = ui.label(f"{n_active}/{n_total} 使用中").classes(
                    "text-caption text-cyan"
                )

            def _upd():
                n = len([c for c in all_cols if c in active_set])
                cnt_lbl.text = f"{n}/{n_total} 使用中"

            with ui.row().classes("q-gutter-sm q-mb-sm"):
                def _all_on():
                    for c in all_cols:
                        active_set.add(c)
                    for _, cb in cb_refs:
                        cb.value = True
                    _upd()

                def _all_off():
                    active_set.clear()
                    for _, cb in cb_refs:
                        cb.value = False
                    _upd()

                ui.button("全ON", on_click=_all_on).props("outline size=sm no-caps color=cyan")
                ui.button("全OFF", on_click=_all_off).props("outline size=sm no-caps color=grey")

                # 検索ボックス
                search_input = ui.input("検索", placeholder="記述子名で絞り込み").props(
                    "dense outlined clearable"
                ).style("width: 250px;")

            ui.separator()

            with ui.scroll_area().style("max-height: 55vh;"):
                rows_container = ui.column().classes("full-width q-gutter-xs")
                with rows_container:
                    for col_name in all_cols:
                        short = desc_shorts.get(col_name, "")
                        with ui.row().classes("items-center q-gutter-xs").style(
                            "min-height: 26px;"
                        ) as row_el:
                            cb = ui.checkbox(
                                col_name,
                                value=(col_name in active_set),
                            ).props("dense").style("min-width: 250px;")

                            def _ch(e, n=col_name):
                                if e.value:
                                    active_set.add(n)
                                else:
                                    active_set.discard(n)
                                _upd()

                            cb.on_value_change(_ch)
                            cb_refs.append((col_name, cb))

                            if short:
                                ui.label(short).classes(
                                    "text-caption text-grey"
                                ).style("font-size: 0.7rem;")

            ui.separator()
            with ui.row().classes("justify-end q-gutter-sm"):
                def _apply2():
                    state["active_descriptors"] = [c for c in all_cols if c in active_set]
                    ui.notify(
                        f"✅ {len(state['active_descriptors'])}/{n_total} 記述子を使用",
                        type="positive",
                    )
                    dlg2.close()

                ui.button("キャンセル", on_click=dlg2.close).props("flat no-caps color=grey")
                ui.button("適用", on_click=_apply2).props("no-caps color=cyan")

        dlg2.open()

    # メインUIに表示するボタン
    with ui.row().classes("items-center q-gutter-sm"):
        ui.button(
            f"📋 使用記述子を確認・変更 ({n_active}/{n_total})",
            on_click=_open_active_dialog,
        ).props("outline no-caps color=cyan")


# ═══════════════════════════════════════════════════════════
# 3. 複数記述子セット（パターン）管理
# ═══════════════════════════════════════════════════════════

def render_descriptor_sets_panel(state: dict) -> None:
    """
    複数の記述子セット(パターン)を管理するUI。
    各セットに名前をつけて保存し、解析時に複数セットを同時に試行できる。
    """
    # セット管理の初期化
    if "descriptor_sets" not in state:
        state["descriptor_sets"] = {
            "デフォルト": {
                "engines": [],  # 使用するエンジン名リスト
                "active": True,
                "descriptors": None,  # Noneなら全記述子
            }
        }
    if "current_set_name" not in state:
        state["current_set_name"] = "デフォルト"

    sets = state["descriptor_sets"]
    current = state["current_set_name"]

    with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
        "border: 1px solid rgba(0,188,212,0.3); border-radius: 8px;"
    ):
        with ui.row().classes("items-center q-gutter-sm full-width"):
            ui.icon("layers").classes("text-cyan")
            ui.label("記述子セット管理").classes("text-subtitle2")
            ui.badge(f"{len(sets)} セット", color="cyan")

        # セット一覧
        with ui.row().classes("q-gutter-xs q-mt-xs flex-wrap"):
            for name, info in sets.items():
                is_current = (name == current)
                color = "cyan" if is_current else "grey"
                props = "no-caps size=sm"
                if not is_current:
                    props += " outline"

                def _switch(n=name):
                    state["current_set_name"] = n
                    if sets[n].get("descriptors"):
                        state["active_descriptors"] = list(sets[n]["descriptors"])
                    ui.notify(f"🔄 セット「{n}」に切替", type="info")

                btn = ui.button(
                    f"{'▶ ' if is_current else ''}{name}",
                    on_click=_switch,
                ).props(props + f" color={color}")

        # セット操作ボタン
        with ui.row().classes("q-gutter-xs q-mt-xs"):
            def _save_current():
                """現在の記述子選択を現在のセットに保存"""
                active = state.get("active_descriptors", [])
                sets[current]["descriptors"] = list(active)
                ui.notify(f"💾 セット「{current}」に {len(active)} 記述子を保存", type="positive")

            def _add_set():
                """新しいセットを追加"""
                new_name = f"セット{len(sets) + 1}"
                active = state.get("active_descriptors", [])
                sets[new_name] = {
                    "engines": [],
                    "active": True,
                    "descriptors": list(active),
                }
                state["current_set_name"] = new_name
                ui.notify(f"➕ セット「{new_name}」を作成", type="positive")

            def _duplicate_set():
                """現在のセットを複製"""
                new_name = f"{current}_コピー"
                sets[new_name] = copy.deepcopy(sets[current])
                state["current_set_name"] = new_name
                ui.notify(f"📋 セット「{new_name}」を複製", type="info")

            def _delete_current():
                """現在のセットを削除（デフォルトは削除不可）"""
                if current == "デフォルト":
                    ui.notify("⚠️ デフォルトセットは削除できません", type="warning")
                    return
                del sets[current]
                state["current_set_name"] = "デフォルト"
                ui.notify(f"🗑️ セット「{current}」を削除", type="info")

            ui.button("💾 保存", on_click=_save_current).props(
                "flat dense size=xs no-caps color=cyan"
            )
            ui.button("➕ 新規", on_click=_add_set).props(
                "flat dense size=xs no-caps color=green"
            )
            ui.button("📋 複製", on_click=_duplicate_set).props(
                "flat dense size=xs no-caps color=blue"
            )
            ui.button("🗑️ 削除", on_click=_delete_current).props(
                "flat dense size=xs no-caps color=red"
            )

        # 全セット同時解析ボタン
        active_sets = [n for n, info in sets.items() if info.get("active", True)]
        if len(active_sets) > 1:
            ui.separator().classes("q-my-xs")
            with ui.row().classes("items-center q-gutter-sm"):
                ui.label(f"🔬 {len(active_sets)} セットで比較解析が可能").classes(
                    "text-caption text-cyan"
                )

                def _toggle_set_active(name, val):
                    sets[name]["active"] = val

                for name in sets:
                    ui.checkbox(
                        f"{name}",
                        value=sets[name].get("active", True),
                        on_change=lambda e, n=name: _toggle_set_active(n, e.value),
                    ).props("dense")
