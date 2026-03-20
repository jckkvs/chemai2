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
# 3. 複数記述子セット（パターン）管理 — カード型ビジュアルUI
# ═══════════════════════════════════════════════════════════

def _get_engine_badges(descriptors: list[str] | None) -> list[str]:
    """記述子名からエンジン名を推定してバッジ用のリストを返す。"""
    if not descriptors:
        return []
    engines = set()
    for d in descriptors:
        if d.startswith("xtb_"):
            engines.add("XTB")
        elif d.startswith("joback_"):
            engines.add("基団寄与")
        elif d.startswith("mu_") or d.startswith("ln_gamma"):
            engines.add("COSMO")
        elif d.startswith("fr_") or d in ("MolWt", "MolLogP", "TPSA", "qed"):
            engines.add("RDKit")
        elif d.startswith("gasteiger_"):
            engines.add("RDKit")
        else:
            engines.add("他")
    return sorted(engines)[:4]

_ENGINE_BADGE_COLORS = {
    "RDKit": "green", "XTB": "orange", "COSMO": "purple",
    "基団寄与": "teal", "Mordred": "blue", "scikit-FP": "indigo",
    "Molfeat": "pink", "他": "grey",
}


def _open_set_compare_dialog(sets: dict, state: dict) -> None:
    """セット比較ダイアログ: セット間の記述子重複・差分を表示。"""
    set_names = list(sets.keys())
    if len(set_names) < 2:
        ui.notify("⚠️ 比較には2つ以上のセットが必要です", type="warning")
        return

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style(
        "width: 85vw; max-width: 1000px; max-height: 85vh;"
    ):
        ui.label("📊 セット比較ダッシュボード").classes("text-h6 q-mb-sm")

        # 各セットの記述子セット
        desc_sets = {}
        for name in set_names:
            descs = sets[name].get("descriptors")
            desc_sets[name] = set(descs) if descs else set()

        # 比較テーブル
        rows = []
        for name in set_names:
            d = desc_sets[name]
            others_union = set()
            for n2 in set_names:
                if n2 != name:
                    others_union |= desc_sets[n2]
            unique = d - others_union
            shared = d & others_union
            rows.append({
                "name": name,
                "total": len(d),
                "unique": len(unique),
                "shared": len(shared),
                "engines": ", ".join(_get_engine_badges(list(d))),
            })

        cols = [
            {"name": "name", "label": "セット名", "field": "name"},
            {"name": "total", "label": "記述子数", "field": "total", "sortable": True},
            {"name": "unique", "label": "固有", "field": "unique", "sortable": True},
            {"name": "shared", "label": "共有", "field": "shared", "sortable": True},
            {"name": "engines", "label": "エンジン", "field": "engines"},
        ]
        ui.table(columns=cols, rows=rows, row_key="name").classes(
            "full-width"
        ).props("dense flat bordered")

        # ペアワイズ重複マトリクス
        if len(set_names) >= 2:
            ui.separator().classes("q-my-sm")
            ui.label("🔗 ペアワイズ重複率 (%)").classes("text-subtitle2")
            matrix_rows = []
            for n1 in set_names:
                row_data = {"name": n1}
                for n2 in set_names:
                    if desc_sets[n1] and desc_sets[n2]:
                        overlap = len(desc_sets[n1] & desc_sets[n2])
                        union = len(desc_sets[n1] | desc_sets[n2])
                        pct = round(overlap / union * 100) if union > 0 else 0
                    else:
                        pct = 0
                    row_data[n2] = f"{pct}%"
                matrix_rows.append(row_data)

            m_cols = [{"name": "name", "label": "", "field": "name"}] + [
                {"name": n, "label": n, "field": n} for n in set_names
            ]
            ui.table(columns=m_cols, rows=matrix_rows, row_key="name").classes(
                "full-width"
            ).props("dense flat bordered")

        ui.separator()
        ui.button("閉じる", on_click=dlg.close).props("outline no-caps color=cyan")

    dlg.open()


def _open_rename_dialog(old_name: str, sets: dict, state: dict) -> None:
    """セット名変更ダイアログ。"""
    if old_name == "デフォルト":
        ui.notify("⚠️ デフォルトセットは名前変更できません", type="warning")
        return

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("min-width: 350px;"):
        ui.label("✏️ セット名を変更").classes("text-h6 q-mb-sm")
        name_input = ui.input("新しい名前", value=old_name).props(
            "outlined dense autofocus"
        ).classes("full-width")

        with ui.row().classes("justify-end q-gutter-sm q-mt-sm"):
            def _apply():
                new_name = name_input.value.strip()
                if not new_name or new_name == old_name:
                    dlg.close()
                    return
                if new_name in sets:
                    ui.notify(f"⚠️ 「{new_name}」は既に存在します", type="warning")
                    return
                sets[new_name] = sets.pop(old_name)
                if state.get("current_set_name") == old_name:
                    state["current_set_name"] = new_name
                ui.notify(f"✏️ 「{old_name}」→「{new_name}」に変更", type="positive")
                dlg.close()

            ui.button("キャンセル", on_click=dlg.close).props("flat no-caps color=grey")
            ui.button("変更", on_click=_apply).props("no-caps color=cyan")

    dlg.open()


def render_descriptor_sets_panel(state: dict) -> None:
    """
    複数の記述子セット(パターン)をカード型で管理するUI。
    各セットの内容（記述子数・エンジン構成・プログレスバー）が一目瞭然。
    """
    # セット管理の初期化
    if "descriptor_sets" not in state:
        state["descriptor_sets"] = {
            "デフォルト": {
                "engines": [],
                "active": True,
                "descriptors": None,
            }
        }
    if "current_set_name" not in state:
        state["current_set_name"] = "デフォルト"

    sets = state["descriptor_sets"]
    current = state["current_set_name"]

    # 全記述子数を取得（プログレスバー用）
    precalc_df = state.get("precalc_df")
    total_available = precalc_df.shape[1] if precalc_df is not None else 0

    # ── メインカード ──
    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(0,188,212,0.3); border-radius: 12px;"
        "background: rgba(0,20,40,0.3);"
    ):
        # ── ヘッダー ──
        with ui.row().classes("items-center justify-between full-width q-mb-sm"):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("layers", color="cyan").classes("text-h5")
                ui.label("記述子セット管理").classes("text-h6")
                ui.badge(f"{len(sets)} セット", color="cyan").props("outline")

            with ui.row().classes("q-gutter-xs"):
                def _add_set():
                    idx = len(sets) + 1
                    name = f"セット{idx}"
                    while name in sets:
                        idx += 1
                        name = f"セット{idx}"
                    active = state.get("active_descriptors", [])
                    sets[name] = {
                        "engines": [],
                        "active": True,
                        "descriptors": list(active) if active else None,
                    }
                    state["current_set_name"] = name
                    ui.notify(f"➕ セット「{name}」を作成", type="positive")

                ui.button("➕ 新規セット", on_click=_add_set).props(
                    "unelevated size=sm no-caps color=green-8"
                ).tooltip("現在の記述子選択を新しいセットとして保存")

                if len(sets) >= 2:
                    ui.button(
                        "📊 比較",
                        on_click=lambda: _open_set_compare_dialog(sets, state),
                    ).props("outline size=sm no-caps color=amber")

        # ── プリセットギャラリー ──
        try:
            from backend.chem.recommender import (
                get_target_categories,
                get_targets_by_category,
            )
            categories = get_target_categories()
            if categories:
                with ui.expansion(
                    "🏪 プリセットから作成", icon="collections_bookmark",
                ).classes("full-width q-mb-sm").props("dense"):
                    ui.label(
                        "予測目標に最適化された記述子セットをワンクリックで作成できます"
                    ).classes("text-caption text-grey q-mb-xs")
                    with ui.row().classes("q-gutter-xs flex-wrap"):
                        for cat_name in categories:
                            cat_recs = get_targets_by_category(cat_name)
                            for rec in cat_recs:
                                def _create_preset(r=rec):
                                    preset_name = f"📌 {r.target_name}"
                                    desc_names = [d.name for d in r.descriptors]
                                    sets[preset_name] = {
                                        "engines": sorted(set(d.library for d in r.descriptors)),
                                        "active": True,
                                        "descriptors": desc_names,
                                    }
                                    state["current_set_name"] = preset_name
                                    state["active_descriptors"] = list(desc_names)
                                    state["_applied_recommendation"] = r
                                    ui.notify(
                                        f"✅ プリセット「{r.target_name}」を作成 ({len(desc_names)}記述子)",
                                        type="positive",
                                    )

                                lib_colors = sorted(set(d.library for d in rec.descriptors))
                                with ui.button(
                                    rec.target_name,
                                    on_click=_create_preset,
                                ).props("outline dense size=sm no-caps color=cyan").classes("text-xs"):
                                    pass
        except ImportError:
            pass

        ui.separator().classes("q-my-sm")

        # ── セットカード一覧 ──
        with ui.scroll_area().style("max-height: 400px;"):
            with ui.row().classes("q-gutter-md flex-wrap full-width"):
                for set_name, info in sets.items():
                    is_current = (set_name == current)
                    descs = info.get("descriptors")
                    n_descs = len(descs) if descs else 0
                    engines = _get_engine_badges(descs)
                    pct = round(n_descs / total_available * 100) if total_available > 0 and descs else 0
                    is_active = info.get("active", True)

                    # カードの色設定
                    border_color = "rgba(0,212,255,0.7)" if is_current else (
                        "rgba(255,255,255,0.15)" if is_active else "rgba(255,255,255,0.05)"
                    )
                    bg_color = "rgba(0,35,60,0.6)" if is_current else "rgba(25,25,35,0.5)"
                    glow = "box-shadow: 0 0 15px rgba(0,200,255,0.15);" if is_current else ""

                    with ui.card().classes("q-pa-sm").style(
                        f"border: 2px solid {border_color}; border-radius: 10px;"
                        f"background: {bg_color}; min-width: 240px; max-width: 320px;"
                        f"transition: all 0.3s ease; {glow}"
                    ):
                        # ── カードヘッダー: 名前 + 操作ボタン ──
                        with ui.row().classes("items-center justify-between full-width no-wrap"):
                            with ui.row().classes("items-center q-gutter-xs no-wrap"):
                                if is_current:
                                    ui.icon("play_arrow", color="cyan").classes("text-body1")
                                else:
                                    ui.icon("folder", color="grey").classes("text-body1")
                                ui.label(set_name).classes(
                                    "text-body1 text-bold" + (" text-cyan" if is_current else "")
                                ).style("max-width: 160px; overflow: hidden; text-overflow: ellipsis;")

                            with ui.row().classes("q-gutter-none no-wrap"):
                                # 名前変更
                                ui.button(
                                    icon="edit",
                                    on_click=lambda n=set_name: _open_rename_dialog(n, sets, state),
                                ).props("flat round dense size=xs color=grey").tooltip("名前変更")

                                # 複製
                                def _dup(n=set_name):
                                    new_name = f"{n}_コピー"
                                    i = 2
                                    while new_name in sets:
                                        new_name = f"{n}_コピー{i}"
                                        i += 1
                                    sets[new_name] = copy.deepcopy(sets[n])
                                    state["current_set_name"] = new_name
                                    ui.notify(f"📋 「{new_name}」を複製", type="info")

                                ui.button(
                                    icon="content_copy",
                                    on_click=_dup,
                                ).props("flat round dense size=xs color=grey").tooltip("複製")

                                # 削除
                                if set_name != "デフォルト":
                                    def _del(n=set_name):
                                        del sets[n]
                                        if state.get("current_set_name") == n:
                                            state["current_set_name"] = "デフォルト"
                                        ui.notify(f"🗑️ 「{n}」を削除", type="info")

                                    ui.button(
                                        icon="delete_outline",
                                        on_click=_del,
                                    ).props("flat round dense size=xs color=red-4").tooltip("削除")

                        # ── 記述子数 + プログレスバー ──
                        with ui.column().classes("full-width q-mt-xs q-gutter-none"):
                            if descs is not None:
                                with ui.row().classes("items-center justify-between full-width"):
                                    ui.label(f"📐 {n_descs} 記述子").classes(
                                        "text-caption" + (" text-cyan" if is_current else " text-grey")
                                    )
                                    if total_available > 0:
                                        ui.label(f"{pct}%").classes("text-caption text-grey")
                                ui.linear_progress(
                                    value=pct / 100, color="cyan" if is_current else "grey",
                                ).props("rounded instant-feedback").style("height: 4px;")
                            else:
                                ui.label("📐 全記述子（未制限）").classes("text-caption text-amber")
                                ui.linear_progress(
                                    value=1.0, color="amber",
                                ).props("rounded instant-feedback").style("height: 4px;")

                        # ── エンジンバッジ ──
                        if engines:
                            with ui.row().classes("q-gutter-xs q-mt-xs flex-wrap"):
                                for eng in engines:
                                    color = _ENGINE_BADGE_COLORS.get(eng, "grey")
                                    ui.badge(eng, color=color).props("dense outline").classes("text-xs")

                        # ── アクションボタン ──
                        with ui.row().classes("q-gutter-xs q-mt-xs justify-center full-width"):
                            if not is_current:
                                def _switch(n=set_name):
                                    state["current_set_name"] = n
                                    if sets[n].get("descriptors"):
                                        state["active_descriptors"] = list(sets[n]["descriptors"])
                                    ui.notify(f"🔄 セット「{n}」に切替", type="info")

                                ui.button(
                                    "▶ 切替", on_click=_switch,
                                ).props("unelevated size=sm no-caps color=cyan")
                            else:
                                def _save():
                                    active = state.get("active_descriptors", [])
                                    sets[current]["descriptors"] = list(active)
                                    ui.notify(
                                        f"💾 セット「{current}」に {len(active)} 記述子を保存",
                                        type="positive",
                                    )

                                ui.button(
                                    "💾 現在の選択を保存", on_click=_save,
                                ).props("unelevated size=sm no-caps color=teal")

                            # アクティブ切替
                            def _toggle(n=set_name, val=not is_active):
                                sets[n]["active"] = val
                                status = "有効" if val else "無効"
                                ui.notify(f"{'✅' if val else '⏸️'} 「{n}」を{status}に", type="info")

                            if is_active:
                                ui.button(
                                    icon="pause_circle_outline", on_click=_toggle,
                                ).props("flat round dense size=xs color=grey").tooltip("比較解析から除外")
                            else:
                                ui.button(
                                    icon="play_circle_outline", on_click=_toggle,
                                ).props("flat round dense size=xs color=green").tooltip("比較解析に含める")

        # ── フッター: アクティブセット数 ──
        active_sets = [n for n, info in sets.items() if info.get("active", True)]
        if len(active_sets) > 1:
            ui.separator().classes("q-my-xs")
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("compare_arrows", color="amber")
                ui.label(
                    f"🔬 {len(active_sets)} セットが比較解析対象（アクティブ）"
                ).classes("text-caption text-amber")
