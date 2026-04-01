"""
frontend_nicegui/components/descriptor_status_bar.py

選択中の記述子を常時表示するフローティングステータスバー。

Implements: F-2-6 | 記述子セット常時表示パネル
設計:
  - メインコンテンツ下部にsticky固定
  - 折りたたみ時: セット名 + 記述子数 + 一括ボタン
  - 展開時: チップ一覧（個別ON/OFF） + セット切替 + 詳細ダイアログ起動
  - SMILES列未設定 or 計算未完了時は非表示
"""
from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)

# グループ→色マッピング（チップの色分け用）
_CHIP_COLORS: dict[str, str] = {
    "rdkit": "green",
    "molai": "purple",
    "xtb": "orange",
    "mordred": "blue",
    "cosmo": "deep-purple",
    "group_contrib": "teal",
    "descriptastorus": "lime",
    "molfeat": "pink",
    "mol2vec": "indigo",
    "chemprop": "red",
    "uma": "amber",
    "padel": "light-blue",
    "unipka": "cyan",
    "morgan": "blue-grey",
    "maccs": "light-green",
    "avalon": "brown",
}


def _guess_group(name: str) -> str:
    """記述子名からグループを推定してチップの色を決める。"""
    nl = name.lower()
    if nl.startswith("molai_") or nl.startswith("cnn_pca_"):
        return "molai"
    if nl.startswith("xtb_") or name in (
        "HomoEnergy", "LumoEnergy", "HomoLumoGap",
        "DipoleMoment", "Polarizability",
    ):
        return "xtb"
    if nl.startswith("mordred_") or nl.startswith("mrd_"):
        return "mordred"
    if nl.startswith("joback_"):
        return "group_contrib"
    if nl.startswith("ds_"):
        return "descriptastorus"
    if nl.startswith("molfeat_"):
        return "molfeat"
    if nl.startswith("mol2vec_"):
        return "mol2vec"
    if nl.startswith("chemprop_"):
        return "chemprop"
    if nl.startswith("uma_"):
        return "uma"
    if nl.startswith("padel_"):
        return "padel"
    if nl.startswith("pka") or name == "pKa_pred":
        return "unipka"
    if nl.startswith("mu_") or nl.startswith("ln_gamma") or nl.startswith("cosmo_"):
        return "cosmo"
    if nl.startswith("morgan"):
        return "morgan"
    if nl.startswith("maccs"):
        return "maccs"
    if nl.startswith("avalon"):
        return "avalon"
    if nl.startswith("fr_"):
        return "rdkit"
    return "rdkit"


def render_descriptor_status_bar(state: dict[str, Any]) -> None:
    """
    記述子セット常時表示フローティングバーを描画する。

    メインタブパネルの直後に配置し、position: sticky で下部固定。
    precalc_done=True の場合のみ表示される。
    """
    bar_container = ui.column().classes("full-width")

    def _rebuild_bar():
        bar_container.clear()

        # 表示条件: SMILES計算完了
        if not state.get("precalc_done") or state.get("precalc_df") is None:
            return

        precalc_df = state["precalc_df"]
        total_available = precalc_df.shape[1]

        # 現在のセット情報
        sets = state.get("descriptor_sets", {})
        current_set = state.get("current_set_name", "デフォルト")
        current_descs = sets.get(current_set, {}).get("descriptors")
        selected = state.get("selected_descriptors", [])
        active = state.get("active_descriptors", selected)

        # 実際に使用する記述子リスト
        use_descs = active if active else selected
        n_use = len(use_descs)

        # サンプル数
        n_samples = len(state["df"]) if state.get("df") is not None else 0

        # 過学習リスク判定
        ratio_warn = n_samples > 0 and n_use > n_samples
        ratio_info = n_use > 500

        with bar_container:
            with ui.card().classes("full-width q-pa-none q-ma-none").style(
                "position: sticky; bottom: 0; z-index: 200;"
                "background: rgba(10, 15, 30, 0.97);"
                "border-top: 1px solid rgba(0, 212, 255, 0.35);"
                "backdrop-filter: blur(12px);"
                "border-radius: 12px 12px 0 0;"
                "box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.4);"
            ):
                # ── 折りたたみヘッダー（常に表示） ──
                with ui.expansion(
                    "",
                    icon="layers",
                ).classes("full-width").props("dense header-class=q-pa-xs").style(
                    "background: transparent;"
                ) as bar_expansion:
                    # ヘッダーのカスタムスロット
                    pass

                # ── ヘッダー行（expansion の外に配置して常に見えるようにする） ──
                # expansion を使わずカスタム展開UIにする
                bar_expansion.delete()

                # 展開状態管理
                _expanded = {"value": False}

                # ── メインバー（常時表示部分） ──
                with ui.row().classes(
                    "items-center full-width q-gutter-sm q-px-md q-py-xs"
                ).style("min-height: 42px;"):
                    # アイコン + セット名
                    ui.icon("layers", color="cyan").classes("text-body1")
                    ui.label("記述子:").classes("text-caption text-grey")

                    # セット名バッジ
                    ui.chip(
                        f"{current_set}",
                        color="cyan" if not ratio_warn else "amber",
                        text_color="white",
                    ).props("dense outline size=sm").classes("text-xs")

                    # 記述子数
                    count_color = "text-green" if not ratio_warn else "text-amber"
                    ui.label(f"{n_use}個").classes(f"text-body2 text-bold {count_color}")
                    ui.label(f"/ {total_available}").classes("text-caption text-grey")

                    # 警告アイコン
                    if ratio_warn:
                        ui.icon("warning", color="amber").classes("text-body2").tooltip(
                            f"記述子数({n_use}) > サンプル数({n_samples})で過学習リスク"
                        )
                    elif ratio_info:
                        ui.icon("info_outline", color="grey").classes("text-caption").tooltip(
                            f"記述子が{n_use}個あります。計算時間が長くなる場合があります。"
                        )

                    ui.space()

                    # ── 一括操作ボタン ──
                    def _all_on():
                        all_cols = list(precalc_df.columns)
                        state["active_descriptors"] = all_cols
                        state["selected_descriptors"] = all_cols
                        ui.notify(f"✅ 全{len(all_cols)}記述子をON", type="positive", timeout=2000)
                        _rebuild_bar()

                    def _all_off():
                        state["active_descriptors"] = []
                        state["selected_descriptors"] = []
                        ui.notify("⬜ 全記述子をOFF", type="info", timeout=2000)
                        _rebuild_bar()

                    ui.button("全ON", on_click=_all_on).props(
                        "flat dense size=xs no-caps color=cyan"
                    )
                    ui.button("全OFF", on_click=_all_off).props(
                        "flat dense size=xs no-caps color=grey"
                    )

                    # セット切替ボタン群
                    if len(sets) > 1:
                        ui.separator().props("vertical")
                        for sn in list(sets.keys())[:5]:  # 最大5セットまで表示
                            is_active = (sn == current_set)
                            s_descs = sets[sn].get("descriptors")
                            s_count = len(s_descs) if s_descs else total_available

                            def _switch_set(name=sn):
                                state["current_set_name"] = name
                                if sets[name].get("descriptors"):
                                    state["active_descriptors"] = list(sets[name]["descriptors"])
                                    state["selected_descriptors"] = list(sets[name]["descriptors"])
                                ui.notify(f"🔄 「{name}」に切替", type="info", timeout=2000)
                                _rebuild_bar()
                                # 外部タブも再描画
                                refresh = state.get("_refresh_tabs")
                                if refresh:
                                    try:
                                        refresh()
                                    except Exception:
                                        pass

                            ui.button(
                                f"{sn} ({s_count})",
                                on_click=_switch_set,
                            ).props(
                                f"{'unelevated' if is_active else 'outline'} dense size=xs no-caps "
                                f"color={'cyan' if is_active else 'grey-6'}"
                            ).classes("text-xs")

                    # 展開/折りたたみトグル
                    def _toggle_expand():
                        _expanded["value"] = not _expanded["value"]
                        _rebuild_bar()

                    ui.button(
                        icon="expand_less" if _expanded["value"] else "expand_more",
                        on_click=_toggle_expand,
                    ).props("flat dense round size=sm color=grey")

                # ── 展開部分: チップ一覧（個別ON/OFF） ──
                if _expanded["value"] and use_descs:
                    ui.separator().classes("q-mx-md")
                    with ui.scroll_area().style(
                        "max-height: 180px;"
                    ).classes("q-px-md q-py-xs"):
                        with ui.element("div").style(
                            "display: flex; flex-wrap: wrap; gap: 4px;"
                        ):
                            # 最大200個まで表示（パフォーマンス）
                            display_descs = use_descs[:200]
                            remaining = len(use_descs) - 200

                            for desc_name in display_descs:
                                group = _guess_group(desc_name)
                                color = _CHIP_COLORS.get(group, "grey")

                                def _remove_desc(name=desc_name):
                                    if name in state.get("active_descriptors", []):
                                        state["active_descriptors"].remove(name)
                                    if name in state.get("selected_descriptors", []):
                                        state["selected_descriptors"].remove(name)
                                    _rebuild_bar()

                                ui.chip(
                                    desc_name,
                                    color=color,
                                    removable=True,
                                    on_remove=_remove_desc,
                                ).props("dense size=sm outline").classes(
                                    "text-xs"
                                ).style("max-width: 200px; overflow: hidden;")

                            if remaining > 0:
                                ui.label(
                                    f"... 他 {remaining}個"
                                ).classes("text-caption text-grey q-pa-xs")

                    # 展開時フッター: 記述子追加ボタン
                    with ui.row().classes(
                        "items-center q-gutter-sm q-px-md q-py-xs"
                    ):
                        # 詳細ダイアログへの導線
                        def _open_detail():
                            from frontend_nicegui.components.descriptor_selector_dialog import (
                                render_selected_descriptors_panel,
                            )
                            # SMILESタブに遷移
                            switch_fn = state.get("_switch_to_data_smiles")
                            if switch_fn:
                                switch_fn()
                            else:
                                ui.notify(
                                    "📂 データ設定 → ⚗️ SMILES特徴量 タブで詳細選択できます",
                                    type="info",
                                )

                        ui.button(
                            "🔬 記述子詳細設定を開く",
                            on_click=_open_detail,
                        ).props("outline size=sm no-caps color=cyan")

                        ui.space()

                        # 現在のセットの統計
                        if n_samples > 0:
                            ratio = n_use / n_samples
                            ratio_text = f"記述子/サンプル比: {ratio:.2f}"
                            ratio_color = (
                                "text-green" if ratio < 0.5 else
                                "text-amber" if ratio < 1.0 else
                                "text-red"
                            )
                            ui.label(ratio_text).classes(f"text-caption {ratio_color}")

    _rebuild_bar()

    # state に再描画関数を登録（タブ切替やデータ変更時に呼ばれる）
    state["_refresh_descriptor_bar"] = _rebuild_bar
