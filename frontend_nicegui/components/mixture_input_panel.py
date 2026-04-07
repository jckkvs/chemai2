"""
frontend_nicegui/components/mixture_input_panel.py

混合物入力パネル — スタンドアロンUIコンポーネント。

- 動的行追加 (2〜任意数)
- 比率タイプ切り替え (重量比/mol比/その他)
- CSVテンプレートダウンロード
- 比率変換プレビュー
- 混合物特徴量計算のトリガー

既存UIへの影響: なし（完全新規コンポーネント）
main.pyのタブパネル内でrender_mixture_panel(state)で呼び出して使用。
"""
from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)


def render_mixture_panel(state: dict[str, Any]) -> None:
    """混合物入力パネルを描画する。"""

    # ── 内部状態 ──
    mixture_state: dict[str, Any] = {
        "components": [],       # [{smiles, name, ratio, row_ref}, ...]
        "ratio_type": "weight",
        "other_unit": "",
        "result": None,
    }

    # ═══════════════════════════════════════════════════════════
    # ヘッダー
    # ═══════════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 24px;"
    ):
        with ui.row().classes("items-center gap-4 w-full"):
            ui.icon("science").classes("text-3xl").style("color: #a78bfa;")
            with ui.column().classes("gap-0"):
                ui.label("🧪 混合物特徴量計算").classes("text-xl font-bold").style(
                    "color: #e0e0f0;"
                )
                ui.label(
                    "複数化合物の混合比を指定し、加重平均記述子を計算"
                ).classes("text-sm").style("color: #a0a0c0;")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # CSVテンプレートダウンロード
    # ═══════════════════════════════════════════════════════════
    with ui.expansion(
        "📥 CSVテンプレート",
        icon="download",
    ).classes("w-full").style(
        "background: rgba(255,255,255,0.02); border-radius: 12px;"
    ):
        ui.markdown(
            "- 同一 `session_id` の行が1混合物として処理されます\n"
            "- `ratio_unit`: `weight`(重量比) / `mole`(mol比) / `other`(その他)\n"
            "- 複数混合物を一括処理できます"
        ).classes("text-sm").style("color: #a0a0c0;")

        with ui.row().classes("gap-4 q-mt-sm"):
            def _download_template():
                try:
                    from backend.chem.mixture_csv_template import generate_template_csv
                    csv_bytes = generate_template_csv()
                    ui.download(
                        csv_bytes,
                        "chemai2_mixture_template_v1.0.csv",
                    )
                    ui.notify("📥 テンプレートをダウンロードしました", type="positive")
                except Exception as e:
                    ui.notify(f"❌ ダウンロード失敗: {e}", type="negative")

            ui.button(
                "⬇️ CSVテンプレートをダウンロード",
                on_click=_download_template,
                icon="download",
            ).props("outline color=cyan")

        # CSVアップロード
        ui.separator().classes("q-my-sm")
        ui.label("📤 CSVファイルをアップロード").classes("text-sm font-bold").style(
            "color: #e0e0f0;"
        )

        async def _on_csv_upload(e):
            try:
                from backend.chem.mixture_csv_template import parse_mixture_csv
                content = e.content.read()
                mixtures = parse_mixture_csv(content)
                state["_mixture_csv_parsed"] = mixtures
                ui.notify(
                    f"✅ {len(mixtures)}件の混合物を読み込みました",
                    type="positive",
                )
                for m in mixtures:
                    for w in m.warnings:
                        ui.notify(f"⚠️ {w}", type="warning")
            except Exception as ex:
                ui.notify(f"❌ CSV解析エラー: {ex}", type="negative")

        ui.upload(
            on_upload=_on_csv_upload,
            label="CSVファイルを選択",
            auto_upload=True,
        ).props("accept=.csv").classes("w-full")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 手動入力セクション
    # ═══════════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        ui.label("✏️ 手動入力").classes("text-lg font-bold").style(
            "color: #e0e0f0;"
        )

        # 比率タイプ選択
        with ui.row().classes("items-center gap-4 q-mt-sm"):
            ui.label("比率タイプ:").style("color: #a0a0c0;")
            ratio_radio = ui.radio(
                options={
                    "weight": "⚖️ 重量比",
                    "mole": "🔢 mol比",
                    "other": "⚙️ その他",
                },
                value="weight",
            ).props("inline")

            other_input = ui.input(
                "単位名",
                placeholder="例: volume_fraction",
            ).classes("w-48")
            other_input.set_visibility(False)

            def _on_ratio_change(e):
                mixture_state["ratio_type"] = e.value
                other_input.set_visibility(e.value == "other")

            ratio_radio.on_value_change(_on_ratio_change)

        # 成分入力テーブル
        ui.separator().classes("q-my-sm")
        components_container = ui.column().classes("w-full gap-2")

        def _add_component(order: int | None = None, smiles: str = "", name: str = "", ratio: float = 1.0):
            if order is None:
                order = len(mixture_state["components"]) + 1

            comp_data: dict[str, Any] = {"order": order}

            with components_container:
                with ui.row().classes("w-full items-end gap-2").style(
                    "background: rgba(255,255,255,0.02); "
                    "border-radius: 8px; padding: 8px;"
                ) as row_ref:
                    ui.label(f"#{order}").classes("w-8 text-center font-bold").style(
                        "color: #00d4ff;"
                    )
                    comp_data["smiles_input"] = ui.input(
                        "SMILES", placeholder="例: CCO", value=smiles,
                    ).classes("flex-grow")
                    comp_data["name_input"] = ui.input(
                        "名称（任意）", placeholder="例: ethanol", value=name,
                    ).classes("w-32")
                    comp_data["ratio_input"] = ui.number(
                        "比率", min=0.001, step=0.1, value=ratio,
                    ).classes("w-24")

                    def _remove(ref=row_ref, data=comp_data):
                        ref.delete()
                        if data in mixture_state["components"]:
                            mixture_state["components"].remove(data)

                    ui.button(
                        icon="delete", on_click=_remove,
                    ).props("flat color=red size=sm round")

                    comp_data["row_ref"] = row_ref

            mixture_state["components"].append(comp_data)

        # 初期行（2行）
        _add_component(1)
        _add_component(2)

        with ui.row().classes("q-mt-sm gap-4"):
            ui.button(
                "➕ 成分を追加",
                on_click=lambda: _add_component(),
                icon="add",
            ).props("outline color=cyan")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 加重方法設定
    # ═══════════════════════════════════════════════════════════
    with ui.expansion(
        "⚙️ 特徴量ごとの加重方法カスタマイズ",
        icon="settings",
    ).classes("w-full").style(
        "background: rgba(255,255,255,0.02); border-radius: 12px;"
    ):
        ui.markdown(
            "各特徴量の加重方法は物理化学的根拠に基づき自動設定されています。\n"
            "必要に応じて個別に上書きできます。\n\n"
            "| カテゴリ | 推奨加重 | 根拠 |\n"
            "|---------|---------|------|\n"
            "| 質量・体積・LogP | 重量比 | 質量保存則・巨視的物性 |\n"
            "| 軌道エネルギー・反応性 | mol比 | 電子状態は分子単位 |\n"
            "| 3D幾何・FP | 文脈依存 | 線形加重が不適切 |"
        ).classes("text-sm").style("color: #a0a0c0;")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 実行ボタン + 結果表示
    # ═══════════════════════════════════════════════════════════
    result_container = ui.column().classes("w-full")

    async def _run_mixture_calc():
        components = []
        for comp in mixture_state["components"]:
            smiles = comp["smiles_input"].value.strip()
            if not smiles:
                ui.notify("❌ SMILESが空の成分があります", type="negative")
                return
            components.append({
                "smiles": smiles,
                "compound_name": comp["name_input"].value.strip() or None,
                "ratio_value": float(comp["ratio_input"].value),
                "ratio_unit": mixture_state["ratio_type"],
            })

        if len(components) < 2:
            ui.notify("❌ 成分は2つ以上必要です", type="negative")
            return

        ui.notify("⏳ 混合物特徴量を計算中...", type="info")

        try:
            from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor
            extractor = MixtureFeatureExtractor()
            result = extractor.extract(components)
            mixture_state["result"] = result

            # 結果表示
            result_container.clear()
            with result_container:
                # 変換情報
                info = result.conversion_info
                with ui.card().classes("w-full q-mt-md").style(
                    "background: rgba(74, 222, 128, 0.05); "
                    "border: 1px solid rgba(74, 222, 128, 0.2); "
                    "border-radius: 12px; padding: 16px;"
                ):
                    ui.label("✅ 計算完了").classes("text-lg font-bold").style(
                        "color: #4ade80;"
                    )

                    # 変換テーブル
                    headers = ["#", "SMILES", "分子量", "重量分率", "モル分率"]
                    rows = []
                    for i in range(len(components)):
                        rows.append({
                            "#": i + 1,
                            "SMILES": components[i]["smiles"],
                            "分子量": f"{info['molecular_weights'][i]:.2f}",
                            "重量分率": f"{info['weight_fractions'][i]*100:.1f}%",
                            "モル分率": f"{info['mole_fractions'][i]*100:.1f}%",
                        })

                    with ui.table(
                        columns=[{"name": h, "label": h, "field": h} for h in headers],
                        rows=rows,
                    ).classes("w-full").props("dense flat"):
                        pass

                    ui.label(
                        f"🧪 混合物特徴量: {len(result.mixture_features)}列"
                    ).classes("q-mt-sm text-sm").style("color: #a0a0c0;")

                # 警告があれば表示
                for w in result.warnings:
                    ui.label(f"⚠️ {w}").classes("text-sm").style("color: #fbbf24;")

                # stateに保存（下流のパイプラインで使用可能）
                state["_mixture_result"] = result

            ui.notify(
                f"✅ {len(result.mixture_features)}列の混合物特徴量を計算しました",
                type="positive",
            )

        except Exception as e:
            ui.notify(f"❌ 計算エラー: {e}", type="negative")
            logger.error("混合物特徴量計算エラー: %s", e, exc_info=True)

    ui.button(
        "🚀 混合物特徴量を計算",
        on_click=_run_mixture_calc,
        icon="play_arrow",
    ).props("color=primary size=lg").classes("w-full").style(
        "background: linear-gradient(135deg, #7b2ff7, #00d4ff) !important; "
        "border-radius: 12px; font-size: 16px; font-weight: 600;"
    )
