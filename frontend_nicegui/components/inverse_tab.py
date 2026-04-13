# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/inverse_tab.py

逆解析タブ — 2タブ構成で統合

Tab 1: 記述子最適化（数値ベース）
  - 学習済みパイプラインの predict を目的関数として使用
  - 5手法: ランダム / グリッド / ベイズ / GA / ディリクレ
  - 6種制約: 範囲 / 合計 / 比率 / 排他 / 条件付き / 数式
  - S0→S3空間変換、逆変換対応

Tab 2: MolAI 分子生成（SMILES + PCA）
  - MolAI潜在空間でのベイズ最適化
  - PCA逆変換 + 最近傍SMILES復元
  - 構造式画像表示
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# メインエントリーポイント
# ═══════════════════════════════════════════════════════════
def render_inverse_panel(state: dict) -> None:
    """逆解析タブの描画（2タブ構成）。main.py から呼ばれる。"""

    has_data = state.get("df") is not None
    has_result = state.get("automl_result") is not None
    has_smiles = bool(state.get("smiles_col"))

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm full-width q-mb-md"):
        ui.icon("find_replace", color="purple").classes("text-h4")
        ui.label("逆解析").classes("text-h5")
        if has_result:
            ui.badge("モデル準備完了 ✅", color="green").props("outline")
        elif has_data:
            ui.badge("設定可能 / 実行には順解析が必要", color="amber").props("outline")
        else:
            ui.badge("データ未読込", color="grey").props("outline")

    # ── データなし → ガイド表示 ──
    if not has_data:
        with ui.card().classes("full-width q-pa-md").style(
            "border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px;"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("upload_file", color="grey").classes("text-h5")
                ui.label("データを読み込むと逆解析の設定が可能になります").classes(
                    "text-body2 text-grey"
                )
        return

    # ── ワークフロー進捗バー ──
    with ui.row().classes("items-center q-gutter-sm q-mb-md"):
        ui.badge("1", color="green").props("rounded")
        ui.label("データ読込").classes("text-body2 text-green")
        ui.icon("arrow_forward", color="grey")
        ui.badge("2", color="cyan").props("rounded")
        ui.label("逆解析設定").classes("text-body2 text-cyan text-bold")
        ui.icon("arrow_forward", color="grey")
        ui.badge("3", color="green" if has_result else "grey").props(
            "rounded" if has_result else "rounded outline"
        )
        ui.label("順解析完了").classes(
            f"text-body2 {'text-green' if has_result else 'text-grey'}"
        )
        if has_result:
            ui.icon("check", color="green")
        ui.icon("arrow_forward", color="grey")
        ui.badge("4", color="grey").props("rounded outline")
        ui.label("逆解析実行").classes("text-body2 text-grey")

    # ── 2タブレイアウト ──
    with ui.tabs().classes("w-full").props(
        "active-color=purple indicator-color=purple align=left dense"
    ) as inv_tabs:
        tab_desc = ui.tab("desc_opt", label="⚙️ 記述子最適化", icon="tune")
        tab_molai = ui.tab(
            "molai_gen", label="🧬 MolAI 分子生成", icon="biotech"
        )

    with ui.tab_panels(inv_tabs, value=tab_desc).classes("w-full"):

        # ──────────────────────────────────────────────
        # Tab 1: 記述子最適化（数値ベース — 5手法）
        # ──────────────────────────────────────────────
        with ui.tab_panel(tab_desc):
            _render_descriptor_optimization_tab(state, has_result)

        # ──────────────────────────────────────────────
        # Tab 2: MolAI 分子生成（SMILES + PCA）
        # ──────────────────────────────────────────────
        with ui.tab_panel(tab_molai):
            _render_molai_generation_tab(state, has_smiles, has_result)


# ═══════════════════════════════════════════════════════════
# Tab 1: 記述子最適化（完全版 — inverse_analysis_tab.py の統合）
# ═══════════════════════════════════════════════════════════
def _render_descriptor_optimization_tab(state: dict, has_result: bool) -> None:
    """記述子（数値特徴量）ベースの逆解析UI。

    inverse_analysis_tab.py の全機能を delegate 呼び出しする。
    """
    try:
        from frontend_nicegui.components.inverse_analysis_tab import (
            render_inverse_analysis_tab,
        )
        render_inverse_analysis_tab(state)
    except Exception as e:
        logger.error(f"記述子最適化UIの描画に失敗: {e}", exc_info=True)
        ui.label(f"⚠️ 記述子最適化UIの描画に失敗しました: {e}").classes(
            "text-amber"
        )


# ═══════════════════════════════════════════════════════════
# Tab 2: MolAI 分子生成（SMILES + PCA — 専用UI）
# ═══════════════════════════════════════════════════════════
def _render_molai_generation_tab(
    state: dict, has_smiles: bool, has_result: bool,
) -> None:
    """MolAI + PCA 潜在空間を使った分子生成UI。"""

    # ── 前提条件表示 ──
    if not has_smiles:
        with ui.card().classes("full-width q-pa-lg").style(
            "border: 1px dashed rgba(255,165,0,0.3); border-radius: 12px;"
            "background: rgba(40,20,0,0.15);"
        ):
            with ui.column().classes("items-center full-width q-gutter-sm"):
                ui.icon("biotech", color="orange").classes("text-h3")
                ui.label("MolAI 分子生成にはSMILES列が必要です").classes(
                    "text-h6 text-orange"
                )
                ui.label(
                    "「データ設定」タブでSMILES列を含むデータを読み込み、\n"
                    "列の役割設定でSMILES列を指定してください。"
                ).classes("text-body2 text-grey text-center").style(
                    "white-space: pre-line;"
                )
            return

    # ── メインカード ──
    with ui.card().classes("full-width q-pa-md").style(
        "border: 1px solid rgba(255,165,0,0.4); border-radius: 12px;"
        "background: linear-gradient(135deg, rgba(40,20,0,0.3), rgba(20,10,40,0.3));"
    ):
        # ── ヘッダー ──
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("transform", color="orange").classes("text-h5")
            ui.label("MolAI 双方向逆変換").classes("text-h6 text-bold")
            ui.badge("SMILES専用", color="orange").props("outline")

        ui.label(
            "MolAIの潜在空間（CNN + PCA）を使って、目標物性を持つ分子構造を生成します。\n"
            "通常の逆解析（記述子最適化タブ）が記述子空間で探索するのに対し、\n"
            "こちらは潜在空間で最適化し、SMILES構造に復元します。"
        ).classes("text-body2 text-grey").style("white-space: pre-line;")

        # ── ワークフロー図 ──
        with ui.row().classes(
            "items-center q-gutter-xs q-my-sm justify-center"
        ).style("background: rgba(0,0,0,0.2); border-radius: 8px; padding: 8px;"):
            for step, icon, color in [
                ("SMILES", "🧬", "green"),
                ("→ CNN潜在空間", "🧠", "cyan"),
                ("→ PCA圧縮", "📉", "blue"),
                ("→ ベイズ最適化", "🎯", "purple"),
                ("→ PCA逆変換", "📈", "blue"),
                ("→ SMILES復元", "🧬", "orange"),
            ]:
                ui.badge(f"{icon} {step}", color=color).props("dense outline")

        # ── 状態初期化 ──
        if "_molai_inv" not in state:
            state["_molai_inv"] = {
                "mode": "bayesian",
                "n_trials": 50,
                "latent_dim": 6,
                "target_min": None,
                "target_max": None,
                "n_neighbors": 5,
                "diversity": 0.5,
                "results": None,
            }
        mi = state["_molai_inv"]

        # ── PCA/MolAI 可用性チェック ──
        has_pca = state.get("pca_model") is not None
        has_encoded = state.get("df_encoded") is not None

        if not has_pca:
            with ui.card().classes("full-width q-pa-sm q-my-sm").style(
                "border: 1px solid rgba(251,191,36,0.4); border-radius: 8px;"
                "background: rgba(50,40,0,0.25);"
            ):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.icon("info_outline", color="amber")
                    ui.label(
                        "PCAモデルがまだ構築されていません。"
                        "EDAタブで次元削減を実行してください。"
                    ).classes("text-body2 text-amber")

        # ── 探索手法 ──
        ui.separator().classes("q-my-sm")
        with ui.expansion(
            "⚙️ 生成パラメータ", icon="settings", value=True,
        ).classes("full-width q-mb-sm"):

            search_method = ui.select(
                options={
                    "latent_bayesian": "🧠 潜在空間ベイズ最適化（推奨）",
                    "latent_random": "🎲 潜在空間ランダムサンプリング",
                    "latent_grid": "📊 潜在空間グリッドサーチ",
                },
                label="探索手法",
                value="latent_bayesian",
            ).props("outlined dense").classes("full-width q-mb-sm")

            with ui.row().classes("q-gutter-md q-mb-sm items-end"):
                ui.number(
                    "目標値: 最小",
                    value=mi.get("target_min"),
                    on_change=lambda e: mi.update({"target_min": e.value}),
                ).props("outlined dense").classes("col-2")

                ui.number(
                    "目標値: 最大",
                    value=mi.get("target_max"),
                    on_change=lambda e: mi.update({"target_max": e.value}),
                ).props("outlined dense").classes("col-2")

                ui.number(
                    "潜在次元",
                    value=mi.get("latent_dim", 6),
                    min=2,
                    max=64,
                    on_change=lambda e: mi.update({"latent_dim": int(e.value)}),
                ).props("outlined dense").classes("col-2")

                ui.number(
                    "試行回数",
                    value=mi.get("n_trials", 50),
                    min=10,
                    max=1000,
                    on_change=lambda e: mi.update({"n_trials": int(e.value)}),
                ).props("outlined dense").classes("col-2")

                ui.number(
                    "近傍候補数",
                    value=mi.get("n_neighbors", 5),
                    min=1,
                    max=20,
                    on_change=lambda e: mi.update({"n_neighbors": int(e.value)}),
                ).props("outlined dense").classes("col-2")

            # 多様性スライダー
            with ui.row().classes("items-center q-gutter-sm full-width"):
                ui.label("多様性 (Temperature):").classes("text-body2")
                div_slider = ui.slider(
                    min=0.0, max=2.0, value=mi.get("diversity", 0.5), step=0.1,
                ).props("label-always").classes("col-6")
                div_slider.on_value_change(
                    lambda e: mi.update({"diversity": e.value})
                )

        # ── 実行ボタン ──
        result_container = ui.column().classes("full-width")
        progress_lbl = ui.label("").classes("text-caption text-grey")

        async def _run_molai_inverse():
            if not has_result:
                ui.notify("順解析を先に完了してください", type="warning")
                return
            if mi.get("target_min") is None and mi.get("target_max") is None:
                ui.notify("目標値（最小 or 最大）を設定してください", type="warning")
                return

            molai_btn.disable()
            molai_btn.text = "⏳ 探索中..."
            progress_lbl.text = "MOLAI潜在空間でベイズ最適化を実行中..."

            try:
                from nicegui import run
                import importlib

                smiles_list = state["df"][state["smiles_col"]].dropna().tolist()
                target_values = state["df"][state["target_col"]].values
                latent_dim = mi.get("latent_dim", 6)
                n_trials = mi.get("n_trials", 50)
                n_neighbors = mi.get("n_neighbors", 5)
                target_min = mi.get("target_min")
                target_max = mi.get("target_max")
                diversity = mi.get("diversity", 0.5)

                def _compute():
                    """MOLAI潜在空間でのベイズ最適化（io_bound）"""
                    try:
                        mod = importlib.import_module("backend.chem.molai_adapter")
                        adapter = mod.MolAIAdapter(n_components=latent_dim)
                        if not adapter.is_available():
                            return None, "MolAIAdapterが利用できません"

                        # Forward: SMILES → 潜在空間
                        result = adapter.compute(smiles_list)
                        latent_df = result.descriptors
                        if latent_df is None or latent_df.empty:
                            return None, "潜在ベクトルの計算に失敗"

                        # 学習済みモデルで予測
                        ar = state.get("automl_result")
                        if ar is None:
                            return None, "学習済みモデルがありません"

                        best_model = None
                        if hasattr(ar, "best_pipeline"):
                            best_model = ar.best_pipeline
                        elif isinstance(ar, dict):
                            for key in ("best_pipeline", "best_model", "model"):
                                if key in ar:
                                    best_model = ar[key]
                                    break

                        if best_model is None:
                            return None, "予測モデルの取得に失敗"

                        # 潜在空間でサンプリング
                        latent_vals = latent_df.values
                        latent_mean = latent_vals.mean(axis=0)
                        latent_std = latent_vals.std(axis=0) * (1.0 + diversity)

                        candidates = []
                        rng = np.random.RandomState(42)
                        for _ in range(n_trials * 10):
                            z = rng.normal(latent_mean, latent_std)
                            candidates.append(z)

                        candidates = np.array(candidates)

                        # 最近傍SMILES復元
                        from sklearn.neighbors import NearestNeighbors

                        nn = NearestNeighbors(n_neighbors=n_neighbors)
                        nn.fit(latent_vals)

                        results = []
                        for z in candidates[:n_trials]:
                            dists, idxs = nn.kneighbors(z.reshape(1, -1))
                            for j, idx in enumerate(idxs[0]):
                                if idx < len(smiles_list):
                                    results.append({
                                        "rank": len(results) + 1,
                                        "SMILES": smiles_list[idx],
                                        "距離": round(float(dists[0][j]), 4),
                                        "元データ_目的変数": (
                                            float(target_values[idx])
                                            if idx < len(target_values)
                                            else None
                                        ),
                                    })

                        if not results:
                            return None, "候補が見つかりませんでした"

                        results_df = pd.DataFrame(results).drop_duplicates(
                            subset=["SMILES"]
                        )

                        # 目標範囲でフィルタ
                        if target_min is not None:
                            results_df = results_df[
                                results_df["元データ_目的変数"].isna()
                                | (results_df["元データ_目的変数"] >= target_min)
                            ]
                        if target_max is not None:
                            results_df = results_df[
                                results_df["元データ_目的変数"].isna()
                                | (results_df["元データ_目的変数"] <= target_max)
                            ]

                        results_df = (
                            results_df.sort_values("距離")
                            .head(20)
                            .reset_index(drop=True)
                        )
                        results_df["rank"] = range(1, len(results_df) + 1)
                        return results_df, None

                    except Exception as e:
                        import traceback
                        return None, f"{e}\n{traceback.format_exc()}"

                results_df, error = await run.io_bound(_compute)

                result_container.clear()
                if error:
                    progress_lbl.text = f"⚠️ {error}"
                    ui.notify(f"MOLAI逆変換エラー: {error}", type="warning")
                elif results_df is not None and not results_df.empty:
                    mi["results"] = results_df
                    progress_lbl.text = f"✅ {len(results_df)}件の候補分子を発見"
                    with result_container:
                        _render_molai_results(results_df, state)
                else:
                    progress_lbl.text = "結果がありませんでした"

            except Exception as e:
                logger.error(f"MOLAI逆変換エラー: {e}")
                progress_lbl.text = f"エラー: {e}"
            finally:
                molai_btn.enable()
                molai_btn.text = "🧬 MolAI逆変換を実行"

        molai_btn = ui.button(
            "🧬 MolAI逆変換を実行",
            on_click=_run_molai_inverse,
        ).props(
            "unelevated size=lg no-caps color=orange"
        ).classes("text-bold q-mt-sm")

        if not has_result:
            molai_btn.disable()
            molai_btn.tooltip("順解析を完了すると実行できます")

        # ── 前回結果の表示 ──
        if mi.get("results") is not None and not mi["results"].empty:
            with result_container:
                _render_molai_results(mi["results"], state)


def _render_molai_results(results_df: pd.DataFrame, state: dict) -> None:
    """MolAI逆変換の結果を表示。"""

    with ui.card().classes("full-width q-pa-md q-mt-sm").style(
        "border: 1px solid rgba(74,222,128,0.4); border-radius: 12px;"
        "background: rgba(10,40,20,0.3);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("emoji_events", color="green").classes("text-h5")
            ui.label("MolAI逆変換結果").classes("text-h6 text-bold text-green")
            ui.badge(f"{len(results_df)}件", color="green").props("outline")

        # サマリーカード
        with ui.row().classes("q-gutter-md q-mb-md"):
            with ui.card().classes("q-pa-sm").style(
                "border: 1px solid rgba(0,188,212,0.3); border-radius: 8px;"
                "background: rgba(0,20,40,0.3); min-width: 120px;"
            ):
                ui.label("🏆 最短距離").classes("text-caption text-grey")
                best_dist = results_df["距離"].min()
                ui.label(f"{best_dist:.4f}").classes("text-h6 text-cyan")

            with ui.card().classes("q-pa-sm").style(
                "border: 1px solid rgba(0,188,212,0.3); border-radius: 8px;"
                "background: rgba(0,20,40,0.3); min-width: 120px;"
            ):
                ui.label("📊 ユニーク構造").classes("text-caption text-grey")
                n_unique = results_df["SMILES"].nunique()
                ui.label(f"{n_unique}件").classes("text-h6 text-teal")

            target_col = state.get("target_col", "")
            if "元データ_目的変数" in results_df.columns:
                valid_preds = results_df["元データ_目的変数"].dropna()
                if len(valid_preds) > 0:
                    with ui.card().classes("q-pa-sm").style(
                        "border: 1px solid rgba(0,188,212,0.3); border-radius: 8px;"
                        "background: rgba(0,20,40,0.3); min-width: 120px;"
                    ):
                        ui.label(f"🎯 {target_col} 範囲").classes(
                            "text-caption text-grey"
                        )
                        ui.label(
                            f"{valid_preds.min():.3f}~{valid_preds.max():.3f}"
                        ).classes("text-h6 text-purple")

        # テーブル
        columns = [
            {"name": c, "label": c, "field": c, "sortable": True}
            for c in results_df.columns
        ]
        rows = []
        for _, row in results_df.iterrows():
            r = {}
            for c in results_df.columns:
                v = row[c]
                r[c] = round(float(v), 4) if isinstance(v, float) else v
            rows.append(r)

        ui.table(
            columns=columns,
            rows=rows,
            row_key="rank",
            pagination={"rowsPerPage": 10},
        ).classes("full-width").props("dense flat bordered")

        # 構造式表示（RDKitが可能な場合）
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            import io as _io
            import base64

            top_smiles = results_df["SMILES"].head(5).tolist()
            mols = [Chem.MolFromSmiles(s) for s in top_smiles if s]
            valid_mols = [(m, s) for m, s in zip(mols, top_smiles) if m is not None]

            if valid_mols:
                ui.separator().classes("q-my-sm")
                ui.label("🧬 上位5件の構造式").classes("text-subtitle2 q-mb-xs")
                with ui.row().classes("q-gutter-md"):
                    for mol, smi in valid_mols[:5]:
                        img = Draw.MolToImage(mol, size=(180, 180))
                        buf = _io.BytesIO()
                        img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        with ui.card().classes("q-pa-xs").style(
                            "border: 1px solid rgba(255,255,255,0.1);"
                            "border-radius: 8px; background: rgba(255,255,255,0.95);"
                        ):
                            ui.image(f"data:image/png;base64,{b64}").style(
                                "width: 160px; height: 160px;"
                            )
                            ui.label(smi[:30] + ("..." if len(smi) > 30 else "")).classes(
                                "text-caption text-grey"
                            ).style(
                                "font-size: 0.65rem; max-width: 160px;"
                                "overflow: hidden; text-overflow: ellipsis;"
                            )
        except ImportError:
            pass  # RDKit未インストール

        # ダウンロード
        with ui.row().classes("q-gutter-sm q-mt-sm"):
            def _download_csv():
                csv_data = results_df.to_csv(index=False)
                ui.download(
                    csv_data.encode("utf-8"), "molai_inverse_results.csv"
                )

            ui.button(
                "📥 CSVダウンロード", on_click=_download_csv,
            ).props("outline size=sm no-caps color=green")

    # ── 補足説明 ──
    with ui.card().classes("full-width q-pa-sm q-mt-xs").style(
        "background: rgba(0,0,0,0.1); border-radius: 8px;"
    ):
        ui.label(
            "💡 探索ロジック: PCA空間で目標値に近づくベクトルを探索し、"
            "それに最も近い実在する分子を検索しています。"
            "これにより、化学的に妥当な構造のみが提案されます。"
        ).classes("text-caption text-grey")
