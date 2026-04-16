# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/quick_eda.py

データ読込直後や役割設定中に、データの全体像を素早く把握するための「クイックEDA」パネル。
包括的な eda_panel.py の軽量版として機能。
"""
from __future__ import annotations
import logging
from typing import Any
import pandas as pd
from nicegui import ui
import plotly.express as px

logger = logging.getLogger(__name__)

def render_quick_eda(state: dict[str, Any]) -> None:
    """クイックEDAパネルを描画する。"""
    
    df = state.get("df")
    if df is None:
        ui.label("⚠️ データを読み込むとここに概要が表示されます").classes("text-amber q-pa-md")
        return

    with ui.column().classes("full-width gap-4 q-pa-md"):
        ui.label("📈 クイックEDA").classes("text-xl font-bold")
        
        # ── 1. 基本統計サマリー ──
        with ui.row().classes("full-width q-gutter-md"):
            with ui.card().classes("glass-card q-pa-md flex-1"):
                ui.label("サンプル数").classes("text-caption text-grey-5")
                ui.label(str(len(df))).classes("text-h5 font-bold")
            with ui.card().classes("glass-card q-pa-md flex-1"):
                ui.label("項目数 (列)").classes("text-caption text-grey-5")
                ui.label(str(len(df.columns))).classes("text-h5 font-bold")
            with ui.card().classes("glass-card q-pa-md flex-1"):
                missing_pct = (df.isna().sum().sum() / df.size) * 100 if df.size > 0 else 0
                ui.label("欠損率").classes("text-caption text-grey-5")
                ui.label(f"{missing_pct:.1f}%").classes("text-h5 font-bold text-amber")

        # ── 2. 数値データの分布（簡易プレビュー） ──
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            with ui.expansion("📊 数値データの基本統計量", icon="analytics").classes("full-width glass-card").props("default-opened"):
                desc = df[num_cols].describe().transpose().reset_index()
                desc.columns = ["列名", "数", "平均", "標準偏差", "最小", "25%", "50%", "75%", "最大"]
                ui.table(
                    columns=[{"name": col, "label": col, "field": col, "align": "left"} for col in desc.columns],
                    rows=desc.to_dict("records"),
                ).classes("full-width text-xs").props("dense flat bordered")

        # ── 3. 相関ヒートマップ（上位10項目など簡易版） ──
        if len(num_cols) >= 2:
            with ui.expansion("🔗 相関ヒートマップ (Top 15)", icon="grid_view").classes("full-width glass-card"):
                # 列が多すぎる場合は制限
                corr_cols = num_cols[:15]
                corr = df[corr_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="ピアソン相関係数"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0f0", size=10),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                ui.plotly(fig).classes("full-width").style("height: 400px")
        
        # ── 4. 詳細EDAへの誘導 ──
        ui.separator().classes("q-my-md")
        with ui.row().classes("full-width items-center justify-between bg-blue-900/10 p-4 border-blue-500/20 rounded-lg"):
            with ui.column():
                ui.label("もっと詳しく分析しますか？").classes("text-subtitle2 font-bold text-blue-3")
                ui.label("分布詳細、多重共線性(VIF)、SHAP重要度、外れ値などは「EDA・可視化」タブで確認できます。").classes("text-caption text-grey-4")
            ui.button("詳細展示へ", on_click=lambda: ui.notify("上の「EDA・可視化」タブをクリックしてください")).props("unelevated color=indigo-6 no-caps")
