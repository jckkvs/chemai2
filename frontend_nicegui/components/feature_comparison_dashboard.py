"""
frontend_nicegui/components/feature_comparison_dashboard.py

解析結果セット（AutoMLの結果群）間の特徴量構成を比較・分析するダッシュボード。
- 特徴量重複率（Jaccard係数）のヒートマップ
- 具体的な記述子構成（RDKit, MolAI 等）の集計
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from nicegui import ui
import plotly.graph_objects as go

from frontend_nicegui.utils.feature_classifier import FeatureClassifier

def render_feature_comparison_dashboard(state: Dict[str, Any]) -> None:
    """共通の比較ダッシュボードを描画する"""
    
    all_results = state.get("automl_results", {})
    if not all_results:
        ui.label("比較対象の解析結果がありません").classes("text-grey-5 q-pa-md")
        return

    # 成功した結果セットのみ（AutoMLResult オブジェクト）
    results = {k: v for k, v in all_results.items() if v is not None}
    if len(results) < 1:
        ui.label("成功した解析結果がありません").classes("text-grey-5 q-pa-md")
        return

    set_names = list(results.keys())

    # --- 1. データの準備 (特徴量セットの抽出) ---
    feature_sets: Dict[str, Set[str]] = {}
    for sn, ar in results.items():
        proc_X = getattr(ar, "processed_X", None)
        if proc_X is not None and hasattr(proc_X, "columns"):
            feature_sets[sn] = set(proc_X.columns)
        else:
            feature_sets[sn] = set()

    # --- 2. 🔗 ペアワイズ重複率ヒートマップ ---
    if len(set_names) >= 2:
        ui.label("🔗 特徴量セット重複率 (Jaccard Overlap %)").classes("text-subtitle2 q-mt-md q-mb-xs")
        ui.label("セット間でどれだけ同じ記述子が共有されているかを示します。100%に近いほど構成が似ています。").classes("text-caption text-grey-6 q-mb-sm")
        
        jaccard_matrix = []
        for sn1 in set_names:
            row = []
            set1 = feature_sets[sn1]
            for sn2 in set_names:
                set2 = feature_sets[sn2]
                if not set1 or not set2:
                    overlap = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap = (intersection / union) * 100 if union > 0 else 0.0
                row.append(overlap)
            jaccard_matrix.append(row)

        fig_overlap = go.Figure(go.Heatmap(
            z=jaccard_matrix,
            x=[sn[:20] for sn in set_names],
            y=[sn[:20] for sn in set_names],
            colorscale="Viridis",
            zmin=0, zmax=100,
            text=[[f"{v:.1f}%" for v in row] for row in jaccard_matrix],
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="Overlap %"),
        ))
        fig_overlap.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        ui.plotly(fig_overlap).classes("full-width")

    # --- 3. 🧪 具体的記述子構成エンジン別内訳 ---
    ui.separator().classes("q-my-md")
    ui.label("🧪 記述子ソース内訳").classes("text-subtitle2 q-mb-sm")
    
    analysis_rows = []
    for sn, f_set in feature_sets.items():
        if not f_set: continue
        
        # 特徴量をソース別に分類
        grouped = FeatureClassifier.group_features_by_set(list(f_set))
        
        # 集計
        counts = {
            "RDKit": 0, "MolAI": 0, "Mordred": 0, "xTB": 0, "Pipeline": 0, "Other": 0
        }
        for g_name, g_info in grouped.items():
            eng = g_info.get("engine")
            f_count = len(g_info.get("features", []))
            
            if eng == "rdkit": counts["RDKit"] += f_count
            elif eng == "molai": counts["MolAI"] += f_count
            elif eng == "mordred": counts["Mordred"] += f_count
            elif eng == "gfn2_xtb": counts["xTB"] += f_count
            elif eng == "pipeline": counts["Pipeline"] += f_count
            else: counts["Other"] += f_count
            
        row = {"セット名": sn, "総数": len(f_set)}
        row.update(counts)
        analysis_rows.append(row)

    cols = [
        {"name": "セット名", "label": "解析セット", "field": "セット名", "align": "left", "sortable": True},
        {"name": "総数", "label": "総記述子", "field": "総数", "sortable": True},
        {"name": "RDKit", "label": "RDKit", "field": "RDKit", "sortable": True},
        {"name": "MolAI", "label": "MolAI", "field": "MolAI", "sortable": True},
        {"name": "xTB", "label": "xTB", "field": "xTB", "sortable": True},
        {"name": "Pipeline", "label": "加工", "field": "Pipeline", "sortable": True},
        {"name": "Other", "label": "その他", "field": "Other", "sortable": True},
    ]
    
    ui.table(columns=cols, rows=analysis_rows).classes("full-width").props("dense flat bordered dark")

    # --- 4. 重要 / 共通記述子のサンプリング ---
    with ui.row().classes("full-width q-gutter-md q-mt-sm"):
        # 全セット共通のトップ記述子 (もしあれば)
        common_features = set.intersection(*feature_sets.values()) if feature_sets else set()
        if common_features:
            with ui.card().classes("col glass-card q-pa-sm"):
                ui.label(f"共通記述子 ({len(common_features)}件)").classes("text-caption text-cyan text-bold")
                display_str = ", ".join(list(common_features)[:15]) + ("..." if len(common_features) > 15 else "")
                ui.label(display_str).classes("text-xs text-grey-5")
        
        # いずれかのセットだけに含まれる記述子の数
        for sn, f_set in feature_sets.items():
            others = set().union(*(s for name, s in feature_sets.items() if name != sn))
            unique = f_set - others
            if unique:
                with ui.card().classes("col glass-card q-pa-sm"):
                    ui.label(f"固有 ({sn[:10]}): {len(unique)}件").classes("text-caption text-grey-4")
                    ui.label(", ".join(list(unique)[:5]) + "...").classes("text-xs text-grey-6")
