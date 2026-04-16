# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/template_manager.py

解析プロジェクト用のテンプレート（CSVヘッダー形式、列役割のデフォルト等）を管理するコンポーネント。
"""
from __future__ import annotations
import logging
from typing import Any
from nicegui import ui

logger = logging.getLogger(__name__)

def render_template_manager(state: dict[str, Any]) -> None:
    """テンプレート管理UIを描画する。"""
    
    with ui.column().classes("full-width gap-4"):
        ui.label("📄 解析用テンプレート管理").classes("text-xl font-bold text-cyan")
        
        with ui.card().classes("full-width glass-card q-pa-md"):
            ui.markdown("""
            **解析プロジェクトの標準化**
            CSVのヘッダー名や、どの列を「目的変数」とするかといった設定をテンプレートとして保存し、
            新しいデータを読み込む際に一括適用できます。
            """).classes("text-sm text-grey-4")

            with ui.row().classes("q-gutter-md q-mt-sm"):
                ui.button("現在の設定をテンプレート化", icon="save").props("unelevated color=indigo no-caps")
                ui.button("テンプレートを選択", icon="list").props("outline color=grey no-caps")

        # ── テンプレートリスト（仮） ──
        ui.separator().classes("q-my-md")
        ui.label("登録済みテンプレート").classes("text-subtitle2 text-grey-5")
        
        with ui.row().classes("full-width q-gutter-md"):
             with ui.card().classes("q-pa-sm bg-blue-900/10 border-blue-500/20").style("width: 200px"):
                 ui.label("標準Solubility用").classes("text-sm font-bold")
                 ui.label("SMILES, Solubility").classes("text-xs text-grey-5")
                 ui.button("適用", icon="play_arrow").props("flat dense color=cyan size=sm")
             
             with ui.card().classes("q-pa-sm bg-blue-900/10 border-blue-500/20").style("width: 200px"):
                 ui.label("融点予測(MP)").classes("text-sm font-bold")
                 ui.label("SMILES, MeltingPoint").classes("text-xs text-grey-5")
                 ui.button("適用", icon="play_arrow").props("flat dense color=cyan size=sm")

        ui.label("※ 現在はサンプルテンプレートのみ表示されています。カスタム保存機能は開発中です。").classes("text-caption text-grey-7 q-mt-md")
