# -*- coding: utf-8 -*-
"""
frontend_nicegui/pages/mixture_template_page.py

混合物設定およびテンプレート管理の専用ページ。
"""
from __future__ import annotations
import logging
from typing import Any
from nicegui import ui

logger = logging.getLogger(__name__)

def render_mixture_template_page(state: dict[str, Any]) -> None:
    """混合物・テンプレート設定ページの内容。"""
    
    with ui.column().classes('w-full q-pa-lg max-w-5xl mx-auto'):
        ui.markdown("""
        ## 🧪 混合物設定とデータテンプレート
        普段の解析フローとは切り離し、特殊なデータ形式や混合物計算のプリセットを管理します。
        
        このページでは以下の操作が可能です：
        - 混合物の重み付けルール（重量比/モル比など）の変更
        - 記述子計算時の結合テンプレート設定
        - よく使う列構成（テンプレート）の保存
        """).classes('text-grey-4 q-mb-md')
        
        with ui.tabs().classes('w-full glass-card') as mix_tabs:
            tab_mix = ui.tab('mixture', label='🧪 混合物設定', icon='science')
            tab_tmp = ui.tab('template', label='📄 テンプレート管理', icon='description')
        
        with ui.tab_panels(mix_tabs, value='mixture').classes('w-full bg-transparent'):
            with ui.tab_panel('mixture'):
                try:
                    from frontend_nicegui.components.mixture_input_panel import render_mixture_panel
                    render_mixture_panel(state)
                except Exception as e:
                    ui.label(f"⚠️ 混合物パネルの表示に失敗: {e}").classes("text-red")
            
            with ui.tab_panel('template'):
                try:
                    from frontend_nicegui.components.template_manager import render_template_manager
                    render_template_manager(state)
                except Exception as e:
                    ui.label(f"⚠️ テンプレート管理の表示に失敗: {e}").classes("text-red")
