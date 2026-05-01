# frontend_nicegui/components/auto_params_ui.py
"""
自動パラメータUI描画ユーティリティ。
render_param_editor, render_model_param_editor, render_adapter_param_editor を提供。
"""

from typing import Any, Optional
from nicegui import ui


def render_param_editor(
    params: dict,
    on_change: Optional[callable] = None,
    key_prefix: str = "param",
) -> Any:
    """
    任意のパラメータ辞書を受け取り、適切な入力ウィジェットを自動生成する。
    戻り値: NiceGUIのUIコンテナ（ui.row() 等）
    """
    with ui.row() as container:
        for name, value in params.items():
            with ui.column():
                ui.label(text=name)
                if isinstance(value, bool):
                    ui.switch(text=name, value=value).props('color=primary')
                elif isinstance(value, (int, float)):
                    ui.number(text=name, value=float(value)).props('step=0.1')
                else:
                    ui.input(text=str(value))
    return container


def render_model_param_editor(
    model_type: str,
    current_params: Optional[dict] = None,
    on_change: Optional[callable] = None,
) -> Any:
    """
    モデルタイプ（'rf', 'svr', 'gpr' 等）に応じたパラメータUIを生成する。
    戻り値: NiceGUIのUIコンテナ
    """
    from backend.models.factory import get_model  # 遅延インポート

    try:
        cls = get_model(model_type)
        params = cls().get_params() if hasattr(cls, 'get_params') else {}
        if current_params:
            params.update(current_params)
        return render_param_editor(params, on_change=on_change, key_prefix=f"model_{model_type}")
    except Exception:
        with ui.row() as container:
            ui.label(text=f"モデル {model_type} のパラメータエディタ（準備中）")
        return container


def render_adapter_param_editor(
    adapter_name: str,
    current_params: Optional[dict] = None,
    on_change: Optional[callable] = None,
) -> Any:
    """
    アダプター名（'rdkit', 'mordred', 'molfeat' 等）に応じたパラメータUIを生成する。
    戻り値: NiceGUIのUIコンテナ
    """
    from backend.chem.descriptors.loader import get_adapter

    try:
        adapter = get_adapter(adapter_name)
        params = adapter.get_params() if hasattr(adapter, 'get_params') else {}
        if current_params:
            params.update(current_params)
        return render_param_editor(params, on_change=on_change, key_prefix=f"adapter_{adapter_name}")
    except Exception:
        with ui.row() as container:
            ui.label(text=f"アダプター {adapter_name} のパラメータエディタ（準備中）")
        return container
