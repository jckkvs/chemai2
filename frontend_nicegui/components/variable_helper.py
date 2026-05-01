"""Variable explanation helper component for EDA page."""

from nicegui import ui
from frontend_nicegui.utils import app_state
import pandas as pd
import numpy as np


def get_column_stats(df: pd.DataFrame, col: str) -> dict:
    """Get basic statistics for a column."""
    if col not in df.columns:
        return {"error": f"列 '{col}' が見つかりません"}

    series = df[col]
    stats = {
        "name": col,
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "nulls": int(series.isna().sum()),
    }

    if pd.api.types.is_numeric_dtype(series):
        stats.update({
            "type": "数値変数",
            "mean": float(series.mean()) if series.count() > 0 else None,
            "std": float(series.std()) if series.count() > 0 else None,
            "min": float(series.min()) if series.count() > 0 else None,
            "max": float(series.max()) if series.count() > 0 else None,
            "median": float(series.median()) if series.count() > 0 else None,
        })
    else:
        stats.update({
            "type": "カテゴリ変数",
            "unique": int(series.nunique()),
            "top_values": series.value_counts().head(3).to_dict() if series.count() > 0 else {},
        })

    return stats


def guess_variable_meaning(col_name: str, target_col: str = None) -> str:
    """Guess what a variable might represent based on its name."""
    col_lower = col_name.lower()

    # Common patterns in materials/chemistry data
    if any(x in col_lower for x in ["refractive", "n_", "ri_", "index"]):
        return "屈折率に関連する変数と思われます"
    if any(x in col_lower for x in ["polar", "alpha", "polarizability"]):
        return "分極率に関連する変数と思われます"
    if any(x in col_lower for x in ["temp", "t_", "temperature"]):
        return "温度に関連する変数と思われます"
    if any(x in col_lower for x in ["press", "p_", "pressure"]):
        return "圧力に関連する変数と思われます"
    if any(x in col_lower for x in ["conc", "conc", "wt_", "weight"]):
        return "濃度または重量に関連する変数と思われます"
    if any(x in col_lower for x in ["feat", "x", "var"]) and any(x in col_lower for x in ["1", "2", "3", "_0", "_1"]):
        return "特徴量として自動生成された変数と思われます（物理的意味はデータによります）"
    if any(x in col_lower for x in ["smiles", "mol", "structure"]):
        return "分子構造に関連する変数と思われます"
    if target_col and col_lower == target_col.lower():
        return "これは予測対象（目標）の変数です"
    return "汎用的な数値変数と思われます。名前からは特定が困難です。"


def show_variable_dialog(col_name: str) -> None:
    """Show dialog with variable explanation."""
    if app_state.data_df is None:
        ui.notify("データが読み込まれていません", type="warning")
        return

    df = app_state.data_df
    stats = get_column_stats(df, col_name)
    guess = guess_variable_meaning(col_name, app_state.target_column)

    with ui.dialog() as dialog:
        with ui.card().classes("w-full max-w-2xl p-6").style("background-color: #1f2937; color: #f9fafb; border: 1px solid #374151;"):
            ui.label(f"🔍 変数の説明: {col_name}").classes("text-xl font-bold text-white mb-4")

            # Basic info
            ui.label(f"タイプ: {stats.get('type', '不明')}").classes("text-sm text-gray-300")
            ui.label(f"データ型: {stats.get('dtype', '不明')}").classes("text-sm text-gray-400")
            ui.label(f"有効値: {stats.get('count', 0)} / 全{len(df)}件").classes("text-sm text-gray-400")
            if stats.get("nulls", 0) > 0:
                ui.label(f"欠損値: {stats['nulls']}件").classes("text-sm text-yellow-400")

            ui.separator().classes("bg-gray-700 my-3")

            # Guess meaning
            with ui.card().classes("w-full bg-blue-900 p-3").style("border: 1px solid #1e40af;"):
                ui.label("💡 推測される意味").classes("text-sm text-blue-300 font-bold")
                ui.label(guess).classes("text-sm text-blue-100")

            # Detailed stats
            if stats.get("type") == "数値変数":
                ui.separator().classes("bg-gray-700 my-3")
                ui.label("📊 統計量").classes("text-sm text-gray-300 font-bold mb-2")
                with ui.row().classes("w-full gap-4"):
                    for label, key in [("平均", "mean"), ("中央値", "median"), ("最小値", "min"), ("最大値", "max")]:
                        if stats.get(key) is not None:
                            with ui.card().classes("flex-1 bg-gray-800 p-2"):
                                ui.label(label).classes("text-xs text-gray-400")
                                ui.label(f"{stats[key]:.4f}").classes("text-sm text-white")

            elif stats.get("type") == "カテゴリ変数":
                ui.separator().classes("bg-gray-700 my-3")
                ui.label(f"ユニーク値数: {stats.get('unique', 0)}").classes("text-sm text-gray-300")
                if stats.get("top_values"):
                    ui.label("頻出値 TOP3").classes("text-sm text-gray-300 mb-1")
                    for val, cnt in stats["top_values"].items():
                        ui.label(f"  {val}: {cnt}件").classes("text-xs text-gray-400")

            # Ask LLM button (if available)
            ui.separator().classes("bg-gray-700 my-3")
            ui.label("💬 AIに質問する").classes("text-sm text-gray-300 font-bold mb-2")
            with ui.row().classes("w-full gap-2"):
                question_input = ui.input(
                    placeholder=f"{col_name}は何の変数ですか？"
                ).classes("flex-1")

                async def ask_llm():
                    question = question_input.value or f"{col_name}は何の変数ですか？"
                    # Try to use LLM if configured
                    try:
                        from backend.llm.llm_manager import LLMManager
                        llm = LLMManager()
                        context = f"データには以下の列があります: {', '.join(df.columns.tolist())}。目標変数は{app_state.target_column}です。"
                        prompt = f"{context}\n\nユーザー質問: {question}\n\n日本語で簡潔に答えてください。"
                        answer = await llm.generate(prompt)
                        ui.notify(answer, type="info", timeout=10)
                    except Exception:
                        # Fallback: show guessed meaning
                        ui.notify(guess, type="info", timeout=5)

                ui.button("質問", icon="send", on_click=ask_llm).props("color=primary size=sm")

            # Close button
            ui.button("閉じる", on_click=dialog.close).props("flat color=gray").classes("w-full mt-4")

    dialog.open()
