"""Test result trend visualization component for NiceGUI."""

import json
from pathlib import Path
from nicegui import ui

HISTORY_FILE = Path(__file__).parent.parent.parent / "test_results_history.json"


def load_history() -> list:
    """Load test result history from JSON file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def render_test_trend() -> None:
    """Render the test result trend chart and summary."""
    history = load_history()

    if not history:
        ui.label("テスト履歴がありません。").classes("text-gray-400 text-sm")
        ui.label("scripts/record_test_results.py を実行して記録を作成してください。").classes(
            "text-gray-500 text-xs"
        )
        return

    # Summary stats
    latest = history[-1]
    if len(history) > 1:
        prev = history[-2]
        delta_pass = latest["passed"] - prev["passed"]
        delta_fail = latest["failed"] - prev["failed"]
        delta_err = latest["errors"] - prev["errors"]
    else:
        delta_pass = delta_fail = delta_err = 0

    # Summary cards
    with ui.row().classes("w-full gap-4 mb-4"):
        with ui.card().classes("flex-1 bg-gray-800 p-3"):
            ui.label("成功 (Passed)").classes("text-xs text-gray-400")
            with ui.row().classes("items-center gap-2"):
                ui.label(str(latest["passed"])).classes("text-2xl font-bold text-green-400")
                if delta_pass != 0:
                    color = "text-green-400" if delta_pass > 0 else "text-red-400"
                    arrow = "↑" if delta_pass > 0 else "↓"
                    ui.label(f"{arrow} {abs(delta_pass)}").classes(f"text-sm {color}")

        with ui.card().classes("flex-1 bg-gray-800 p-3"):
            ui.label("失敗 (Failed)").classes("text-xs text-gray-400")
            with ui.row().classes("items-center gap-2"):
                ui.label(str(latest["failed"])).classes("text-2xl font-bold text-red-400")
                if delta_fail != 0:
                    color = "text-green-400" if delta_fail < 0 else "text-red-400"
                    arrow = "↓" if delta_fail < 0 else "↑"
                    ui.label(f"{arrow} {abs(delta_fail)}").classes(f"text-sm {color}")

        with ui.card().classes("flex-1 bg-gray-800 p-3"):
            ui.label("エラー (Errors)").classes("text-xs text-gray-400")
            with ui.row().classes("items-center gap-2"):
                ui.label(str(latest["errors"])).classes("text-2xl font-bold text-yellow-400")
                if delta_err != 0:
                    color = "text-green-400" if delta_err < 0 else "text-red-400"
                    arrow = "↓" if delta_err < 0 else "↑"
                    ui.label(f"{arrow} {abs(delta_err)}").classes(f"text-sm {color}")

        with ui.card().classes("flex-1 bg-gray-800 p-3"):
            ui.label("スキップ (Skipped)").classes("text-xs text-gray-400")
            ui.label(str(latest["skipped"])).classes("text-2xl font-bold text-gray-500")

    # Trend chart using plotly
    if len(history) >= 1:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            dates = [h["timestamp"][:10] for h in history]
            passed = [h["passed"] for h in history]
            failed = [h["failed"] for h in history]
            errors = [h["errors"] for h in history]
            skipped = [h["skipped"] for h in history]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=passed,
                    mode="lines+markers",
                    name="成功 (Passed)",
                    line=dict(color="#4ade80", width=3),
                    marker=dict(size=8),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=failed,
                    mode="lines+markers",
                    name="失敗 (Failed)",
                    line=dict(color="#f87171", width=2),
                    marker=dict(size=6),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=errors,
                    mode="lines+markers",
                    name="エラー (Errors)",
                    line=dict(color="#fbbf24", width=2),
                    marker=dict(size=6),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=skipped,
                    mode="lines+markers",
                    name="スキップ (Skipped)",
                    line=dict(color="#9ca3af", width=2, dash="dot"),
                    marker=dict(size=6),
                )
            )

            fig.update_layout(
                title="テスト結果の推移",
                xaxis_title="日付",
                yaxis_title="件数",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")

            ui.plotly(fig).classes("w-full")

        except ImportError:
            # Fallback: simple table
            ui.label("Plotlyがインストールされていません。テーブル表示します。").classes(
                "text-yellow-400 text-sm mb-2"
            )
            _render_history_table(history)

    # History table
    _render_history_table(history)


def _render_history_table(history: list) -> None:
    """Render a table of test result history."""
    ui.label("履歴詳細").classes("text-sm text-gray-400 mt-4 mb-2")

    columns = [
        {"name": "date", "label": "日時", "field": "date", "align": "left"},
        {"name": "passed", "label": "成功", "field": "passed", "align": "right"},
        {"name": "failed", "label": "失敗", "field": "failed", "align": "right"},
        {"name": "errors", "label": "エラー", "field": "errors", "align": "right"},
        {"name": "skipped", "label": "スキップ", "field": "skipped", "align": "right"},
        {"name": "duration", "label": "時間(秒)", "field": "duration", "align": "right"},
    ]

    rows = []
    for h in reversed(history[-20:]):  # Show last 20 entries
        rows.append(
            {
                "date": h["timestamp"][:19].replace("T", " "),
                "passed": h["passed"],
                "failed": h["failed"],
                "errors": h["errors"],
                "skipped": h["skipped"],
                "duration": f"{h['duration_seconds']:.1f}",
            }
        )

    ui.table(columns=columns, rows=rows, row_key="date").classes("w-full")
