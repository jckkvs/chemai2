"""
Report Export UI Panel
"""
from nicegui import ui

def render_export_panel(state: dict):
    with ui.card().classes("w-full q-pa-md"):
        ui.label("📤 解析レポート エクスポート").classes("text-lg font-bold hero-gradient q-mb-md")
        ui.label("この機能は、完了した解析結果を基にPDF、Word、Jupyter Notebook等の形式でレポートを生成します。").classes("text-caption text-grey-5 q-mb-lg")

        if state.get("automl_result") is None:
            ui.label("⚠️ まだ解析が完了していません。「データ設定」から解析を実行してください。").classes("text-amber q-mb-md")

        # フォーマット選択
        format_select = ui.select(
            options={
                "pdf": "📄 PDF (ReportLab/WeasyPrint)",
                "docx": "📝 Word (.docx)",
                "zip": "🖼️ 図表一括 (.zip)",
                "ipynb": "📓 Jupyter Notebook"
            },
            label="出力形式",
            value="pdf"
        ).props("outlined dense").classes("w-full q-mb-md")

        with ui.expansion("⚙️ 詳細設定", icon="settings").classes("w-full q-mb-md glass-card"):
            ui.checkbox("ソースコード（前処理・学習スクリプト）を含める", value=True)
            ui.checkbox("生データサンプル（上位5行）を含める", value=False)
            ui.select(options=["standard", "academic", "executive"], label="テンプレート", value="standard").props("outlined dense").classes("q-mt-sm")

        def on_export():
            ui.notify("🔄 レポート生成は現在モック状態です。バックエンドの実装後に有効化されます。", type="info")

        ui.button("🚀 エクスポート実行", on_click=on_export).classes("btn-primary w-full").props("size=md icon=launch")
