from nicegui import ui, run
import plotly.express as px
from backend.data.dim_reduction import compute_dim_reduction_and_importance

@ui.refreshable
def render_dim_reduction_panel(state: dict):
    """次元削減＆重要度タブの描画関数"""
    df = state.get("df")
    if df is None:
        ui.label("⚠️ データが読み込まれていません。先にCSV/Excelをアップロードしてください。").classes("text-warning bg-warning/10 p-2 rounded")
        return

    # 未計算または再計算フラグが立っている場合は計算実行
    if state.get("dim_red_computing", False):
        with ui.column().classes("w-full items-center p-4 gap-2"):
            ui.spinner(color="primary", size="lg")
            ui.label("次元削減計算中...（データ規模により数十秒かかる場合があります）").classes("text-grey-6")
        return

    results = state.get("dim_red_results")
    if results is None:
        # 初回アクセス時に非同期計算トリガー
        state["dim_red_computing"] = True
        
        async def _run():
            try:
                res = await run.io_bound(compute_dim_reduction_and_importance, df)
                state["dim_red_results"] = res
            finally:
                state["dim_red_computing"] = False
                render_dim_reduction_panel.refresh(state)

        ui.timer(0.05, _run, once=True)
        render_dim_reduction_panel.refresh(state)
        return

    # 計算結果のステータス分岐
    if results.get("status") == "skip":
        ui.label(f"ℹ️ {results.get('message')}").classes("text-info bg-info/10 p-2 rounded")
        return
    if results.get("status") == "error":
        ui.label(f"❌ {results.get('message')}").classes("text-negative bg-negative/10 p-2 rounded")
        return

    # 成功時の描画
    with ui.tabs().classes("w-full") as dim_tabs:
        ui.tab("PCA 散布図")
        ui.tab("t-SNE 散布図")
        ui.tab("特徴量重要度 (PCA)")
        ui.tab("特徴量重要度 (t-SNE)")

    with ui.tab_panels(dim_tabs, value="PCA 散布図").classes("w-full"):
        with ui.tab_panel("PCA 散布図"):
            ev = results["explained_var"]
            fig = px.scatter(results["pca_coords"], x="PC1", y="PC2",
                             title=f"PCA 2次元投影 (PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%})",
                             labels={"PC1": f"PC1 ({ev[0]:.1%})", "PC2": f"PC2 ({ev[1]:.1%})"})
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            ui.plotly(fig).classes("w-full")

        with ui.tab_panel("t-SNE 散布図"):
            fig = px.scatter(results["tsne_coords"], x="t-SNE1", y="t-SNE2",
                             title="t-SNE 非線形埋め込み")
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            ui.plotly(fig).classes("w-full")

        with ui.tab_panel("特徴量重要度 (PCA)"):
            imp = results["pca_importance"].sort_values("PC1", ascending=False).head(15)
            fig = px.bar(imp, x=imp.index, y="PC1", title="PC1への寄与度 上位15特徴量")
            fig.update_layout(xaxis_tickangle=-45, height=300)
            ui.plotly(fig).classes("w-full")

        with ui.tab_panel("特徴量重要度 (t-SNE)"):
            imp = results["tsne_importance"].sort_values("t-SNE1", ascending=False).head(15)
            fig = px.bar(imp, x=imp.index, y="t-SNE1", title="t-SNE1との相関 上位15特徴量")
            fig.update_layout(xaxis_tickangle=-45, height=300)
            ui.plotly(fig).classes("w-full")
