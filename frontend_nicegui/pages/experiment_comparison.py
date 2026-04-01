"""
Experiment Comparison Dashboard UI
"""
from nicegui import ui

def render_experiment_comparison(state: dict):
    with ui.card().classes("w-full q-pa-md"):
        ui.label("🔬 実験比較ダッシュボード").classes("text-lg font-bold hero-gradient q-mb-md")
        ui.label("過去の解析履歴・実験結果を選択し、モデル精度の比較やハイパーパラメータの違いを可視化します。").classes("text-caption text-grey-5 q-mb-md")

        if not state.get("automl_result"):
            ui.label("💡 比較可能なデータがありません。データの読み込み・保存を行ってください。").classes("text-info q-mb-md")

        # ダミー比較リスト
        dummy_options = {
            "exp_001": "2023-11-01 [SMILES] MolAI + XGBoost",
            "exp_002": "2023-11-02 [SMILES] Mordred + RF",
            "exp_003": "2023-11-03 [SMILES] UniPKa fine-tuning"
        }
        
        ui.select(
            options=dummy_options,
            label="比較対象実験を選択（複数可）",
            multiple=True,
            value=["exp_001", "exp_002"]
        ).props("outlined use-chips dense").classes("w-full q-mb-md")

        viz_type = ui.radio(
            options={"bar": "📊 棒グラフ", "radar": "🕸️ レーダーチャート", "table": "📋 表形式"},
            label="表示形式",
            value="bar"
        ).props("inline").classes("q-mb-md")

        def on_compare():
            ui.notify("🔄 ダッシュボードの実装はバックエンドDB連携待ちです。", type="info")

        ui.button("🔄 比較実行", on_click=on_compare).classes("btn-primary w-full").props("size=md icon=dashboard")
