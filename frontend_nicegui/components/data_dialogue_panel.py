"""
frontend_nicegui/components/data_dialogue_panel.py

Data Dialogue Panel — データとの対話、自動サジェスト
"""
from nicegui import ui

def render_data_dialogue(state: dict):
    """
    データ全体のコンテキストに基づいた「対話的」なサジェストを表示する。
    """
    df = state.get("df")
    if df is None: return

    with ui.card().classes("full-width q-pa-lg glass-card animate-slide-up").style("border-left: 5px solid var(--accent-blue);"):
        with ui.row().classes("items-center q-gutter-md"):
            ui.icon("chat_bubble", color="cyan", size="md")
            ui.label("💬 Data Dialogue").classes("text-h6 text-cyan font-bold")
        
        ui.separator().classes("q-my-sm")
        
        with ui.column().classes("q-gutter-sm"):
            # データの複雑性
            ui.label("現在のデータの状況:").classes("text-caption text-grey-5")
            
            n_rows, n_cols = df.shape
            if n_rows < 100:
                ui.markdown("- **小規模データセット**: モデルの単純化または「転移学習」が有効かもしれません。").classes("text-body2")
            else:
                ui.markdown("- **十分なデータ量**: 高度な非線形モデル（GBTやDNN）の性能が期待できます。").classes("text-body2")

            # 特定のパターン
            ui.markdown("- **SMILES列を検出しました**: 記述子生成タブで「Morgan」「RDKit」セットを計算することをお勧めします。").classes("text-body2")
            
            if state.get("target_col"):
                ui.markdown(f"- **目的変数「{state['target_col']}」を設定済み**: EDAタブで「Conflict解析」を実行してデータの整合性を確認してください。").classes("text-body2")
            else:
                ui.markdown("- **目的変数が未設定です**: 予測ターゲットを選択すると、より具体的な分析アドバイスが可能になります。").classes("text-body2 text-amber")

        with ui.row().classes("q-mt-md justify-end"):
            ui.button("次のステップを提案", icon="auto_awesome").props("flat color=cyan no-caps")
