"""
ChemAI Nexus - テーマ設定
ダークテーマとカラーパレットを一元管理
"""

from nicegui import ui

# カラーパレット
THEME = {
    "dark": {
        "bg_primary": "#1a1a2e",
        "bg_secondary": "#16213e",
        "bg_tertiary": "#0f3460",
        "text_primary": "#e8e8e8",
        "text_secondary": "#b8b8b8",
        "accent_primary": "#e94560",
        "accent_secondary": "#533483",
        "success": "#00d9a3",
        "warning": "#ffc107",
        "error": "#ff4757",
        "info": "#2ed573",
        "border": "rgba(255, 255, 255, 0.08)",
    }
}

def apply_dark_theme():
    """ダークテーマを適用"""
    ui.add_head_html("""
    <style>
    :root {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-tertiary: #0f3460;
        --text-primary: #e8e8e8;
        --text-secondary: #b8b8b8;
        --accent-primary: #e94560;
        --accent-secondary: #533483;
        --success: #00d9a3;
        --warning: #ffc107;
        --error: #ff4757;
        --info: #2ed573;
        --border-color: rgba(255, 255, 255, 0.08);
    }
    
    body {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .nicegui-content {
        background-color: var(--bg-primary) !important;
    }
    
    /* ヘッダー - グラデーション */
    .nicegui-header {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    /* サイドバー */
    .nicegui-drawer {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    /* タイトル - 落ち着いた白/グレーグラデーション */
    .app-title {
        background: linear-gradient(135deg, #ffffff 0%, #a8b2d1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    
    /* ボタン */
    .q-btn {
        text-transform: none !important;
        letter-spacing: 0.02em;
    }
    
    /* 主要アクションボタン */
    .btn-primary-action {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.3) !important;
    }
    
    .btn-primary-action:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4) !important;
    }
    
    /* カード */
    .q-card {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* 入力フィールド */
    .q-field__control {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
    }
    
    /* テーブル */
    .q-table {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .q-table__card {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* チャートコンテナ */
 .chart-container {
        background-color: var(--bg-secondary) !important;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--border-color);
    }
    
    /* スクロールバー */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
    
    /* 通知 */
    .q-notification {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    /* ダイアログ */
    .q-dialog__card {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 12px !important;
    }
    
    /* タブ */
    .q-tabs {
        background-color: transparent !important;
    }
    
    .q-tab--active {
        color: var(--accent-primary) !important;
    }
    
    /* 区切り線 */
    .q-separator {
        background-color: var(--border-color) !important;
    }
    
    /* アイコン */
    .q-icon {
        color: var(--text-secondary);
    }
    
    /* リンク */
    a {
        color: var(--accent-primary);
    }
    
    a:hover {
        color: var(--info);
    }
    
    /* アニメーション */
    .transition-all {
        transition: all 0.3s ease;
    }
    
    /* ホバーエフェクト */
    .hover-lift:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    </style>
    """)

def setup_theme():
    """テーマを初期設定"""
    apply_dark_theme()
    
    # NiceGUIのクワイアント設定
    ui.query('body').classes('bg-dark text-white')
