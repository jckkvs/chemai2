"""F-11/F-03/F-14: main.pyにキーバインド + 色覚対応CSS + アイコンCSS追加"""

fp = 'C:/Users/horie/chemai2/frontend_nicegui/main.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. F-11: Ctrl+Enterキーバインドを解析ボタンの後に追加
old1 = '''    # ═════════════════════════════════════════════════
    # スマートデフォルト（データ特性に基づく自動設定）
    # ═════════════════════════════════════════════════'''

new1 = '''    # F-11: キーボードショートカット登録
    ui.keyboard(
        on_key=lambda e: (
            _run_analysis() if e.key == 'Enter' and e.modifiers.ctrl and not e.action.repeat else
            main_tabs.set_value('data') if e.key == '1' and e.modifiers.ctrl else
            main_tabs.set_value('results') if e.key == '2' and e.modifiers.ctrl else
            main_tabs.set_value('inverse') if e.key == '3' and e.modifiers.ctrl else
            None
        ),
    )

    # ═════════════════════════════════════════════════
    # スマートデフォルト（データ特性に基づく自動設定）
    # ═════════════════════════════════════════════════'''

if old1 in content:
    content = content.replace(old1, new1, 1)
    changes += 1
else:
    print("WARNING: old1 not found")

# 2. F-03: WCAG色覚対応CSSとF-14: アイコン統一CSSをCUSTOM_CSSの末尾に追加
old2 = '''@keyframes status-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
"""'''

new2 = '''@keyframes status-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* F-03: WCAG 2.1 色覚対応 — コントラスト比4.5:1以上 */
/* 色覚多様性(CVD)対応: 赤/緑の区別に依存しないよう形状・アイコンで補完 */
.color-safe-success { color: var(--status-success); }
.color-safe-success::before { content: "✅ "; }
.color-safe-warning { color: var(--status-warning); }
.color-safe-warning::before { content: "⚠️ "; }
.color-safe-error { color: var(--status-error); }
.color-safe-error::before { content: "❌ "; }
.color-safe-info { color: var(--status-info); }
.color-safe-info::before { content: "ℹ️ "; }

/* F-03: 高コントラストモード（OSの設定連携） */
@media (prefers-contrast: high) {
    :root {
        --bg-card: rgba(255, 255, 255, 0.12);
        --border: rgba(255, 255, 255, 0.3);
        --text-primary: #ffffff;
    }
    .glass-card { border-width: 2px !important; }
}

/* F-14: Material Icon統一ルール */
/* アイコンサイズの一貫性: ボタン=20px, ラベル=16px, タイトル=24px */
.q-btn .q-icon { font-size: 20px !important; }
.text-caption .q-icon { font-size: 16px !important; }
.text-h5 .q-icon, .text-h6 .q-icon { font-size: 24px !important; }
"""'''

if old2 in content:
    content = content.replace(old2, new2, 1)
    changes += 1
else:
    print("WARNING: old2 not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} replacements")
