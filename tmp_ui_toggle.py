# -*- coding: utf-8 -*-
"""descriptor_plugins_ui.pyにカウント/密度トグルスイッチ追加"""

fp = 'C:/Users/horie/chemai2/frontend_nicegui/components/descriptor_plugins_ui.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# ヘッダーブロック直後にトグルを追加
old = '''    # ─────────────────────────────────────────────────────
    # セクション1: 計算状態 + 推薦記述子
    # ─────────────────────────────────────────────────────'''

new = '''    # ── 数え上げ記述子の正規化設定 ──
    with ui.row().classes("items-center q-gutter-sm full-width q-mb-sm").style(
        "background: rgba(6,182,212,0.08); border-radius: 8px; padding: 6px 12px;"
    ):
        ui.icon("tune", color="cyan").classes("text-body1")
        ui.label("カウント系記述子:").classes("text-body2")

        # デフォルト値の設定
        if "count_normalization" not in state:
            state["count_normalization"] = "density"

        norm_toggle = ui.toggle(
            {"density": "密度 (個数/分子量)", "raw": "個数 (そのまま)"},
            value=state.get("count_normalization", "density"),
        ).props("dense no-caps color=cyan size=sm").tooltip(
            "数え上げ系記述子(原子数/環数/官能基数等)を\\n"
            "分子量で割った密度に変換するか、生の個数のまま使うか。\\n"
            "密度モードは分子サイズの影響を除外し、\\n"
            "異なるサイズの分子間での公平な比較が可能。"
        )

        def _on_norm_change(e):
            state["count_normalization"] = e.value
            mode_label = "密度(個数/分子量)" if e.value == "density" else "個数(そのまま)"
            ui.notify(f"カウント系記述子: {mode_label}モード", type="info", timeout=2000)

        norm_toggle.on("update:model-value", _on_norm_change)

        ui.label(
            "fr_*, Num*, *Count 等の整数カウント記述子に適用"
        ).classes("text-caption text-grey")

    # ─────────────────────────────────────────────────────
    # セクション1: 計算状態 + 推薦記述子
    # ─────────────────────────────────────────────────────'''

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
else:
    print("WARNING: target not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes")
