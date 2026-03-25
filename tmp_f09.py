"""F-09: descriptor_selector_dialog.pyに検索ボックス+フィルタリングロジックを追加"""

fp = 'C:/Users/horie/chemai2/frontend_nicegui/components/descriptor_selector_dialog.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. 全解除ボタン後に検索ボックスを追加
old1 = '''            ui.button("全解除", on_click=_deselect_all).props(
                "outline size=sm no-caps color=grey"
            )

        ui.separator()'''

new1 = '''            ui.button("全解除", on_click=_deselect_all).props(
                "outline size=sm no-caps color=grey"
            )

            ui.space()

            # F-09: 検索ボックス
            search_input = ui.input(
                placeholder="🔍 記述子を検索...",
            ).props(
                'dense outlined clearable'
            ).style("min-width: 280px;").tooltip(
                "記述子名・カテゴリ名・日本語説明で絞り込み"
            )

            # 検索フィルタ用リスト
            _filter_rows: list[tuple] = []
            _filter_expansions: list[tuple] = []

            def _on_search(e):
                """検索テキストで記述子行の表示/非表示を切り替え"""
                query = (e.value or "").lower().strip()
                for row_el, text in _filter_rows:
                    if not query or query in text:
                        row_el.style(remove="display: none")
                    else:
                        row_el.style(add="display: none")
                for exp_el, cat_text, child_texts in _filter_expansions:
                    if not query:
                        exp_el.style(remove="display: none")
                    elif query in cat_text or any(query in t for t in child_texts):
                        exp_el.style(remove="display: none")
                        if query and query not in cat_text:
                            exp_el.value = True
                    else:
                        exp_el.style(add="display: none")

            search_input.on('update:model-value', _on_search)

        ui.separator()'''

if old1 in content:
    content = content.replace(old1, new1, 1)
    changes += 1
else:
    print("WARNING: old1 not found")
    # debug
    idx = content.find('全解除')
    if idx > 0:
        print(f"  Found '全解除' at index {idx}")
        print(f"  Context: {repr(content[idx:idx+200])}")

# 2. カテゴリ展開をexp_panelに変更
old2 = '''                with ui.expansion(
                    f"{cat_name}  ({n_cat_sel}/{len(cat_actual)})",
                    icon="folder",
                ).classes("full-width q-mb-xs"):'''

new2 = '''                exp_panel = ui.expansion(
                    f"{cat_name}  ({n_cat_sel}/{len(cat_actual)})",
                    icon="folder",
                ).classes("full-width q-mb-xs")
                # F-09: カテゴリのフィルタ登録
                cat_search_texts = [d["name"].lower() + " " + d.get("short", "").lower() for d in cat_actual]
                _filter_expansions.append((exp_panel, cat_name.lower(), cat_search_texts))
                with exp_panel:'''

if old2 in content:
    content = content.replace(old2, new2, 1)
    changes += 1
else:
    print("WARNING: old2 not found")

# 3. 各記述子行をdesc_rowに変更
old3 = '''                    # 個別記述子チェックボックス
                    for desc in cat_actual:
                        dname = desc["name"]
                        short = desc.get("short", "")
                        with ui.row().classes("items-center q-gutter-xs").style(
                            "min-height: 28px;"
                        ):'''

new3 = '''                    # 個別記述子チェックボックス
                    for desc in cat_actual:
                        dname = desc["name"]
                        short = desc.get("short", "")
                        desc_row = ui.row().classes("items-center q-gutter-xs").style(
                            "min-height: 28px;"
                        )
                        # F-09: 行のフィルタ登録
                        _filter_rows.append((desc_row, (dname + " " + short).lower()))
                        with desc_row:'''

if old3 in content:
    content = content.replace(old3, new3, 1)
    changes += 1
else:
    print("WARNING: old3 not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} replacements")
