"""
Domain Knowledge Input Component for ChemAI2
Allows users to input casual domain knowledge easily
"""

from nicegui import ui
from frontend_nicegui.utils.domain_knowledge import domain_knowledge


def show_domain_knowledge_dialog(page_name: str = "") -> None:
    """Show dialog for inputting casual domain knowledge"""

    with ui.dialog() as dialog:
        with ui.card().classes('w-full max-w-2xl p-6'):
            ui.label('ドメイン知識を保存').classes('text-xl font-bold text-white mb-4')

            # Knowledge type
            knowledge_type = ui.select(
                options={
                    'variable_property': '変数の性質',
                    'constraint': '制約',
                    'system': '系全体に関する知識',
                    'sample': 'サンプルに関する知識',
                    'other': 'その他',
                },
                label='種類',
                value='variable_property'
            ).classes('w-full mb-4')

            # Content input
            content = ui.textarea(
                label='知識内容（くだけでOK）',
                placeholder='例：温度が上がると屈折率は下がる\n例：フッ素導入は避けたい\n例：比率0.6超で相分離が始まる'
            ).classes('w-full mb-4').props('autogrow')

            # Context
            context = ui.input(
                label='文脈（任意）',
                placeholder='例：屈折率予測'
            ).classes('w-full mb-4')

            # Buttons
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('キャンセル', on_click=dialog.close).props('flat')

                def save_and_close():
                    if content.value.strip():
                        domain_knowledge.add_knowledge(
                            knowledge_type=knowledge_type.value,
                            content=content.value.strip(),
                            context=context.value,
                            page=page_name,
                        )
                        ui.notify('知識を保存しました', type='positive')
                        dialog.close()
                    else:
                        ui.notify('内容を入力してください', type='warning')

                ui.button('保存', on_click=save_and_close).props('color=primary')

    dialog.open()


def domain_knowledge_button(page_name: str = "") -> None:
    """Small button to open domain knowledge input dialog"""
    ui.button(
        icon='lightbulb',
        on_click=lambda: show_domain_knowledge_dialog(page_name)
    ).props('flat round color=amber').classes('absolute top-2 right-2').tooltip('ドメイン知識を保存')


def show_domain_knowledge_summary() -> None:
    """Show summary of saved domain knowledge"""
    items = domain_knowledge.get_all()

    if not items:
        ui.label('保存された知識はありません').classes('text-gray-400 italic')
        return

    ui.label(f'{len(items)}件の知識が保存されています').classes('text-sm text-gray-300 mb-2')

    with ui.scroll_area().classes('h-64 w-full'):
        for item in items:
            with ui.card().classes('w-full bg-gray-800 p-3 mb-2').style('border: 1px solid #374151;'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label({
                        'variable_property': '変数の性質',
                        'constraint': '制約',
                        'system': '系全体',
                        'sample': 'サンプル',
                        'other': 'その他',
                    }.get(item['type'], item['type'])).classes('text-xs text-blue-300')
                    ui.label(item['timestamp'][:19].replace('T', ' ')).classes('text-xs text-gray-500')

                ui.label(item['content']).classes('text-sm text-white mt-1')
                if item['context']:
                    ui.label(f'文脈: {item["context"]}').classes('text-xs text-gray-400')
