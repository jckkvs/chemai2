# prototyping/agent_demo.py
# 🧪 MI 自立解析アシスタント（プロトタイプ）

import os
import sys
from pathlib import Path
from nicegui import ui

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

# Model Configuration (Adjust based on your local setup)
# The user mentioned RTX 5080 (16GB VRAM)
MODEL_DIR = Path.home() / ".cache" / "chemai" / "llm"
# Fallback to current dir if not found
if not MODEL_DIR.exists():
    MODEL_DIR = Path("models")

# We prioritize Qwen2.5-7B for the prototype speed
MODEL_PATH = MODEL_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf"

# Global LLM instance
llm = None

def init_llm():
    global llm
    if not HAS_LLAMA:
        ui.notify("Error: llama-cpp-python is not installed.", type="negative")
        return False
    
    if not MODEL_PATH.exists():
        ui.notify(f"Error: Model not found at {MODEL_PATH}", type="negative")
        return False
    
    try:
        ui.notify("LLMを初期化中... (RTX 5080 最適設定)", type="info")
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_gpu_layers=100,  # RTX 5080: Offload everything
            flash_attn=True,   # Enabled as requested
            verbose=False
        )
        ui.notify("LLMの初期化が完了しました。", type="positive")
        return True
    except Exception as e:
        ui.notify(f"LLM初期化エラー: {str(e)}", type="negative")
        return False

def analyze_material(query: str):
    """Intent Extraction -> Workflow Suggestion (Prototype)"""
    if llm is None:
        return "LLMが初期化されていません。モデルファイルの有無を確認してください。"
    
    prompt = f"""
    [化学ドメイン特化プロンプト]
    あなたはマテリアルズインフォマティクスの専門家です。
    以下の研究者の要望に対して、
    1) 推奨される解析ワークフロー (データ前処理, 特徴量生成, モデル選定, 評価)
    2) 必要なデータ形式 (CSV, SDF, POSCAR等)
    3) 注意すべき統計的落とし穴 (過学習, データリーク, 外れ値)
    を、専門用語を避け平易な日本語で説明してください。
    
    研究者の要望: "{query}"
    """
    
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"解析エラーが発生しました: {str(e)}"

@ui.page('/')
def main():
    ui.colors(primary='#38bdf8', secondary='#0ea5e9', accent='#f43f5e')
    
    with ui.header().classes('items-center justify-between bg-slate-900 text-white p-4'):
        ui.label('🧪 ChemAI Autonomous MI Platform').classes('text-2xl font-bold')
        ui.button('LLM初期化', on_click=init_llm, icon='refresh').props('flat color=white')

    with ui.column().classes('w-full max-w-4xl mx-auto p-8 gap-6'):
        with ui.card().classes('w-full p-6 shadow-lg bg-slate-50'):
            ui.markdown('### 🔍 何を解析したいですか？')
            ui.markdown('自然言語で指示を入力すると、AIエージェントが最適なワークフローを提案します。')
            
            with ui.row().classes('w-full items-center gap-4'):
                input_box = ui.input(
                    label='解析リクエスト',
                    placeholder='例: ペロブスカイト太陽電池の効率予測モデルを作りたい'
                ).classes('flex-grow text-lg')
                
                ui.button(
                    '解析プランを生成', 
                    on_click=lambda: update_result(input_box.value)
                ).classes('h-14 px-8 text-lg font-bold').props('rounded elevated color=primary')

        result_area = ui.card().classes('w-full p-6 hidden')
        with result_area:
            ui.markdown('### 📋 提案された解析プラン')
            result_text = ui.markdown('')
            
            with ui.row().classes('justify-end w-full mt-4'):
                ui.button('このプランで実行 (Coming Soon)', icon='play_arrow').props('outline color=secondary')

        with ui.expansion('💡 使用例 / ヒント', icon='lightbulb').classes('w-full border rounded'):
            ui.markdown('''
            - **物性予測**: "新しい触媒材料の水素吸着エネルギーを予測したい"
            - **記述子選定**: "有機薄膜太陽電池の効率に最も寄与する分子構造は？"
            - **データ品質**: "実験データに不自然な外れ値がないか統計的にチェックして"
            - **モデル解釈**: "なぜこのモデルはこの物質の導電率を高く予測したの？"
            ''')

    async def update_result(query):
        if not query:
            ui.notify("リクエストを入力してください", type="warning")
            return
        
        result_area.set_visibility(True)
        result_text.set_content("⏳ 解析プランを生成中...")
        
        # Run in executor to avoid blocking UI
        import asyncio
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, analyze_material, query)
        
        result_text.set_content(content)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='MI Assistant Prototype', port=8085, dark=True)
