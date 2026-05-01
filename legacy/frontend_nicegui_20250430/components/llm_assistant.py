# frontend_nicegui/components/llm_assistant.py

"""
NiceGUI component for LLM Assistant (Chat & Analysis)

Provides a chat interface to interact with the model and request
automated chemical analysis reports.
"""
from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional
import httpx

from nicegui import ui

API_BASE = "http://localhost:8000"


class LLMAssistantUI:
    """Chat-based assistant for chemical analysis"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.is_loading = False
    
    def render(self):
        """Render the assistant UI"""
        with ui.column().classes('w-full h-full max-w-4xl mx-auto q-pa-md gap-4'):
            # Header
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('🤖 ChemAI Assistant').classes('text-2xl font-bold hero-gradient')
                with ui.row().classes('gap-2'):
                    self.status_chip = ui.chip('Uninitialized', color='gray').props('outline')
                    ui.button('⚙️ Initialize', on_click=self._initialize).props('dense outline')
            
            # Chat Area
            with ui.scroll_area().classes('flex-1 w-full border rounded-lg bg-gray-50 q-pa-sm') as self.chat_scroll:
                self.chat_container = ui.column().classes('w-full gap-2')
            
            # Input Area
            with ui.row().classes('w-full items-end gap-2'):
                self.input_field = ui.textarea(
                    placeholder='Ask about chemical properties, analysis results, or model selection...',
                ).classes('flex-1').props('autogrow outlined')
                
                with ui.column().classes('gap-1'):
                    ui.button(icon='send', on_click=self._send_message).props('round color=primary')
                    ui.button(icon='delete_sweep', on_click=self._clear_chat).props('round flat color=grey')
        
        # Initial status check
        ui.timer(1.0, self._update_status, once=True)
    
    async def _update_status(self):
        """Check LLM status from backend"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{API_BASE}/api/v1/llm/status")
                if resp.status_code == 200:
                    data = resp.json()
                    state = data.get('state', 'unknown')
                    self.status_chip.text = state.capitalize()
                    
                    colors = {
                        'ready': 'green',
                        'loading_model': 'orange',
                        'error': 'red',
                        'uninitialized': 'gray'
                    }
                    self.status_chip.props(f'color={colors.get(state, "gray")}')
                else:
                    self.status_chip.text = "Offline"
                    self.status_chip.props('color=red')
        except Exception:
            self.status_chip.text = "Error"
            self.status_chip.props('color=red')
    
    async def _initialize(self):
        """Trigger LLM initialization"""
        ui.notify("Initializing LLM engine...", type='info')
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(f"{API_BASE}/api/v1/llm/initialize")
                resp.raise_for_status()
                ui.notify("LLM ready!", type='positive')
                await self._update_status()
        except Exception as e:
            ui.notify(f"Initialization failed: {e}", type='negative')
            await self._update_status()
    
    async def _send_message(self):
        """Send message to assistant and handle streaming response"""
        text = self.input_field.value.strip()
        if not text or self.is_loading:
            return
        
        self.input_field.value = ""
        self._add_message('user', text)
        
        # Placeholder for AI response
        ai_msg_container = self._add_message('assistant', "...")
        ai_text = ""
        
        self.is_loading = True
        try:
            # Note: Real implementation should use WebSockets or Server-Sent Events for streaming
            # Here we simulate or use a simpler POST if streaming endpoint isn't ready
            async with httpx.AsyncClient(timeout=60.0) as client:
                # We'll assume a /chat endpoint exists or will be added
                # For now, let's mock the streaming behavior
                resp = await client.post(
                    f"{API_BASE}/api/v1/llm/chat", # We might need to add this to router
                    json={"message": text}
                )
                
                if resp.status_code == 200:
                    # Non-streaming fallback for demonstration
                    ai_text = resp.json().get('response', "I'm sorry, I couldn't process that.")
                    ai_msg_container.set_content(ai_text)
                else:
                    ai_msg_container.set_content(f"Error: {resp.status_code}")
                    
        except Exception as e:
            ai_msg_container.set_content(f"Failed to connect: {e}")
        finally:
            self.is_loading = False
            await self._scroll_to_bottom()
    
    def _add_message(self, role: str, content: str):
        """Add message bubble to chat container"""
        with self.chat_container:
            align = 'end' if role == 'user' else 'start'
            bg = 'bg-blue-100' if role == 'user' else 'bg-white'
            
            with ui.column().classes(f'w-full items-{align}'):
                card = ui.card().classes(f'q-pa-sm {bg} max-w-[80%] shadow-sm')
                with card:
                    msg_label = ui.markdown(content)
        
        self._scroll_to_bottom_sync()
        
        # Return a helper to update content (for streaming)
        class ContentSetter:
            def set_content(self, new_content):
                msg_label.content = new_content
        
        return ContentSetter()

    def _scroll_to_bottom_sync(self):
        """Scroll to bottom of chat"""
        self.chat_scroll.scroll_to(percent=1.0)
        
    async def _scroll_to_bottom(self):
        """Async scroll to bottom"""
        await asyncio.sleep(0.1)
        self.chat_scroll.scroll_to(percent=1.0)
    
    def _clear_chat(self):
        """Clear all messages"""
        self.chat_container.clear()
        ui.notify("Chat history cleared locally", type='info')
