# frontend_nicegui/components/llm_benchmark.py

"""
NiceGUI component for running and viewing LLM benchmarks

Allows users to test models on their hardware and apply optimal settings.
"""
from __future__ import annotations

import json
import asyncio
from typing import Optional, Dict, Any
import httpx

from nicegui import ui, events

API_BASE = "http://localhost:8000"


class LLMBenchmarkUI:
    """Interactive benchmark runner for NiceGUI"""
    
    def __init__(self):
        self.is_running = False
        self.results: Dict[str, Any] = {}
    
    def render(self):
        """Render benchmark UI components"""
        with ui.card().classes('w-full'):
            ui.label('🔬 Model Benchmark').classes('text-lg font-bold mb-2')
            
            # Model selection
            with ui.row().classes('w-full items-end gap-2'):
                self.model_select = ui.select(
                    {},  # Populated on mount
                    label='Model to Test',
                    value=None
                ).classes('flex-1')
                ui.button('▶ Run Benchmark', on_click=self._run_benchmark).props('dense')
                ui.button('🔄 Refresh', on_click=self._refresh_models).props('dense outline')
            
            # Test prompt customization
            with ui.expansion('⚙️ Advanced Options', icon='settings').classes('w-full mt-2'):
                self.test_prompt = ui.textarea(
                    value='Explain the relationship between molecular weight and solubility.',
                    label='Test Prompt (optional)',
                ).classes('w-full')
                self.max_tokens = ui.number(
                    value=128, min=32, max=512, step=32,
                    label='Tokens to Generate'
                ).classes('w-32')
            
            # Results display
            self.result_area = ui.column().classes('w-full mt-4 space-y-2 hidden')
            with self.result_area:
                self.speed_label = ui.label()
                self.memory_label = ui.label()
                self.quality_label = ui.label()
                self.recommendation_area = ui.column()
                
                with ui.row():
                    ui.button('✅ Use This Model', on_click=self._apply_recommendation).props('dense')
                    ui.button('🗑️ Clear Results', on_click=self._clear_results).props('dense outline')
            
            # Cached benchmarks list
            with ui.expansion('📚 Cached Benchmarks', icon='history').classes('w-full mt-2'):
                self.cached_list = ui.column().classes('w-full space-y-1')
                ui.button('🔄 Refresh Cache', on_click=self._load_cached).props('dense')
            
            # Pre-surveyed profiles reference
            with ui.expansion('🖥️ Reference Hardware Profiles', icon='memory').classes('w-full mt-2'):
                self.profile_list = ui.column().classes('w-full space-y-1')
                ui.button('🔄 Load Profiles', on_click=self._load_profiles).props('dense')
        
        # Auto-load on mount
        ui.timer(0.1, self._on_mount, once=True)
    
    async def _on_mount(self):
        """Initialize component"""
        await self._refresh_models()
        await self._load_cached()
        await self._load_profiles()
    
    async def _refresh_models(self):
        """Populate model dropdown from API"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{API_BASE}/api/v1/llm/status")
                resp.raise_for_status()
                data = resp.json()
            
            models = data.get('available_models', {})
            self.model_select.options = {
                name: f"{info['description']} ({info['size_gb']})"
                for name, info in models.items()
            }
        except Exception as e:
            ui.notify(f"Failed to load models: {e}", type='negative')
    
    async def _run_benchmark(self):
        """Run benchmark for selected model"""
        if self.is_running:
            return
        
        model = self.model_select.value
        if not model:
            ui.notify("Please select a model first", type='warning')
            return
        
        self.is_running = True
        self.result_area.classes('hidden')
        ui.notify(f"Starting benchmark for {model}... (this may take 1-3 minutes)", type='info')
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/v1/llm/benchmarks/run",
                    params={
                        "model_name": model,
                        "test_prompt": self.test_prompt.value if self.test_prompt.value else None,
                    }
                )
                resp.raise_for_status()
                self.results = resp.json()
            
            # Display results
            self.result_area.classes('')  # Show
            self.speed_label.text = f"⚡ Speed: {self.results['speed_tps']:.1f} tokens/sec"
            self.memory_label.text = f"💾 Memory: {self.results['memory_gb']:.2f} GB peak"
            self.quality_label.text = f"🎯 Estimated Quality: {self.results.get('quality_score', 0.0):.2f}/1.0"
            
            # Show recommendation if available
            if self.results.get('recommendation'):
                rec = self.results['recommendation']
                self.recommendation_area.clear()
                with self.recommendation_area:
                    ui.label(f"🏆 Recommended: {rec['model_name']}").classes('font-bold text-green-700')
                    ui.label(f"Expected: {rec['expected_speed_tps']:.1f} t/s, {rec['expected_memory_gb']:.1f}GB")
            
            ui.notify("Benchmark completed successfully!", type='positive')
            
        except Exception as e:
            ui.notify(f"Benchmark failed: {e}", type='negative')
        finally:
            self.is_running = False
    
    async def _load_cached(self):
        """Load and display cached benchmark results"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{API_BASE}/api/v1/llm/benchmarks")
                resp.raise_for_status()
                data = resp.json()
            
            self.cached_list.clear()
            cached = data.get('cached_results', [])
            
            if not cached:
                self.cached_list.add(ui.label('No cached benchmarks yet.').classes('text-gray-500'))
            else:
                for result in cached:
                    with ui.row().classes('items-center gap-2'):
                        ui.label(f"{result['model_name']}: {result['speed_tps']:.1f} t/s").classes('text-sm')
                        ui.label(f"({result['memory_peak_gb']:.1f}GB)").classes('text-xs text-gray-500')
            
            # Show recommendation
            if data.get('recommendation'):
                rec = data['recommendation']
                with ui.card().classes('w-full bg-green-50 mt-2'):
                    ui.label(f"🏆 Best for your hardware: {rec['model_name']}").classes('font-bold')
                    ui.label(f"{rec['expected_speed_tps']:.1f} t/s • {rec['expected_memory_gb']:.1f}GB • Q:{rec['estimated_quality']:.2f}")
                    
        except Exception as e:
            self.cached_list.clear()
            self.cached_list.add(ui.label(f"Failed to load cache: {e}").classes('text-red-500'))
    
    async def _load_profiles(self):
        """Load pre-surveyed hardware profiles for reference"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{API_BASE}/api/v1/llm/hardware/profiles")
                resp.raise_for_status()
                data = resp.json()
            
            self.profile_list.clear()
            for name, profile in list(data.items())[:5]:  # Show top 5
                with ui.expansion(f"{profile['specs']['gpu'] or 'CPU'} • {profile['specs']['ram_gb']}GB RAM", 
                               icon='memory').classes('w-full'):
                    ui.label(f"Category: {profile['category']}").classes('text-sm')
                    ui.label(f"Cores: {profile['specs']['cpu_cores']} • VRAM: {profile['specs'].get('vram_gb', 'N/A')}GB").classes('text-xs')
                    ui.label(f"Recommended: {', '.join(profile['recommended_models'][:3])}").classes('text-xs text-blue-600')
                    if profile['notes']:
                        with ui.element('ul').classes('text-xs text-gray-600 list-disc list-inside'):
                            for note in profile['notes'][:2]:
                                ui.element('li').text(note)
            
        except Exception as e:
            self.profile_list.clear()
            self.profile_list.add(ui.label(f"Failed to load profiles: {e}").classes('text-red-500'))
    
    async def _apply_recommendation(self):
        """Apply benchmark-recommended model settings"""
        if not self.results.get('recommendation'):
            ui.notify("No recommendation available", type='warning')
            return
        
        rec = self.results['recommendation']
        try:
            # This would trigger model re-initialization with recommended settings
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{API_BASE}/api/v1/llm/initialize",
                    params={"preferred_model": rec['model_name']}
                )
                resp.raise_for_status()
            
            ui.notify(f"✅ Applied recommendation: {rec['model_name']}", type='positive')
            # Could trigger UI refresh here
        except Exception as e:
            ui.notify(f"Failed to apply: {e}", type='negative')
    
    def _clear_results(self):
        """Clear displayed benchmark results"""
        self.result_area.classes('hidden')
        self.results = {}
        ui.notify("Results cleared", type='info')
