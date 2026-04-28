"""
backend/llm/model_registry.py

2026年4月時点のOllamaモデルレジストリ。
ハードウェアスペックに応じた最適モデルを選択する。
定期的に更新する（月1回程度）。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

REGISTRY_VERSION = "2026-04"


@dataclass
class OllamaModelInfo:
    """Ollamaで使用するモデルの情報"""
    ollama_name: str          # ollama pull <name> で使う名前
    display_name: str
    size_gb: float            # ディスク容量
    vram_required_gb: float   # GPU推論時の最低VRAM
    ram_required_gb: float    # CPU推論時の最低RAM
    quality_score: int        # 1-10 (主観的な品質スコア)
    speed_score: int          # 1-10 (トークン生成速度)
    japanese_support: bool    # 日本語対応
    reasoning: bool           # 推論特化
    context_length: int       # コンテキスト長
    description: str
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2026年4月時点のモデルカタログ
# Ollama公式: https://ollama.com/library
# ---------------------------------------------------------------------------
OLLAMA_MODELS: List[OllamaModelInfo] = [

    # ── 超軽量 (CPU / 低スペック GPU) ──────────────────────────────────────
    OllamaModelInfo(
        ollama_name="qwen3:0.6b",
        display_name="Qwen3 0.6B",
        size_gb=0.4,
        vram_required_gb=1.0,
        ram_required_gb=4.0,
        quality_score=4,
        speed_score=10,
        japanese_support=True,
        reasoning=False,
        context_length=8192,
        description="超高速・超軽量。単純な質問応答に適す。",
        tags=["cpu", "ultralight"],
    ),
    OllamaModelInfo(
        ollama_name="qwen3:1.7b",
        display_name="Qwen3 1.7B",
        size_gb=1.0,
        vram_required_gb=2.0,
        ram_required_gb=6.0,
        quality_score=5,
        speed_score=9,
        japanese_support=True,
        reasoning=False,
        context_length=8192,
        description="軽量でも日本語品質が高い。RAM8GBのCPU環境推奨。",
        tags=["cpu", "light"],
    ),
    OllamaModelInfo(
        ollama_name="gemma3:1b",
        display_name="Gemma 3 1B",
        size_gb=0.8,
        vram_required_gb=1.5,
        ram_required_gb=4.0,
        quality_score=4,
        speed_score=10,
        japanese_support=False,
        reasoning=False,
        context_length=4096,
        description="Google製超軽量モデル。英語専用ユースケース向け。",
        tags=["cpu", "ultralight", "en"],
    ),

    # ── 軽量 (4-6GB VRAM / RAM16GB CPU) ───────────────────────────────────
    OllamaModelInfo(
        ollama_name="qwen3:4b",
        display_name="Qwen3 4B",
        size_gb=2.5,
        vram_required_gb=4.0,
        ram_required_gb=10.0,
        quality_score=7,
        speed_score=8,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="MI解析に十分な品質。4GB VRAM / RAM16GB環境の第一選択。",
        tags=["recommended", "light", "reasoning"],
    ),
    OllamaModelInfo(
        ollama_name="phi4-mini",
        display_name="Phi-4 Mini",
        size_gb=2.5,
        vram_required_gb=4.0,
        ram_required_gb=10.0,
        quality_score=7,
        speed_score=8,
        japanese_support=True,
        reasoning=True,
        context_length=16384,
        description="Microsoft製。コード生成と推論が得意。",
        tags=["light", "reasoning", "code"],
    ),
    OllamaModelInfo(
        ollama_name="gemma3:4b",
        display_name="Gemma 3 4B",
        size_gb=3.0,
        vram_required_gb=4.0,
        ram_required_gb=10.0,
        quality_score=7,
        speed_score=8,
        japanese_support=True,
        reasoning=False,
        context_length=8192,
        description="Google製バランス型。マルチリンガル対応。",
        tags=["light"],
    ),
    OllamaModelInfo(
        ollama_name="llama3.2:3b",
        display_name="Llama 3.2 3B",
        size_gb=2.0,
        vram_required_gb=3.0,
        ram_required_gb=8.0,
        quality_score=6,
        speed_score=9,
        japanese_support=False,
        reasoning=False,
        context_length=8192,
        description="Meta製。英語に強い。日本語は限定的。",
        tags=["light", "en"],
    ),

    # ── 中量級 (6-10GB VRAM) ──────────────────────────────────────────────
    OllamaModelInfo(
        ollama_name="qwen3:8b",
        display_name="Qwen3 8B",
        size_gb=5.2,
        vram_required_gb=6.0,
        ram_required_gb=14.0,
        quality_score=8,
        speed_score=7,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="MI解析の主力モデル。6GB VRAM以上で高品質な日本語MI解析。",
        tags=["recommended", "balanced", "reasoning"],
    ),
    OllamaModelInfo(
        ollama_name="deepseek-r1:7b",
        display_name="DeepSeek-R1 7B",
        size_gb=4.5,
        vram_required_gb=6.0,
        ram_required_gb=12.0,
        quality_score=8,
        speed_score=6,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="推論特化。数値解析・統計解釈が非常に得意。MI解析に最適。",
        tags=["recommended", "reasoning", "math"],
    ),
    OllamaModelInfo(
        ollama_name="llama3.1:8b",
        display_name="Llama 3.1 8B",
        size_gb=4.7,
        vram_required_gb=6.0,
        ram_required_gb=12.0,
        quality_score=7,
        speed_score=7,
        japanese_support=False,
        reasoning=False,
        context_length=16384,
        description="Meta製汎用モデル。英語での解析に適す。",
        tags=["balanced", "en"],
    ),
    OllamaModelInfo(
        ollama_name="phi4",
        display_name="Phi-4 14B",
        size_gb=9.0,
        vram_required_gb=8.0,
        ram_required_gb=18.0,
        quality_score=9,
        speed_score=5,
        japanese_support=True,
        reasoning=True,
        context_length=16384,
        description="Microsoft製。サイズ以上の品質。数値推論・コード生成が優秀。",
        tags=["recommended", "reasoning", "code"],
    ),
    OllamaModelInfo(
        ollama_name="mistral:7b",
        display_name="Mistral 7B",
        size_gb=4.1,
        vram_required_gb=6.0,
        ram_required_gb=12.0,
        quality_score=7,
        speed_score=8,
        japanese_support=False,
        reasoning=False,
        context_length=8192,
        description="高速でバランスが取れた汎用モデル。英語専用。",
        tags=["balanced", "fast", "en"],
    ),

    # ── 中高級 (10-16GB VRAM) ─────────────────────────────────────────────
    OllamaModelInfo(
        ollama_name="qwen3:14b",
        display_name="Qwen3 14B",
        size_gb=9.0,
        vram_required_gb=10.0,
        ram_required_gb=22.0,
        quality_score=9,
        speed_score=5,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="高精度な日本語MI解析。12GB VRAM環境向け最上位。",
        tags=["high-quality", "reasoning"],
    ),
    OllamaModelInfo(
        ollama_name="gemma3:12b",
        display_name="Gemma 3 12B",
        size_gb=8.0,
        vram_required_gb=10.0,
        ram_required_gb=20.0,
        quality_score=9,
        speed_score=5,
        japanese_support=True,
        reasoning=True,
        context_length=16384,
        description="Google製高精度モデル。マルチリンガル性能が高い。",
        tags=["high-quality"],
    ),
    OllamaModelInfo(
        ollama_name="deepseek-r1:14b",
        display_name="DeepSeek-R1 14B",
        size_gb=9.0,
        vram_required_gb=10.0,
        ram_required_gb=22.0,
        quality_score=9,
        speed_score=4,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="推論特化の最高峰。複雑なMI戦略立案に最適。",
        tags=["high-quality", "reasoning", "math"],
    ),

    # ── 高性能 (16-24GB VRAM) ─────────────────────────────────────────────
    OllamaModelInfo(
        ollama_name="qwen3:30b-a3b",
        display_name="Qwen3 30B-A3B (MoE)",
        size_gb=17.0,
        vram_required_gb=16.0,
        ram_required_gb=36.0,
        quality_score=10,
        speed_score=6,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="MoEアーキテクチャで30Bクラスの品質を16GB VRAMで実現。",
        tags=["top-tier", "reasoning", "moe"],
    ),
    OllamaModelInfo(
        ollama_name="deepseek-r1:32b",
        display_name="DeepSeek-R1 32B",
        size_gb=20.0,
        vram_required_gb=20.0,
        ram_required_gb=44.0,
        quality_score=10,
        speed_score=3,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="ローカル最高峰の推論性能。24GB VRAM推奨。",
        tags=["top-tier", "reasoning"],
    ),
    OllamaModelInfo(
        ollama_name="mistral-small:22b",
        display_name="Mistral Small 22B",
        size_gb=13.0,
        vram_required_gb=14.0,
        ram_required_gb=28.0,
        quality_score=9,
        speed_score=4,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="Mistral製高精度モデル。バランスが優秀。",
        tags=["high-quality"],
    ),

    # ── 最高級 (24GB+ VRAM) ───────────────────────────────────────────────
    OllamaModelInfo(
        ollama_name="qwen3:32b",
        display_name="Qwen3 32B",
        size_gb=20.0,
        vram_required_gb=22.0,
        ram_required_gb=48.0,
        quality_score=10,
        speed_score=3,
        japanese_support=True,
        reasoning=True,
        context_length=32768,
        description="日本語MI解析の最高峰。24GB VRAM環境で最高品質。",
        tags=["top-tier", "reasoning"],
    ),
    OllamaModelInfo(
        ollama_name="llama3.3:70b",
        display_name="Llama 3.3 70B",
        size_gb=43.0,
        vram_required_gb=40.0,
        ram_required_gb=80.0,
        quality_score=10,
        speed_score=2,
        japanese_support=False,
        reasoning=True,
        context_length=32768,
        description="Meta製最大モデル。英語では最高品質。複数GPU推奨。",
        tags=["top-tier", "en"],
    ),
]


def get_recommended_models(
    vram_gb: float,
    ram_gb: float,
    is_apple_silicon: bool = False,
    require_japanese: bool = True,
) -> List[OllamaModelInfo]:
    """
    ハードウェアスペックから推奨モデルリストを返す（スコア降順）。
    Apple Siliconは統合メモリのため vram=ram として計算。
    """
    if is_apple_silicon:
        # Unified memory: VRAMとしてRAMの80%まで使用可能
        effective_vram = ram_gb * 0.80
    else:
        effective_vram = vram_gb

    candidates = []
    for m in OLLAMA_MODELS:
        # CPUモードの場合
        if effective_vram < 1.0:
            if m.ram_required_gb <= ram_gb * 0.7:
                if not require_japanese or m.japanese_support:
                    candidates.append(m)
        else:
            if m.vram_required_gb <= effective_vram * 0.9:
                if not require_japanese or m.japanese_support:
                    candidates.append(m)

    # quality_score 降順、同スコアはspeed_score降順
    candidates.sort(key=lambda x: (x.quality_score, x.speed_score), reverse=True)
    return candidates[:5]  # 上位5件


def get_model_by_name(ollama_name: str) -> Optional[OllamaModelInfo]:
    """Ollamaモデル名でモデル情報を取得"""
    for m in OLLAMA_MODELS:
        if m.ollama_name == ollama_name:
            return m
    return None


def get_tier_label(model: OllamaModelInfo) -> str:
    """モデルのティアラベルを返す"""
    if model.quality_score >= 9:
        return "高品質"
    elif model.quality_score >= 7:
        return "バランス"
    else:
        return "軽量"
