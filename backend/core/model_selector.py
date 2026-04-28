"""
backend/core/model_selector.py
検出ハードウェアに基づき最適LLMを自動選定
既存機能と共存：既存のモデル設定は維持し、拡張として実装
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .hardware_detector import HardwareSpec

logger = logging.getLogger(__name__)


@dataclass
class ModelRecommendation:
    """推奨モデル情報"""
    model_id: str
    model_name: str
    quantization: str
    expected_tps: float
    context_max: int
    vram_needed_gb: float
    ram_needed_gb: float
    confidence: float  # 0.0-1.0
    notes: List[str]
    download_url: Optional[str] = None
    is_moe: bool = False


class ModelSelector:
    """
    ハードウェア仕様から最適LLMを自動選定するクラス
    """
    
    def __init__(self, dict_path: str = 'backend/config/llm_hardware_dict.json'):
        self.dict_path = Path(dict_path)
        self.dict_data = self._load_dict()
    
    def _load_dict(self) -> Dict:
        """辞書ファイルを読み込み"""
        if not self.dict_path.exists():
            logger.warning(f"辞書ファイルが見つかりません: {self.dict_path}")
            return {"hardware_profiles": [], "model_catalog": {}}
        
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def select_best_model(self, hardware: HardwareSpec,
                         task_hint: Optional[str] = None,
                         quality_priority: bool = False) -> ModelRecommendation:
        """
        ハードウェアから最適モデルを選定
        
        Args:
            hardware: 検出されたハードウェア仕様
            task_hint: 分析タスクのヒント（'property_prediction', 'smiles_parsing'等）
            quality_priority: 品質優先モード（速度より精度を重視）
        
        Returns:
            ModelRecommendation: 推奨モデル情報
        """
        # 1. 適合するハードウェアプロファイルを検索
        matching_profiles = self._find_matching_profiles(hardware)
        
        if not matching_profiles:
            # 適合プロファイルがない場合、CPUフォールバック
            return self._get_fallback_recommendation(hardware)
        
        # 2. 推奨モデルを取得
        recommended_model_id = matching_profiles[0]['recommended_model']
        
        # 3. モデルカタログから詳細情報を取得
        model_info = self._get_model_info(recommended_model_id)
        if not model_info:
            return self._get_fallback_recommendation(hardware)
        
        # 4. 互換モデルリストから詳細を取得
        compatible = next(
            (m for m in matching_profiles[0]['compatible_models'] 
             if f"{m['model']}-{m['quantization']}" == recommended_model_id),
            None
        )
        if not compatible:
            return self._get_fallback_recommendation(hardware)
        
        # 5. 信頼度スコア計算
        confidence = self._calculate_confidence(hardware, compatible, quality_priority)
        
        # 6. 注記の追加
        notes = [compatible.get('notes', '')]
        if hardware.inference_tier == 'cpu_only':
            notes.append("CPU推論のため速度が低速です。可能であればGPU環境を推奨")
        if compatible.get('vram_needed_gb', 0) > hardware.ram_available_gb:
            notes.append("RAM不足の可能性があります。他のアプリを閉じてください")
        
        return ModelRecommendation(
            model_id=recommended_model_id,
            model_name=compatible['model'],
            quantization=compatible['quantization'],
            expected_tps=compatible['expected_tps'],
            context_max=compatible['context_max'],
            vram_needed_gb=compatible['vram_needed_gb'],
            ram_needed_gb=compatible['ram_needed_gb'],
            confidence=confidence,
            notes=[n for n in notes if n],
            download_url=self._get_download_url(model_info),
            is_moe=model_info.get('architecture') == 'moe'
        )
    
    def _find_matching_profiles(self, hardware: HardwareSpec) -> List[Dict]:
        """ハードウェアに適合するプロファイルを検索"""
        matching = []
        
        for profile in self.dict_data.get('hardware_profiles', []):
            req = profile.get('requirements', {})
            
            # GPU要件チェック
            gpu_req = req.get('gpu')
            if gpu_req:
                if not hardware.gpus:
                    continue  # GPU必須だが検出されず
                
                # ベンダーチェック
                if gpu_req.get('vendor') and not any(
                    g.get('vendor') == gpu_req['vendor'] for g in hardware.gpus
                ):
                    continue
                
                # VRAMチェック
                vram_min = gpu_req.get('vram_min_gb', 0)
                if not any(g.get('vram_total_gb', 0) >= vram_min for g in hardware.gpus):
                    continue
                
                # Compute Capabilityチェック
                cc_min = gpu_req.get('min_cc')
                if cc_min and not any(
                    g.get('compute_capability', '0.0') >= cc_min 
                    for g in hardware.gpus if g.get('compute_capability')
                ):
                    continue
            
            # RAM要件チェック
            ram_min = req.get('ram_min_gb', 0)
            if hardware.ram_total_gb < ram_min:
                continue
            
            # OSチェック
            os_req = req.get('os', [])
            if os_req and hardware.os.lower() not in [o.lower() for o in os_req]:
                continue
            
            # CPUアーキテクチャチェック
            arch_req = req.get('cpu_arch')
            if arch_req and hardware.cpu_arch != arch_req:
                continue
            
            # CPUフラグチェック（AVX2等）
            flags_req = req.get('cpu_flags', [])
            if flags_req and not all(f in hardware.cpu_flags for f in flags_req):
                continue
            
            matching.append(profile)
        
        # 優先順位: user_profile > tier一致 > VRAM余裕
        matching.sort(key=lambda p: (
            -int(p.get('user_profile', False)),  # ユーザー環境優先
            -({'cpu_only':0,'entry_gpu':1,'mid_gpu':2,'high_gpu':3,'multi_gpu':4,'apple_silicon':5,'datacenter':6}
              .get(p.get('tier',''), -1)),  # 高ティア優先
            -sum(g.get('vram_total_gb',0) for g in hardware.gpus)  # VRAM多い順
        ))
        
        return matching[:3]  # 上位3件を返す
    
    def _get_model_info(self, model_id: str) -> Optional[Dict]:
        """モデルカタログから情報を取得"""
        # model_idからモデル名を抽出（例: "Qwen3.5-9B-Q4_K_M" → "Qwen3.5-9B"）
        model_name = '-'.join(model_id.split('-')[:-1]) if '-' in model_id else model_id
        return self.dict_data.get('model_catalog', {}).get(model_name)
    
    def _get_download_url(self, model_info: Dict) -> Optional[str]:
        """HuggingFaceダウンロードURLを生成"""
        repo = model_info.get('huggingface_repo')
        if not repo:
            return None
        return f"https://huggingface.co/{repo}"
    
    def _calculate_confidence(self, hardware: HardwareSpec, 
                             model_config: Dict,
                             quality_priority: bool) -> float:
        """推奨信頼度を計算"""
        score = 0.7  # ベーススコア
        
        # VRAM余裕度
        total_vram = sum(g.get('vram_total_gb', 0) for g in hardware.gpus) or hardware.ram_total_gb
        vram_needed = model_config.get('vram_needed_gb', 0) + model_config.get('ram_needed_gb', 0)
        if total_vram >= vram_needed * 1.5:
            score += 0.2  # 余裕あり
        elif total_vram >= vram_needed:
            score += 0.1  # ぎりぎり
        else:
            score -= 0.2  # 不足
        
        # 品質優先モードの調整
        if quality_priority and model_config['quantization'] in ['Q6_K', 'Q8_0', 'FP16']:
            score += 0.1
        
        # MoEモデルの速度優位性
        if model_config.get('is_moe', False) and hardware.inference_tier in ['mid_gpu', 'high_gpu']:
            score += 0.05
        
        return min(max(score, 0.0), 1.0)
    
    def _get_fallback_recommendation(self, hardware: HardwareSpec) -> ModelRecommendation:
        """適合モデルがない場合のフォールバック推奨"""
        if hardware.ram_total_gb >= 32:
            return ModelRecommendation(
                model_id="Qwen3.5-9B-Q4_K_M",
                model_name="Qwen3.5-9B",
                quantization="Q4_K_M",
                expected_tps=1.8,
                context_max=8192,
                vram_needed_gb=0,
                ram_needed_gb=6.5,
                confidence=0.6,
                notes=["CPU推論フォールバック", "化学タスクに十分な性能"],
                download_url="https://huggingface.co/Qwen/Qwen3.5-9B-Instruct"
            )
        else:
            return ModelRecommendation(
                model_id="Qwen3.5-4B-Q4_K_M",
                model_name="Qwen3.5-4B",
                quantization="Q4_K_M",
                expected_tps=2.5,
                context_max=8192,
                vram_needed_gb=0,
                ram_needed_gb=3.2,
                confidence=0.5,
                notes=["軽量モデルでフォールバック", "大規模モデルには非対応"],
                download_url="https://huggingface.co/Qwen/Qwen3.5-4B-Instruct"
            )
    
    def list_compatible_models(self, hardware: HardwareSpec) -> List[ModelRecommendation]:
        """適合する全モデルを一覧表示"""
        profiles = self._find_matching_profiles(hardware)
        recommendations = []
        
        for profile in profiles:
            for model_config in profile.get('compatible_models', []):
                model_info = self._get_model_info(model_config['model'])
                if not model_info:
                    continue
                
                recommendations.append(ModelRecommendation(
                    model_id=f"{model_config['model']}-{model_config['quantization']}",
                    model_name=model_config['model'],
                    quantization=model_config['quantization'],
                    expected_tps=model_config['expected_tps'],
                    context_max=model_config['context_max'],
                    vram_needed_gb=model_config['vram_needed_gb'],
                    ram_needed_gb=model_config['ram_needed_gb'],
                    confidence=0.8,
                    notes=[model_config.get('notes', '')],
                    download_url=self._get_download_url(model_info),
                    is_moe=model_info.get('architecture') == 'moe'
                ))
        
        # 速度順にソート
        recommendations.sort(key=lambda r: r.expected_tps, reverse=True)
        return recommendations
