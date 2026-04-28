#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM動作辞書 自動更新モジュール
- 40環境ごとの適合モデルを自動判定
- 新モデルリリースを自動検出・辞書に反映
- 実機ベンチマーク値の記録・更新
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict
import requests
from huggingface_hub import HfApi, ModelInfo

# 内部モジュール（chemai2既存構造に準拠）
from backend.llm.hardware_detector import detect_hardware, HardwareProfile
from config import CATALOG_PATH, BENCHMARK_CACHE_PATH

logger = logging.getLogger(__name__)


# ========== 型定義 ==========
class ModelEntry(TypedDict):
    """辞書内の1モデルエントリ"""
    model: str           # e.g. "Qwen3.5-7B"
    quant: str           # e.g. "Q4_K_M"
    file_size_gb: float  # e.g. 5.1
    min_vram_gb: Optional[float]  # GPU必要量（CPUのみの場合はNone）
    min_ram_gb: float    # システムメモリ必要量
    expected_tps: str    # e.g. "22-38 (GPU)"
    use_case: str        # e.g. "化学構造解析"
    huggingface: str     # e.g. "Qwen/Qwen3.5-7B-GGUF"
    priority: str        # "primary" | "high_quality" | "speed" | "japanese"
    benchmark: Optional[Dict[str, float]]  # 実機ベンチマーク値


class EnvironmentSpec(TypedDict):
    """40環境の1つを定義"""
    env_id: str          # e.g. "ENV020"
    name: str            # e.g. "ユーザー環境（RTX 5080 16GB + 32GB）"
    specs: Dict[str, any]  # gpu, vram_gb, ram_gb, cpu, platform
    execution_mode: str  # "gpu_full" | "gpu_preferred" | "cpu_only" | "metal"


class LLMDictionaryUpdater:
    """LLM動作辞書の自動更新クラス"""
    
    # 更新ソース（新モデル検出用）
    UPDATE_SOURCES = [
        {"type": "huggingface", "url": "https://huggingface.co/api/models", "params": {"library": "gguf", "sort": "trending", "limit": 100}},
        {"type": "github", "url": "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"},
        {"type": "unsloth", "url": "https://unsloth.ai/api/model-catalog"}  # 仮定エンドポイント
    ]
    
    # 量子化形式の品質順位（高→低）と圧縮率目安
    QUANTIZATION_RANK = {
        "Q8_0": {"quality": 10, "compression": 0.5},
        "Q6_K": {"quality": 9, "compression": 0.4},
        "Q5_K_M": {"quality": 8, "compression": 0.35},
        "Q4_K_M": {"quality": 7, "compression": 0.3},
        "Q4_K_S": {"quality": 6, "compression": 0.28},
        "Q3_K_M": {"quality": 5, "compression": 0.25},
        "IQ3_M": {"quality": 4, "compression": 0.22},
        "Q2_K": {"quality": 3, "compression": 0.2},
        "IQ2_XXS": {"quality": 2, "compression": 0.15}
    }
    
    # 40環境定義（抜粋：代表8環境。完全版は別ファイルで管理）
    ENVIRONMENTS: List[EnvironmentSpec] = [
        {
            "env_id": "ENV001",
            "name": "エントリーノート（Intel UHD + 8GB）",
            "specs": {"gpu": "integrated", "vram_gb": 0, "ram_gb": 8, "cpu": "entry", "platform": "windows"},
            "execution_mode": "cpu_only"
        },
        {
            "env_id": "ENV007",
            "name": "標準ノート（RTX 3060 6GB + 16GB）",
            "specs": {"gpu": "RTX 3060", "vram_gb": 6, "ram_gb": 16, "cpu": "mid", "platform": "windows"},
            "execution_mode": "gpu_preferred"
        },
        {
            "env_id": "ENV020",  # ユーザー環境
            "name": "ユーザー環境（RTX 5080 16GB + 32GB）",
            "specs": {"gpu": "RTX 5080", "vram_gb": 16, "ram_gb": 32, "cpu": "high", "platform": "windows"},
            "execution_mode": "gpu_full"
        },
        {
            "env_id": "ENV033",
            "name": "Mac Studio（M3 Max + 64GB UM）",
            "specs": {"gpu": "Apple M3 Max", "vram_gb": 64, "ram_gb": 64, "cpu": "apple_silicon", "platform": "macos"},
            "execution_mode": "metal"
        },
        {
            "env_id": "ENV029",
            "name": "CPU専用ワークステーション（Ryzen 7 + 64GB）",
            "specs": {"gpu": "none", "vram_gb": 0, "ram_gb": 64, "cpu": "high", "platform": "linux"},
            "execution_mode": "cpu_only"
        },
        # ... 残り35環境は省略（完全版は `config/environments_40.json` で管理）
    ]
    
    # 化学ドメイン適性評価ルール（簡易版）
    CHEM_DOMAIN_RULES = {
        "japanese_support": {"keywords": ["ja", "japanese", "nihongo"], "weight": 2},
        "reasoning_capability": {"keywords": ["reasoning", "math", "logic"], "weight": 2},
        "code_generation": {"keywords": ["code", "python", "function"], "weight": 1},
        "science_domain": {"keywords": ["science", "chemistry", "material"], "weight": 3}
    }
    
    def __init__(self, catalog_path: str = CATALOG_PATH):
        self.catalog_path = Path(catalog_path)
        self.api = HfApi()
        self._load_benchmark_cache()
    
    def _load_benchmark_cache(self):
        """実機ベンチマーク値をキャッシュから読み込み"""
        cache_file = Path(BENCHMARK_CACHE_PATH)
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.benchmark_cache = json.load(f)
        else:
            self.benchmark_cache = {}
    
    def _save_benchmark_cache(self):
        """ベンチマークキャッシュを保存"""
        cache_file = Path(BENCHMARK_CACHE_PATH)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.benchmark_cache, f, ensure_ascii=False, indent=2)
    
    def _fetch_huggingface_models(self) -> List[ModelInfo]:
        """HuggingFaceからGGUFモデル一覧を取得"""
        try:
            models = self.api.list_models(
                library="gguf",
                sort="trending",
                limit=100,
                full=False
            )
            # 日本語対応・化学関連・最新リリースを優先フィルタ
            filtered = []
            for model in models:
                tags = model.tags or []
                card = self.api.model_info(model.id).cardData or {}
                # 簡易フィルタ: 日本語タグまたは化学関連キーワードを含むもの
                if any(kw in str(card).lower() for kw in ["japanese", "ja", "chemistry", "material", "smiles"]):
                    filtered.append(model)
                elif "gguf" in model.id.lower():
                    filtered.append(model)  # GGUF形式なら一旦保持
            return filtered[:50]  # 上位50モデルに絞る
        except Exception as e:
            logger.warning(f"HuggingFace API fetch failed: {e}")
            return []
    
    def _parse_gguf_files(self, model_id: str) -> List[Dict]:
        """モデルリポジトリからGGUFファイル一覧を解析"""
        try:
            files = self.api.list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            results = []
            for fname in gguf_files:
                # ファイル名から量子化形式を抽出（例: "qwen3.5-7b-q4_k_m.gguf" → "Q4_K_M"）
                quant = self._extract_quant_from_filename(fname)
                if not quant:
                    continue
                # ファイルサイズ取得（APIで取得できない場合は推定）
                size_info = self.api.model_info(model_id)
                file_size_bytes = None
                for sibling in size_info.siblings or []:
                    if sibling.rfilename == fname:
                        file_size_bytes = sibling.size
                        break
                if file_size_bytes is None:
                    # 推定: ベースパラメータ数 × 量子化圧縮率
                    param_count = self._extract_param_count(model_id)  # e.g. 7 for 7B
                    compression = self.QUANTIZATION_RANK.get(quant, {}).get("compression", 0.3)
                    file_size_bytes = int(param_count * 1e9 * 2 * compression)  # 2 bytes/param (FP16) × compression
                results.append({
                    "filename": fname,
                    "quant": quant,
                    "file_size_gb": round(file_size_bytes / (1024**3), 2),
                    "huggingface_path": f"{model_id}/{fname}"
                })
            return results
        except Exception as e:
            logger.warning(f"Failed to parse GGUF files for {model_id}: {e}")
            return []
    
    def _extract_quant_from_filename(self, filename: str) -> Optional[str]:
        """ファイル名から量子化形式を抽出"""
        filename_lower = filename.lower()
        for quant in self.QUANTIZATION_RANK.keys():
            if quant.lower() in filename_lower:
                return quant
        return None
    
    def _extract_param_count(self, model_id: str) -> float:
        """モデルIDからパラメータ数（単位: B）を抽出"""
        # 例: "Qwen3.5-7B-GGUF" → 7.0, "DeepSeek-V3.2-14B" → 14.0
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*[bB](?!y)', model_id)
        if match:
            return float(match.group(1))
        # デフォルト値（推定）
        if "3b" in model_id.lower():
            return 3.0
        elif "7b" in model_id.lower():
            return 7.0
        elif "14b" in model_id.lower():
            return 14.0
        elif "24b" in model_id.lower():
            return 24.0
        return 7.0  # デフォルト
    
    def _check_environment_compatibility(self, env: EnvironmentSpec, model_entry: ModelEntry) -> bool:
        """環境とモデルの適合性を判定"""
        specs = env["specs"]
        
        # RAMチェック（常に必須）
        if specs["ram_gb"] < model_entry["min_ram_gb"]:
            return False
        
        # GPU環境の場合: VRAMチェック
        if env["execution_mode"] in ["gpu_full", "gpu_preferred", "metal"]:
            if model_entry.get("min_vram_gb") is not None:
                if specs["vram_gb"] < model_entry["min_vram_gb"]:
                    # CPUオフロードが可能かチェック（llama.cppは部分オフロード対応）
                    if env["execution_mode"] != "gpu_preferred":
                        return False
                    # GPU_preferredならCPUフォールバック可能
                    if specs["ram_gb"] < model_entry["min_ram_gb"] + model_entry["file_size_gb"]:
                        return False
        
        # Apple Silicon 環境: Metal backend 対応モデルのみ
        if env["execution_mode"] == "metal":
            if "metal" not in model_entry.get("huggingface", "").lower():
                # Metal非対応でもCPU実行は可能だが、速度低下を考慮して優先度下げ
                if model_entry["priority"] == "primary":
                    model_entry["priority"] = "alternative"
        
        return True
    
    def _evaluate_chem_domain_suitability(self, model_id: str, card_data: dict) -> Dict[str, float]:
        """化学ドメイン適性をスコア評価"""
        score = 0.0
        details = {}
        
        for rule_name, rule in self.CHEM_DOMAIN_RULES.items():
            keywords = rule["keywords"]
            weight = rule["weight"]
            # card_data（モデルカード）からキーワード検索
            card_text = json.dumps(card_data).lower()
            if any(kw in card_text for kw in keywords):
                score += weight
                details[rule_name] = True
            else:
                details[rule_name] = False
        
        # 追加: 日本語対応は重み付け
        if "japanese_support" in details and details["japanese_support"]:
            score += 1  # 追加ボーナス
        
        return {"total_score": score, "details": details}
    
    def _estimate_expected_tps(self, env: EnvironmentSpec, model_entry: ModelEntry) -> str:
        """期待推論速度（tokens/sec）を推定"""
        # ベンチマークキャッシュがあれば優先使用
        cache_key = f"{env['env_id']}:{model_entry['model']}:{model_entry['quant']}"
        if cache_key in self.benchmark_cache:
            tps = self.benchmark_cache[cache_key]
            return f"{tps:.1f} ({env['execution_mode'].replace('_', '-').upper()})"
        
        # 推定ロジック（簡易版）
        specs = env["specs"]
        param_count = self._extract_param_count(model_entry["huggingface"])
        quant_quality = self.QUANTIZATION_RANK.get(model_entry["quant"], {}).get("quality", 5)
        
        # 基本速度係数（環境別）
        base_tps = {
            "cpu_only": 2.0,      # CPU 8core 目安
            "gpu_preferred": 15.0, # entry-dGPU
            "gpu_full": 30.0,      # high-dGPU (RTX 4080/5080)
            "metal": 20.0          # Apple Silicon
        }.get(env["execution_mode"], 5.0)
        
        # パラメータ数・量子化品質で補正
        tps = base_tps * (7.0 / param_count) * (quant_quality / 7.0)
        
        # VRAM/RAM余裕度で微調整
        if env["execution_mode"] in ["gpu_full", "metal"]:
            vram_margin = specs["vram_gb"] - model_entry.get("min_vram_gb", 0)
            if vram_margin > 4:
                tps *= 1.2  # 余裕あれば高速化
        
        return f"{max(1, int(tps*0.8))}-{int(tps*1.2)} ({env['execution_mode'].replace('_', '-').upper()})"
    
    def _generate_model_entry(self, model_id: str, gguf_file: Dict, env_list: List[EnvironmentSpec]) -> Optional[ModelEntry]:
        """1モデルエントリを生成"""
        param_count = self._extract_param_count(model_id)
        quant = gguf_file["quant"]
        
        # 最小必要メモリ計算（簡易モデル）
        # 推論時: モデルサイズ + KVキャッシュ(2GB) + 作業領域(1GB)
        model_size_gb = gguf_file["file_size_gb"]
        min_ram_gb = model_size_gb + 3.0  # 余裕3GB
        min_vram_gb = model_size_gb + 1.5 if quant in ["Q5_K_M", "Q6_K", "Q8_0"] else model_size_gb + 1.0
        
        # 使用用途の自動判定（モデル名・タグベース）
        use_case = "汎用タスク"
        priority = "alternative"
        if "qwen" in model_id.lower() and "3.5" in model_id.lower():
            if param_count <= 3:
                use_case = "軽量チャット・SMILES入力支援"
                priority = "speed"
            elif param_count <= 7:
                use_case = "化学構造解析・コード生成"
                priority = "primary"
            else:
                use_case = "高精度推論・複雑タスク"
                priority = "high_quality"
        elif "deepseek" in model_id.lower():
            use_case = "数学的推論・物性計算支援"
            priority = "reasoning"
        elif "solar" in model_id.lower():
            use_case = "日本語特化・レポート作成"
            priority = "japanese"
        elif "gemma" in model_id.lower():
            use_case = "多言語対応・実験計画支援"
            priority = "versatile"
        
        entry = ModelEntry(
            model=model_id.split("/")[-1].replace("-GGUF", "").replace("-gguf", ""),
            quant=quant,
            file_size_gb=gguf_file["file_size_gb"],
            min_vram_gb=round(min_vram_gb, 1),
            min_ram_gb=round(min_ram_gb, 1),
            expected_tps="",  # 後で環境別計算
            use_case=use_case,
            huggingface=gguf_file["huggingface_path"],
            priority=priority,
            benchmark=None
        )
        return entry
    
    def monthly_update(self) -> Dict[str, any]:
        """月次更新：新モデル検出→適合性判定→辞書生成"""
        logger.info("Starting monthly LLM dictionary update...")
        update_log = {
            "timestamp": datetime.now().isoformat(),
            "new_models_added": 0,
            "environments_updated": 0,
            "errors": []
        }
        
        # 1. 新モデル検出（HuggingFaceから）
        new_models = self._fetch_huggingface_models()
        logger.info(f"Found {len(new_models)} trending GGUF models")
        
        # 2. 既存辞書読み込み（マージ用）
        existing_catalog = {}
        if self.catalog_path.exists():
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                existing_catalog = json.load(f)
        
        # 3. 各モデルを処理
        for model in new_models:
            model_id = model.id
            # 既存エントリがあればスキップ（重複防止）
            if any(model_id in env_data.get("models", {}) for env_data in existing_catalog.values()):
                continue
            
            # GGUFファイル一覧取得
            gguf_files = self._parse_gguf_files(model_id)
            if not gguf_files:
                continue
            
            # 化学ドメイン適性評価
            card_data = self.api.model_info(model_id).cardData or {}
            chem_score = self._evaluate_chem_domain_suitability(model_id, card_data)
            if chem_score["total_score"] < 2:  # 閾値未満はスキップ
                logger.info(f"Skipping {model_id}: low chem-domain score ({chem_score['total_score']})")
                continue
            
            # 40環境それぞれに対して適合モデルを生成
            for env in self.ENVIRONMENTS:
                env_id = env["env_id"]
                if env_id not in existing_catalog:
                    existing_catalog[env_id] = {
                        "env_id": env_id,
                        "name": env["name"],
                        "specs": env["specs"],
                        "execution_mode": env["execution_mode"],
                        "recommended": [],
                        "last_updated": None
                    }
                
                for gguf in gguf_files:
                    entry = self._generate_model_entry(model_id, gguf, self.ENVIRONMENTS)
                    if not entry:
                        continue
                    
                    # 適合性チェック
                    if not self._check_environment_compatibility(env, entry):
                        continue
                    
                    # 期待速度計算
                    entry["expected_tps"] = self._estimate_expected_tps(env, entry)
                    
                    # 既存リストに重複なく追加
                    if not any(e["model"] == entry["model"] and e["quant"] == entry["quant"] 
                              for e in existing_catalog[env_id]["recommended"]):
                        existing_catalog[env_id]["recommended"].append(entry)
                        # 優先度でソート（primary → high_quality → speed → japanese → alternative）
                        priority_order = {"primary": 0, "high_quality": 1, "speed": 2, "reasoning": 3, "japanese": 4, "versatile": 5, "alternative": 6}
                        existing_catalog[env_id]["recommended"].sort(key=lambda x: priority_order.get(x["priority"], 99))
                        update_log["new_models_added"] += 1
                
                existing_catalog[env_id]["last_updated"] = datetime.now().isoformat()
                update_log["environments_updated"] += 1
        
        # 4. 辞書ファイル保存
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, 'w', encoding='utf-8') as f:
            json.dump(existing_catalog, f, ensure_ascii=False, indent=2)
        
        # 5. ベンチマークキャッシュ保存
        self._save_benchmark_cache()
        
        logger.info(f"Update completed: {update_log['new_models_added']} new entries added")
        return update_log
    
    def on_new_release(self, model_name: str) -> Dict[str, any]:
        """新モデルリリース検出時の即時更新"""
        logger.info(f"Processing new release: {model_name}")
        result = {"success": False, "added_entries": 0, "errors": []}
        
        try:
            # 1. GGUF形式の有無を確認
            gguf_files = self._parse_gguf_files(model_name)
            if not gguf_files:
                result["errors"].append("No GGUF files found")
                return result
            
            # 2. 化学ドメイン適性評価
            card_data = self.api.model_info(model_name).cardData or {}
            chem_score = self._evaluate_chem_domain_suitability(model_name, card_data)
            if chem_score["total_score"] < 2:
                result["errors"].append(f"Low chem-domain score: {chem_score['total_score']}")
                return result
            
            # 3. 既存辞書読み込み
            existing_catalog = {}
            if self.catalog_path.exists():
                with open(self.catalog_path, 'r', encoding='utf-8') as f:
                    existing_catalog = json.load(f)
            
            # 4. 40環境に適合するエントリを追加
            for env in self.ENVIRONMENTS:
                env_id = env["env_id"]
                if env_id not in existing_catalog:
                    continue  # 未知の環境はスキップ
                
                for gguf in gguf_files:
                    entry = self._generate_model_entry(model_name, gguf, self.ENVIRONMENTS)
                    if not entry:
                        continue
                    if not self._check_environment_compatibility(env, entry):
                        continue
                    entry["expected_tps"] = self._estimate_expected_tps(env, entry)
                    
                    # 重複チェックして追加
                    if not any(e["model"] == entry["model"] and e["quant"] == entry["quant"]
                              for e in existing_catalog[env_id]["recommended"]):
                        existing_catalog[env_id]["recommended"].append(entry)
                        # ソート再適用
                        priority_order = {"primary": 0, "high_quality": 1, "speed": 2, "reasoning": 3, "japanese": 4, "versatile": 5, "alternative": 6}
                        existing_catalog[env_id]["recommended"].sort(key=lambda x: priority_order.get(x["priority"], 99))
                        result["added_entries"] += 1
                
                existing_catalog[env_id]["last_updated"] = datetime.now().isoformat()
            
            # 5. 辞書保存
            with open(self.catalog_path, 'w', encoding='utf-8') as f:
                json.dump(existing_catalog, f, ensure_ascii=False, indent=2)
            
            result["success"] = True
            logger.info(f"Added {result['added_entries']} entries for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            result["errors"].append(str(e))
        
        return result
    
    def register_benchmark_result(self, env_id: str, model: str, quant: str, tps: float):
        """実機ベンチマーク結果を登録（手動/自動テストから呼び出し）"""
        cache_key = f"{env_id}:{model}:{quant}"
        self.benchmark_cache[cache_key] = tps
        self._save_benchmark_cache()
        logger.info(f"Registered benchmark: {cache_key} = {tps:.2f} tps")
    
    def get_recommended_models(self, hardware_profile: HardwareProfile) -> List[ModelEntry]:
        """実行時: 現在のハードウェアに適合する推奨モデルリストを返す"""
        # 1. 最も適合する環境を特定
        best_match = None
        best_score = -1
        for env in self.ENVIRONMENTS:
            score = self._calculate_env_match_score(hardware_profile, env)
            if score > best_score:
                best_score = score
                best_match = env
        
        if not best_match:
            # 適合環境なし: CPU専用フォールバック
            best_match = next((e for e in self.ENVIRONMENTS if e["execution_mode"] == "cpu_only"), self.ENVIRONMENTS[0])
        
        # 2. 辞書から推奨リスト取得
        if not self.catalog_path.exists():
            logger.warning("Catalog file not found. Running initial update...")
            self.monthly_update()
        
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
        
        env_data = catalog.get(best_match["env_id"], {})
        return env_data.get("recommended", [])[:5]  # 上位5件を返す
    
    def _calculate_env_match_score(self, hw: HardwareProfile, env: EnvironmentSpec) -> float:
        """ハードウェアプロファイルと環境定義の一致度をスコア化"""
        score = 0.0
        # GPU名一致
        if hw.gpu_name and env["specs"]["gpu"].lower() in hw.gpu_name.lower():
            score += 3.0
        # VRAM範囲内
        if env["specs"]["vram_gb"] > 0:
            if abs(hw.vram_gb - env["specs"]["vram_gb"]) <= 2:
                score += 2.0
        # RAM範囲内
        if abs(hw.ram_gb - env["specs"]["ram_gb"]) <= 8:
            score += 1.5
        # CPUクラス一致
        if hw.cpu_class == env["specs"]["cpu"]:
            score += 1.0
        # platform一致
        if hw.platform == env["specs"]["platform"]:
            score += 0.5
        return score


# ========== ユーティリティ関数（モジュール外からも利用可能） ==========
def auto_select_model(hardware_profile: HardwareProfile, catalog_path: str = CATALOG_PATH) -> Optional[ModelEntry]:
    """簡易インターフェース: ハードウェアから最適モデルを1つ選択"""
    updater = LLMDictionaryUpdater(catalog_path)
    candidates = updater.get_recommended_models(hardware_profile)
    if not candidates:
        return None
    # 優先度順で最初の「primary」または「high_quality」を返す
    for candidate in candidates:
        if candidate["priority"] in ["primary", "high_quality"]:
            return candidate
    return candidates[0]  # なければ先頭を返す


if __name__ == "__main__":
    # 単体テスト用エントリーポイント
    logging.basicConfig(level=logging.INFO)
    updater = LLMDictionaryUpdater()
    
    # 月次更新実行（テスト用）
    # result = updater.monthly_update()
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 新モデル即時追加テスト
    # result = updater.on_new_release("Qwen/Qwen3.5-7B-GGUF")
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 現在環境の推奨モデル取得テスト
    hw = detect_hardware()
    recommended = updater.get_recommended_models(hw)
    print(f"Recommended models for your environment ({hw.gpu_name}):")
    for i, model in enumerate(recommended, 1):
        print(f"{i}. {model['model']} ({model['quant']}) - {model['expected_tps']} - {model['use_case']}")
