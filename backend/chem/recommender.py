# backend/chem/recommender.py — 精緻化版 (目的変数別推奨記述子エンジン)

from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
import logging
import hashlib
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class DescriptorRecommender:
    """
    Recommend chemical descriptors based on target property with robust fallback
    
    Features:
    - Rule-based recommendations for known properties
    - Similarity-based fallback for unknown properties
    - LRU cache for repeated queries
    - Graceful degradation when data is missing
    """
    
    # 【修正点4】デフォルト推奨セット（未知プロパティ用）
    DEFAULT_RECOMMENDATIONS: Dict[str, List[Dict[str, float]]] = {
        'default': [
            {'plugin': 'rdkit_basic', 'priority': 1.0, 'reason': 'general purpose'},
            {'plugin': 'mordred_2d', 'priority': 0.8, 'reason': 'broad coverage'},
            {'plugin': 'xtb_energy', 'priority': 0.6, 'reason': 'quantum features'},
        ]
    }
    
    # 既知プロパティ用のルールベース推奨
    PROPERTY_RULES: Dict[str, List[Dict[str, float]]] = {
        'solubility': [
            {'plugin': 'rdkit_logp', 'priority': 1.0, 'reason': 'lipophilicity correlation'},
            {'plugin': 'rdkit_tpsa', 'priority': 0.9, 'reason': 'polar surface area'},
            {'plugin': 'cosmo_sigma', 'priority': 0.7, 'reason': 'solvation effects'},
        ],
        'logp': [
            {'plugin': 'rdkit_logp', 'priority': 1.0, 'reason': 'direct calculation'},
            {'plugin': 'mordred_hydrophobic', 'priority': 0.8, 'reason': 'hydrophobic descriptors'},
        ],
        'pka': [
            {'plugin': 'unipka', 'priority': 1.0, 'reason': 'specialized pKa predictor'},
            {'plugin': 'rdkit_acidic_groups', 'priority': 0.7, 'reason': 'functional group count'},
        ],
    }
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self._cache: Dict[str, List[Dict]] = {}
        self._cache_order: List[str] = []
    
    def _make_cache_key(self, property_name: str, params: Dict) -> str:
        """
        Generate deterministic cache key independent of dict order
        
        【修正点3】パラメータ順序に依存しないキー生成
        """
        key_data = {
            'property': property_name.lower().strip(),
            'params': dict(sorted(params.items()))  # 【修正点3】キーをソート
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _update_cache(self, key: str, value: List[Dict]):
        """LRU cache update with size limit"""
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self.cache_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def recommend(
        self,
        property_name: str,
        available_plugins: List[str],
        params: Optional[Dict] = None,
        min_priority: float = 0.3,
        max_results: int = 10
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Recommend descriptors for a target property with robust fallback
        """
        params = params or {}
        cache_key = self._make_cache_key(property_name, params)
        
        # Cache check
        if cache_key in self._cache:
            logger.debug(f"Cache hit for property '{property_name}'")
            return self._filter_recommendations(
                self._cache[cache_key], available_plugins, min_priority, max_results
            )
        
        # 【修正点4】プロパティ名の正規化と未知プロパティ処理
        prop_normalized = property_name.lower().strip()
        
        # 1. 完全一致ルール検索
        if prop_normalized in self.PROPERTY_RULES:
            recommendations = self.PROPERTY_RULES[prop_normalized].copy()
        
        # 2. 部分一致フォールバック（類似プロパティ検索）
        else:
            matched = self._find_similar_property(prop_normalized)
            if matched:
                logger.info(f"Using similar property rules for '{prop_normalized}': '{matched}'")
                recommendations = self.PROPERTY_RULES[matched].copy()
            else:
                # 【修正点4】未知プロパティはデフォルト推奨セットを使用
                logger.warning(
                    f"Unknown property '{property_name}'. Using default recommendations."
                )
                recommendations = self.DEFAULT_RECOMMENDATIONS['default'].copy()
        
        # 3. 利用可能プラグインでフィルタリング
        filtered = self._filter_recommendations(
            recommendations, available_plugins, min_priority, max_results
        )
        
        # 【修正点2】優先度の正規化を固定範囲[0,1]に統一（データ依存を排除）
        if filtered:
            max_p = max(r['priority'] for r in filtered)
            if max_p > 0:
                for r in filtered:
                    r['priority'] = min(1.0, r['priority'] / max_p)
        
        # Cache and return
        self._update_cache(cache_key, filtered)
        return filtered
    
    def _find_similar_property(self, query: str, threshold: float = 0.6) -> Optional[str]:
        """
        Find similar property name using simple string similarity
        
        【修正点1】空文字列・None対策を追加
        """
        if not query or not isinstance(query, str):
            return None
        
        query = query.strip().lower()
        if not query:
            return None
        
        best_match = None
        best_score = threshold
        
        for known_prop in self.PROPERTY_RULES.keys():
            # 簡易類似度: 部分一致＋編集距離ベース
            if query in known_prop or known_prop in query:
                score = 1.0
            else:
                score = self._simple_string_similarity(query, known_prop)
            
            if score > best_score:
                best_score = score
                best_match = known_prop
        
        return best_match
    
    def _simple_string_similarity(self, s1: str, s2: str) -> float:
        """Simple character-level similarity ratio"""
        if not s1 or not s2:
            return 0.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        common = sum(min(s1.count(c), s2.count(c)) for c in set(s1) | set(s2))
        return 2.0 * common / (len1 + len2)
    
    def _filter_recommendations(
        self,
        recommendations: List[Dict],
        available_plugins: List[str],
        min_priority: float,
        max_results: int
    ) -> List[Dict]:
        """Filter and sort recommendations"""
        available_set = set(available_plugins)
        
        filtered = [
            r for r in recommendations
            if r.get('plugin') in available_set and r.get('priority', 0) >= min_priority
        ]
        
        # Sort by priority descending, then by plugin name for determinism
        filtered.sort(key=lambda x: (-x['priority'], x['plugin']))
        
        return filtered[:max_results]
    
    def clear_cache(self, property_name: Optional[str] = None):
        """Clear cache for specific property or all"""
        if property_name:
            key = self._make_cache_key(property_name, {})
            if key in self._cache:
                del self._cache[key]
                self._cache_order.remove(key)
        else:
            self._cache.clear()
            self._cache_order.clear()
