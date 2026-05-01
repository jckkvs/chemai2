"""
backend/chem/adaptive_feature_selector.py

目的変数の予測タスクと計算予算に応じて、
最適な特徴量セットを自動推奨するエンジン。

既存の recommender.py（目的変数ごとの推奨記述子）とは異なり、
こちらは「計算コスト制約」と「タスク種別」を考慮した
**xTB派生特徴量の選択最適化**に特化する。

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


# ============================================================
# recommender.py 知識ベースの統合
# ============================================================

def _get_recommender_features(task_type: str) -> list[str]:
    """
    recommender.py の知識ベースから、指定タスク種別に関連する
    特徴量名のリストを返す。
    """
    try:
        from backend.chem.recommender import (
            get_target_recommendation_by_name,
            get_all_target_recommendations,
        )
    except ImportError:
        logger.debug("recommender.py not available")
        return []

    # task_type から recommender のターゲット名を推定
    # recommender.py のターゲット名は日本語だが、部分一致で検索する
    rec = get_target_recommendation_by_name(task_type)
    if rec is None:
        # 全ターゲットからカテゴリ一致を探す
        all_recs = get_all_target_recommendations()
        for r in all_recs:
            if task_type.lower() in r.category.lower():
                rec = r
                break

    if rec is None:
        return []

    # DescriptorInfo から特徴量名を抽出
    # library 名を adaptive_feature_selector の特徴量名にマップ
    feature_names = []
    library_map = {
        "RDKit": "rdkit_2d",
        "XTB": "xtb_sp",
        "COSMO-RS": "cosmo",
        "GroupContribution": "group_contrib",
        "Uni-pKa": "unipka",
    }
    for desc in rec.descriptors:
        mapped = library_map.get(desc.library)
        if mapped and mapped not in feature_names:
            feature_names.append(mapped)
        elif desc.name not in feature_names:
            # そのまま追加
            feature_names.append(desc.name)

    return feature_names


# ============================================================
# 特徴量の計算コスト定義
# ============================================================

@dataclass
class FeatureCost:
    """1つの特徴量（群）の計算コスト情報。"""
    name: str
    category: str
    requires_xtb: bool = False
    requires_opt: bool = False        # 構造最適化が必要か
    requires_freq: bool = False       # 振動計算が必要か
    requires_3d_coords: bool = False  # 3D座標が必要か
    approx_time_per_mol_s: float = 0.0  # 概算計算時間（秒/分子）
    description_jp: str = ""


# 特徴量コストカタログ
_FEATURE_COST_CATALOG: list[FeatureCost] = [
    # === RDKit由来（高速）===
    FeatureCost(
        "rdkit_2d", "2D記述子",
        approx_time_per_mol_s=0.01,
        description_jp="RDKit 217記述子 + Gasteiger電荷（高速）",
    ),
    FeatureCost(
        "morgan_fp", "フィンガープリント",
        approx_time_per_mol_s=0.005,
        description_jp="Morgan FP (ECFP4, 2048bit)",
    ),
    FeatureCost(
        "maccs_keys", "フィンガープリント",
        approx_time_per_mol_s=0.003,
        description_jp="MACCS Keys (166bit)",
    ),

    # === xTB由来（中速）===
    FeatureCost(
        "xtb_sp", "量子化学(単点)",
        requires_xtb=True,
        approx_time_per_mol_s=10.0,
        description_jp="xTB単点計算: エネルギー・軌道・双極子",
    ),
    FeatureCost(
        "xtb_opt", "量子化学(最適化)",
        requires_xtb=True, requires_opt=True,
        approx_time_per_mol_s=90.0,
        description_jp="xTB構造最適化 + 電子状態",
    ),
    FeatureCost(
        "xtb_ml_derived", "派生特徴量",
        requires_xtb=True,
        approx_time_per_mol_s=0.1,
        description_jp="硬さ/軟らかさ/親電子性（xTB結果から派生）",
    ),

    # === 3D幾何（中速、xTB opt必要）===
    FeatureCost(
        "3d_geometry", "3D幾何特徴量",
        requires_xtb=True, requires_opt=True, requires_3d_coords=True,
        approx_time_per_mol_s=0.5,
        description_jp="慣性モーメント・分子サイズ・非球面性",
    ),

    # === 振動（低速）===
    FeatureCost(
        "vibrational", "振動特徴量",
        requires_xtb=True, requires_opt=True, requires_freq=True,
        approx_time_per_mol_s=300.0,
        description_jp="振動数統計・ゼロ点エネルギー・熱力学量",
    ),

    # === 反応性指標（中速）===
    FeatureCost(
        "fukui_approx", "反応性指標",
        requires_xtb=True,
        approx_time_per_mol_s=0.1,
        description_jp="簡易福井関数（Mulliken電荷ベース）",
    ),

    # === アンサンブル（高コスト）===
    FeatureCost(
        "conformer_ensemble", "アンサンブル",
        requires_xtb=True,
        approx_time_per_mol_s=500.0,
        description_jp="複数conformerの統計量（エネルギー、電荷分布等）",
    ),
]


# ============================================================
# タスク別推奨テンプレート
# ============================================================

@dataclass
class TaskRecommendation:
    """タスク種別に対する特徴量推奨。"""
    task_type: str
    description_jp: str
    must_have: list[str]
    recommended: list[str]
    optional: list[str]
    exclude: list[str] = field(default_factory=list)


_TASK_RECOMMENDATIONS: list[TaskRecommendation] = [
    TaskRecommendation(
        task_type="solubility",
        description_jp="溶解度予測（LogS, 水溶性）",
        must_have=["rdkit_2d", "morgan_fp"],
        recommended=["xtb_sp", "xtb_ml_derived"],
        optional=["3d_geometry"],
        exclude=["vibrational", "conformer_ensemble"],
    ),
    TaskRecommendation(
        task_type="reactivity",
        description_jp="反応性予測（反応速度、選択性）",
        must_have=["xtb_opt", "xtb_ml_derived", "fukui_approx"],
        recommended=["rdkit_2d", "3d_geometry"],
        optional=["vibrational"],
    ),
    TaskRecommendation(
        task_type="toxicity",
        description_jp="毒性予測",
        must_have=["rdkit_2d", "morgan_fp", "maccs_keys"],
        recommended=["xtb_sp", "xtb_ml_derived"],
        optional=["fukui_approx"],
        exclude=["vibrational", "conformer_ensemble"],
    ),
    TaskRecommendation(
        task_type="binding_affinity",
        description_jp="結合親和性予測（タンパク質-リガンド等）",
        must_have=["rdkit_2d", "morgan_fp", "xtb_opt"],
        recommended=["3d_geometry", "xtb_ml_derived"],
        optional=["conformer_ensemble"],
    ),
    TaskRecommendation(
        task_type="optical",
        description_jp="光学物性予測（屈折率、吸収等）",
        must_have=["xtb_opt", "xtb_ml_derived", "rdkit_2d"],
        recommended=["3d_geometry"],
        optional=["vibrational"],
    ),
    TaskRecommendation(
        task_type="thermal",
        description_jp="熱物性予測（Tg, Tm, 耐熱性）",
        must_have=["rdkit_2d", "xtb_sp", "xtb_ml_derived"],
        recommended=["morgan_fp", "3d_geometry"],
        optional=["vibrational"],
    ),
    TaskRecommendation(
        task_type="mechanical",
        description_jp="力学物性予測（弾性率、引張強度）",
        must_have=["rdkit_2d", "xtb_sp"],
        recommended=["xtb_ml_derived", "morgan_fp"],
        optional=["3d_geometry"],
        exclude=["vibrational", "conformer_ensemble"],
    ),
    TaskRecommendation(
        task_type="general",
        description_jp="汎用（タスク不明）",
        must_have=["rdkit_2d", "morgan_fp"],
        recommended=["xtb_sp", "xtb_ml_derived"],
        optional=["3d_geometry", "fukui_approx"],
    ),
]


# ============================================================
# セレクターエンジン
# ============================================================

@dataclass
class FeatureSelectionResult:
    """特徴量選択の結果。"""
    selected_features: list[str]
    estimated_time_per_mol_s: float
    estimated_total_minutes: float
    requires_xtb: bool
    requires_opt: bool
    requires_freq: bool
    task_type: str
    budget_met: bool
    notes: list[str] = field(default_factory=list)


class AdaptiveFeatureSelector:
    """
    目的変数のタスク種別と計算予算に基づいて、
    最適な特徴量セットを自動選択するエンジン。

    使い方::

        selector = AdaptiveFeatureSelector()
        result = selector.select(
            task_type="solubility",
            n_molecules=200,
            max_time_per_mol_s=30,
        )
        print(result.selected_features)
        # → ["rdkit_2d", "morgan_fp", "xtb_sp", "xtb_ml_derived"]
    """

    def __init__(self) -> None:
        self._cost_catalog = {f.name: f for f in _FEATURE_COST_CATALOG}
        self._task_recommendations = {r.task_type: r for r in _TASK_RECOMMENDATIONS}

    @property
    def available_tasks(self) -> list[str]:
        """利用可能なタスク種別のリスト。"""
        return list(self._task_recommendations.keys())

    @property
    def available_features(self) -> list[str]:
        """利用可能な特徴量名のリスト。"""
        return list(self._cost_catalog.keys())

    def get_task_description(self, task_type: str) -> str | None:
        """タスク種別の日本語説明を返す。"""
        rec = self._task_recommendations.get(task_type)
        return rec.description_jp if rec else None

    def select(
        self,
        task_type: str = "general",
        n_molecules: int = 100,
        max_time_per_mol_s: float = 120.0,
        xtb_available: bool = True,
        force_include: list[str] | None = None,
        force_exclude: list[str] | None = None,
    ) -> FeatureSelectionResult:
        """
        計算予算とタスク種別に基づいて特徴量を選択する。

        Args:
            task_type: タスク種別（"solubility", "reactivity", 等）。
            n_molecules: データセットの分子数。
            max_time_per_mol_s: 1分子あたりの最大許容計算時間（秒）。
            xtb_available: xTBバイナリが利用可能か。
            force_include: 強制的に含める特徴量。
            force_exclude: 強制的に除外する特徴量。

        Returns:
            FeatureSelectionResult。
        """
        rec = self._task_recommendations.get(task_type)
        if rec is None:
            rec = self._task_recommendations["general"]
            logger.info("未知のタスク '%s' → 'general' にフォールバック", task_type)

        force_include = set(force_include or [])
        force_exclude = set(force_exclude or [])

        # 0.5. recommender.py の知識ベースから追加推奨を取得
        recommender_features = _get_recommender_features(task_type)
        if recommender_features:
            # rec.must_have / recommended / optional に統合
            # 既存の推奨と重複しないように追加
            existing = set(rec.must_have + rec.recommended + rec.optional)
            for f in recommender_features:
                if f not in existing and f not in force_exclude:
                    rec.recommended.append(f)
            logger.info(
                "recommender.py から %d 件の特徴量を追加推奨: %s",
                len(recommender_features), recommender_features[:5],
            )

        # 1. 候補リスト構築（must → recommended → optional の順）
        candidates_ordered = []
        for f_name in rec.must_have:
            if f_name not in force_exclude and f_name not in rec.exclude:
                candidates_ordered.append(("must", f_name))
        for f_name in rec.recommended:
            if f_name not in force_exclude and f_name not in rec.exclude:
                candidates_ordered.append(("recommended", f_name))
        for f_name in rec.optional:
            if f_name not in force_exclude and f_name not in rec.exclude:
                candidates_ordered.append(("optional", f_name))

        # force_include を候補の先頭に追加
        for f_name in force_include:
            if f_name in self._cost_catalog and f_name not in force_exclude:
                candidates_ordered.insert(0, ("forced", f_name))

        # 2. 予算制約内で貪欲法で選択
        selected: list[str] = []
        total_cost = 0.0
        notes: list[str] = []
        seen = set()

        for priority, f_name in candidates_ordered:
            if f_name in seen:
                continue
            seen.add(f_name)

            cost_info = self._cost_catalog.get(f_name)
            if cost_info is None:
                continue

            # xTB不要チェック
            if cost_info.requires_xtb and not xtb_available:
                notes.append(f"'{f_name}' はxTBが必要ですが利用不可のため除外")
                continue

            # 予算チェック
            new_cost = total_cost + cost_info.approx_time_per_mol_s
            if new_cost > max_time_per_mol_s and priority not in ("must", "forced"):
                notes.append(
                    f"'{f_name}' は予算超過のため除外 "
                    f"(+{cost_info.approx_time_per_mol_s:.0f}s → "
                    f"合計{new_cost:.0f}s > 上限{max_time_per_mol_s:.0f}s)"
                )
                continue

            selected.append(f_name)
            total_cost = new_cost

        # 3. 結果の集約
        requires_xtb = any(
            self._cost_catalog[f].requires_xtb
            for f in selected
            if f in self._cost_catalog
        )
        requires_opt = any(
            self._cost_catalog[f].requires_opt
            for f in selected
            if f in self._cost_catalog
        )
        requires_freq = any(
            self._cost_catalog[f].requires_freq
            for f in selected
            if f in self._cost_catalog
        )

        return FeatureSelectionResult(
            selected_features=selected,
            estimated_time_per_mol_s=total_cost,
            estimated_total_minutes=(total_cost * n_molecules) / 60.0,
            requires_xtb=requires_xtb,
            requires_opt=requires_opt,
            requires_freq=requires_freq,
            task_type=task_type,
            budget_met=total_cost <= max_time_per_mol_s,
            notes=notes,
        )

    def get_cost_summary(self) -> list[dict[str, str | float]]:
        """全特徴量のコストサマリーを返す（UI表示用）。"""
        return [
            {
                "name": f.name,
                "category": f.category,
                "time_per_mol_s": f.approx_time_per_mol_s,
                "requires_xtb": f.requires_xtb,
                "description": f.description_jp,
            }
            for f in _FEATURE_COST_CATALOG
        ]

    def select_with_correlation(
        self,
        df: "pd.DataFrame",
        target_column: str,
        task_type: str = "general",
        n_molecules: int = 100,
        max_time_per_mol_s: float = 120.0,
        xtb_available: bool = True,
        top_n_features: int | None = None,
        force_include: list[str] | None = None,
        force_exclude: list[str] | None = None,
        llm_advisor: Any | None = None,
    ) -> FeatureSelectionResult:
        """
        相関係数とタスクベースの推奨を組み合わせて特徴量を選択する。

        Args:
            df: 特徴量と目的変数を含むDataFrame。
            target_column: 目的変数の列名。
            task_type: タスク種別。
            n_molecules: データセットの分子数。
            max_time_per_mol_s: 1分子あたりの最大許容計算時間（秒）。
            xtb_available: xTBバイナリが利用可能か。
            top_n_features: 相関係数で選択する上位N件。
            force_include: 強制的に含める特徴量。
            force_exclude: 強制的に除外する特徴量。

        Returns:
            FeatureSelectionResult。
        """
        from backend.chem.correlation_selector import (
            CorrelationBasedSelector, CorrelationMethod, CorrelationResult,
        )

        notes: list[str] = []

        # 1. 相関係数ベースの選択
        corr_selector = CorrelationBasedSelector(method=CorrelationMethod.PEARSON)
        try:
            corr_result = corr_selector.compute_correlations(df, target_column=target_column)
            if corr_result.rankings:
                n = top_n_features or min(10, len(corr_result.rankings))
                corr_selected = corr_selector.select_top_features(corr_result, n_features=n)
                notes.append(
                    f"相関係数トップ{n}件: {', '.join(corr_selected[:5])}"
                    + (f" 他{len(corr_selected)-5}件" if len(corr_selected) > 5 else "")
                )
            else:
                corr_selected = []
                notes.append("相関係数計算: 有効な数値特徴量がないためスキップ")
        except Exception as e:
            corr_selected = []
            notes.append(f"相関係数計算でエラー: {e}")

        # 2. LLM推奨（オプション）
        llm_selected = []
        if llm_advisor is not None:
            try:
                llm_rec = llm_advisor.recommend(
                    df=df,
                    target_column=target_column,
                    task_type=task_type,
                    n_features=top_n_features or 10,
                )
                if llm_rec.feature_names:
                    llm_selected = llm_rec.feature_names
                    notes.append(
                        f"LLM推奨: {', '.join(llm_selected[:5])}"
                        + (f" 他{len(llm_selected)-5}件" if len(llm_selected) > 5 else "")
                        + (f" (確信度: {llm_rec.confidence:.1%})" if llm_rec.confidence else "")
                    )
                else:
                    notes.append("LLM推奨: 有効な推奨が得られませんでした")
            except Exception as e:
                notes.append(f"LLM推奨でエラー: {e}")

        # 3. タスクベースの選択（既存ロジック）
        task_result = self.select(
            task_type=task_type,
            n_molecules=n_molecules,
            max_time_per_mol_s=max_time_per_mol_s,
            xtb_available=xtb_available,
            force_include=force_include,
            force_exclude=force_exclude,
        )

        # 4. 統合: 相関上位 + LLM推奨 + タスク推奨の重複を除きながら統合
        integrated_selected = list(dict.fromkeys(
            corr_selected + llm_selected + task_result.selected_features
        ))

        # 3.5. top_n_features が指定されている場合は制限
        if top_n_features and len(integrated_selected) > top_n_features:
            integrated_selected = integrated_selected[:top_n_features]
            notes.append(f"統合結果を上位{top_n_features}件に制限(correlation)")

        # 4. 予算チェック（統合後の特徴量が予算内か）
        total_cost = 0.0
        budget_met = True
        for f_name in integrated_selected:
            cost_info = self._cost_catalog.get(f_name)
            if cost_info:
                total_cost += cost_info.approx_time_per_mol_s
                if total_cost > max_time_per_mol_s:
                    budget_met = False
                    notes.append(
                        f"統合後予算超過: {total_cost:.1f}s > {max_time_per_mol_s:.1f}s"
                    )
                    break

        integrated_result = FeatureSelectionResult(
            selected_features=integrated_selected,
            estimated_time_per_mol_s=total_cost,
            estimated_total_minutes=(total_cost * n_molecules) / 60.0,
            requires_xtb=any(
                self._cost_catalog[f].requires_xtb
                for f in integrated_selected
                if f in self._cost_catalog
            ),
            requires_opt=any(
                self._cost_catalog[f].requires_opt
                for f in integrated_selected
                if f in self._cost_catalog
            ),
            requires_freq=any(
                self._cost_catalog[f].requires_freq
                for f in integrated_selected
                if f in self._cost_catalog
            ),
            task_type=task_type,
            budget_met=budget_met,
            notes=task_result.notes + notes,
        )
        return integrated_result
