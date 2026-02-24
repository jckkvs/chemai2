"""
backend/models/tuner.py

ハイパーパラメータ最適化モジュール。
GridSearch, RandomSearch, HalvingSearch, Optuna, BayesSearchCV に対応。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)

from backend.utils.config import RANDOM_STATE
from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

_optuna = safe_import("optuna", "optuna")
_skopt = safe_import("skopt", "scikit-optimize")

# HalvingSearchCV (sklearn 0.24+)
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    _halving_available = True
except ImportError:
    _halving_available = False


@dataclass
class TunerConfig:
    """チューニング設定。"""
    method: str = "random"
    # "grid" | "random" | "halving_grid" | "halving_random" | "optuna" | "bayes"
    param_grid: dict[str, Any] = field(default_factory=dict)
    n_iter: int = 50              # RandomSearch / Optuna の試行回数
    cv: int = 5                   # 内部CV folds
    scoring: str = "neg_root_mean_squared_error"
    n_jobs: int = -1
    verbose: int = 0
    refit: bool = True
    optuna_direction: str = "maximize"
    optuna_timeout: int | None = None
    random_state: int = RANDOM_STATE


def tune(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    config: TunerConfig,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    指定された手法でハイパーパラメータを最適化して結果を返す。

    Args:
        model: sklearn互換の推定器
        X: 特徴量行列
        y: 目的変数
        config: TunerConfig インスタンス
        groups: グループラベル（GroupKFold等で必要）

    Returns:
        {
            "best_estimator": 最良モデル,
            "best_params": 最良パラメータ,
            "best_score": 最良スコア,
            "cv_results": 全試行の結果（DataFrameに変換可能）
        }
    """
    method = config.method.lower()

    if method == "grid":
        return _run_grid(model, X, y, config, groups)
    elif method == "random":
        return _run_random(model, X, y, config, groups)
    elif method == "halving_grid":
        return _run_halving_grid(model, X, y, config, groups)
    elif method == "halving_random":
        return _run_halving_random(model, X, y, config, groups)
    elif method == "optuna":
        return _run_optuna(model, X, y, config, groups)
    elif method == "bayes":
        return _run_bayes(model, X, y, config, groups)
    else:
        raise ValueError(
            f"未知のチューニング手法 '{method}'。"
            f"利用可能: grid, random, halving_grid, halving_random, optuna, bayes"
        )


def _run_grid(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """GridSearchCV による全探索。"""
    gs = GridSearchCV(
        model,
        cfg.param_grid,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
    )
    gs.fit(X, y, groups=groups)
    return _extract_results(gs)


def _run_random(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """RandomizedSearchCV によるランダム探索。"""
    rs = RandomizedSearchCV(
        model,
        cfg.param_grid,
        n_iter=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    rs.fit(X, y, groups=groups)
    return _extract_results(rs)


def _run_halving_grid(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """HalvingGridSearchCV による段階的全探索。"""
    if not _halving_available:
        logger.warning("HalvingGridSearchCV 未対応 → GridSearchCVで代替")
        return _run_grid(model, X, y, cfg, groups)
    hs = HalvingGridSearchCV(
        model,
        cfg.param_grid,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    hs.fit(X, y, groups=groups)
    return _extract_results(hs)


def _run_halving_random(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """HalvingRandomSearchCV による段階的ランダム探索。"""
    if not _halving_available:
        logger.warning("HalvingRandomSearchCV 未対応 → RandomizedSearchCVで代替")
        return _run_random(model, X, y, cfg, groups)
    hs = HalvingRandomSearchCV(
        model,
        cfg.param_grid,
        n_candidates=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    hs.fit(X, y, groups=groups)
    return _extract_results(hs)


def _run_optuna(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """Optuna によるベイズ最適化。param_grid は optuna の suggest形式で指定。"""
    if not _optuna:
        logger.warning("optuna 未インストール → RandomizedSearchCVで代替")
        return _run_random(model, X, y, cfg, groups)

    import optuna  # type: ignore
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        params: dict[str, Any] = {}
        for param_name, spec in cfg.param_grid.items():
            if isinstance(spec, dict):
                suggest_type = spec.get("type", "float")
                if suggest_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, spec["low"], spec["high"],
                        step=spec.get("step", 1)
                    )
                elif suggest_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, spec["low"], spec["high"],
                        log=spec.get("log", False)
                    )
                elif suggest_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, spec["choices"]
                    )
            elif isinstance(spec, list):
                params[param_name] = trial.suggest_categorical(param_name, spec)

        m = model.__class__(**{**model.get_params(), **params})
        scores = cross_val_score(
            m, X, y,
            cv=cfg.cv,
            scoring=cfg.scoring,
            n_jobs=cfg.n_jobs,
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=cfg.optuna_direction,
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
    )
    study.optimize(objective, n_trials=cfg.n_iter, timeout=cfg.optuna_timeout)

    best_params = study.best_params
    best_model = model.__class__(**{**model.get_params(), **best_params})
    if cfg.refit:
        best_model.fit(X, y)

    return {
        "best_estimator": best_model,
        "best_params": best_params,
        "best_score": study.best_value,
        "cv_results": study.trials_dataframe(),
    }


def _run_bayes(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """BayesSearchCV (scikit-optimize) によるベイズ最適化。"""
    if not _skopt:
        logger.warning("scikit-optimize 未インストール → RandomizedSearchCVで代替")
        return _run_random(model, X, y, cfg, groups)

    from skopt import BayesSearchCV  # type: ignore

    bs = BayesSearchCV(
        model,
        cfg.param_grid,
        n_iter=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    bs.fit(X, y, groups=groups)
    return _extract_results(bs)


def _extract_results(search_obj: Any) -> dict[str, Any]:
    """sklearn Search系オブジェクトから結果を抽出する。"""
    import pandas as pd
    return {
        "best_estimator": search_obj.best_estimator_,
        "best_params": search_obj.best_params_,
        "best_score": float(search_obj.best_score_),
        "cv_results": pd.DataFrame(search_obj.cv_results_),
    }
