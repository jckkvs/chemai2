"""
backend/models/factory.py

機械学習モデルのファクトリーモジュール。
新モデルの追加はこのファイルの辞書への登録のみで完結する設計（Open/Closed原則）。
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV,
    ElasticNet, ElasticNetCV, BayesianRidge,
    HuberRegressor, TheilSenRegressor, RANSACRegressor,
    ARDRegression,
    LogisticRegression,
)
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    BaggingRegressor, BaggingClassifier,
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cross_decomposition import PLSRegression

from backend.utils.config import RANDOM_STATE
from backend.utils.optional_import import safe_import, is_available

logger = logging.getLogger(__name__)

# オプショナルライブラリ
_xgb = safe_import("xgboost", "xgboost")
_lgb = safe_import("lightgbm", "lightgbm")
_cat = safe_import("catboost", "catboost")


def _xgb_regressor(**kw: Any) -> Any:
    from xgboost import XGBRegressor  # type: ignore
    return XGBRegressor(random_state=RANDOM_STATE, eval_metric="rmse", **kw)


def _xgb_classifier(**kw: Any) -> Any:
    from xgboost import XGBClassifier  # type: ignore
    return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **kw)


def _lgb_regressor(**kw: Any) -> Any:
    from lightgbm import LGBMRegressor  # type: ignore
    return LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **kw)


def _lgb_classifier(**kw: Any) -> Any:
    from lightgbm import LGBMClassifier  # type: ignore
    return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **kw)


def _cat_regressor(**kw: Any) -> Any:
    from catboost import CatBoostRegressor  # type: ignore
    return CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, **kw)


def _cat_classifier(**kw: Any) -> Any:
    from catboost import CatBoostClassifier  # type: ignore
    return CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **kw)


# ============================================================
# 回帰モデルのレジストリ
# ============================================================
_REGRESSION_REGISTRY: dict[str, dict[str, Any]] = {
    "linear": {
        "name": "Linear Regression",
        "class": LinearRegression,
        "default_params": {},
        "available": True,
        "tags": ["linear", "interpretable"],
    },
    "ridge": {
        "name": "Ridge Regression",
        "class": Ridge,
        "default_params": {"alpha": 1.0, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized"],
    },
    "ridge_cv": {
        "name": "Ridge CV",
        "class": RidgeCV,
        "default_params": {},
        "available": True,
        "tags": ["linear", "regularized", "cv"],
    },
    "lasso": {
        "name": "Lasso",
        "class": Lasso,
        "default_params": {"alpha": 1.0, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized", "sparse"],
    },
    "lasso_cv": {
        "name": "Lasso CV",
        "class": LassoCV,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized", "cv"],
    },
    "elasticnet": {
        "name": "ElasticNet",
        "class": ElasticNet,
        "default_params": {"alpha": 1.0, "l1_ratio": 0.5, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized"],
    },
    "elasticnet_cv": {
        "name": "ElasticNet CV",
        "class": ElasticNetCV,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized", "cv"],
    },
    "bayesian_ridge": {
        "name": "Bayesian Ridge",
        "class": BayesianRidge,
        "default_params": {},
        "available": True,
        "tags": ["linear", "bayesian"],
    },
    "ard": {
        "name": "ARD Regression",
        "class": ARDRegression,
        "default_params": {},
        "available": True,
        "tags": ["linear", "bayesian", "sparse"],
    },
    "huber": {
        "name": "Huber Regressor",
        "class": HuberRegressor,
        "default_params": {},
        "available": True,
        "tags": ["linear", "robust"],
    },
    "theilsen": {
        "name": "TheilSen Regressor",
        "class": TheilSenRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "robust"],
    },
    "ransac": {
        "name": "RANSAC Regressor",
        "class": RANSACRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "robust"],
    },
    "svr_rbf": {
        "name": "SVR (RBF)",
        "class": SVR,
        "default_params": {"kernel": "rbf", "C": 1.0},
        "available": True,
        "tags": ["kernel", "nonlinear"],
    },
    "svr_linear": {
        "name": "SVR (Linear)",
        "class": SVR,
        "default_params": {"kernel": "linear", "C": 1.0},
        "available": True,
        "tags": ["kernel", "linear"],
    },
    "knn": {
        "name": "KNN Regressor",
        "class": KNeighborsRegressor,
        "default_params": {"n_neighbors": 5},
        "available": True,
        "tags": ["instance-based"],
    },
    "dt": {
        "name": "Decision Tree",
        "class": DecisionTreeRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["tree", "interpretable"],
    },
    "rf": {
        "name": "Random Forest",
        "class": RandomForestRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "et": {
        "name": "Extra Trees",
        "class": ExtraTreesRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "gbm": {
        "name": "Gradient Boosting",
        "class": GradientBoostingRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "hgbm": {
        "name": "HistGradientBoosting",
        "class": HistGradientBoostingRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting", "missing_ok"],
    },
    "adaboost": {
        "name": "AdaBoost",
        "class": AdaBoostRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "bagging": {
        "name": "Bagging Regressor",
        "class": BaggingRegressor,
        "default_params": {"n_estimators": 10, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble"],
    },
    "xgb": {
        "name": "XGBoost",
        "factory": _xgb_regressor,
        "default_params": {"n_estimators": 100},
        "available": bool(_xgb),
        "tags": ["ensemble", "boosting"],
    },
    "lgbm": {
        "name": "LightGBM",
        "factory": _lgb_regressor,
        "default_params": {"n_estimators": 100},
        "available": bool(_lgb),
        "tags": ["ensemble", "boosting"],
    },
    "catboost": {
        "name": "CatBoost",
        "factory": _cat_regressor,
        "default_params": {"iterations": 100},
        "available": bool(_cat),
        "tags": ["ensemble", "boosting"],
    },
    "mlp": {
        "name": "MLP Regressor",
        "class": MLPRegressor,
        "default_params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["neural_network"],
    },
    "gp": {
        "name": "Gaussian Process",
        "class": GaussianProcessRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["probabilistic"],
    },
    "pls": {
        "name": "PLS Regression",
        "class": PLSRegression,
        "default_params": {"n_components": 2},
        "available": True,
        "tags": ["linear", "dimensionality"],
    },
}

# ============================================================
# 分類モデルのレジストリ
# ============================================================
_CLASSIFICATION_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic": {
        "name": "Logistic Regression",
        "class": LogisticRegression,
        "default_params": {
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["linear", "interpretable"],
    },
    "svc_rbf": {
        "name": "SVC (RBF)",
        "class": SVC,
        "default_params": {
            "kernel": "rbf",
            "C": 1.0,
            "probability": True,
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["kernel", "nonlinear"],
    },
    "linearsvc": {
        "name": "LinearSVC",
        "class": LinearSVC,
        "default_params": {"C": 1.0, "max_iter": 2000, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear"],
    },
    "knn_c": {
        "name": "KNN Classifier",
        "class": KNeighborsClassifier,
        "default_params": {"n_neighbors": 5},
        "available": True,
        "tags": ["instance-based"],
    },
    "dt_c": {
        "name": "Decision Tree",
        "class": DecisionTreeClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["tree", "interpretable"],
    },
    "rf_c": {
        "name": "Random Forest",
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "et_c": {
        "name": "Extra Trees",
        "class": ExtraTreesClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "gbm_c": {
        "name": "Gradient Boosting",
        "class": GradientBoostingClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "hgbm_c": {
        "name": "HistGradientBoosting",
        "class": HistGradientBoostingClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting", "missing_ok"],
    },
    "adaboost_c": {
        "name": "AdaBoost",
        "class": AdaBoostClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "bagging_c": {
        "name": "Bagging Classifier",
        "class": BaggingClassifier,
        "default_params": {"n_estimators": 10, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble"],
    },
    "xgb_c": {
        "name": "XGBoost",
        "factory": _xgb_classifier,
        "default_params": {"n_estimators": 100},
        "available": bool(_xgb),
        "tags": ["ensemble", "boosting"],
    },
    "lgbm_c": {
        "name": "LightGBM",
        "factory": _lgb_classifier,
        "default_params": {"n_estimators": 100},
        "available": bool(_lgb),
        "tags": ["ensemble", "boosting"],
    },
    "catboost_c": {
        "name": "CatBoost",
        "factory": _cat_classifier,
        "default_params": {"iterations": 100},
        "available": bool(_cat),
        "tags": ["ensemble", "boosting"],
    },
    "mlp_c": {
        "name": "MLP Classifier",
        "class": MLPClassifier,
        "default_params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["neural_network"],
    },
    "gnb": {
        "name": "Gaussian NB",
        "class": GaussianNB,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic"],
    },
    "bnb": {
        "name": "Bernoulli NB",
        "class": BernoulliNB,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic"],
    },
    "gp_c": {
        "name": "Gaussian Process",
        "class": GaussianProcessClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["probabilistic"],
    },
}


# ============================================================
# Factory API
# ============================================================

def get_model(
    model_key: str,
    task: str = "regression",
    **override_params: Any,
) -> Any:
    """
    指定された model_key に対応する学習済みモデルインスタンスを返す。

    Args:
        model_key: モデルのキー（例: "rf", "xgb", "logistic"）
        task: "regression" or "classification"
        **override_params: デフォルトパラメータを上書きするキーワード引数

    Returns:
        sklearn互換の推定器インスタンス

    Raises:
        ValueError: 未知のmodel_key、またはライブラリ未インストール
    """
    registry = _get_registry(task)
    if model_key not in registry:
        raise ValueError(
            f"未知のモデルキー '{model_key}' (task={task})。"
            f"利用可能: {list(registry.keys())}"
        )

    entry = registry[model_key]
    if not entry.get("available", True):
        raise ValueError(
            f"モデル '{model_key}' ({entry['name']}) は現在の環境で"
            f"ライブラリがインストールされていません。"
        )

    params = {**entry.get("default_params", {}), **override_params}

    if "factory" in entry:
        return entry["factory"](**params)
    else:
        return entry["class"](**params)


def list_models(
    task: str = "regression",
    available_only: bool = True,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    利用可能なモデルの一覧を返す。

    Args:
        task: "regression" or "classification"
        available_only: True の場合はインストール済みのみ返す
        tags: 指定した場合、そのタグを持つモデルのみ返す

    Returns:
        [{key, name, tags, available}] のリスト
    """
    registry = _get_registry(task)
    results = []
    for key, entry in registry.items():
        if available_only and not entry.get("available", True):
            continue
        if tags:
            model_tags = entry.get("tags", [])
            if not any(t in model_tags for t in tags):
                continue
        results.append({
            "key": key,
            "name": entry["name"],
            "tags": entry.get("tags", []),
            "available": entry.get("available", True),
        })
    return results


def get_default_automl_models(task: str = "regression") -> list[str]:
    """
    AutoMLモードで使用するデフォルトモデルキーのリストを返す。
    使用可能なモデルから代表的なものを選択する。
    """
    regression_defaults = ["linear", "ridge", "lasso", "rf", "et", "gbm", "hgbm",
                           "xgb", "lgbm", "svr_rbf"]
    classification_defaults = ["logistic", "rf_c", "et_c", "gbm_c", "hgbm_c",
                                "xgb_c", "lgbm_c", "svc_rbf", "knn_c", "dt_c"]

    registry = _get_registry(task)
    defaults = regression_defaults if task == "regression" else classification_defaults
    return [k for k in defaults if k in registry and registry[k].get("available", True)]


def _get_registry(task: str) -> dict[str, dict[str, Any]]:
    """タスク種別に応じてレジストリを返す。"""
    if task == "regression":
        return _REGRESSION_REGISTRY
    elif task == "classification":
        return _CLASSIFICATION_REGISTRY
    else:
        raise ValueError(f"未知のタスク '{task}'。'regression' または 'classification' を指定してください。")
