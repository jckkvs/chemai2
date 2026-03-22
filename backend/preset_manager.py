"""
backend/preset_manager.py

パイプライン設定プリセットの保存・読込・一覧・削除。
YAML形式で ~/.chemai2/presets/ に保存する。
3フレームワーク共通で使えるバックエンド関数。

Implements: 設定プリセット保存/読込（UI課題レポート §5.5, §7.2）
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── デフォルト保存先 ──
DEFAULT_PRESET_DIR = Path.home() / ".chemai2" / "presets"

# ── 保存可能なstateキー（パイプライン設定に関するもののみ） ──
PIPELINE_KEYS = [
    # ユーザーモード
    "user_mode",
    # 列役割
    "task_type", "exclude_cols",
    # パイプライン: CV
    "cv_key", "cv_folds", "timeout",
    # パイプライン: 前処理
    "num_scaler", "num_imputer", "num_transform",
    "cat_encoder", "cat_imputer",
    # パイプライン: 特徴量
    "do_polynomial", "feature_selector",
    # パイプライン: モデル
    "selected_models", "model_params",
    "monotonic_constraints",
    # パイプライン: フラグ
    "do_eda", "do_prep", "do_eval", "do_pca", "do_shap",
]


def _ensure_dir(preset_dir: Path | None = None) -> Path:
    """プリセットディレクトリを確保して返す。"""
    d = preset_dir or DEFAULT_PRESET_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_preset(
    name: str,
    state: dict[str, Any],
    *,
    description: str = "",
    tags: list[str] | None = None,
    preset_dir: Path | None = None,
) -> Path:
    """パイプライン設定をYAMLプリセットとして保存する。

    Args:
        name: プリセット名（ファイル名になる）
        state: 共有ステート辞書
        description: 説明文
        tags: タグリスト
        preset_dir: 保存先ディレクトリ（デフォルト: ~/.chemai2/presets/）

    Returns:
        保存先パス

    Raises:
        ValueError: name が空
    """
    if not name or not name.strip():
        raise ValueError("プリセット名は必須です")

    try:
        import yaml
    except ImportError:
        # PyYAML がなければ JSON fallback
        import json
        d = _ensure_dir(preset_dir)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
        filepath = d / f"{safe_name}.json"
        config = {k: state.get(k) for k in PIPELINE_KEYS if state.get(k) is not None}
        data = {
            "name": name.strip(),
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "config": _make_serializable(config),
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("プリセット保存 (JSON): %s", filepath)
        return filepath

    d = _ensure_dir(preset_dir)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    filepath = d / f"{safe_name}.yaml"

    config = {k: state.get(k) for k in PIPELINE_KEYS if state.get(k) is not None}
    data = {
        "name": name.strip(),
        "description": description,
        "tags": tags or [],
        "created_at": datetime.now().isoformat(),
        "config": _make_serializable(config),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("プリセット保存: %s", filepath)
    return filepath


def load_preset(
    name: str,
    state: dict[str, Any],
    *,
    preset_dir: Path | None = None,
) -> dict[str, Any]:
    """YAMLプリセットを読み込んでstateに適用する。

    Args:
        name: プリセット名
        state: 書き込み先の共有ステート辞書
        preset_dir: 保存先ディレクトリ

    Returns:
        読み込んだプリセットのメタデータ

    Raises:
        FileNotFoundError: プリセットが見つからない
    """
    d = _ensure_dir(preset_dir)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())

    # YAML → JSON の順で探す
    filepath = None
    for ext in (".yaml", ".yml", ".json"):
        candidate = d / f"{safe_name}{ext}"
        if candidate.exists():
            filepath = candidate
            break

    if filepath is None:
        raise FileNotFoundError(f"プリセット '{name}' が見つかりません: {d}")

    if filepath.suffix == ".json":
        import json
        data = json.loads(filepath.read_text(encoding="utf-8"))
    else:
        import yaml
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    config = data.get("config", {})
    for k, v in config.items():
        if k in PIPELINE_KEYS:
            state[k] = v

    logger.info("プリセット読込: %s (%d件の設定)", name, len(config))
    return {
        "name": data.get("name", name),
        "description": data.get("description", ""),
        "tags": data.get("tags", []),
        "created_at": data.get("created_at", ""),
        "keys_loaded": list(config.keys()),
    }


def list_presets(
    *,
    preset_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """利用可能なプリセット一覧を返す。

    Returns:
        各プリセットの {name, description, tags, created_at, filepath} のリスト
    """
    d = _ensure_dir(preset_dir)
    results: list[dict[str, Any]] = []

    for filepath in sorted(d.iterdir()):
        if filepath.suffix not in (".yaml", ".yml", ".json"):
            continue
        try:
            if filepath.suffix == ".json":
                import json
                data = json.loads(filepath.read_text(encoding="utf-8"))
            else:
                import yaml
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

            results.append({
                "name": data.get("name", filepath.stem),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "created_at": data.get("created_at", ""),
                "filepath": str(filepath),
                "n_settings": len(data.get("config", {})),
            })
        except Exception as ex:
            logger.warning("プリセット読込エラー: %s - %s", filepath, ex)

    return results


def delete_preset(
    name: str,
    *,
    preset_dir: Path | None = None,
) -> bool:
    """プリセットを削除する。

    Returns:
        True if 削除成功, False if ファイルが見つからなかった
    """
    d = _ensure_dir(preset_dir)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())

    for ext in (".yaml", ".yml", ".json"):
        candidate = d / f"{safe_name}{ext}"
        if candidate.exists():
            candidate.unlink()
            logger.info("プリセット削除: %s", candidate)
            return True

    return False


def export_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """stateから設定サマリーを抽出する（表示用）。

    Returns:
        人間可読な設定サマリー辞書
    """
    summary: dict[str, Any] = {}
    for k in PIPELINE_KEYS:
        v = state.get(k)
        if v is not None and v != [] and v != {}:
            summary[k] = v
    return summary


def _make_serializable(obj: Any) -> Any:
    """YAML/JSONシリアライズ可能な形式に変換する。"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)
