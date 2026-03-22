"""
frontend_nicegui/components/analysis_runner.py

解析実行コンポーネント: AutoMLEngine呼び出しとリアルタイム進捗表示。
run.io_bound で重い計算をバックグラウンドスレッドにオフロードし、
NiceGUI の WebSocket heartbeat をブロックしない。
"""
from __future__ import annotations

import logging
import queue
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui, run

logger = logging.getLogger(__name__)

# 解析ロック（二重実行防止）
_analysis_running = False


def _run_engine_sync(
    df_work: pd.DataFrame,
    target_col: str,
    smiles_col: str | None,
    group_col: str | None,
    task: str,
    model_keys: list[str] | None,
    cv_folds: int,
    timeout: int,
    selected_desc: list[str] | None,
    progress_queue: queue.Queue,
    *,
    cv_key: str = "auto",
    model_params: dict[str, dict] | None = None,
    preprocess_params: dict[str, Any] | None = None,
    monotonic_constraints: dict[str, int] | None = None,
) -> Any:
    """
    バックグラウンドスレッドで AutoMLEngine を実行する同期関数。

    進捗情報は progress_queue に送信される。
    NiceGUI の run.io_bound から呼ばれるため、
    この関数内で UI 操作を行ってはいけない。

    Implements: WebSocket切断修正 + パイプライン設定統合
    注意点: run.cpu_bound ではなく run.io_bound を使用。
            AutoMLEngine 内部で numpy/sklearn が GIL リリースするため、
            io_bound（スレッド）で十分かつ pickle 不要で簡潔。
    """
    from backend.models.automl import AutoMLEngine

    def progress_callback(step: int, total: int, msg: str) -> None:
        """進捗をキューに送信（スレッドセーフ）"""
        try:
            progress_queue.put_nowait(("progress", step, total, msg))
        except queue.Full:
            pass  # キューが満杯なら進捗をスキップ

    engine = AutoMLEngine(
        task=task,
        cv_folds=cv_folds,
        cv_key=cv_key,
        model_keys=model_keys if model_keys else None,
        model_params=model_params,
        preprocess_params=preprocess_params,
        timeout_seconds=timeout,
        progress_callback=progress_callback,
        selected_descriptors=selected_desc,
        monotonic_constraints_dict=monotonic_constraints,
    )

    result = engine.run(
        df_work,
        target_col=target_col,
        smiles_col=smiles_col,
        group_col=group_col,
    )

    return result


async def run_analysis(state: dict[str, Any], status_container, on_complete=None) -> None:
    """
    AutoML解析を実行し、結果をstateに保存する。

    重い計算は run.io_bound でバックグラウンドスレッドにオフロードし、
    NiceGUI の WebSocket heartbeat をブロックしない。
    進捗は queue.Queue + ui.timer でポーリング更新する。

    Args:
        state: 共有ステート辞書
        status_container: 進捗表示を描画するUIコンテナ
        on_complete: 完了時のコールバック
    """
    global _analysis_running

    df = state.get("df")
    target_col = state.get("target_col")

    if df is None or not target_col:
        ui.notify("データと目的変数を設定してください", type="warning")
        return

    if _analysis_running:
        ui.notify("⏳ 解析が既に実行中です", type="info")
        return

    _analysis_running = True

    # 進捗表示の構築
    status_container.clear()
    with status_container:
        with ui.card().classes("full-width glass-card q-pa-md q-mb-sm"):
            progress_header = ui.row().classes("items-center full-width justify-between")
            with progress_header:
                progress_label = ui.label("⏳ 解析を開始しています...").classes("text-lg")
                progress_pct = ui.label("").classes("text-h6 text-bold hero-gradient")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("q-mb-xs").props("color=cyan rounded")
            with ui.row().classes("justify-between full-width"):
                progress_detail = ui.label("").classes("text-caption text-grey-5")
                progress_eta = ui.label("").classes("text-caption text-grey-5")

    # 進捗キュー（スレッドセーフ）
    progress_queue: queue.Queue = queue.Queue(maxsize=100)
    _start_time = time.time()

    # 進捗ポーリングタイマー（最新のステップのみ表示、羅列しない）
    def _poll_progress():
        """キューから進捗情報を取得してUI更新（最新のみ）"""
        latest = None
        while True:
            try:
                item = progress_queue.get_nowait()
                if item[0] == "progress":
                    latest = item
            except queue.Empty:
                break
        if latest:
            _, step, total, msg = latest
            pct = step / total if total > 0 else 0
            progress_bar.value = pct
            progress_pct.text = f"{int(pct * 100)}%"
            progress_label.text = f"⏳ {msg}"
            progress_detail.text = f"ステップ {step}/{total}"
            # 推定残り時間
            elapsed = time.time() - _start_time
            if pct > 0.05:
                eta_sec = elapsed / pct * (1 - pct)
                if eta_sec < 60:
                    progress_eta.text = f"残り約{eta_sec:.0f}秒"
                else:
                    progress_eta.text = f"残り約{eta_sec/60:.1f}分"
            else:
                progress_eta.text = "推定中..."

    timer = ui.timer(0.5, _poll_progress)

    try:
        # タスク判定
        task = state.get("task_type", "auto")

        # モデル選択
        model_keys = state.get("selected_models")
        if not model_keys:
            from backend.models.factory import get_default_automl_models
            effective_task = task
            if effective_task == "auto":
                effective_task = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
            model_keys = get_default_automl_models(task=effective_task)

        # SMILES列
        smiles_col = state.get("smiles_col") or None

        # 除外列の処理
        exclude_cols = state.get("exclude_cols", [])
        df_work = df.copy()
        if exclude_cols:
            df_work = df_work.drop(columns=[c for c in exclude_cols if c in df_work.columns], errors="ignore")

        # ── パイプライン設定をstateから抽出 ──
        preprocess_params = {}
        for key in [
            "num_scaler", "num_imputer", "num_transform",
            "cat_encoder", "cat_imputer",
            "feature_selector", "n_features_to_select",
            "do_polynomial", "poly_degree", "poly_interaction_only",
        ]:
            if key in state:
                preprocess_params[key] = state[key]

        model_params = state.get("model_params") or None
        mono_raw = state.get("monotonic_constraints", {})
        monotonic_constraints = {k: v for k, v in mono_raw.items() if v != 0} or None
        cv_key = state.get("cv_key", "auto")

        # ══════════════════════════════════════════════════════
        # 複数セット対応: active=True のセットをループ
        # ══════════════════════════════════════════════════════
        desc_sets = state.get("descriptor_sets", {})
        active_sets = {
            name: info for name, info in desc_sets.items()
            if info.get("active", True)
        }

        # activeセットがなければ現在の選択で1回実行
        if not active_sets:
            active_sets = {"デフォルト": {"descriptors": state.get("selected_descriptors")}}

        all_results: dict[str, Any] = {}
        best_result = None
        best_set_name = ""
        best_score = -float("inf")
        total_sets = len(active_sets)

        for set_idx, (set_name, set_info) in enumerate(active_sets.items()):
            set_descs = set_info.get("descriptors")
            # None = 全記述子
            selected_desc = list(set_descs) if set_descs else state.get("selected_descriptors")

            # 進捗更新
            progress_label.text = f"⏳ [{set_idx + 1}/{total_sets}] セット「{set_name}」を解析中..."
            progress_bar.value = set_idx / total_sets
            progress_pct.text = f"{int(set_idx / total_sets * 100)}%"

            set_queue: queue.Queue = queue.Queue(maxsize=100)

            # セット固有の進捗ポーリング
            def _poll_set_progress(sq=set_queue, sn=set_name, si=set_idx):
                latest = None
                while True:
                    try:
                        item = sq.get_nowait()
                        if item[0] == "progress":
                            latest = item
                    except queue.Empty:
                        break
                if latest:
                    _, step, total, msg = latest
                    set_pct = (si + step / max(total, 1)) / total_sets
                    progress_bar.value = set_pct
                    progress_pct.text = f"{int(set_pct * 100)}%"
                    progress_label.text = f"⏳ [{si + 1}/{total_sets}] {sn}: {msg}"
                    progress_detail.text = f"セット {si + 1}/{total_sets} | ステップ {step}/{total}"
                    elapsed = time.time() - _start_time
                    if set_pct > 0.05:
                        eta_sec = elapsed / set_pct * (1 - set_pct)
                        progress_eta.text = f"残り約{eta_sec:.0f}秒" if eta_sec < 60 else f"残り約{eta_sec/60:.1f}分"

            set_timer = ui.timer(0.5, _poll_set_progress)

            try:
                result = await run.io_bound(
                    _run_engine_sync,
                    df_work,
                    target_col,
                    smiles_col if smiles_col and smiles_col in df_work.columns else None,
                    state.get("group_col"),
                    task,
                    model_keys if model_keys else None,
                    state.get("cv_folds", 5),
                    state.get("timeout", 300),
                    selected_desc,
                    set_queue,
                    cv_key=cv_key,
                    model_params=model_params,
                    preprocess_params=preprocess_params if preprocess_params else None,
                    monotonic_constraints=monotonic_constraints,
                )
                all_results[set_name] = result

                # ベストスコア追跡
                if hasattr(result, "best_score") and result.best_score > best_score:
                    best_score = result.best_score
                    best_result = result
                    best_set_name = set_name

            except Exception as set_ex:
                logger.warning(f"セット「{set_name}」の解析エラー: {set_ex}")
                all_results[set_name] = None
            finally:
                set_timer.deactivate()

        # ── 結果の保存 ──
        state["automl_results"] = all_results  # 全セット結果
        # 後方互換: 最良セットを automl_result にも保存
        if best_result:
            state["automl_result"] = best_result
            state["best_set_name"] = best_set_name
            state["pipeline_result"] = type("PipelineResult", (), {"elapsed": best_result.elapsed_seconds})()

        # 成功表示
        elapsed_total = time.time() - _start_time
        progress_bar.value = 1.0
        progress_pct.text = "100%"

        n_success = sum(1 for v in all_results.values() if v is not None)
        if best_result:
            progress_label.text = (
                f"✅ 解析完了！ {n_success}/{total_sets}セット成功 | "
                f"最良: {best_set_name} → {best_result.best_model_key}"
            )
            n_models = len(best_result.model_scores)
            proc_X = getattr(best_result, "processed_X", None)
            n_feats = proc_X.shape[1] if proc_X is not None and hasattr(proc_X, "shape") else "?"
            progress_detail.text = (
                f"スコア: {best_result.best_score:.4f} | "
                f"所要時間: {elapsed_total:.1f}秒 | "
                f"{n_models}モデル比較 | {n_feats}特徴量"
            )
            progress_eta.text = f"タスク: {best_result.task}"
            ui.notify(
                f"✅ {n_success}セット解析完了！ 最良: {best_set_name} ({best_result.best_score:.4f})",
                type="positive",
                timeout=5000,
            )
        else:
            progress_label.text = "❌ 全セットの解析に失敗しました"
            progress_detail.text = "設定を確認して再実行してください"

        if on_complete:
            on_complete()

        # 解析履歴を自動記録
        try:
            from backend.preset_manager import record_analysis
            if best_result:
                record_analysis(state, best_result)
        except Exception as hist_ex:
            logger.warning("解析履歴の保存に失敗: %s", hist_ex)

    except Exception as ex:
        error_msg = str(ex)
        tb_text = traceback.format_exc()
        short_msg = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg

        # エラー種別に応じた対処法
        remedy = _get_error_remedy(error_msg, tb_text)

        progress_pct.text = "❌"
        progress_label.text = "❌ エラーが発生しました"
        progress_detail.text = short_msg
        progress_eta.text = ""

        # 対処法パネル
        with status_container:
            with ui.card().classes("full-width q-pa-sm q-mt-sm").style(
                "border: 1px solid rgba(248,113,113,0.3); border-radius: 8px; background: rgba(60,20,20,0.2);"
            ):
                ui.label("💡 対処法").classes("text-subtitle2 text-bold text-amber")
                ui.label(remedy).classes("text-caption")
                with ui.expansion("🔍 詳細エラー情報", icon="bug_report").classes("full-width q-mt-xs"):
                    ui.code(tb_text[-1000:]).classes("full-width").style("font-size: 0.7rem;")

        ui.notify(f"解析エラー: {short_msg}", type="negative", timeout=8000)
        logger.error(f"AutoML実行エラー: {tb_text}")

    finally:
        _analysis_running = False
        timer.deactivate()


def _get_error_remedy(error_msg: str, tb_text: str) -> str:
    """エラーメッセージからユーザー向けの対処法を生成する。"""
    msg_lower = error_msg.lower()
    tb_lower = tb_text.lower()

    if "memory" in msg_lower or "memoryerror" in tb_lower:
        return (
            "メモリ不足です。以下を試してください:\n"
            "• データの行数を減らす（サンプリング）\n"
            "• 記述子エンジンを少なくする\n"
            "• 使わない列を「除外」に設定する"
        )
    elif "smiles" in msg_lower or "rdkit" in tb_lower or "mol" in msg_lower:
        return (
            "SMILES列の解析に失敗しました:\n"
            "• SMILES列に無効な分子構造が含まれている可能性があります\n"
            "• SMILES列を「なし」に設定して再解析するか、データを確認してください"
        )
    elif "target" in msg_lower and ("not found" in msg_lower or "見つかりません" in msg_lower):
        return (
            "目的変数が見つかりません:\n"
            "• 「列の役割」タブで目的変数が正しく設定されているか確認してください"
        )
    elif "timeout" in msg_lower or "timed out" in msg_lower:
        return (
            "解析がタイムアウトしました:\n"
            "• タイムアウト値を増やしてください（パイプライン設定）\n"
            "• モデル数を減らすと高速化できます"
        )
    elif "fit" in msg_lower or "convergence" in tb_lower:
        return (
            "モデルの学習に失敗しました:\n"
            "• データに欠損値やInfが多い可能性があります — EDAタブで確認\n"
            "• スケーラーを「robust」に変更してみてください"
        )
    elif "nan" in msg_lower or "inf" in msg_lower or "missing" in msg_lower:
        return (
            "データに NaN/Inf が含まれています:\n"
            "• EDAタブの「欠損行削除」や「高欠損列削除」を試してください\n"
            "• 前処理の欠損値補完方法を変更してみてください"
        )
    elif "shape" in msg_lower or "dimension" in msg_lower:
        return (
            "データの次元に問題があります:\n"
            "• 全行が同じ値の定数列があれば「除外」してください\n"
            "• 特徴量数がサンプル数より大きい場合、特徴量選択を有効にしてください"
        )
    else:
        return (
            "予期しないエラーが発生しました:\n"
            "• データの形式やサイズを確認してください\n"
            "• 詳細エラー情報（下）を展開して原因を確認してください\n"
            "• 問題が解決しない場合は、設定を変更して再試行してください"
        )
