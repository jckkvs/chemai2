"""
core/views.py
ChemAI ML Studio - Django ビュー
"""
from __future__ import annotations

import io
import json
import traceback
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# backendへのパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .models import AnalysisSession


# ─────────────────────────────────────────────
# ページビュー
# ─────────────────────────────────────────────
def index(request):
    """ダッシュボード：セッション一覧 + 新規作成"""
    sessions = AnalysisSession.objects.all()[:20]
    return render(request, "core/index.html", {"sessions": sessions})


def session_detail(request, session_id):
    """セッション詳細（ステップ別タブ構成）"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    step = request.GET.get("step", "data")
    return render(request, "core/session.html", {
        "session": session,
        "step": step,
    })


def new_session(request):
    """新規セッション作成"""
    session = AnalysisSession.objects.create(name="新しい解析")
    return redirect("session_detail", session_id=session.id)


# ─────────────────────────────────────────────
# API: データアップロード
# ─────────────────────────────────────────────
@csrf_exempt
@require_POST
def upload_data(request, session_id):
    """CSVファイルをアップロードしてセッションに紐付け"""
    session = get_object_or_404(AnalysisSession, id=session_id)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "ファイルが選択されていません"}, status=400)

    try:
        # ファイルを保存
        session.uploaded_file = uploaded_file
        session.original_filename = uploaded_file.name

        # DataFrameとして読み込み
        content = uploaded_file.read()
        uploaded_file.seek(0)

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            return JsonResponse({"error": "CSV/Excelファイルのみ対応"}, status=400)

        session.n_rows = len(df)
        session.n_cols = len(df.columns)
        session.status = "data_loaded"
        session.save()

        # プレビューデータ
        preview = df.head(10).to_dict(orient="records")
        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in columns}

        return JsonResponse({
            "success": True,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": columns,
            "dtypes": dtypes,
            "preview": preview,
        })

    except Exception as e:
        return JsonResponse({"error": str(e), "trace": traceback.format_exc()}, status=500)


@csrf_exempt
@require_POST
def set_columns(request, session_id):
    """目的変数・SMILES列を設定"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    data = json.loads(request.body)
    session.target_col = data.get("target_col", "")
    session.smiles_col = data.get("smiles_col", "")
    session.task_type = data.get("task_type", "regression")
    session.save()
    return JsonResponse({"success": True})


@csrf_exempt
@require_POST
def load_sample(request, session_id):
    """デバッグ用サンプルデータを読み込み"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    data = json.loads(request.body)
    sample_type = data.get("type", "regression")
    include_smiles = data.get("include_smiles", True)

    np.random.seed(42)
    n = 25

    if sample_type == "regression":
        if include_smiles:
            smiles_list = [
                "CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCCO", "CC=O",
                "c1ccc(O)cc1", "CC(=O)OC", "CCOC", "CCN",
                "CC(C)(C)O", "c1ccc(N)cc1", "OC(=O)c1ccccc1", "CCOCC",
                "CC(O)CC", "c1ccc(Cl)cc1", "CC(=O)N", "CCCCCO",
                "c1ccc(F)cc1", "CC(C)=O", "OCCO", "c1ccncc1",
                "CC(=O)CC", "CCCCO", "c1ccc(C)cc1"
            ]
            df = pd.DataFrame({
                "SMILES": smiles_list[:n],
                "target_value": np.random.randn(n) * 2 + 5,
            })
            session.smiles_col = "SMILES"
        else:
            df = pd.DataFrame({
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n) * 2,
                "target_value": np.random.randn(n) * 2 + 5,
            })
        session.target_col = "target_value"
        session.task_type = "regression"
    else:
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n) * 2,
            "target_class": np.random.choice(["A", "B"], n),
        })
        session.target_col = "target_class"
        session.task_type = "classification"

    # CSVとして保存
    csv_path = Path(f"media/uploads/sample_{session.id}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    session.uploaded_file = f"uploads/sample_{session.id}.csv"
    session.original_filename = f"sample_{sample_type}.csv"
    session.n_rows = len(df)
    session.n_cols = len(df.columns)
    session.status = "data_loaded"
    session.save()

    return JsonResponse({
        "success": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "preview": df.head(10).to_dict(orient="records"),
    })


# ─────────────────────────────────────────────
# API: 記述子計算
# ─────────────────────────────────────────────
@csrf_exempt
@require_POST
def calculate_descriptors(request, session_id):
    """全エンジンで記述子を自動計算"""
    session = get_object_or_404(AnalysisSession, id=session_id)

    if not session.smiles_col:
        return JsonResponse({"error": "SMILES列が設定されていません"}, status=400)

    try:
        # データ読み込み
        df = pd.read_csv(session.uploaded_file.path)
        smiles_list = df[session.smiles_col].tolist()

        from backend.chem import ADAPTER_REGISTRY, get_available_adapters
        available = get_available_adapters()

        results = {}
        all_dfs = []

        for name, adapter_cls in available.items():
            try:
                adapter = adapter_cls()
                desc_df = adapter.compute(smiles_list)
                if desc_df is not None and len(desc_df.columns) > 0:
                    all_dfs.append(desc_df)
                    results[name] = {
                        "n_descriptors": len(desc_df.columns),
                        "columns": list(desc_df.columns)[:10],
                    }
            except Exception as e:
                results[name] = {"error": str(e)}

        if all_dfs:
            combined = pd.concat(all_dfs, axis=1)
            # NaN列を除去
            combined = combined.dropna(axis=1, how="all")
            # 保存
            save_path = Path(f"media/descriptors/{session.id}.parquet")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(save_path)
            session.precalc_data_path = str(save_path)
            session.status = "descriptors_calculated"
            session.save()

            return JsonResponse({
                "success": True,
                "total_descriptors": len(combined.columns),
                "engines": results,
                "columns": list(combined.columns),
            })
        else:
            return JsonResponse({"error": "計算可能な記述子がありませんでした"}, status=500)

    except Exception as e:
        return JsonResponse({"error": str(e), "trace": traceback.format_exc()}, status=500)


def help_page(request):
    """ヘルプページ"""
    return render(request, "core/help.html")
