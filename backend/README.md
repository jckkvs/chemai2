# Backend Module

`backend/` は ChemAI ML Studio のコアエンジンを含みます。すべてのフロントエンド（Streamlit / NiceGUI / Django）から共通で利用されます。

---

## モジュール構成

### 1. Data (`backend/data/`)

データの読み込み・前処理・探索的データ分析を担当します。

| ファイル | 説明 |
|---------|------|
| `loader.py` | CSV / Excel / Parquet / JSON / SQLite / SDF / MOL からのデータ読み込み・保存 |
| `type_detector.py` | 各列の変数型（数値/カテゴリ/バイナリ/定数/SMILES/日時/テキスト/周期）を自動判定 |
| `preprocessor.py` | `ColumnTransformer` を利用した変数型別前処理パイプライン自動構築 |
| `feature_engineer.py` | 交互作用特徴量 / グループ集約 / 日時特徴量抽出 / ラグ・ローリング特徴量 |
| `data_cleaner.py` | 列削除 / 欠損行除去 / 定数列除去 / 外れ値クリッピング / 重複行除去 |
| `eda.py` | 基本統計量 / 相関分析 / 外れ値検出 / 分布可視化 / 目的変数分析 |
| `dim_reduction.py` | 次元削減（PCA / t-SNE / UMAP） |
| `leakage_detector.py` | 学習データ・テストデータ間のリーケージリスク検出 |
| `benchmark.py` | モデルベンチマーク・評価指標計算 |
| `benchmark_datasets.py` | ESOL / FreeSolv / Lipophilicity のダウンロード・キャッシュ |

### 2. Models (`backend/models/`)

機械学習モデルの構築・最適化・評価を担当します。

| ファイル | 説明 |
|---------|------|
| `factory.py` | モデルレジストリ（Ridge / RF / XGBoost / LightGBM / CatBoost / SVR 等のファクトリ） |
| `automl.py` | ワンボタンAutoML実行エンジン（タスク自動判定→前処理→全モデル比較→最適選択） |
| `tuner.py` | Optuna によるハイパーパラメータ最適化 |
| `cv_manager.py` | CV戦略管理（KFold / Stratified / Group / WalkForward / Repeated） |
| `cv_bias_evaluator.py` | Tibshirani / BBC-CV によるCV偏りバイアス補正 |
| `linear_tree.py` | LinearTree / LinearForest / LinearBoost（フルスクラッチ実装） |
| `rgf.py` | Regularized Greedy Forest（フルスクラッチ実装） |
| `monotonic_kernel.py` | カーネルモデルへのソフト単調性制約ラッパー |

### 3. Pipeline (`backend/pipeline/`)

scikit-learn Pipeline の組立・グリッド探索を担当します。

| ファイル | 説明 |
|---------|------|
| `pipeline_builder.py` | 入力列選択→前処理→特徴量生成→特徴量選択→推定器 の5段Pipeline構築 |
| `pipeline_grid.py` | 各ステップの複数候補からデカルト積でPipeline候補を生成 |
| `column_selector.py` | mlxtend ラッパー、列メタ情報（単調性/グループ）管理 |
| `col_preprocessor.py` | 変数型ルール別の前処理Transformer |
| `feature_generator.py` | 多項式・交互作用特徴量生成Transformer |
| `feature_selector.py` | Lasso / RF / SelectKBest / Boruta / ReliefF 等の特徴量選択 |

### 4. Interpret (`backend/interpret/`)

モデル解釈・説明性を担当します。

| ファイル | 説明 |
|---------|------|
| `shap_explainer.py` | SHAP（Tree / Linear / Kernel / Deep）+ 各種プロット |
| `sri.py` | SHAP SRI分解（Synergy / Redundancy / Independence） |

### 5. Chem (`backend/chem/`)

化学情報学に特化した記述子計算を担当します。

| ファイル | 説明 |
|---------|------|
| `base.py` | `BaseChemAdapter` 抽象基底クラス |
| `rdkit_adapter.py` | RDKit 記述子（200種類+ / フィンガープリント） |
| `xtb_adapter.py` | GFN2-xTB 量子化学記述子（HOMO/LUMO/双極子モーメント等） |
| `mordred_adapter.py` | Mordred 1,800+記述子 |
| `uma_adapter.py` | Meta UMA (fairchem) 学習済み分子表現 |
| `cosmo_adapter.py` | COSMO-RS 溶媒和エネルギー |
| `unipka_adapter.py` | UniPKa pKa/LogD予測 |
| `group_contrib_adapter.py` | 基団寄与法 |
| `molai_adapter.py` | CNN+PCA 潜在ベクトル |
| `smiles_transformer.py` | SMILES→記述子DataFrame変換パイプライン |
| `charge_config.py` | 分子電荷・スピン多重度設定 |
| `protonation.py` | pH依存プロトン化状態の適用 |

### 6. Optim (`backend/optim/`)

ベイズ最適化・探索空間定義を担当します。

| ファイル | 説明 |
|---------|------|
| `bayesian_optimizer.py` | ガウス過程ベースのベイズ最適化エンジン |
| `search_space.py` | 連続/離散/カテゴリ変数の探索空間定義 |
| `constraints.py` | 制約条件管理（線形/非線形/範囲制約） |
| `bo_visualizer.py` | 最適化履歴・獲得関数の可視化 |

### 7. Utils (`backend/utils/`)

共通ユーティリティを提供します。

| ファイル | 説明 |
|---------|------|
| `config.py` | グローバル設定（RANDOM_STATE / パス / AutoML / SHAP / MLflow） |
| `optional_import.py` | 安全import（未インストールライブラリのフォールバック） |
| `param_schema.py` | ハイパーパラメータスキーマ定義 |

---

## 主要アルゴリズム

### 次元削減
- **PCA**: 線形な分散最大化方向への投影
- **t-SNE**: 高次元の局所的な構造を維持する非線形埋め込み
- **UMAP**: 局所構造と大局構造のバランスに優れた高速非線形埋め込み

### フルスクラッチ実装モデル
- **LinearTree / LinearForest / LinearBoost**: 葉にリニアモデルを持つ決定木系アンサンブル
- **RGF (Regularized Greedy Forest)**: 正則化付き貪欲森林（L1/L2正則化対応）

### CV偏りバイアス補正
- **Tibshirani法**: 楽観バイアスの推定
- **BBC-CV (Bootstrap Bias Corrected CV)**: ブートストラップによる偏り補正

---

## FastAPI エンドポイント

### データ読み込み・前処理

**POST `/api/upload`**
- ファイル（CSV/Excel）をアップロード、自動型検出
- レスポンス: `{"data": DataFrame, "meta": {"filename", "shape", "dtypes"}}`

**POST `/api/detect-types`**
```python
{
  "data": DataFrame,
}
# レスポンス: {"types": {"col1": "numeric", "col2": "category", ...}}
```

**POST `/api/preprocess`**
```python
{
  "data": DataFrame,
  "target_col": "y",
  "task": "regression|classification"  # 自動判定
}
# レスポンス: {"preprocessed_data": DataFrame, "pipeline": ColumnTransformer}
```

**POST `/api/clean-data`**
```python
{
  "data": DataFrame,
  "remove_duplicates": true,
  "remove_constants": true,
  "handle_missing": "drop|mean|forward_fill"
}
```

### AutoML エンドポイント

**POST `/api/automl/run`**
```python
{
  "data": DataFrame,
  "target_col": "y",
  "task": "auto|regression|classification",
  "cv_folds": 5,
  "n_trials": 20,
  "metric": "rmse|r2|accuracy|f1",
  "models": ["rf", "xgboost", "lightgbm", "catboost"]
}
# レスポンス: {
#   "best_model": EstimatorPipeline,
#   "best_score": 0.95,
#   "results": DataFrame with scores,
#   "feature_importance": Series
# }
```

**POST `/api/automl/predict`**
```python
{
  "model": Pipeline,
  "X": DataFrame
}
# レスポンス: {"predictions": ndarray, "probabilities": ndarray (分類時)}
```

### 解釈性・説明エンドポイント

**POST `/api/shap/explain`**
```python
{
  "model": Pipeline,
  "X": DataFrame,
  "explainer_type": "tree|linear|kernel"
}
# レスポンス: {
#   "shap_values": ndarray,
#   "feature_importance": Series,
#   "base_value": float
# }
```

**POST `/api/visualize/dimension-reduction`**
```python
{
  "data": DataFrame,
  "method": "pca|tsne|umap",
  "n_components": 2,
  "color_column": "optional"
}
# レスポンス: {"embedding": ndarray (n, 2), "explained_variance": float (PCA時)}
```

**POST `/api/visualize/correlation`**
```python
{
  "data": DataFrame
}
# レスポンス: {"corr_matrix": DataFrame}
```

### 化学記述子エンドポイント

**POST `/api/chem/descriptors`**
```python
{
  "smiles_list": ["CCO", "CC(C)O", ...],
  "descriptor_set": "rdkit|mordred|uma|xtb|cosmo|all"
}
# レスポンス: {"descriptors": DataFrame}
```

---

## データフロー

### 1️⃣ データアップロード～前処理フロー

```
ユーザー (NiceGUI)
    ↓
[FileUploadZone] → CSV/Excel アップロード
    ↓
backend/data/loader.py → DataLoader
    ├─ データ読み込み
    ├─ 型自動判定 (TypeDetector)
    └─ メタデータ抽出 (shape, dtypes, missing%)
    ↓
backend/data/eda.py → EDAAnalyzer
    ├─ 基本統計量計算
    ├─ 外れ値検出
    ├─ 相関分析
    └─ 品質評価スコア
    ↓
frontend: データ品質表示 + クリーニング提案
```

### 2️⃣ AutoML実行フロー

```
[AutoMLページ] (ユーザー設定)
    ├─ target_col: 目的変数
    ├─ task: 回帰/分類 (自動判定)
    ├─ cv_folds: 5（デフォルト）
    ├─ n_trials: 20（デフォルト）
    └─ metrics: RMSE/R2/Accuracy/F1
    ↓
backend/models/automl.py → AutoMLEngine
    ├─ 1. DataLoader: 前処理パイプライン構築
    ├─ 2. ColumnTransformer: 型別変数処理
    ├─ 3. PipelineBuilder: 5段Pipeline生成
    ├─ 4. BayesianOptimizer: Optuna ハイパーパラメータ最適化
    ├─ 5. CVManager: KFold/Stratified交差検証
    └─ 6. ModelFactory: RF/XGBoost/LightGBM/CatBoost 評価
    ↓
backend/interpret/shap_explainer.py → ShapExplainer
    ├─ SHAP値計算（TreeExplainer）
    ├─ 特徴量重要度算出
    └─ Dependence プロット作成
    ↓
frontend: 結果表示
    ├─ 最良スコア表示
    ├─ SHAP特徴量重要度グラフ
    ├─ モデル保存 (joblib)
    └─ PDF レポート出力
```

### 3️⃣ LLM Assistant フロー

```
[LLMアシスタント] (チャット入力)
    ↓
モード選択:
    ├─ 外部LLM向けプロンプト生成
    │   └─ システムコンテキスト + 分析履歴 → マークダウン形式プロンプト → コピー
    │
    └─ ローカルLLM (Ollama) チャット
        └─ backend/llm/engine.py → LLMEngine.stream_chat()
            ├─ Ollama API 呼び出し (http://localhost:11434)
            ├─ ストリーミング応答
            └─ ui.markdown でリアルタイム表示
```

### 4️⃣ データクリーニングフロー

```
[データアップロードページ]
    ├─ ファイルアップロード
    └─ [クリーニング分析] ボタン
        ↓
backend/data/data_cleaner.py → DataCleanerLLM
    ├─ 自動検出: 欠損値/外れ値/重複/定数列/型エラー
    ├─ LLM生成: 修正提案コード
    └─ 信頼度スコア: 0.0-1.0
    ↓
frontend: 提案ダイアログ表示
    └─ ユーザー: コード確認 → コピー → 自分の環境で実行
```

---

## 設定・環境変数

`.env` ファイルで以下を指定：

```bash
# データベース
DATABASE_URL=sqlite:///./chemiai.db
# DATABASE_URL=postgresql://user:pass@localhost:5432/chemiai

# Redis（キャッシュ）
REDIS_URL=redis://localhost:6379/0

# LLM 設定
LLM_MODE=prompt_only|local|api
LLM_API_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

# ローカルLLM (Ollama)
OLLAMA_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama2

# AutoML設定
AUTOML_DEFAULT_CV_FOLDS=5
AUTOML_DEFAULT_N_TRIALS=20
AUTOML_TIMEOUT_SECONDS=3600

# ストレージ
DATA_DIR=./data
EXPORT_DIR=./exports
CACHE_DIR=./.cache
UPLOAD_MAX_SIZE_MB=50

# MLflow
MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
MLFLOW_EXPERIMENT_NAME=ChemAI-Default
```

---

## テスト・開発

### テスト実行

```bash
# すべてのテスト
python -m pytest tests/ -v --tb=short

# 特定モジュールのテスト
python -m pytest tests/test_automl.py -v

# カバレッジ測定
python -m pytest tests/ --cov=backend --cov-report=html
```

### ローカルサーバー起動

```bash
# FastAPI バックエンド（ポート 8000）
python -m uvicorn backend.api.main:app --reload

# API ドキュメント: http://localhost:8000/docs
```

### デバッグ

```python
# ログレベル設定（backend/utils/config.py）
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.debug("デバッグメッセージ")
```

---

## デプロイメント

### Docker コンテナ化

```bash
# バックエンド Dockerfile
docker build -f backend_fastapi/Dockerfile -t chemai-backend .

# コンテナ起動
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/chemiai \
  -e REDIS_URL=redis://redis:6379 \
  chemai-backend
```

### Docker Compose

```bash
# 開発環境
docker-compose up

# 本番環境
docker-compose -f docker-compose.prod.yml up -d
```

### NGINX リバースプロキシ

```nginx
upstream backend {
    server backend_fastapi:8000;
}

server {
    listen 443 ssl http2;
    server_name example.com;
    
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;  # AutoML長時間対応
    }
}
```

詳細: `infra/nginx/prod.conf`

---

## パフォーマンス最適化

### 遅延インポート
重いライブラリ（RDKit, Mordred, UMAP 等）は必要時のみインポート：

```python
from backend.utils.optional_import import import_optional

torch = import_optional("torch")
if torch is None:
    logger.warning("PyTorch not installed, deep learning models unavailable")
```

### キャッシング戦略
- 型検出結果: Redis キャッシュ（24時間）
- 化学記述子: ローカルファイルキャッシュ（更新日ベース）
- SHAP値: MLflow にトラッキング

### マルチプロセッシング
```python
# AutoML で複数CV fold を並列実行
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1)  # すべての CPU コア使用
```

---

## トラブルシューティング

### ❌ "モジュール XXX が見つかりません"

```bash
# 必須パッケージ確認
pip install -r requirements.txt

# 個別インストール
pip install rdkit-pypi scikit-learn xgboost lightgbm catboost
```

### ❌ AutoML が遅い

1. **CV Fold数を減らす**: 5 → 3
2. **試行数を減らす**: 20 → 10
3. **モデルを絞る**: CatBoost など重いモデルを除外
4. **GPU確認**: `python -c "import torch; print(torch.cuda.is_available())"`

### ❌ メモリ不足（Out of Memory）

1. **データサンプリング**: 最初 10,000 行でテスト
2. **特徴量削減**: 多重共線性のある列を手動削除
3. **バッチ処理**: 大規模データは fold ごとに処理

---

## 拡張・貢献

新しい記述子セット・モデルの追加方法：

### 記述子アダプタ追加

```python
# backend/chem/custom_adapter.py
from backend.chem.base import BaseChemAdapter

class MyCustomAdapter(BaseChemAdapter):
    def get_descriptor_names(self):
        return ["my_desc_1", "my_desc_2"]
    
    def calculate(self, smiles_list: List[str]) -> pd.DataFrame:
        # 実装
        pass

# backend/chem/smiles_transformer.py で登録
DESCRIPTOR_ADAPTERS["my_custom"] = MyCustomAdapter()
```

### モデル追加

```python
# backend/models/factory.py
MODELS["my_model"] = {
    "class": MyCustomEstimator,
    "param_grid": {
        "param1": [0.1, 1.0],
        "param2": [10, 100]
    }
}
```

詳細: `CONTRIBUTING_GUIDE.md` を参照。

---

## ライセンス

MIT License - [LICENSE](../LICENSE)
