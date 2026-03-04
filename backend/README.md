# Backend Module

`backend/` は解析プラットフォームのコアエンジンを含みます。

## モジュール構成

### 1. Data (`backend/data/`)
データの読み込み、前処理、可視化を担当します。
- `loader.py`: CSV, Excel, Parquet, SQLite からのデータ読み込み。
- `preprocessor.py`: カテゴリカル変数のエンコーディング、スケーリング、欠損値補完。
- `dim_reduction.py`: 次元削減アルゴリズム（PCA, t-SNE, UMAP）。
- `eda.py`: 基本統計量、相関分析。
- `type_detector.py`: カラムのデータ型（数値、カテゴリ、テキスト、SMILES等）を自動判定。

### 2. Models (`backend/models/`)
機械学習モデルの構築と最適化を担当します。
- `factory.py`: 各種推定器（XGBoost, Random Forest 等）の生成。
- `cv_manager.py`: クロスバリデーションの制御。
- `tuner.py`: Optuna を用いたハイパーパラメータ最適化。
- `automl.py`: データセットに対する最適なモデルとパラメータの自動探索。

### 3. Chem (`backend/chem/`)
化学情報学に特化した機能を提供します。
- `descriptors.py`: Mordred, RDKit を用いた記述子計算。

## 主要アルゴリズムの解説

### 次元削減 (Dimensionality Reduction)
- **PCA (Principal Component Analysis)**: 線形な分散最大化方向への投影。
- **t-SNE (t-distributed Stochastic Neighbor Embedding)**: 高次元の局所的な構造を維持する非線形埋め込み。
- **UMAP (Uniform Manifold Approximation and Projection)**: 局所構造と大局構造のバランスに優れ、高速な非線形埋め込み。

## 開発者向け情報

### 遅延インポート
`backend.utils.optional_import` を使用して、重いライブラリ（UMAP, Mordred 等）は必要になるまでインポートされません。これにより、一部の依存関係がない環境でも他の機能は動作します。
