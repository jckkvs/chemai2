# Chemical ML Platform (chemai2)

化学構造データ（SMILES）やテーブルデータの解析、機械学習モデルの構築、次元削減、モデル解釈を行うための統合プラットフォームです。
ユーザーフレンドリーなワンクリックUIと、専門家向けの高度な記述子選択・機械学習カスタマイズ機能を両立しています。

## ✨ 主な機能

- **化学情報学（ケモインフォマティクス）**
  - 分子構造 (SMILES) からの自動記述子生成
  - **RDKit**: 物理化学的記述子、MorganFP、RDKitFP、MACCS Keys 等 (~4000記述子)
  - **Mordred**: 有機化学やポリマー物性予測に有用な2Dトポロジカル記述子セットをデフォルトで厳選提供
  - 多角的な記述子選択 UI（「目的変数のカテゴリ」「記述子の物理的意味」「計算ライブラリ」の3つの視点から人間が考えて柔軟に選択可能）
- **高度な機械学習構築 (AutoML)**
  - Random Forest, LightGBM, XGBoost, CatBoost など主要アルゴリズムを網羅
  - 回帰・分類の自動判定と、Optunaによる自動ハイパーパラメータチューニング
- **データ解析・前処理**
  - EDA（探索的データ解析）、相関分析、自動欠損値補完、特徴量エンジニアリング
  - PCA, t-SNE, UMAP などの次元削減と可視化
- **モデル解釈 (XAI)**
  - SHAP 値計算による特徴量重要度の可視化

## 📂 ディレクトリ構成

- `backend/`: 解析・学習のコアロジック
  - `chem/`: 化学記述子計算（RDKit, Mordred, 未統合スタブ群）と推奨変数データベース
  - `data/`: データロード、前処理、EDA、次元削減
  - `models/`: MLモデル、AutoMLエンジン、チューナー
  - `pipeline/`: 特徴量選択、前処理変換パイプライン構築
  - `interpret/`: モデル解釈 (SHAP等)
- `frontend_streamlit/`: Streamlit を用いた主要なユーザーインターフェース
- `frontend_django/`: 管理・拡張用バックエンドAPI
- `tests/`: ユニットテスト・統合テスト（`pytest`）
- `examples/`: 機能別サンプルスクリプト（記述子計算の実例など）

## 🚀 セットアップと実行

**1. 依存関係のインストール**
Python 3.10 以上を推奨します。Mordred等の記述子エンジンも自動でインストールされます。
```bash
pip install -r requirements.txt
pip install mordred  # 高度な記述子を利用する場合
```

**2. アプリケーションの起動 (Streamlit)**
```bash
cd frontend_streamlit
streamlit run app.py
```

## 🧪 提供される記述子計算ライブラリ

本プラットフォームはスケーラブルなアダプタ構成を採用しており、以下の記述子計算エンジンをサポート・または将来の統合前提としています。

*   **RDKit (`RDKitAdapter`)**: 標準搭載。高速で安定した記述子・FP計算。
*   **Mordred (`MordredAdapter`)**: 実装済み。`pip install mordred` を入れるだけで、QSAR領域で実績のある高精度な2D/3D記述子が自動的に利用可能になります。
*   **XTB, COSMO-RS, UniPka 等 (`*Adapter`)**: バックエンドに柔軟なスタブフレームワークを実装済み。利用不能な環境では自動でフォールバックされエラーを防ぎます。

## 📝 サンプルスクリプト

UIを介さず、Pythonスクリプトから直接各種エンジンを呼び出す例が含まれています。
```bash
python examples/descriptor_calculation_example.py
```
（入力されたSMILESからRDKitとMordredの記述子をまとめて計算し、CSVとして出力します）
