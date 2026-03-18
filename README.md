# ChemAI ML Studio (chemai2)

化学構造データ（SMILES）とテーブルデータを統合して、探索的データ解析 → 機械学習モデル構築 → モデル解釈までを一気通貫で行う **ケモインフォマティクス統合プラットフォーム** です。

量子化学計算（GFN2-xTB）、pKa予測（UniPKa）、溶媒和計算（COSMO-RS）による高精度記述子と、
直感的な Streamlit UI でワンクリック AutoML を実現します。

---

## ✨ 主な機能

### 🧪 ケモインフォマティクス

- **SMILES → 自動記述子生成**（7種の計算エンジン搭載）
  - **RDKit**: 物理化学記述子（MolWt, LogP, TPSA 等）＋ Morgan/RDKit/MACCS フィンガープリント
  - **Mordred**: 1,800+ QSAR 記述子（2Dトポロジカル）
  - **GFN2-xTB**: 半経験的量子化学記述子 — HOMO/LUMO/双極子モーメント/Mulliken電荷
  - **UniPKa**: pKa (酸性/塩基性) / LogD / 溶媒和エネルギー
  - **COSMO-RS**: σ-プロファイルによる溶媒和自由エネルギー
  - **MolAI**: CNN + PCA 分子潜在空間ベクトル
  - **GroupContrib**: 基団寄与法による熱物性推定

- **⚡ 分子電荷・スピン設定UI**
  - 形式電荷（SMILES自動読取 or 手動指定）・スピン多重度の設定
  - pH依存プロトン化（UniPKa pKa + Henderson-Hasselbalch式）
  - 5モード: as_is / neutral / auto_ph / max_acid / max_base
  - Gasteiger部分電荷の色付きSVG可視化
  - 分子ごとの個別電荷設定（上級者向け）
  - XTB に `--chrg` / `--uhf` を自動連携

- **高度な記述子選択UI**（5つの視点）
  - 相関係数、数え上げ変数、目的変数系統、物理的意味、計算ライブラリ

### 🤖 機械学習

- **AutoML**（ワンクリック）: Random Forest, LightGBM, XGBoost, CatBoost, SVR, KNN 等
- **Optuna 自動チューニング** + Grid Search
- **全組み合わせ Pipeline 探索**: 前処理・特徴選択・推定器の直積をCV評価
- **高度なモデル群**:
  - LinearTree / LinearForest / LinearBoost（区分線形モデル）
  - RGF (Regularized Greedy Forest)
  - imodels（解釈可能モデル）
  - 単調制約付きカーネルモデル
- **交差検証**: KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit 等

### 📊 データ分析・前処理

- EDA（探索的データ解析）・相関分析・外れ値検出
- 自動欠損値補完・カテゴリカルエンコーディング
- PCA, t-SNE, UMAP, PHATE 次元削減・可視化

### 🔬 モデル解釈 (XAI)

- **SHAP**: Summary / Individual / Dependence / Heatmap / Interaction
- **SAGE**: 特徴量重要度のゲーム理論的評価
- **shapiq**: SRI 分解による特徴量相互作用
- 伝統的手法: Feature Importance / Permutation Importance / PDP

---

## 📂 ディレクトリ構成

```
chemai2/
├── backend/                    # 解析・学習コアロジック
│   ├── chem/                   # 化学記述子エンジン
│   │   ├── charge_config.py    # 電荷設定データクラス
│   │   ├── protonation.py      # pH依存プロトン化変換
│   │   ├── rdkit_adapter.py    # RDKit + Gasteiger電荷
│   │   ├── xtb_adapter.py      # GFN2-xTB + Mulliken電荷
│   │   ├── mordred_adapter.py  # Mordred 記述子
│   │   ├── unipka_adapter.py   # UniPKa pKa/LogD
│   │   ├── cosmo_adapter.py    # COSMO-RS 溶媒和
│   │   ├── molai_adapter.py    # CNN + PCA 潜在空間
│   │   ├── recommender.py      # 目的変数別推奨記述子DB
│   │   └── smiles_transformer.py
│   ├── data/                   # ロード・前処理・EDA・次元削減
│   ├── models/                 # ML モデル・AutoML・チューナー
│   │   ├── automl.py
│   │   ├── linear_tree.py      # LinearTree/LinearForest/LinearBoost
│   │   ├── rgf.py              # Regularized Greedy Forest
│   │   └── monotonic_kernel.py # 単調制約カーネル
│   ├── pipeline/               # 前処理パイプライン・特徴選択
│   ├── interpret/              # SHAP・SAGE・shapiq
│   └── utils/
├── frontend_streamlit/         # Streamlit UI
│   ├── app.py                  # メインアプリ（3タブ構造）
│   └── components/
│       ├── charge_config_ui.py # 電荷設定UIコンポーネント
│       ├── interpretability_ui.py
│       └── pipeline_config_ui.py
├── tests/                      # テスト
├── tools/                      # 外部バイナリ（xTB等）
├── examples/                   # サンプルスクリプト
├── REPRODUCE.md                # 完全再現手順
├── environment.yml             # Conda環境定義
├── requirements.txt            # pip依存関係
└── pyproject.toml              # ビルド設定
```

---

## 🚀 セットアップ

### 1. 環境構築（Conda推奨）

```bash
conda env create -f environment.yml
conda activate ml_gui_app
```

または pip:

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. オプション依存パッケージ

```bash
pip install mordred            # Mordred 記述子
pip install unipka             # UniPKa pKa/LogD
pip install torch --index-url https://download.pytorch.org/whl/cpu  # MolAI用
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git    # COSMO-RS
pip install sage-importance shapiq imodels   # 解釈性ツール
```

### 3. xTB バイナリセットアップ（量子化学記述子を使う場合）

> [!IMPORTANT]
> xTB は外部バイナリが必要です。`tools/xtb-6.7.1/` に同梱されています。

```powershell
# Windows: PATH に追加
$xtbBin = "$PWD\tools\xtb-6.7.1\bin"
$env:PATH = "$xtbBin;$env:PATH"
xtb --version  # 動作確認
```

手動ダウンロード: [grimme-lab/xtb releases](https://github.com/grimme-lab/xtb/releases)

### 4. アプリ起動

3 つのフロントエンドが利用可能です:

| 版 | コマンド | ポート | 特徴 |
|---|---------|-------|------|
| **NiceGUI** ⭐ | `python frontend_nicegui/main.py` | **8080** | Pure Python UI、2クリック解析 |
| Streamlit | `streamlit run frontend_streamlit/app.py` | 8501 | データ分析向け |
| Django | `python frontend_django/manage.py runserver` | 8000 | REST API、ユーザー認証 |

```bash
# NiceGUI 版（推奨）
python frontend_nicegui/main.py
# → http://localhost:8080

# Streamlit 版
cd frontend_streamlit
streamlit run app.py
# → http://localhost:8501
```

> [!TIP]
> 各フロントエンドの詳細は以下を参照:
> - [NiceGUI 版 README](frontend_nicegui/README.md) — 起動・設定・デプロイガイド
> - [バックエンド README](backend/README.md) — API ドキュメント

---

## 🧪 テスト

```bash
# 全テスト実行
python -m pytest tests/ -v --tb=short

# 電荷設定モジュールのみ
python -m pytest tests/test_charge_config.py tests/test_charge_extended.py -v

# カバレッジ付き
python -m pytest tests/ --cov=backend --cov-branch --cov-report=term-missing
```

---

## 🧪 提供される記述子エンジン

| エンジン | アダプター | インストール | 主な記述子 |
|----------|-----------|-------------|-----------|
| RDKit | `RDKitAdapter` | 標準搭載 | 分子量・LogP・FP・Gasteiger電荷 |
| Mordred | `MordredAdapter` | `pip install mordred` | 1800+ QSAR記述子 |
| **xTB** | `XTBAdapter` | バイナリ配置 | HOMO/LUMO/双極子/Mulliken電荷 |
| UniPKa | `UniPkaAdapter` | `pip install unipka` | pKa / LogD / 溶媒和エネルギー |
| COSMO-RS | `CosmoAdapter` | git+... | 溶媒和自由エネルギー |
| MolAI | `MolAIAdapter` | `pip install torch` | CNN+PCA分子潜在空間 |
| GroupContrib | `GroupContribAdapter` | 標準搭載 | 基団寄与法熱物性 |

---

## 📝 サンプル

```bash
python examples/descriptor_calculation_example.py
```

> [!NOTE]
> 詳しい再現手順は [REPRODUCE.md](REPRODUCE.md) を参照してください。
