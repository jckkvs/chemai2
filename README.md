# Chemical ML Platform (chemai2)

化学構造データ（SMILES）やテーブルデータの解析、機械学習モデルの構築、次元削減、モデル解釈を行うための統合プラットフォームです。
ユーザーフレンドリーなワンクリックUIと、専門家向けの高度な記述子選択・機械学習カスタマイズ機能を両立しています。

## ✨ 主な機能

- **化学情報学（ケモインフォマティクス）**
  - 分子構造 (SMILES) からの自動記述子生成
  - **RDKit**: 物理化学的記述子、MorganFP、RDKitFP、MACCS Keys 等 (~4000記述子)
  - **Mordred**: 有機化学やポリマー物性予測に有用な2Dトポロジカル記述子セット
  - **xTB (GFN2-xTB)**: 半経験的量子化学計算による電子状態記述子（HOMO/LUMO/双極子モーメント等）
  - **Uni-pKa**: pKa_acidic / pKa_basic / LogD_7.4 / 溶媒和エネルギー
  - **COSMO-RS**: σ-プロファイルを用いた溶媒和自由エネルギー計算（opencosmorspy）
  - **高度な記述子選択 UI**: 5つの視点からターゲットに最適な記述子を選定可能
    - `相関係数から選ぶ`: 目的変数との相関 (Pearson) に基づくランキング
    - `数え上げ変数から選ぶ`: 原子数・環数等の厳密抽出
    - `目的変数の系統から選ぶ`: 溶解度・熱物性・毒性等の推奨記述子セット
    - `記述子の物理的意味から選ぶ`: 極性・柔軟性・大きさ等のカテゴリ選択
    - `計算ライブラリから選ぶ`: RDKit / Mordred / xTB 等エンジン別選択
  - **統合相関ヒートマップ**: SMILES由来の記述子と元データ数値列を動的に統合・可視化

- **全組み合わせ Pipeline 探索（新機能）**
  - 解析実行タブの「⚙️ Pipeline 前処理・推定器を選ぶ」で各ステップを複数選択
  - 選択数に応じたパイプライン数をリアルタイム表示（例: 2×2×3×15 = 180通り）
  - 除外列（Excluder）を UI から直接指定可能

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
  - `chem/`: 化学記述子計算（RDKit, Mordred, xTB, UniPka, COSMO-RS）と推奨変数データベース
  - `data/`: データロード、前処理、EDA、次元削減
  - `models/`: MLモデル、AutoMLエンジン、チューナー
  - `pipeline/`: 特徴量選択、前処理変換パイプライン構築（全組み合わせ探索機能含む）
  - `interpret/`: モデル解釈 (SHAP等)
- `frontend_streamlit/`: Streamlit を用いた主要なユーザーインターフェース
- `tools/`: 外部バイナリ（xTB等）の配置ディレクトリ
- `tests/`: ユニットテスト・統合テスト（`pytest`）
- `examples/`: 機能別サンプルスクリプト

## 🚀 セットアップと実行

**1. 依存関係のインストール**

Python 3.10 以上を推奨します。

```bash
pip install -r requirements.txt
pip install mordred          # Mordred記述子
pip install unipka           # Uni-pKa（pKa/LogD計算）
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git  # COSMO-RS
pip install ase              # ASE（原子シミュレーション環境、xTBと連携）
```

**2. xTB バイナリのセットアップ（量子化学記述子を使う場合）**

> [!IMPORTANT]
> xTB は外部バイナリが必要です。以下の手順で配置してください。

```powershell
# Windows の場合（推奨）: tools/ に配置済みのバイナリを PATH に追加
$xtbBin = "C:\path\to\chemai2\tools\xtb-6.7.1\bin"
[Environment]::SetEnvironmentVariable("Path", "$env:Path;$xtbBin", "User")
```

もしくは公式から手動ダウンロード:
- **GitHub Releases**: https://github.com/grimme-lab/xtb/releases
- ファイル名: `xtb-6.7.1pre-windows-x86_64.zip`（Windows x64）
- `tools/xtb-6.7.1/` に解凍し、`bin/` を PATH に追加する

配置後、以下で動作確認:
```bash
xtb --version
```

> [!NOTE]
> `tools/xtb-6.7.1/` にはすでに xTB 6.7.1 Windows 版が同梱されています。
> PATH に追加するだけで使用できます。

**3. アプリケーションの起動 (Streamlit)**

```bash
cd frontend_streamlit
streamlit run app.py
```

## 🧪 提供される記述子計算ライブラリ

| ライブラリ | アダプター | インストール | 主な記述子 |
|----------|----------|------------|----------|
| RDKit | `RDKitAdapter` | 標準搭載 | 分子量・LogP・FP等 |
| Mordred | `MordredAdapter` | `pip install mordred` | 1800+ QSAR記述子 |
| **xTB** | `XTBAdapter` | バイナリ配置（上記） | HOMO/LUMO/双極子/電荷 |
| Uni-pKa | `UniPkaAdapter` | `pip install unipka` | pKa / LogD / 溶媒和エネルギー |
| COSMO-RS | `CosmoAdapter` | `pip install git+...` | 溶媒和自由エネルギー |
| MolAI | `MolAIAdapter` | `pip install torch` | PCA分子潜在空間 |

## 📝 サンプルスクリプト

UIを介さず、Pythonスクリプトから直接各種エンジンを呼び出す例が含まれています。
```bash
python examples/descriptor_calculation_example.py
```
（入力されたSMILESからRDKitとMordredの記述子をまとめて計算し、CSVとして出力します）
