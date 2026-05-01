# ChemAI ML Studio - NiceGUI フロントエンド

## 概要

ChemAI ML Studio は、**化学研究者向けの材料インフォマティクス (MI) プラットフォーム**です。
NiceGUI フロントエンドは、コーディング知識がない研究者でも簡単に利用できるUI/UXを提供します。

## 🚀 クイックスタート

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/jckkvs/chemai2.git
cd chemai2

# 環境構築（方法A: ワンクリック）
.\tools\setup_all.bat

# または手動設定
conda activate ml_gui_app
pip install -r requirements.txt
```

### 起動

```bash
# NiceGUI フロントエンド（推奨）
python frontend_nicegui/main.py
# → http://localhost:8080 でアクセス

# Docker Compose で起動
docker-compose up -d
```

---

## 📋 使用方法

### 1️⃣ Data Upload & Cleaning タブ

#### ファイルのアップロード
- **対応形式**: CSV, Excel (.xlsx, .xls, .xlsm)
- **最大サイズ**: 50MB
- **操作**: ドラッグ&ドロップまたはクリックで選択

#### データ品質評価
- 自動で行われます
- 問題があれば警告が表示されます

#### LLM支援クリーニング
- `🔧 LLM支援でクリーニング` ボタンをクリック
- 自動検出される問題：
  - 列名の正規化
  - 欠損値処理
  - 型変換
  - 誤字・表記ゆれ
- 提案されたコードをコピーして適用可能

**次のステップ**: `📊 このデータで解析へ` → AutoML タブに遷移

---

### 2️⃣ LLM Assistant タブ

#### 🔐 外部LLM向けプロンプト生成（セキュア）
**推奨**: 社内セキュリティ要件が厳しい環境向け

1. モードを「外部高精度LLM」に選択
2. `📋 プロンプト生成` をクリック
3. 生成されたプロンプトをコピー
4. ChatGPT, Claude.ai など任意のLLMに貼り付け

**長所**:
- データが社外に出ない
- 高精度LLMが使用可能
- カスタマイズが容易

#### 💬 ローカルLLM チャット（実験的）
**前提条件**: Ollama 等のローカルLLMがインストールされていること

1. モードを「ローカルLLMで実行」に選択
2. チャット入力欄に質問を入力
3. 送信ボタンをクリック
4. 応答がリアルタイムで表示

**注意**:
- 初回実行時は遅い場合があります
- GPU があれば高速化されます

---

### 3️⃣ AutoML タブ

ChemAI の中核機能です。自動機械学習で最適なモデルを探索します。

#### ステップ1: データ確認
- アップロード済みデータが表示されます
- 行数・列数・メモリ使用量を確認

#### ステップ2: 目的変数選択
- 予測対象の列を選択
- 自動的に回帰/分類が判定されます

#### ステップ3: AutoML設定
- **CV Fold数**: デフォルト 5（3-10推奨）
- **評価指標**: RMSE, MAE, R2, Accuracy, F1 から選択
- **最大試行**: モデル評価の試行数（5-100）
- **使用モデル**: チェックボックスで選択

#### ステップ4: 実行
- `▶ AutoMLを実行` をクリック
- 進捗がリアルタイムで表示されます
- 完了後、結果が表示されます

#### 結果確認
- **📊 特徴量重要度**: SHAP値による特徴量の寄与度
- **💾 モデルを保存**: 最良モデルをpickle形式で保存
- **📄 レポート出力**: 結果をPDF形式で出力

---

### 4️⃣ Visualization タブ

データの多角的な可視化が可能です。

#### 🔄 次元削減手法
- **PCA**: 高速（説明分散率を表示）
- **t-SNE**: 非線形（グループ分けに最適）
- **UMAP**: バランス型（高速かつ解釈性あり）

#### 操作方法
1. 次元削減手法を選択
2. （オプション）色分け列を選択
3. `🔄 次元削減プロットを生成`をクリック
4. Plotly インタラクティブグラフが表示

#### 📈 相関ヒートマップ
- すべての数値列間の相関係数をヒートマップ表示
- マウスホバーで詳細を確認可能

---

## ⚙️ 設定

### LLM設定（⚙️ ボタン）

```
[LLM設定ダイアログ]

動作モード:
  ▪ プロンプトのみ生成（セキュア推奨）
  ▪ ローカルLLMを使用
  ▪ 外部APIを使用

[外部APIモード時]
  - APIエンドポイント: https://api.openai.com/v1
  - APIキー: sk-...
  - モデル名: gpt-4o-mini
  - Temperature: 0.0-1.0（デフォルト: 0.1）
  - Max Tokens: 128-8000（デフォルト: 2000）

詳細設定:
  ☐ LLM生成コードの自動実行（危険）
```

---

## 🔧 トラブルシューティング

### ❌ "モジュール XXX が見つかりません"

```bash
# 解決策: 必須パッケージをインストール
pip install -r requirements.txt

# または個別インストール
pip install rdkit-pypi mordred unipka
```

### ❌ "ローカルLLMが接続できない"

```bash
# Ollama がインストール・起動されているか確認
curl http://localhost:11434/api/tags

# Ollama が起動していなければ
ollama serve

# または Docker で実行
docker run -d -p 11434:11434 ollama/ollama
ollama pull llama2  # モデルダウンロード
```

### ❌ AutoML が遅い

1. **CV Fold数を減らす** (5 → 3)
2. **最大試行を減らす** (20 → 10)
3. **使用モデルを絞る** (重いモデルを外す)
4. **GPU が利用可能か確認**

```bash
# GPU確認（CUDA対応NVIDIAの場合）
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📚 API ドキュメント

フロントエンドの背後には FastAPI バックエンドがあります。

**API ドキュメント**: http://localhost:8000/docs （起動時）

---

## 🔐 セキュリティ考慮事項

### 開発環境
- 本ファイル内の `SECRET_KEY` はダミー（開発用）
- HTTPS は不要
- `DEBUG=true` で詳細なエラーメッセージ

### 本番環境
- `.env` ファイルで `SECRET_KEY` を変更
- SSL/TLS 証明書を設定
- `DEBUG=false` に設定
- `ALLOWED_HOSTS` を明示的に指定
- Docker Compose + nginx でデプロイ

詳しくは [デプロイメントガイド](../docs/DEPLOYMENT.md) を参照。

---

## 📖 ユースケース

### 例1: 物性値予測
```
1. SMILES リストを CSV で用意
2. [Data Upload] でアップロード
3. [AutoML] で「溶解度」を目的変数に選択 → 実行
4. [Visualization] で特徴量の相関を確認
5. [LLM Assistant] で結果を化学的に解釈
```

### 例2: 材料分類
```
1. 材料データ（CSV）をアップロード
2. [Data Cleaning] で問題点を修正
3. [AutoML] で分類モデルを構築
4. SHAP で分類判定根拠を確認
```

---

## 🤝 貢献

バグ報告や機能リクエストは GitHub Issues で。

## 📄 ライセンス

MIT License

---

**Powered by ChemAI ML Studio** 🧪✨
