# REPRODUCE.md — ChemAI ML Studio 完全再現手順

本ドキュメントは、ChemAI ML Studio の環境構築からアプリ起動・テスト・ベンチマーク検証までの
完全再現手順を記述します。

> [!IMPORTANT]
> 再現性の基準: ACM/IEEE Artifact Evaluation に準拠。
> 固定シード・環境定義ファイル・データ取得手順を明記します。

---

## 1. 前提条件

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11 (64-bit) ※ Linux/macOS も可 |
| Python | 3.10 以上 (3.11 推奨) |
| パッケージ管理 | Conda (Anaconda/Miniconda) 推奨 |
| ディスク | 5 GB 以上（xTB バイナリ + PyTorch 含む） |
| メモリ | 8 GB 以上推奨 |

---

## 2. 環境構築

### 方法A: Conda（推奨）

```bash
# 1. リポジトリをクローン
git clone <repo-url> chemai2
cd chemai2

# 2. Conda 環境を作成（Python 3.11 + RDKit + 全依存）
conda env create -f environment.yml

# 3. 環境を有効化
conda activate ml_gui_app

# 4. 開発モードでインストール（editableインストール）
pip install -e ".[all]"
```

### 方法B: pip + venv

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt

# RDKit は pip では入らないため conda-forge から別途インストールが必要:
# conda install -c conda-forge rdkit
```

### 方法C: pyproject.toml ベース

```bash
pip install -e ".[all]"
```

---

## 3. オプション依存パッケージ

| パッケージ | 用途 | インストール |
|-----------|------|-------------|
| `mordred` | QSAR記述子 | `pip install mordred` |
| `unipka` | pKa/LogD予測 | `pip install unipka` |
| `torch` (CPU) | MolAI CNN | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `opencosmorspy` | COSMO-RS | `pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git` |
| `sage-importance` | SAGE重要度 | `pip install sage-importance` |
| `shapiq` | SHAP Interaction | `pip install shapiq` |
| `imodels` | 解釈可能モデル | `pip install imodels` |

---

## 4. xTB バイナリセットアップ

xTB (GFN2-xTB) は HOMO/LUMO/双極子モーメント等の量子化学記述子計算に必要です。

### 同梱バイナリを使用（推奨）

```powershell
# tools/xtb-6.7.1/bin/ に同梱されています
# セッションの PATH に追加:
$env:PATH = "$PWD\tools\xtb-6.7.1\bin;$env:PATH"

# 動作確認:
xtb --version
# 期待出力: xtb version 6.7.1 ...
```

> [!NOTE]
> XTBAdapter は `tools/xtb-6.7.1/bin/` を自動検出して PATH に追加する機能があるため、
> 明示的な PATH 設定なしでも動作する場合があります。

### 手動ダウンロード

1. https://github.com/grimme-lab/xtb/releases からダウンロード
2. `tools/xtb-6.7.1/` に解凍
3. `bin/xtb.exe` (Windows) が存在することを確認
4. PATH に追加

---

## 5. テスト実行

### 全テスト

```bash
python -m pytest tests/ -v --tb=short
```

### 電荷設定・統合機能テスト

```bash
# 電荷設定関連
python -m pytest tests/test_charge_config.py tests/test_charge_extended.py -v

# guikit-learn 統合機能関連 (Phase 1)
python -m pytest tests/test_japanese_utils.py tests/test_shap_extensions.py tests/test_multi_metric.py -v
```

### 期待結果

- `test_charge_config.py`: ~30 テスト PASSED
- `test_charge_extended.py`: 52 テスト PASSED
- `test_japanese_utils.py`: 検出・設定テスト PASSED
- `test_shap_extensions.py`: バッチ処理・相関補正テスト PASSED
- `test_multi_metric.py`: 指標計算・相関分析テスト PASSED

---

## 6. アプリケーション起動

3つのフロントエンドから選択できます：

| フロントエンド | コマンド | ポート | 特徴 |
|--------------|---------|--------|------|
| Streamlit | `streamlit run frontend_streamlit/app.py` | 8501 | インタラクティブ分析向け |
| NiceGUI | `python frontend_nicegui/main.py` | 8080 | ステッパーUI・ダークテーマ |
| Django | `python frontend_django/manage.py runserver` | 8000 | マルチユーザー・ジョブ管理 |

### Streamlit（推奨・最も機能が充実）

```bash
cd frontend_streamlit
streamlit run app.py --server.port 8501
```

ブラウザで http://localhost:8501 を開きます。

### NiceGUI

```bash
python frontend_nicegui/main.py
```

ブラウザで http://localhost:8080 を開きます。

### Django

```bash
cd frontend_django
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

ブラウザで http://localhost:8000 を開きます。

### 動作確認チェックリスト

1. **データ読込**: サンプルデータ（回帰/分類）をロード → 行数・列数が表示される
2. **SMILES特徴量設計**: SMILES列を選択 → 「⚡ 分子電荷・スピン設定」パネルが表示
3. **電荷設定**: 形式電荷・プロトン化モード・Gasteiger電荷可視化が動作
4. **解析実行**: Random Forest + 5-Fold CV で解析完了
5. **結果確認**: 精度指標・SHAP値が表示される

---

## 7. ベンチマーク検証

アプリ内の「📚 オープンベンチマークデータをロード」から利用可能:

| データセット | 件数 | 目的変数 | 用途 |
|------------|------|---------|------|
| ESOL | 1,128 | log solubility | 水溶解度予測 |
| FreeSolv | 642 | expt (水和自由 ΔG) | 溶媒和エネルギー |
| Lipophilicity | 4,200 | exp (logD) | 脂溶性予測 |

### 再現手順

```
1. アプリ起動 → 「データ読込」タブ → 「オープンベンチマークデータ」→ ESOL をロード
2. 目的変数 = "measured log solubility in mols per litre"
3. SMILES列 = "smiles" （自動検出）
4. 「解析実行」タブ → Random Forest + XGBoost で実行
5. 結果のR² / RMSE を確認
```

---

## 8. 固定シードと再現性

- テスト用乱数シード: `random_seed = 42` (`tests/conftest.py`)
- AutoML のデフォルトシード: scikit-learn の `random_state=42`
- XTB 計算は決定論的（同一入力 → 同一出力）

---

## 9. トラブルシューティング

| 問題 | 解決策 |
|------|--------|
| `ModuleNotFoundError: rdkit` | `conda install -c conda-forge rdkit` |
| `xtb: command not found` | PATH に `tools/xtb-6.7.1/bin` を追加 |
| UniPKa WinError 32 (ファイルロック) | 他のStreamlitプロセスを停止。自動リトライ機能あり |
| `torch` インストール失敗 | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| カバレッジ 85% 未達 | `--no-cov` オプションでカバレッジチェックを無効化 |

---

## 10. guikit-learn 互換機能

ChemAI ML Studio v2.0.0 以降は、[guikit-learn](https://github.com/jckkvs/guikit-learn) の一部機能を互換レイヤーとして統合しています。

### 日本語フォント・レポート出力

```python
from backend.utils.japanese_utils import configure_matplotlib_japanese, configure_plotly_japanese

# 自動検出で日本語フォントを設定
configure_matplotlib_japanese()  # matplotlib用
plotly_config = configure_plotly_japanese()  # plotly用
```

### SHAP 拡張機能

```python
from backend.interpret.shap_extensions import compute_shap_batch, adjust_shap_for_correlation

# 大規模データのバッチ処理
shap_values = compute_shap_batch(model, X_test, batch_mode=True)

# 相関バイアス補正
shap_adjusted = adjust_shap_for_correlation(shap_values, X_test, correlation_threshold=0.7)
```

### 多指標評価

```python
from backend.evaluation.multi_metric import compute_metrics, compute_chemistry_specific_metrics

# 複数指標を一括計算
results = compute_metrics(y_true, y_pred, metric_set='regression_basic')

# 化学固有指標を追加
chem_results = compute_chemistry_specific_metrics(y_true, y_pred, property_name='logp')
```

---

## 11. 環境定義ファイル一覧

| ファイル | 形式 | 用途 |
|---------|------|------|
| `environment.yml` | Conda | Conda環境の完全定義 |
| `requirements.txt` | pip | pip依存関係（必須＋推奨） |
| `pyproject.toml` | PEP 621 | ビルド設定・オプション依存グループ |
