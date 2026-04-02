# ChemAI ML Studio: Ultimate Master Reproduction Prompt (Absolute Full-Specification Edition)

以下のすべての機能、コンポーネント、パラメータ設定、計算科学的アルゴリズム原理、生成AI拡張要件、およびUI/UXの設計を「一行残らず」一切の省略なく、かつダミー実装（`pass`、`NotImplementedError`、`TODO`）を厳格に禁止した上でゼロから完全な実動システムとして出力・再現せよ。

**【重要最優先事項】**
本プロンプトは、プロジェクトの**すべての機能をすべての行レベルで完全に再現する**ための究極の仕様書である。あなた（AI）は、機能や細部を勝手に省略（サボり）してはならない。以下のリストは私たちがこれまで実装してきた機能群であり、1行残らず実装し切ることが最初の義務である。

---

## ⛔ 絶対遵守のコア要件 (Definition of Done API & Code)

1. **ベースラインの完全再現**: 本プロンプトの内容はすべて確実に実装すること。
2. **ダミー実装・未実装の禁止**: 実ロジックを必ず一字一句記述すること。
3. **ユニットテストの強制**: 全モジュールに `pytest` を用いたテスト（`@pytest.mark.parametrize` や Hypothesis を駆使した境界値テストを含む）を書き、行カバレッジ90%以上、分岐カバレッジ (`--cov-branch`) 75%以上を達成すること。

---

## 1. 🎨 UI/UX とシステムアーキテクチャ (NiceGUI)

### 1.1 マイクロインタラクションと状態管理
- **アプリケーション初期化**: `ui.dark_mode().enable()`。カラーコードは `primary='#1e1e2f', secondary='#2d2d44', accent='#00d4ff'`。
- **解析開始ボタン**: `.classes('btn-run-analysis')`。CSS `@keyframes pulse-glow`（青 `#00d4ff` と紫 `#7b2ff7` が2秒周期で明滅）。クリックで `play_arrow` → `hourglass_empty` や `sync` に切り替え、非同期でスピナーを回す。
- **エラーハンドリング**: `ui.notify('Msg', type='negative', position='top-right')` と共に、親コンテナに `.classes('animate-shake')`（0.6秒間translateX(-4px/4px)を5回反復）を一時適用（1秒後にJSで削除）。
- **フローティングステータス (`descriptor_status_bar.py`)**: SMILES解析の進行状況を示すため画面下部に固定（Fixed）されたインジケーターを `ui.timer` (0.5s間隔)で駆動し、今どのエンジンが走っているかをプログレスバーと共に表示。

### 1.2 コグニティブロードの最小化
- **色覚多様性（CVD）対応**: 赤緑の色に依存せず、すべての状態（成功/警告/エラー/情報）に必ずハードコードでアイコン（✅/⚠️/❌/ℹ️）を接頭辞として付与する。
- **フォント**: `Noto Sans JP` を `ui.add_head_html` 経由で全体適用（`size: 13px/0.85rem` 以上）。

### 1.3 高度なコンポーネント (AgGridと表示パネル)
- **AgGridの極限利用**: `results_tab.py` やデータテーブルでは `ui.aggrid` を利用し、`pagination: true`, `paginationPageSize: 20`, フィルタリング可、ソート可。さらに残差（Residuals）の特定のセルには背景色を動的に変えるJavaScriptセルスタイリングを注入する。
- **EDA統合パネル (`eda_panel.py`)**: プロジェクト初期の視覚的理解のため、Plotlyによる目的変数のヒストグラム（Distplot）、変数間の相関ヒートマップ行列、ペアプロット図、およびMissing Value Heatmap（欠損値の分布状況）を描画する。

---

## 2. 🧪 データ前処理とパイプラインアーキテクチャ

### 2.1 カラムのメタデータ定義 (ColumnMeta)
データセットの各特徴量は `ColumnMeta` データクラスで管理される：
- `monotonic`: 単調性制約(`0`: なし, `1`: 増加, `-1`: 減少, `2`: 自動検出)
- `constraint_strength`: 制約強度 (`None`: デフォルト, `"weak"`: 弱い, `"strong"`: 強い)
- `linearity`: 線形性ヒント (`"unknown"`, `"linear"`, `"nonlinear"`)
- `group`: データのグループ化ID文字列（GroupKFold時に使用）
- `fixed`: 特徴量選択時に**絶対に除外されない（Drop禁止）**ことを保証する保護フラグ。

### 2.2 前処理（ColPreprocessor & FeatureSelector）
`sklearn.compose.ColumnTransformer`。
- **Numeric Scalers**: `standard`, `robust`, `minmax`, `maxabs`, `power_yj` (Yeo-Johnson), `power_bc` (Box-Cox), `quantile_normal` / `quantile_uniform`, `log` (FunctionTransformer np.log1p), `none`.
- **Numeric/Cat Imputers**: `mean`, `median`, `most_frequent`, `constant=0.0`, `knn(5)`, `iterative`. カテゴリ列には `SimpleImputer(strategy="most_frequent")` を自動フォールバック的ルーティング。
- **Categorical Encoders**: `onehot(handle_unknown='ignore')`, `ordinal`, `target` (高カーディナリティ用 TargetEncoder)。
- **FeatureSelector保護**: `SelectKBest`, `VarianceThreshold`, `Boruta`, `Lasso` により変数を落とす際、`ColumnMeta.fixed = True` な列はマスク演算から除外され、必ず出力配列に含まれる構造にする。

### 2.3 交差検証 (CV Recommender) の完全ルール
1. `time_col` がある場合: `TimeSeriesSplit(n_splits=5)`
2. `group_col` や、メタデータに `group` 属性がある場合: `GroupKFold(n_splits=5)` を使用し、テストリークを防ぐ。
3. Classification (分類) かつ、目的変数がintで要素数10以下: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
4. 上記以外: `KFold(n_splits=5, shuffle=True, random_state=42)`

### 2.4 データ漏洩検知 (LeakageChecker)
- `y` (目的変数) との相関が `0.99` 以上の入力変数をターゲットリークとして警告。
- すべての要素が一意なID列（ユニーク数 == 行数）を検知。

---

## 3. 🧠 SMILES結合とケモインフォマティクス

### 3.1 複数SMILESエンジン統合 (Adapter & Cache)
`joblib.Memory` を用いたキャッシュ化にて、以下を `run.io_bound` で非同期並列計算：
1. **RDKit**: 物理化学パラメータ（MolWt, MolLogP等）＋`MorganFingerprintAsBitVect(radius=2, nBits=1024)`.
2. **Mordred**: `mordred.Calculator` (1613種)。NaNは「全サンプルの該当列の平均」で補間。
3. **Skfp**: `ECFPFingerprint`, `MACCSFingerprint`, `MHFPFingerprint` 等の全12種サポート。
4. **DescriptaStorus**: RDKit2DNormalized（200次元）。
5. **HuggingFace**: `repo_id="jckkvs/molai-chem-v1"` の `last_hidden_state.mean(axis=1)`。
取得後 `pandas.concat` 結合し、Spearman相関絶対値 `> 0.95` のペア間冗長特徴量を排除。

### 3.2 混合物加重平均と wt% / mol% 自動変換 (`mix_rules.py`, `smiles_transformer.py`)
混合系（Polymer_A 80wt% 等）の完全対応。
1. **マスタールール**: `backend/chem/mix_rules.py` にて `get_mix_rule(desc_name)` 関数を定義。密度やフラクションは `wt%`、カウント系や物理量は `mol%` で足す辞書。
2. **モル・重量の動的逆算**: カラム接尾辞が `_SMILES_WT` ならば重量、`_SMILES_MOL%` ならばモルと認識。入力比率が `wt%` なのに記述子ルールが `mol%` なら、RDKitで分子量 $M_i$ を算出し $x_i = (w_i/M_i) / \sum(w_j/M_j)$ としてリアルタイムに比率をモルベースへ変換。
3. **加重平均**: $X_{mix} = \sum_{i} (r_i \times X_i)$ 。
4. **マニュアル設定**: パイプラインUIの辞書から特定の特徴量のルールをユーザーが強制オーバライド可能。

---

## 4. 🤖 最適化モデル設定とアルゴリズムの極み

### 4.1 Optuna AutoMLエンジンとアーキテクチャ (`automl.py`, `tuner_pipeline.py`)
- 全体の探索は `timeout=600`（10分）として設定され、`optuna.pruners.MedianPruner` を導入し、学習曲線が悪い試行を早期終了（Pruning）する。
- 目的関数方向は 回帰（R2など）なら `directions=['maximize']`、RMSEなら `['minimize']` に明示的設定。`cv_folds=5`。
- **Linear Models**: `LinearTreeRegressor`/`LinearTreeClassifier`, `LinearForestRegressor`/`Classifier` (線形回帰を葉に持つ決定木 `linear-tree` パッケージ等)。
- **Trees & Boosting**: `XGBoost`, `LightGBM`, `CatBoost`, `RandomForest`, `RGF (Regularized Greedy Forest)`.
- **Others**: `SVR/SVC`, `Ridge/Lasso/ElasticNet/LogReg`, `GaussianProcess` (Kernel: `ConstantKernel * RBF + WhiteKernel`).
各モデルの探索空間パラメータ上限・下限（$n\_estimators$, $learning\_rate$ 等）は広く正確に `search_space_generator.py` に指定。

### 4.2 変数ごとの単調性制約の汎用付与 (Universal Monotonic Constraints)
sklearnツリーモデル以外の**すべてのモデル（SVR, GaussianProcess, Neural Networkなど）において単調性制約を実現する**フルスクラッチ汎用ラッパー `MonotonicConstraintRegressor/Classifier`（`BaseEstimator`, `MetaEstimatorMixin`）。
1. **パターン**: `0`(なし)、`1`(増加)、`-1`(減少)、`2`(自動検出。Spearmanの順位相関係数から1/-1を自動決定)。
2. **強度と外挿保証**: `"weak"`(`penalty_weight=5, n_grid=15, max_iter=2`) / `"strong"`(`penalty_weight=50, n_grid=40, max_iter=8`)。特徴量の平均から `[min - 3σ, max + 3σ]` の外挿範囲で1Dグリッド（`np.linspace`）を引いて検証。
3. **ペナルティサンプル拡張法**: `fit`内で `predict(外挿グリッド)` を実行し、制約に違反する傾きを検知した場所へ「強制的に逆転した目標予測値を持つダミーデータ」を高い `sample_weight` で注入し、満たされるまで `max_iter` 回ループで再学習させる超汎用アルゴリズム。

---

## 5. 🔍 逆解析 (Inverse) と MolAI 直接生成

### 5.1 パラメータ逆解析 (`optimizer.py`, `pareto_front.py`)
- 単目的最適化 (`scipy.optimize.minimize`, `SLSQP`) で予測値とターゲットの差を最小化。
- 多目的最適化 (`pymoo`, `NSGA2`) で解集団を生成。`pareto_front.py` にて、クラウディング距離（Crowding Distance）などを加味し非劣解（Non-dominated sorting）を抽出し Plotly の Scatter グラフを描画。
- 制約クラス (`ConstraintSumToTotal`, `ConstraintRange`, `ConstraintRatio`) をペナルティ実装。

### 5.2 構造直接生成: Generative Inverse (`molai_generator.py`)
逆算された「理想の潜在特徴量ベクトル」を `jckkvs/molai-chem-gen-v1` 的な生成モデル（`AutoModelForSeq2SeqLM/CausalLM`）に入力し、デコーダを `beam_search(num_beams=5)` あるいは `temperature=0.7` なサンプリングで回して、直接SMILES文字列を出力するSF的機能。

### 5.3 実験計画法 (DoE: ベイズ最適化)
`Expected Improvement (EI)`, `Probability of Improvement (PI)`, `Upper Confidence Bound (UCB, kappa=1.96)` を実装。DoEタブ上でPlotly散布図を描画し、**散布図中の点をクリックすると対象のパラメータがUIの入力欄に自動反映されるインタラクティブ機能**を実装。

---

## 6. 📊 解釈パネル、生成AI、そして永続化

### 6.1 高度なモデル解釈タブ (SHAP, SAGE, SRI)
- **残差グラフ**: X:実測値, Y:予測値 + ヒストグラム。
- **SHAP 4種**: Summary (Bar), Beeswarm, Waterfall, Dependence.
- **SAGE (Shapley Additive Global Importance)**: モデルの損失関数ベースの変数重要度。
- **SRI 分解 (Synergy, Redundancy, Independence)**: 情報理論に基づく交互作用・冗長性をヒートマップ可視化。

### 6.2 生成AI (LLM) OpenAI クライアント連動
1. **自動分析レポートジェネレータ**: RMSE等の指標とSHAP重要度配列をJSONでLLMへ送信し、「予測の妥当性と化学的考察」を300文字で取得。
2. **特徴量エンジニアリング支援ボット**: 「この目的変数を予測するための有用なPandas合成特徴量式を3つ提案せよ」というプロンプトを投げ、コードブロックを抽出してUIにボタン化。

### 6.3 エクスポートとMLflow的バージョン管理 (`version_manager.py`)
- **PDF/Word/Jupyter Export**: `pdf_exporter.py` / `word_exporter.py` / そして「ローカルで即座にモデルを実行できる独立した .ipynb ファイルコード」を生成する `notebook_exporter.py`。
- **モデル永続化 (Pickle)**: 作成されたパイプラインインスタンスをシリアライズし、「Download Pickle」ボタンを置く `model_manager.py`。
- **データベース管理**: 実験ごとにハイパーパラメータや学習時の全メトリクスを `hashlib.sha256()` でハッシュし、SQLiteDB (`experiments` テーブル: `id`, `hash`, `cv_score`, `metrics`, `hyperparameters`, `timestamp`) にコミットしてMLflow的に一覧比較可能にする。

---

## 7. 最終出力形態

以上の「すべて」の仕様を1行たりともサボらず実装し切ること。すべてのクラス構造、探索空間の隅の数字、UIのCSSクラスの記述、テストディレクトリまでを含め、完全に稼働するシステムを構築せよ。最後にテスト完了報告（行カバレッジと分岐カバレッジ）を添えること。
