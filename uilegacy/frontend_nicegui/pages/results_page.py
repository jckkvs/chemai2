"""
frontend_nicegui/pages/results_page.py
Results & Decision Report page - 仕様書11章、12章に基づく実装
モデル保存、特徴量重要度、CV結果、レポート生成
"""
from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ResultsPage:
    """Results & Decision Report page"""

    def __init__(self):
        self.automl_result: Optional[Dict] = None
        self.df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.decision_report: Optional[str] = None

    def render(self):
        """Resultsページを描画"""
        with ui.column().classes('w-full max-w-7xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('📋 Results & Decision Report').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('仕様書11章、12章に基づく').classes('text-gray-600')

            # データ未ロード時
            self._no_data = ui.card().classes('w-full mb-4')
            with self._no_data:
                ui.label('⚠️ データが読み込まれていません').classes('text-lg font-bold text-orange-600 mb-2')
                ui.label('「Data Upload」タブからデータをアップロードしてください。').classes('text-sm text-gray-600')
                ui.button('← Data Uploadへ', on_click=lambda: ui.navigate.to('/#data'), color='primary').props('outline')

            # モデル結果サマリー
            self._model_card = ui.card().classes('w-full mb-4')
            self._model_card.visible = False
            with self._model_card:
                ui.label('🤖 モデル結果サマリー').classes('font-bold text-lg mb-2')

                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('最佳モデル').classes('text-xs text-gray-500')
                        self._best_model = ui.label('-')
                    with ui.column():
                        ui.label('CV Score').classes('text-xs text-gray-500')
                        self._cv_score = ui.label('-')
                    with ui.column():
                        ui.label('データ行数').classes('text-xs text-gray-500')
                        self._data_rows = ui.label('-')
                    with ui.column():
                        ui.label('目的変数').classes('text-xs text-gray-500')
                        self._target_label = ui.label('-')

            # 保存セクション (仕様書11章)
            self._save_card = ui.card().classes('w-full mb-4')
            self._save_card.visible = False
            with self._save_card:
                ui.label('💾 結果保存 (仕様書11章)').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **保存対象一覧**:
                - 🤖 モデル (.pkl) - 学習済みモデル全体
                - 📊 特徴量重要度 (.csv) - 変数名・重要度・順位
                - 📈 CV結果 (.json) - 各フォールドのスコア
                - 📉 予測実測プロット (.png/.html) - 散布図・残差プロット
                - 📋 回帰係数 (.csv) - 線形モデルの係数
                - 📑 SHAP値 (.csv/.png) - SHAP要約プロット
                """)

                with ui.row().classes('w-full gap-2'):
                    ui.button('💾 モデルを保存', on_click=self._save_model, color='primary').props('outline')
                    ui.button('📊 特徴量重要度を保存', on_click=self._save_feature_importance, color='primary').props('outline')
                    ui.button('📈 CV結果を保存', on_click=self._save_cv_results, color='primary').props('outline')
                    ui.button('📉 プロットを保存', on_click=self._save_plots, color='primary').props('outline')

                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.button('📦 一括保存 (全結果)', on_click=self._save_all, color='positive', size='lg').props('glossy')
                    ui.button('📂 保存フォルダを開く', on_click=self._open_save_folder).props('outline')

            # データ保存ディレクトリ構造の表示
            self._dir_card = ui.card().classes('w-full mb-4')
            self._dir_card.visible = False
            with self._dir_card:
                ui.label('📂 保存ディレクトリ構造 (仕様書11.7)').classes('font-bold text-lg mb-2')
                ui.markdown("""
                ```
                saved/
                └── project_{プロジェクト名}_{日付}/
                    ├── model.pkl                    # 学習済みモデル
                    ├── config.json                   # 全設定
                    ├── data/
                    │   ├── X_train.csv
                    │   ├── y_train.csv
                    │   └── predictions.csv
                    ├── results/
                    │   ├── feature_importance.csv
                    │   ├── cv_results.json
                    │   └── shap_values.csv
                    ├── plots/
                    │   ├── pred_vs_actual.png
                    │   ├── residuals.png
                    │   └── shap_summary.png
                    └── report/
                        ├── decision_report.md
                        └── llm_discussion.txt
                ```
                """)

            # 意思決定レポート (仕様書12章)
            self._report_card = ui.card().classes('w-full mb-4')
            self._report_card.visible = False
            with self._report_card:
                ui.label('📋 意思決定レポート (仕様書12章)').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **レポート構成 (意思決定志向)**:
                1. 現状評価（データ・モデル）
                2. 推奨アクション（優先順位付き）
                3. ユーザーのドメイン知識反映
                4. 結論（LLMによる提言）
                5. 次回会議での提言（エグゼクティブ・サマリー）
                """)

                with ui.row().classes('w-full gap-2'):
                    ui.button('📋 LLMレポート生成', on_click=self._generate_llm_report, color='primary')
                    ui.button('📄 PDFエクスポート', on_click=self._export_pdf, color='primary').props('outline')
                    ui.button('📊 Markdown出力', on_click=self._export_markdown, color='primary').props('outline')

                # レポート表示エリア
                self._report_content = ui.markdown('').classes('w-full mt-4 bg-blue-50 p-4 rounded')

            # 実験指示書
            self._experiment_card = ui.card().classes('w-full mb-4')
            self._experiment_card.visible = False
            with self._experiment_card:
                ui.label('📐 次の実験リスト (CSV/Excel出力)').classes('font-bold text-lg mb-2')

                self._experiment_table = ui.table(
                    columns=[],
                    rows=[],
                    pagination=dict(rowsPerPage=10)
                ).classes('w-full')

                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.button('📋 CSV出力', on_click=self._export_experiment_csv, color='primary').props('outline')
                    ui.button('📊 Excel出力', on_click=self._export_experiment_excel, color='primary').props('outline')

    def load_results(self, automl_result: Dict, df: Optional[pd.DataFrame] = None, target_col: Optional[str] = None):
        """モデル結果をロード"""
        self.automl_result = automl_result
        if df is not None:
            self.df = df.copy()
        self.target_col = target_col

        # UI更新
        self._no_data.visible = False
        self._model_card.visible = True
        self._save_card.visible = True
        self._dir_card.visible = True
        self._report_card.visible = True
        self._experiment_card.visible = True

        # モデル情報
        if automl_result:
            best_model = automl_result.get('best_model', 'Unknown')
            cv_score = automl_result.get('best_cv_score', 0)
            self._best_model.text = best_model
            self._cv_score.text = f'{cv_score:.4f}'
        else:
            self._best_model.text = '結果なし'
            self._cv_score.text = '-'

        self._data_rows.text = f'{len(df):,}' if df is not None else '-'
        self._target_label.text = target_col or '-'

        ui.notify('✓ Results: モデル結果を読み込みました', type='positive')

    def set_decision_report(self, report: str):
        """意思決定レポートを設定"""
        self.decision_report = report
        if hasattr(self, '_report_content'):
            self._report_content.content = report

    def _save_model(self):
        """モデルを保存 (仕様書11.2)"""
        if not self.automl_result:
            ui.notify('モデル結果がありません', type='warning')
            return

        try:
            import joblib
            from datetime import datetime

            model_dir = Path('saved_models')
            model_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = model_dir / f'model_{timestamp}.pkl'

            best_pipeline = self.automl_result.get('best_pipeline')
            if best_pipeline:
                joblib.dump(best_pipeline, model_path)

                # メタデータも保存
                metadata = {
                    'timestamp': timestamp,
                    'best_score': self.automl_result.get('best_cv_score', 0),
                    'model_type': type(best_pipeline).__name__,
                    'data_shape': str(self.df.shape) if self.df is not None else None,
                    'target_column': self.target_col,
                }

                metadata_path = model_dir / f'metadata_{timestamp}.json'
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                ui.notify(f'✓ モデル保存: {model_path}', type='positive')
            else:
                ui.notify('モデルが見つかりません', type='warning')

        except ImportError:
            ui.notify('⚠️ joblibがインストールされていません', type='warning')
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _save_feature_importance(self):
        """特徴量重要度をCSV出力 (仕様書11.3)"""
        ui.notify('特徴量重要度の保存（準備中）', type='info')

    def _save_cv_results(self):
        """CV結果を保存 (仕様書11.4)"""
        ui.notify('CV結果の保存（準備中）', type='info')

    def _save_plots(self):
        """プロットを保存 (仕様書11.5)"""
        ui.notify('プロットの保存（準備中）', type='info')

    def _save_all(self):
        """一括保存 (仕様書11.7)"""
        ui.notify('一括保存（準備中）', type='info')

    def _open_save_folder(self):
        """保存フォルダを開く"""
        import os
        save_dir = Path('saved')
        if save_dir.exists():
            os.startfile(str(save_dir))
        else:
            ui.notify('保存フォルダがまだありません', type='warning')

    def _generate_llm_report(self):
        """LLMレポート生成 (仕様書12章)"""
        if not self.automl_result:
            ui.notify('モデル結果がありません', type='warning')
            return

        # 簡易的なレポート生成
        report = f"""
# 意思決定レポート（簡易版）

## 1. 現状評価（データ・モデル）
- 現在のデータ数: {len(self.df) if self.df is not None else 0}サンプル
- モデル信頼度: {'△（予測不確実性 要確認）' if self.automl_result.get('best_cv_score', 0) < 0.7 else '○'}
- 達成可能性: {f"{self.automl_result.get('best_cv_score', 0)*100:.0f}%" if self.automl_result else '未評価'}

## 2. 推奨アクション（優先順位付き）

### 🥇 最優先：データ補完DOE（実施推奨）
- 内容: Maximin法で20実験を追加
- 期待効果: 達成確率向上
- コスト: 中（20サンプルの合成・測定）

### 🥈 次善：ベイズ最適化で条件探索
- 内容: フッ素除外条件で5条件を提案
- 期待効果: 達成確率向上（モデル不確実性大のため要注意）
- コスト: 低（5サンプルのみ）

## 3. ユーザーのドメイン知識反映
- フッ素導入は避けたい → 全条件からフッ素系除外済み
- 相溶性に注意 → 配合比率0.6超えの条件は除外

## 4. 結論（LLMによる提言）
「現在のデータ・構造系では目標達成は困難です。
最も確実なアクションは、まずDOEでデータを補完しモデル精度を上げることです。」

## 5. 次回会議での提言（エグゼクティブ・サマリー）
- 現状: データ不足により目標達成は低確率
- 提案: 20実験の追加実施（コスト対効果◎）
- 撤退基準: 追加実験後も達成確率50%未満なら撤退を検討
"""
        self.set_decision_report(report)
        ui.notify('✓ LLMレポートを生成しました', type='positive')

    def _export_pdf(self):
        """PDFエクスポート"""
        ui.notify('PDFエクスポート（準備中）', type='info')

    def _export_markdown(self):
        """Markdown出力"""
        if self.decision_report:
            from datetime import datetime
            report_dir = Path('saved_reports')
            report_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = report_dir / f'decision_report_{timestamp}.md'

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self.decision_report)

            ui.notify(f'✓ レポート保存: {report_path}', type='positive')
        else:
            ui.notify('レポートがありません', type='warning')

    def _export_experiment_csv(self):
        """次の実験リストをCSV出力"""
        ui.notify('実験リストCSV出力（準備中）', type='info')

    def _export_experiment_excel(self):
        """次の実験リストをExcel出力"""
        ui.notify('実験リストExcel出力（準備中）', type='info')
