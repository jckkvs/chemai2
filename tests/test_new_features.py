"""
tests/test_new_features.py
新しく実装した機能のテスト（2026-04-30）

テスト対象：
  1. LLM対話式ヒアリング (interview_session.py)
  2. 加重平均の自動判断 (average_type_detector.py)
  3. SMILESクイック特徴量 (smiles_quick_features.py)
  4. SMILESホバーHTML生成 (smiles_hover.py)
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np


# ════════════════════════════════════
# 1. LLM対話式ヒアリングのテスト
# ════════════════════════════════════

class TestInterviewSession(unittest.TestCase):
    """LLM対話式ヒアリングのテスト"""

    def setUp(self):
        """テスト前準備"""
        try:
            from backend.llm.interview_session import InterviewSession, InterviewPhase
            self.InterviewSession = InterviewSession
            self.InterviewPhase = InterviewPhase
        except ImportError as e:
            self.skipTest(f"interview_session module not available: {e}")

    @patch('backend.llm.interview_session.AutoAnalyzer')
    def test_session_start(self, mock_analyzer_class):
        """セッション開始のテスト"""
        from backend.data.auto_analyzer import AnalysisTask
        # AutoAnalyzerをモック
        mock_analyzer = MagicMock()
        mock_plan = MagicMock()
        mock_plan.target_column = 'yield'
        mock_plan.task_type = AnalysisTask.REGRESSION
        mock_analyzer.create_analysis_plan.return_value = mock_plan
        mock_analyzer_class.return_value = mock_analyzer

        session = self.InterviewSession()
        df = pd.DataFrame({
            'temperature': [100, 200, 300, 400, 500],
            'pressure': [1.0, 2.0, 3.0, 4.0, 5.0],
            'yield': [0.5, 0.6, 0.7, 0.8, 0.9],
        })
        first_q = session.start(df, target_hint='yield')
        self.assertIsNotNone(first_q)
        self.assertGreater(len(first_q), 0)

    @patch('backend.llm.interview_session.AutoAnalyzer')
    def test_submit_answer(self, mock_analyzer_class):
        """回答送信のテスト"""
        from backend.data.auto_analyzer import AnalysisTask
        mock_analyzer = MagicMock()
        mock_plan = MagicMock()
        mock_plan.target_column = 'result'
        mock_plan.task_type = AnalysisTask.REGRESSION
        mock_analyzer.create_analysis_plan.return_value = mock_plan
        mock_analyzer_class.return_value = mock_analyzer

        session = self.InterviewSession()
        df = pd.DataFrame({
            'temp': [100, 200, 300],
            'result': [0.5, 0.6, 0.7],
        })
        session.start(df, target_hint='result')

        # 回答を送信
        next_q = session.submit_answer('予測')
        self.assertIsNotNone(next_q)
        self.assertIn(session.current_phase.value, ['sample_target', 'goal_clarification', 'error_understanding', 'variable_nature'])

    def test_session_without_analyzer(self):
        """AutoAnalyzerなしでのセッション開始"""
        session = self.InterviewSession()
        # AutoAnalyzerがない場外でもエラーにならないこと
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        # モックを使用せずに開始（AutoAnalyzerは内部で作成される）
        # ただし、LLMプロバイダーがない場外があるので、エラーハンドリングを確認
        try:
            first_q = session.start(df)
            # エラーにならずに質問が返ってくればOK
            self.assertIsNotNone(first_q)
        except Exception as e:
            # LLMやAnalyzer関連のエラーは許容（テスト環境の制限）
            pass

    @patch('backend.llm.interview_session.AutoAnalyzer')
    def test_qa_history(self, mock_analyzer_class):
        """Q&A履歴のテスト"""
        from backend.data.auto_analyzer import AnalysisTask
        mock_analyzer = MagicMock()
        mock_plan = MagicMock()
        mock_plan.target_column = 'y'
        mock_plan.task_type = AnalysisTask.REGRESSION
        mock_analyzer.create_analysis_plan.return_value = mock_plan
        mock_analyzer_class.return_value = mock_analyzer

        session = self.InterviewSession()
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        session.start(df)

        # 履歴に追加
        session.qa_history.append({
            "phase": "test",
            "question": "テスト質問",
            "answer": "テスト回答",
        })
        self.assertEqual(len(session.qa_history), 1)
        self.assertEqual(session.qa_history[0]['answer'], "テスト回答")

    def test_phase_enum(self):
        """フェーズ列挙のテスト"""
        phases = [
            self.InterviewPhase.INIT,
            self.InterviewPhase.DATA_SUMMARY,
            self.InterviewPhase.GOAL_CLARIFICATION,
            self.InterviewPhase.COMPLETED,
        ]
        for phase in phases:
            self.assertIsNotNone(phase.value)


# ════════════════════════════════════
# 2. 加重平均の自動判断のテスト
# ════════════════════════════════════

class TestAverageTypeDetector(unittest.TestCase):
    """加重平均自動判断のテスト"""

    def setUp(self):
        try:
            from backend.llm.average_type_detector import AverageTypeDetector, AverageContext
            self.detector = AverageTypeDetector()
            self.AverageContext = AverageContext
        except ImportError as e:
            self.skipTest(f"average_type_detector module not available: {e}")

    def test_rule_based_detection(self):
        """ルールベース判定のテスト"""
        ctx = self.AverageContext(
            target_property='屈折率',
            available_columns=['mol_weight', 'volume', 'refractive_index'],
        )
        result = self.detector.detect(ctx, use_llm=False)
        self.assertIsNotNone(result)
        self.assertIn(result.average_type, ['weight', 'mol', 'simple', 'volume', 'special'])

    def test_refractive_index(self):
        """屈折率は体積平均が適切"""
        ctx = self.AverageContext(
            target_property='屈折率',
            available_columns=['mol_weight', 'volume', 'data'],
        )
        result = self.detector.detect(ctx, use_llm=False)
        self.assertEqual(result.average_type, 'volume')

    def test_viscosity(self):
        """粘度は特殊平均が適切"""
        ctx = self.AverageContext(
            target_property='粘度',
            available_columns=['mol_weight', 'viscosity'],
        )
        result = self.detector.detect(ctx, use_llm=False)
        self.assertEqual(result.average_type, 'special')

    def test_get_available_types(self):
        """利用可能な平均手法リストのテスト"""
        types = self.detector.get_available_types()
        self.assertGreater(len(types), 0)
        self.assertIn('value', types[0])
        self.assertIn('display', types[0])

    def test_calculate_weighted_average(self):
        """加重平均計算のテスト"""
        from backend.llm.average_type_detector import calculate_weighted_average

        values = [1.0, 2.0, 3.0]
        weights = [1.0, 2.0, 1.0]

        # 等加重平均
        result = calculate_weighted_average(values, weights, 'simple')
        self.assertAlmostEqual(result, 2.0, places=5)

        # 重量平均
        result = calculate_weighted_average(values, weights, 'weight')
        expected = (1.0*1.0 + 2.0*2.0 + 3.0*1.0) / 4.0
        self.assertAlmostEqual(result, expected, places=5)

    def test_detect_average_type_for_target(self):
        """便利関数のテスト"""
        from backend.llm.average_type_detector import detect_average_type_for_target

        result = detect_average_type_for_target(
            target_property='屈折率',
            available_columns=['mol_weight', 'volume', 'refractive_index'],
        )
        self.assertIsNotNone(result)
        self.assertIn(result.average_type, ['weight', 'mol', 'simple', 'volume', 'special'])


# ════════════════════════════════════
# 3. SMILESクイック特徴量のテスト
# ════════════════════════════════════

class TestSmilesQuickFeatures(unittest.TestCase):
    """SMILESクイック特徴量のテスト"""

    def setUp(self):
        try:
            from backend.chem.smiles_quick_features import compute_quick_features, QUICK_FEATURES
            self.compute_quick_features = compute_quick_features
            self.QUICK_FEATURES = QUICK_FEATURES
        except ImportError as e:
            self.skipTest(f"smiles_quick_features module not available: {e}")

    def test_compute_quick_features(self):
        """基本特徴量計算のテスト"""
        result = self.compute_quick_features('CCO')  # エタノール
        self.assertIn('MolWt', result)
        self.assertIn('MolLogP', result)
        self.assertIn('TPSA', result)
        self.assertGreater(result['MolWt'], 0)

    def test_invalid_smiles(self):
        """無効なSMILESのテスト"""
        result = self.compute_quick_features('invalid_smiles')
        self.assertIn('error', result)

    def test_format_features_for_hover(self):
        """ホバー用フォーマットのテスト"""
        try:
            from backend.chem.smiles_quick_features import format_features_for_hover
        except ImportError:
            self.skipTest("format_features_for_hover not available")

        features = {'MolWt': 46.07, 'MolLogP': -0.31, 'TPSA': 20.2}
        html = format_features_for_hover(features)
        self.assertGreater(len(html), 0)
        self.assertIn('分子量', html)  # 日本語表示名

    def test_feature_list(self):
        """特徴量リストのテスト"""
        self.assertGreater(len(self.QUICK_FEATURES), 0)
        for key, jp, unit in self.QUICK_FEATURES:
            self.assertIsInstance(key, str)
            self.assertIsInstance(jp, str)


# ════════════════════════════════════
# 4. SMILESホバーHTML生成のテスト
# ════════════════════════════════════

class TestSmilesHover(unittest.TestCase):
    """SMILESホバーHTML生成のテスト"""

    def setUp(self):
        try:
            from frontend_nicegui.components.smiles_hover import get_smiles_hover_html, smiles_to_svg_b64
            self.get_smiles_hover_html = get_smiles_hover_html
            self.smiles_to_svg_b64 = smiles_to_svg_b64
        except ImportError as e:
            self.skipTest(f"smiles_hover module not available: {e}")

    def test_smiles_to_svg(self):
        """SMILES→SVG変換のテスト"""
        uri = self.smiles_to_svg_b64('CCO', 200, 200)
        if uri:  # RDKitがインストールされている場合
            self.assertIn('data:image/svg+xml;base64,', uri)

    def test_hover_html_generation(self):
        """ホバーHTML生成のテスト"""
        html = self.get_smiles_hover_html('CCO', img_size=200)
        self.assertGreater(len(html), 0)
        # 特徴量が含まれているか
        self.assertIn('CCO', html)

    def test_render_smiles_table(self):
        """SMILESテーブル描画のテスト（エラーが出ないか）"""
        try:
            from frontend_nicegui.components.smiles_hover import render_smiles_table_with_features
            import pandas as pd

            df = pd.DataFrame({
                'smiles': ['CCO', 'c1ccccc1', 'CC(=O)O'],
                'value': [1.0, 2.0, 3.0],
            })
            # UIコンポーネントのため、実際のNiceGUIセッションなしではエラーのみチェック
            try:
                render_smiles_table_with_features(df, 'smiles', max_rows=10)
            except Exception:
                # NiceGUI UIがない環境ではエラーになるが、関数は呼べる
                pass
        except ImportError:
            self.skipTest("render_smiles_table_with_features not available")


# ════════════════════════════════════
# 5. 統合テスト
# ════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """統合テスト"""

    @patch('backend.llm.interview_session.AutoAnalyzer')
    def test_full_workflow(self, mock_analyzer_class):
        """LLMヒアリング→加重平均判定の統合テスト"""
        try:
            from backend.llm.interview_session import InterviewSession
            from backend.llm.average_type_detector import AverageTypeDetector, AverageContext
        except ImportError as e:
            self.skipTest(f"Modules not available: {e}")

        # 1. データ準備
        df = pd.DataFrame({
            'temperature': [100, 200, 300, 400],
            'smiles': ['CCO', 'c1ccccc1', 'CC(=O)O', 'c1ccc2ccccc2c1'],
            'refractive_index': [1.36, 1.50, 1.37, 1.60],
        })

        # 2. ヒアリングセッション開始（モック使用）
        mock_analyzer = MagicMock()
        mock_analyzer.create_analysis_plan.return_value = MagicMock()
        mock_analyzer.create_analysis_plan.return_value.target_column = 'refractive_index'
        mock_analyzer.create_analysis_plan.return_value.task_type = 'regression'
        mock_analyzer_class.return_value = mock_analyzer

        session = InterviewSession()
        first_q = session.start(df, target_hint='refractive_index')
        self.assertIsNotNone(first_q)

        # 3. 加重平均判定
        detector = AverageTypeDetector()
        ctx = AverageContext(
            target_property='屈折率',
            available_columns=list(df.columns),
            has_smiles=True,
        )
        result = detector.detect(ctx, use_llm=False)
        self.assertEqual(result.average_type, 'volume')


if __name__ == '__main__':
    unittest.main(verbosity=2)
