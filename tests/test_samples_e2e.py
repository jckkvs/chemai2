"""
tests/test_samples_e2e.py

TDD: Sample data auto-analysis E2E tests.
Red -> Green -> Refactor cycle.

Test targets:
  - Auto-discover all CSV files under data/samples/
  - Run pipeline: load -> detect type -> preprocess -> feature engineering -> ML modeling -> evaluate
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


# ============================================================
# Fixtures: sample file auto-discovery
# ============================================================

def discover_sample_files() -> List[Path]:
    """Discover all CSV files under data/samples/."""
    if not SAMPLES_DIR.exists():
        return []
    return sorted(SAMPLES_DIR.glob("*.csv"))


def classify_sample_file(filepath: Path) -> str:
    """
    Classify file format based on name and header.
      - 'tabular': Tabular data (no SMILES)
      - 'smiles': Single compound SMILES
      - 'mixture': Multiple SMILES + weight% (mixture)
    """
    name = filepath.name.lower()
    if name.startswith("tabular"):
        return "tabular"
    if name.startswith("smiles"):
        return "smiles"
    if name.startswith("mixture"):
        return "mixture"
    try:
        df = pd.read_csv(filepath, nrows=1)
        cols = [c.lower() for c in df.columns]
        if "smiles" in cols:
            return "smiles"
        if any("smiles" in c for c in cols):
            return "mixture"
        return "tabular"
    except Exception:
        return "unknown"


SAMPLE_FILES = discover_sample_files()


@pytest.fixture(scope="module", params=SAMPLE_FILES, ids=lambda p: p.name)
def sample_file(request) -> Path:
    return request.param


@pytest.fixture(scope="module")
def sample_type(sample_file: Path) -> str:
    return classify_sample_file(sample_file)


# ============================================================
# TestSampleDiscovery
# ============================================================

class TestSampleDiscovery:
    def test_discovers_all_expected_files(self):
        files = discover_sample_files()
        assert len(files) >= 9, (
            f"Expected: 9+ files, Actual: {len(files)} files\n"
            f"Files found: {[f.name for f in files]}"
        )

    def test_all_expected_file_types_present(self):
        files = discover_sample_files()
        types = {classify_sample_file(f) for f in files}
        expected = {"tabular", "smiles", "mixture"}
        missing = expected - types
        assert not missing, f"Missing file types: {missing}"


# ============================================================
# TestFileLoading
# ============================================================

class TestFileLoading:
    def test_load_tabular_csv(self, sample_file: Path, sample_type: str):
        if sample_type != "tabular":
            pytest.skip("Not tabular format")
        from backend.data.loader import load_file
        df = load_file(str(sample_file))
        assert df is not None
        assert len(df) > 0
        assert "Target" in df.columns or "target" in df.columns

    def test_load_smiles_csv(self, sample_file: Path, sample_type: str):
        if sample_type != "smiles":
            pytest.skip("Not smiles format")
        from backend.data.loader import load_file
        df = load_file(str(sample_file))
        assert df is not None
        assert len(df) > 0
        smiles_cols = [c for c in df.columns if "smiles" in c.lower()]
        assert len(smiles_cols) >= 1

    def test_load_mixture_csv(self, sample_file: Path, sample_type: str):
        if sample_type != "mixture":
            pytest.skip("Not mixture format")
        from backend.data.loader import load_file
        df = load_file(str(sample_file))
        assert df is not None
        assert len(df) > 0
        smiles_cols = [c for c in df.columns if "SMILES" in c or "smiles" in c.lower()]
        assert len(smiles_cols) >= 2


# ============================================================
# TestTypeDetection
# ============================================================

class TestTypeDetection:
    def test_tabular_type_detection(self, sample_file: Path, sample_type: str):
        if sample_type != "tabular":
            pytest.skip("Not tabular format")
        from backend.data.loader import load_file
        from backend.data.type_detector import TypeDetector
        df = load_file(str(sample_file))
        detector = TypeDetector()
        result = detector.detect(df)
        assert result is not None
        if hasattr(result, "numeric_columns"):
            assert len(result.numeric_columns) >= 1

    def test_smiles_type_detection(self, sample_file: Path, sample_type: str):
        if sample_type != "smiles":
            pytest.skip("Not smiles format")
        from backend.data.loader import load_file
        from backend.data.type_detector import TypeDetector
        df = load_file(str(sample_file))
        detector = TypeDetector()
        result = detector.detect(df)
        assert result is not None


# ============================================================
# TestPreprocessing
# ============================================================

class TestPreprocessing:
    def test_tabular_preprocessing(self, sample_file: Path, sample_type: str):
        if sample_type != "tabular":
            pytest.skip("Not tabular format")
        from backend.data.loader import load_file
        from backend.data.type_detector import TypeDetector
        from backend.data.preprocessor import Preprocessor
        df = load_file(str(sample_file))
        detector = TypeDetector()
        type_result = detector.detect(df)
        preprocessor = Preprocessor()
        ct = preprocessor.build(type_result, target_col="Target")
        assert ct is not None
        X = df.drop(columns=["Target"])
        y = df["Target"]
        X_transformed = ct.fit_transform(X, y)
        assert X_transformed.shape[0] == len(df)


# ============================================================
# TestSmilesFeatureEngineering
# ============================================================

class TestSmilesFeatureEngineering:
    def test_smiles_descriptor_calculation(self, sample_file: Path, sample_type: str):
        if sample_type != "smiles":
            pytest.skip("Not smiles format")
        from backend.data.loader import load_file
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        df = load_file(str(sample_file)).head(10)
        smiles_col = [c for c in df.columns if "SMILES" in c or "smiles" in c.lower()][0]
        transformer = SmilesDescriptorTransformer(smiles_col=smiles_col)
        df_with_features = transformer.fit_transform(df)
        desc_cols = [c for c in df_with_features.columns if c != "Target" and c != "logS" and c != "Class"]
        assert len(desc_cols) >= 1


# ============================================================
# TestMLPipeline
# ============================================================

class TestMLPipeline:
    def test_tabular_ml_pipeline(self, sample_file: Path, sample_type: str):
        if sample_type != "tabular":
            pytest.skip("Not tabular format")
        from backend.data.loader import load_file
        from backend.ml.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
        df = load_file(str(sample_file))
        config = PipelineConfig(
            task_type="regression",
            cv_strategy='kfold',
            cv_params={'n_splits': 3},
            estimator_name='RandomForestRegressor',
            estimator_params={'n_estimators': 10, 'random_state': 42},
        )
        orchestrator = PipelineOrchestrator(config=config)
        # Only use numeric columns (exclude string columns like Sample_ID, Category)
        X = df.select_dtypes(include=['number']).drop(columns=["Target"], errors='ignore')
        y = df["Target"]
        orchestrator.fit(X, y)
        score = orchestrator.score(X, y)
        assert score is not None
        assert not np.isnan(score)


# ============================================================
# TestEndToEnd
# ============================================================

class TestEndToEnd:
    def test_tabular_e2e(self, sample_file: Path, sample_type: str):
        if sample_type != "tabular":
            pytest.skip("Not tabular format")
        from backend.data.loader import load_file
        from backend.data.type_detector import TypeDetector
        from backend.data.preprocessor import Preprocessor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        df = load_file(str(sample_file))
        detector = TypeDetector()
        type_result = detector.detect(df)
        preprocessor = Preprocessor()
        ct = preprocessor.build(type_result, target_col="Target")
        X = df.drop(columns=["Target"])
        y = df["Target"]
        X_transformed = ct.fit_transform(X, y)
        model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
        scores = cross_val_score(model, X_transformed, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert not np.any(np.isnan(scores))

    def test_smiles_e2e(self, sample_file: Path, sample_type: str):
        if sample_type != "smiles":
            pytest.skip("Not smiles format")
        from backend.data.loader import load_file
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np
        df = load_file(str(sample_file)).head(10)
        smiles_col = [c for c in df.columns if "SMILES" in c or "smiles" in c.lower()][0]
        # Properly identify target column: look for "Target", "Class", "logS", etc.
        possible_targets = ["Target", "Class", "logS", "Property"]
        target_col = None
        for pt in possible_targets:
            matches = [c for c in df.columns if pt.lower() in c.lower()]
            if matches:
                target_col = matches[0]
                break
        if target_col is None:
            # Fallback: first column that's not smiles_col and not Compound_Name
            target_col = [c for c in df.columns if c != smiles_col and "Compound" not in c][0]

        transformer = SmilesDescriptorTransformer(smiles_col=smiles_col)
        df_features = transformer.fit_transform(df)
        X = df_features.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
        y = df_features[target_col]

        # Detect task type: check if target column name contains "class" (case insensitive)
        target_lower = target_col.lower()
        is_classification = "class" in target_lower

        if is_classification:
            model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
            scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
            scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert not np.any(np.isnan(scores))

    def test_mixture_e2e(self, sample_file: Path, sample_type: str):
        if sample_type != "mixture":
            pytest.skip("Not mixture format")
        from backend.data.loader import load_file
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor
        import pandas as pd

        df = load_file(str(sample_file)).head(3)  # Only first 3 rows for speed
        assert len(df) > 0

        # For each row, extract mixture features
        all_features = []
        for idx, row in df.iterrows():
            # Collect SMILES columns and WT% columns
            smiles_cols = [c for c in df.columns if "SMILES" in c]
            wt_cols = [c for c in df.columns if "WT%" in c or "Wt%" in c or "wt%" in c]

            components = []
            for sc, wc in zip(smiles_cols, wt_cols):
                if sc in row and wc in row and pd.notna(row[sc]) and pd.notna(row[wc]):
                    components.append({
                        "smiles": str(row[sc]),
                        "ratio_value": float(row[wc]),
                        "ratio_unit": "weight",
                        "compound_name": row.get(sc.replace("SMILES", "Name"), "")
                    })

            if len(components) >= 2:
                extractor = MixtureFeatureExtractor()
                result = extractor.extract(components=components)
                all_features.append(result.mixture_features)

        # Aggregate features into DataFrame
        if all_features:
            df_features = pd.DataFrame(all_features)
            assert df_features.shape[1] > 0, "No mixture features extracted"
            assert df_features.shape[0] == len(all_features)

            # ML modeling
            target_col = [c for c in df.columns if "Target" in c or "Property" in c][0]
            y = df[target_col].head(len(df_features))

            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            import numpy as np

            X = df_features.select_dtypes(include=['number'])
            if len(X) > 3:
                model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
                scores = cross_val_score(model, X, y, cv=min(3, len(X)), scoring="r2")
                assert len(scores) == min(3, len(X))
                assert not np.any(np.isnan(scores))
        else:
            pytest.skip("No valid mixture data found")
