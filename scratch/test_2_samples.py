
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.models.automl import AutoMLEngine
from backend.utils.config import default_config

# Disable logging to verify "no warnings"
logging.basicConfig(level=logging.ERROR)

def test_2_samples():
    print("\n--- Testing with 2 samples ---")
    df = pd.DataFrame({
        "SMILES": ["CCO", "CCC"],
        "target": [10.0, 20.0],
    })
    
    engine = AutoMLEngine(task="regression", cv_folds=5)
    try:
        # Should work without raising ValueError for size < 10 or < 3
        # Should work with cv_folds auto-adjusted to 2
        result = engine.run(df, target_col="target", smiles_col="SMILES")
        print(f"Success! Best model: {result.best_model_key}, Score: {result.best_score}")
        print(f"Final CV folds used: {engine.cv_folds}")
    except Exception as e:
        print(f"Failed with 2 samples: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_2_samples()
