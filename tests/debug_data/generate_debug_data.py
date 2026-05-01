"""
Generate debug test datasets for ChemAI ML Studio.
Creates various SMILES and numeric data patterns for development testing.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent

# 40 known valid SMILES
SMILES_LIST = [
    "CC(=O)O", "CCO", "CCCCO", "c1ccccc1", "CC(=O)c1ccccc1",
    "CCN", "CC(C)O", "c1ccccc1O", "CC(=O)N", "CCCCN",
    "c1ccc2ccccc2c1", "CC(C)C", "CC(C)CO", "c1ccccc1C", "CCOC(=O)C",
    "Nc1ccccc1", "CSc1ccccc1", "CC(=O)OC", "Cc1ccccc1O", "CC(=O)c1ccc2ccccc2c1",
    "CN1C=NC2=CC=CC=C12", "CC(C)N", "C1=CC=CC=C1", "Clc1ccccc1", "Fc1ccccc1",
    "CC(=O)Oc1ccccc1", "CC(C)COc1ccccc1", "Cc1ccc2c(c1)CCO2", "CC(C)Cc1ccccc1",
    "C1=CC=C(C)C=C1", "CC(C)Nc1ccccc1", "Cc1ccc2ccccc2c1O", "CC(=O)Cc1ccccc1",
]

def generate_pure_smiles(n=100):
    """Pure SMILES + target (regression)"""
    smiles = np.random.choice(SMILES_LIST, size=n, replace=True)
    target = np.random.randn(n) * 2 + 5
    df = pd.DataFrame({"SMILES": smiles, "target": target})
    df.to_csv(OUTPUT_DIR / "pure_smiles.csv", index=False)
    print(f"Created pure_smiles.csv: {df.shape}")

def generate_pure_numeric(n=100):
    """Pure numeric features + target (no SMILES)"""
    data = {}
    for i in range(5):
        data[f"feature_{i}"] = np.random.randn(n) * 10 + 50
    data["target"] = (
        2.5 * data["feature_0"]
        - 1.3 * data["feature_1"]
        + 0.8 * data["feature_2"]
        + np.random.randn(n) * 2
    )
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_DIR / "pure_numeric.csv", index=False)
    print(f"Created pure_numeric.csv: {df.shape}")

def generate_smiles_plus_numeric(n=100):
    """1 SMILES + 3 numeric features + target"""
    smiles = np.random.choice(SMILES_LIST, size=n, replace=True)
    df = pd.DataFrame({
        "SMILES": smiles,
        "numeric_1": np.random.randn(n) * 10 + 50,
        "numeric_2": np.random.randn(n) * 5 + 20,
        "numeric_3": np.random.rand(n) * 100,
    })
    df["target"] = (
        1.5 * df["numeric_1"]
        - 0.8 * df["numeric_2"]
        + np.random.randn(n) * 3
    )
    df.to_csv(OUTPUT_DIR / "smiles_plus_numeric.csv", index=False)
    print(f"Created smiles_plus_numeric.csv: {df.shape}")

def generate_multi_smiles_mixture(n=50):
    """2 SMILES columns per sample + target (mixture data)"""
    df = pd.DataFrame({
        "SMILES_A": np.random.choice(SMILES_LIST, size=n, replace=True),
        "SMILES_B": np.random.choice(SMILES_LIST, size=n, replace=True),
        "target": np.random.randn(n) * 3 + 10,
    })
    df.to_csv(OUTPUT_DIR / "multi_smiles_mixture.csv", index=False)
    print(f"Created multi_smiles_mixture.csv: {df.shape}")

def generate_multi_smiles_weighted(n=50):
    """2 SMILES + 2 weight columns + target (weighted average)"""
    df = pd.DataFrame({
        "SMILES_A": np.random.choice(SMILES_LIST, size=n, replace=True),
        "SMILES_B": np.random.choice(SMILES_LIST, size=n, replace=True),
        "weight_A": np.random.rand(n),
        "weight_B": None,  # calculated below
        "target": np.random.randn(n) * 2 + 7,
    })
    df["weight_B"] = 1.0 - df["weight_A"]
    df.to_csv(OUTPUT_DIR / "multi_smiles_weighted_avg.csv", index=False)
    print(f"Created multi_smiles_weighted_avg.csv: {df.shape}")

def generate_invalid_smiles(n=100):
    """Pure SMILES with 10% invalid SMILES"""
    smiles = np.random.choice(SMILES_LIST, size=n, replace=True)
    # Make 10% invalid
    invalid_indices = np.random.choice(n, size=n // 10, replace=False)
    for idx in invalid_indices:
        smiles[idx] = "C1C"  # Invalid SMILES
    df = pd.DataFrame({
        "SMILES": smiles,
        "target": np.random.randn(n) * 2 + 5,
    })
    df.to_csv(OUTPUT_DIR / "invalid_smiles.csv", index=False)
    print(f"Created invalid_smiles.csv: {df.shape}")

def generate_missing_values(n=100):
    """SMILES + numeric with 5% missing values"""
    df = pd.DataFrame({
        "SMILES": np.random.choice(SMILES_LIST, size=n, replace=True),
        "numeric_1": np.random.randn(n) * 10 + 50,
        "numeric_2": np.random.randn(n) * 5 + 20,
        "numeric_3": np.random.rand(n) * 100,
        "target": np.random.randn(n) * 3 + 10,
    })
    # Introduce 5% missing values
    for col in ["numeric_1", "numeric_2", "numeric_3"]:
        mask = np.random.rand(n) < 0.05
        df.loc[mask, col] = np.nan
    df.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)
    print(f"Created missing_values.csv: {df.shape}")

def generate_small_dataset(n=10):
    """Small dataset for quick testing"""
    df = pd.DataFrame({
        "SMILES": np.random.choice(SMILES_LIST, size=n, replace=True),
        "feature_1": np.random.randn(n) * 10,
        "feature_2": np.random.rand(n) * 100,
        "target": np.random.randn(n) * 2 + 5,
    })
    df.to_csv(OUTPUT_DIR / "small_dataset.csv", index=False)
    print(f"Created small_dataset.csv: {df.shape}")

if __name__ == "__main__":
    print("Generating debug test datasets...")
    generate_pure_smiles()
    generate_pure_numeric()
    generate_smiles_plus_numeric()
    generate_multi_smiles_mixture()
    generate_multi_smiles_weighted()
    generate_invalid_smiles()
    generate_missing_values()
    generate_small_dataset()
    print(f"\nAll debug datasets created in: {OUTPUT_DIR}")
