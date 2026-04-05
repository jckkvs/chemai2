#!/usr/bin/env python3
"""
サンプルデータ生成スクリプト
- SMILES系データ：RDKit記述子から擬似目的変数を生成
- テーブルデータ：scikit-learnのmake_regression + 人工的ノイズ・欠損値追加
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from sklearn.datasets import make_regression, make_classification
import os

# ========== SMILESデータ生成 ==========
SMILES_POOL = [
    # 医薬品
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Naproxen
    # 溶媒・試薬
    "CCO", "CC(C)O", "CC(C)(C)O", "c1ccccc1", "CC1=CC=CC=C1",
    # 天然物・複雑構造
    "CC1=CC2=C(C=C1C=O)C(=O)C3=C(O2)C=CC(=C3)O",  # Simple flavonoid
    "CCC1=CC(=C(C=C1)O)C=O",  # Vanillin
    # 極端ケース
    "C"*50,  # 長鎖アルカン（エラーテスト）
    "C1CC1C1CC1C1CC1",  # 複雑な環構造
]

def calc_pseudo_targets(smiles: str) -> dict:
    """SMILESから擬似目的変数を計算（物性値の相関を模擬）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"logS": np.nan, "pIC50": np.nan, "class": -1}
    
    # RDKit記述子
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    
    # 擬似logS（水溶性）: MW↑・logP↑・TPSA↑ → 溶解度↓
    logS = -0.01*mw - 0.3*logp - 0.005*tpsa + np.random.normal(0, 0.3)
    
    # 擬似pIC50（活性）: 複雑な構造ほど高い活性を模擬
    pIC50 = 3 + 0.005*mw + 0.2*logp - 0.1*hbd + np.random.normal(0, 0.5)
    
    # 分類ラベル（logP閾値ベース）
    cls = 1 if logp > 2.5 else 0
    
    return {
        "logS": round(logS, 3),
        "pIC50": round(pIC50, 3),
        "class": cls,
        "MW": round(mw, 2),
        "LogP": round(logp, 2)
    }

def generate_smiles_samples(n: int, filename: str):
    data = []
    for i in range(n):
        smiles = SMILES_POOL[i % len(SMILES_POOL)]
        targets = calc_pseudo_targets(smiles)
        data.append({
            "SMILES": smiles,
            "Compound_Name": f"Compound_{i+1:03d}",
            "logS": targets["logS"],
            "pIC50": targets["pIC50"],
            "Class": targets["class"],
            "Source": "Synthetic",
            "Notes": "Generated for testing"
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")

# ========== 通常テーブルデータ生成 ==========
def generate_tabular_samples(n: int, filename: str, task: str = "regression"):
    if task == "regression":
        X, y = make_regression(
            n_samples=n, n_features=8, n_informative=5,
            noise=10, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Target"] = y
        df["Sample_ID"] = [f"S{i+1:04d}" for i in range(n)]
        
        # カテゴリカル特徴量の追加
        df["Category"] = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n)
        
        # 欠損値の人工的追加（5%）
        mask = np.random.random((n, 3)) < 0.05
        for col_idx in range(3):
            df.iloc[mask[:, col_idx], col_idx] = np.nan
            
    else:  # classification
        X, y = make_classification(
            n_samples=n, n_features=10, n_informative=7,
            n_redundant=2, n_classes=3, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Target_Class"] = y
        df["Sample_ID"] = [f"S{i+1:04d}" for i in range(n)]
    
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")

# ========== 実行 ==========
if __name__ == "__main__":
    os.makedirs("data/samples", exist_ok=True)
    
    # SMILESデータ
    generate_smiles_samples(25, "data/samples/smiles_25_quick.csv")
    generate_smiles_samples(100, "data/samples/smiles_100_ml.csv")
    generate_smiles_samples(500, "data/samples/smiles_500_stress.csv")
    
    # 通常テーブルデータ
    generate_tabular_samples(50, "data/samples/tabular_50_simple.csv", "regression")
    generate_tabular_samples(200, "data/samples/tabular_200_complex.csv", "regression")
    generate_tabular_samples(1000, "data/samples/tabular_1000_large.csv", "regression")
    
    print("\n🎉 All sample data generated successfully!")
