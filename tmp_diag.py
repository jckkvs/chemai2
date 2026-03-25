# -*- coding: utf-8 -*-
"""SMARTSパターンの診断"""
import sys
sys.path.insert(0, '.')
from rdkit import Chem

# 問題のあるSMILES
test = {
    'O':        'Water',
    'CC(=O)O':  'AceticAcid',
    'CC(=O)C':  'Acetone',
}

for smi, name in test.items():
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    print(f"\n=== {name} ({smi}) ===")
    for a in mol.GetAtoms():
        print(f"  idx={a.GetIdx()} sym={a.GetSymbol()} H={a.GetTotalNumHs()} X={a.GetDegree()}")

    # COOH pattern test
    patterns = {
        'COOH': "[CX3](=O)[OX2H1]",
        'OH generic': "[OX2H1]",
        'OH excl': "[OX2H1;!$([OX2H1][CX3]=O)]",
        'Ketone': "[CX3H0](=O)([#6])[#6]",
        'Ketone v2': "[CX3](=O)([#6])[#6]",
        'C=O generic': "[CX3]=O",
        'water OH': "[OX2H2]",
    }
    for pname, smart in patterns.items():
        pat = Chem.MolFromSmarts(smart)
        if pat is None:
            print(f"  {pname}: INVALID SMARTS")
            continue
        matches = mol.GetSubstructMatches(pat)
        print(f"  {pname}: {len(matches)} matches -> {matches}")
