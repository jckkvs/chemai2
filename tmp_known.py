# -*- coding: utf-8 -*-
"""既知化合物テーブルを_compute_hsp_group_contributionに追加"""

fp = 'C:/Users/horie/chemai2/backend/hsp/hsp_predictor.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 既知化合物テーブルを追加
old = '''        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("無効なSMILES: %s", smiles[:50])
            return None

        # 水素を付加して正確な原子数を得る
        mol = Chem.AddHs(mol)'''

new = '''        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("無効なSMILES: %s", smiles[:50])
            return None

        # -- 既知化合物テーブル (Hansen 2007, Appendix A) --
        # GC法が不得意な小分子/特殊分子は文献値を直接返す
        canonical = Chem.MolToSmiles(mol)
        _KNOWN_HSP = {
            # canonical SMILES: (delta_d, delta_p, delta_h)
            "O":             (15.5, 16.0, 42.3),    # Water
            "ClC(Cl)Cl":     (17.8,  3.1,  5.7),    # Chloroform
            "Cl":            (15.8,  2.0,  0.2),    # HCl
            "ClCCl":         (16.4,  6.3,  3.0),    # DCM
            "ClC(Cl)(Cl)Cl": (15.8,  0.0,  0.0),    # CCl4
            "CS(C)=O":       (18.4, 16.4, 10.2),    # DMSO
            "CN(C)C=O":      (17.4, 13.7, 11.3),    # DMF
            "O=CO":          (15.6,  5.1,  8.4),    # Formic acid
            "N":             (15.5, 13.0,  5.2),    # NH3
        }
        if canonical in _KNOWN_HSP:
            d, p, h = _KNOWN_HSP[canonical]
            return {
                "delta_d": d, "delta_p": p, "delta_h": h,
                "delta_total": float(np.sqrt(d**2 + p**2 + h**2)),
                "molar_volume": None,
                "method": "known_compound",
                "confidence": "reference",
                "matched_groups": [],
                "unmatched_atoms": 0,
                "coverage": 1.0,
            }

        # 水素を付加して正確な原子数を得る
        mol = Chem.AddHs(mol)'''

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
else:
    print("WARNING: target not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes")
