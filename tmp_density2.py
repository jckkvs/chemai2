# -*- coding: utf-8 -*-
"""transform()に密度変換呼び出しを追加"""

fp = 'C:/Users/horie/chemai2/backend/chem/smiles_transformer.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# transform()内でX_chem計算直後に密度変換を挿入
old = '''        smiles_list = X[self.smiles_col].tolist()
        X_chem = self._compute_descriptors(smiles_list)

        # 記述子カラムをfitで記憶した順序・列に揃える（推論時の列不一致を防ぐ）'''

new = '''        smiles_list = X[self.smiles_col].tolist()
        X_chem = self._compute_descriptors(smiles_list)
        X_chem = self._apply_count_normalization(X_chem, smiles_list)

        # 記述子カラムをfitで記憶した順序・列に揃える（推論時の列不一致を防ぐ）'''

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
else:
    print("WARNING: transform target not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes")
