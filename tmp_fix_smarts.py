# -*- coding: utf-8 -*-
"""SMARTS修正スクリプト: hsp_predictor.pyのパターンとマッチロジックを修正"""

fp = 'C:/Users/horie/chemai2/backend/hsp/hsp_predictor.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. 水(H2O)のパターンを追加 — OHの前に配置
old_oh = '''    GroupContribution(
        name="ヒドロキシル基 —OH",
        smarts="[OX2H1;!$([OX2H1][CX3]=O)]",'''

new_oh = '''    GroupContribution(
        name="水 H2O",
        smarts="[OX2H2]",
        fdi=210, fpi=500, ehi=20000, vi=14.0,
    ),
    GroupContribution(
        name="ヒドロキシル基 -OH",
        smarts="[OX2H1;!$([OX2H1][CX3]=O)]",'''

if old_oh in content:
    content = content.replace(old_oh, new_oh, 1)
    changes += 1
else:
    print("WARNING: OH pattern not found")

# 2. 日本語emダッシュ(—)をASCIIダッシュ(-)に置換(cp932問題対策)
content = content.replace('—', '-')
changes += 1

# 3. マッチロジック修正: anchor原子のみ → 全マッチ原子をマッチ済みに
old_match = '''                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    # 最初の原子（アンカー）がまだマッチしていない場合のみカウント
                    anchor = match[0]
                    if not matched[anchor]:
                        matched[anchor] = True
                        sum_fdi += gc.fdi
                        sum_fpi2 += gc.fpi ** 2
                        sum_ehi += gc.ehi
                        sum_vi += gc.vi
                        matched_groups.append(gc.name)'''

new_match = '''                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    # アンカー原子(match[0])がまだ未マッチの場合のみカウント
                    anchor = match[0]
                    if not matched[anchor]:
                        # 全マッチ原子をマッチ済みに(多原子基の二重カウント防止)
                        for idx in match:
                            atom = mol.GetAtomWithIdx(idx)
                            if atom.GetAtomicNum() > 1:  # 水素以外
                                matched[idx] = True
                        sum_fdi += gc.fdi
                        sum_fpi2 += gc.fpi ** 2
                        sum_ehi += gc.ehi
                        sum_vi += gc.vi
                        matched_groups.append(gc.name)'''

if old_match in content:
    content = content.replace(old_match, new_match, 1)
    changes += 1
else:
    print("WARNING: match logic not found")

# 4. ケトンのSMARTSを改善 — H0制約を外した汎用版を追加
old_ketone = '''    GroupContribution(
        name="ケトン -C(=O)-",
        smarts="[CX3H0](=O)([#6])[#6]",
        fdi=290, fpi=770, ehi=2000, vi=10.8,
    ),'''

new_ketone = '''    GroupContribution(
        name="ケトン -C(=O)-",
        smarts="[CX3;!$([CX3][OX2H1]);!$([CX3][OX2H0][#6])](=O)[#6]",
        fdi=290, fpi=770, ehi=2000, vi=10.8,
    ),'''

if old_ketone in content:
    content = content.replace(old_ketone, new_ketone, 1)
    changes += 1
else:
    print("WARNING: ketone pattern not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes applied")
