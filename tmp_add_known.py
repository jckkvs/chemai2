# -*- coding: utf-8 -*-
"""アセトン等を既知テーブルに追加"""

fp = 'C:/Users/horie/chemai2/backend/hsp/hsp_predictor.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# アセトン、メタノール、エタノールをテーブルに追加
old = '''            "N":             (15.5, 13.0,  5.2),    # NH3
        }'''

new = '''            "N":             (15.5, 13.0,  5.2),    # NH3
            "CC(C)=O":       (15.5, 10.4,  7.0),    # Acetone
            "CO":            (15.1, 12.3, 22.3),    # Methanol
            "CCO":           (15.8,  8.8, 19.4),    # Ethanol
            "CC(C)O":        (15.1,  6.1, 16.4),    # IPA
            "CCCCCC":        (14.9,  0.0,  0.0),    # Hexane
            "c1ccccc1":      (18.4,  0.0,  2.0),    # Benzene
            "Cc1ccccc1":     (18.0,  1.4,  2.0),    # Toluene
        }'''

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
else:
    print("WARNING: target not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes")
