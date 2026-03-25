# -*- coding: utf-8 -*-
"""HSP Group Contribution法の検証スクリプト"""
import sys
sys.path.insert(0, '.')
from backend.hsp.hsp_predictor import HSPPredictor

pred = HSPPredictor()

# テスト化合物と文献値 (MPa^0.5)
test_cases = [
    ('CCO',         'Ethanol',     15.8, 8.8,  19.4),
    ('Cc1ccccc1',   'Toluene',     18.0, 1.4,  2.0),
    ('CC(=O)C',     'Acetone',     15.5, 10.4, 7.0),
    ('CC(=O)O',     'AceticAcid',  14.5, 8.0,  13.5),
    ('c1ccccc1',    'Benzene',     18.4, 0.0,  2.0),
    ('O',           'Water',       15.6, 16.0, 42.3),
    ('CCCCCC',      'Hexane',      14.9, 0.0,  0.0),
    ('ClC(Cl)Cl',   'Chloroform',  17.8, 3.1,  5.7),
    ('CCOC(=O)C',   'EthylAcetate',15.8, 5.3,  7.2),
]

print("=" * 100)
print("HSP van Krevelen/Hoftyzer GC Verification")
print("=" * 100)

for smi, name, lit_d, lit_p, lit_h in test_cases:
    try:
        r = pred.predict(smi)
        d, p, h = r['delta_d'], r['delta_p'], r['delta_h']
        conf = r.get('confidence', '?')
        cov = r.get('coverage', 0)
        groups = r.get('matched_groups', [])
        err_d = abs(d - lit_d)
        err_p = abs(p - lit_p)
        err_h = abs(h - lit_h)
        status = "OK" if max(err_d, err_p, err_h) < 5.0 else "WARN"
        print(f"[{status}] {name:15s} ({smi:20s})")
        print(f"  calc: dD={d:6.2f}  dP={p:6.2f}  dH={h:6.2f}")
        print(f"  lit:  dD={lit_d:6.2f}  dP={lit_p:6.2f}  dH={lit_h:6.2f}")
        print(f"  err:  dD={err_d:6.2f}  dP={err_p:6.2f}  dH={err_h:6.2f}  conf={conf} cov={cov:.0%}")
        print(f"  groups: {', '.join(groups[:5])}")
        print()
    except Exception as ex:
        print(f"[ERR] {name:15s} ({smi:20s}) -> {ex}")
        print()
