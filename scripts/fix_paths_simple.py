#!/usr/bin/env python3
"""Fix hardcoded paths in test_eda_samples.py - simple version."""

with open("tests/test_eda_samples.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'C:' in line and 'schemai2_cc' in line and 'samples' in line:
        # Extract the filename
        if 'tabular_50_safe' in line:
            line = '    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_safe.csv"\n'
        elif 'tabular_50_simple' in line:
            line = '    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_simple.csv"\n'
        elif 'smiles_25_regression' in line:
            line = '    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_25_regression.csv"\n'
        elif 'mixture_30_simple' in line:
            line = '    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_30_simple.csv"\n'
    new_lines.append(line)

with open("tests/test_eda_samples.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Paths fixed")
