#!/usr/bin/env python3
"""Fix hardcoded paths in test_eda_samples.py - v2."""

# Read the file as text
with open("tests/test_eda_samples.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Process each line
new_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("p = Path(") and "tabular_50_safe" in line:
        line = '    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_safe.csv"\n'
    elif stripped.startswith("p = Path(") and "tabular_50_simple" in line:
        line = '    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_simple.csv"\n'
    elif stripped.startswith("p = Path(") and "smiles_25_regression" in line:
        line = '    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_25_regression.csv"\n'
    elif stripped.startswith("p = Path(") and "mixture_30_simple" in line:
        line = '    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_30_simple.csv"\n'
    new_lines.append(line)

# Write back
with open("tests/test_eda_samples.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Paths fixed v2")
