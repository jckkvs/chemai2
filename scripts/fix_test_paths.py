#!/usr/bin/env python3
"""Fix hardcoded paths in test_eda_samples.py."""

with open("tests/test_eda_samples.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the fixture paths to use relative paths
old1 = r'    p = Path(r"C:\Users\horie\cc\chemai2_cc\data\samples\tabular_50_safe.csv")'
new1 = '    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_safe.csv"'
content = content.replace(old1, new1)

old2 = r'    p = Path(r"C:\Users\horie\cc\chemai2_cc\data\samples\tabular_50_simple.csv")'
new2 = '    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_simple.csv"'
content = content.replace(old2, new2)

old3 = r'    p = Path(r"C:\Users\horie\cc\chemai2_cc\data\samples\smiles_25_regression.csv")'
new3 = '    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_25_regression.csv"'
content = content.replace(old3, new3)

old4 = r'    p = Path(r"C:\Users\horie\cc\chemai2_cc\data\samples\mixture_30_simple.csv")'
new4 = '    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_30_simple.csv"'
content = content.replace(old4, new4)

with open("tests/test_eda_samples.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Paths fixed successfully")
