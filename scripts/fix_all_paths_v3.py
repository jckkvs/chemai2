#!/usr/bin/env python3
"""Fix all hardcoded paths in test files - v3."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "tests"

# Map of filename -> list of (old_path_pattern, new_path)
fixes = {
    "test_tabular_samples.py": [
        ("tabular_50_safe.csv", 'Path(__file__).parent.parent / "data" / "samples" / "tabular_50_safe.csv"'),
        ("tabular_50_simple.csv", 'Path(__file__).parent.parent / "data" / "samples" / "tabular_50_simple.csv"'),
        ("tabular_1000_large.csv", 'Path(__file__).parent.parent / "data" / "samples" / "tabular_1000_large.csv"'),
        ("tabular_200_complex.csv", 'Path(__file__).parent.parent / "data" / "samples" / "tabular_200_complex.csv"'),
        ("smiles_25_regression.csv", 'Path(__file__).parent.parent / "data" / "samples" / "smiles_25_regression.csv"'),
        ("smiles_25_classification.csv", 'Path(__file__).parent.parent / "data" / "samples" / "smiles_25_classification.csv"'),
        ("smiles_100_regression.csv", 'Path(__file__).parent.parent / "data" / "samples" / "smiles_100_regression.csv"'),
        ("smiles_100_classification.csv", 'Path(__file__).parent.parent / "data" / "samples" / "smiles_100_classification.csv"'),
        ("mixture_30_simple.csv", 'Path(__file__).parent.parent / "data" / "samples" / "mixture_30_simple.csv"'),
        ("mixture_30_regression.csv", 'Path(__file__).parent.parent / "data" / "samples" / "mixture_30_regression.csv"'),
        ("mixture_3_30_regression.csv", 'Path(__file__).parent.parent / "data" / "samples" / "mixture_3_30_regression.csv"'),
        ("mixture_50_debug_numeric.csv", 'Path(__file__).parent.parent / "data" / "samples" / "mixture_50_debug_numeric.csv"'),
    ],
    "test_eda_samples.py": [
        ("tabular_50_safe.csv", 'Path(__file__).parent.parent / "data" / "samples" / "tabular_50_safe.csv"'),
        ("tabular_50_simple.csv", 'Path(__file__).parent.parent / "data" / "samples" / "tabular_50_simple.csv"'),
        ("smiles_25_regression.csv", 'Path(__file__).parent.parent / "data" / "samples" / "smiles_25_regression.csv"'),
        ("mixture_30_simple.csv", 'Path(__file__).parent.parent / "data" / "samples" / "mixture_30_simple.csv"'),
    ],
}

fixed_count = 0
for filename, replacements in fixes.items():
    filepath = TEST_DIR / filename
    if not filepath.exists():
        print(f"Not found: {filename}")
        continue

    content = filepath.read_text(encoding="utf-8")
    original = content

    for csv_file, new_path in replacements:
        # Find lines containing the CSV filename and a hardcoded path
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if csv_file in line and "Path(" in line:
                # Replace the entire Path() call
                # Find the start of Path( and replace until the closing )
                start = line.find("Path(")
                if start == -1:
                    start = line.find("p = Path(")
                    if start >= 0:
                        start = line.find("Path(", start)

                if start >= 0:
                    # Find the closing )
                    end = line.rfind(")")
                    if end >= 0:
                        line = line[:start] + new_path + line[end+1:]

            new_lines.append(line)

        content = "\n".join(new_lines)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        fixed_count += 1
        print(f"Fixed: {filename}")
    else:
        print(f"No changes: {filename}")

print(f"\nTotal fixed: {fixed_count}")
