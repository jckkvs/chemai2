#!/usr/bin/env python3
"""Fix all hardcoded paths in test files."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_file(filepath: Path):
    """Fix hardcoded paths in a test file."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Pattern: Path(r"C:\Users\horie\cc\chemai2_cc\data\samples\filename")
    # Replace with: Path(__file__).parent.parent / "data" / "samples" / "filename"

    # Find all Path(...) patterns with hardcoded C:\Users...data\samples
    pattern = r'Path\(r"C:\\\\Users\\\\horie\\\\cc\\\\chemai2_cc\\\\data\\\\samples\\\\([^"]+)"\)'

    def replace_fn(match):
        filename = match.group(1)
        return f'Path(__file__).parent.parent / "data" / "samples" / "{filename}"'

    new_content = re.sub(pattern, replace_fn, content)

    if new_content != original:
        filepath.write_text(new_content, encoding="utf-8")
        return True
    return False


# Fix all test files
test_files = [
    PROJECT_ROOT / "tests" / "test_tabular_samples.py",
    PROJECT_ROOT / "tests" / "test_samples_e2e.py",
    PROJECT_ROOT / "tests" / "test_smiles_25_regression.py",
]

fixed = []
for tf in test_files:
    if tf.exists():
        if fix_file(tf):
            fixed.append(tf.name)
            print(f"Fixed: {tf.name}")
    else:
        print(f"Not found: {tf.name}")

print(f"\nTotal fixed: {len(fixed)}")
