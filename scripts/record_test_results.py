#!/usr/bin/env python3
"""Record pytest results to JSON history for trend visualization."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path(__file__).parent.parent / "test_results_history.json"


def run_tests():
    """Run pytest and return parsed results."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_samples_e2e.py",
            "tests/test_tabular_samples.py",
            "tests/test_smiles_25_regression.py",
            "tests/test_eda_samples.py",
            "--tb=no", "-q",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    output = result.stdout + result.stderr
    return parse_results(output)


def parse_results(output: str) -> dict:
    """Parse pytest output to extract counts."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "warnings": 0,
        "duration_seconds": 0.0,
        "raw_output": output[-2000:] if len(output) > 2000 else output,
    }

    # Parse summary line like "62 passed, 54 failed, 90 skipped, 951 warnings, 116 errors in 108.12s"
    for line in output.splitlines():
        line = line.strip()
        if "passed" in line and "failed" in line:
            # Extract numbers
            import re
            passed = re.search(r"(\d+)\s+passed", line)
            failed = re.search(r"(\d+)\s+failed", line)
            skipped = re.search(r"(\d+)\s+skipped", line)
            errors = re.search(r"(\d+)\s+errors?", line)
            warnings = re.search(r"(\d+)\s+warnings?", line)
            duration = re.search(r"in\s+([\d.]+)s", line)

            if passed:
                data["passed"] = int(passed.group(1))
            if failed:
                data["failed"] = int(failed.group(1))
            if skipped:
                data["skipped"] = int(skipped.group(1))
            if errors:
                data["errors"] = int(errors.group(1))
            if warnings:
                data["warnings"] = int(warnings.group(1))
            if duration:
                data["duration_seconds"] = float(duration.group(1))
            break

    return data


def load_history() -> list:
    """Load existing history or create empty."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list):
    """Save history to JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def main():
    print("Running tests and recording results...")
    result = run_tests()
    history = load_history()
    history.append(result)
    save_history(history)
    print(f"Recorded: {result['passed']} passed, {result['failed']} failed, "
          f"{result['skipped']} skipped, {result['errors']} errors "
          f"({result['duration_seconds']:.1f}s)")
    print(f"Total records: {len(history)}")


if __name__ == "__main__":
    main()
