"""
frontend_nicegui/components/estimator_config_dialog.py

Stub module for estimator configuration dialog.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class EstimatorConfig:
    """Configuration for an estimator."""
    model_key: str = "rf"
    model_cls: Any = None
    default_params: Dict[str, Any] = field(default_factory=dict)
    grid_space: Dict[str, Any] = field(default_factory=dict)
    optuna_space: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_key": self.model_key,
            "default_params": self.default_params,
            "grid_space": self.grid_space,
            "optuna_space": self.optuna_space,
        }


def _parse_value_list(value_str: str) -> List[Any]:
    """
    Parse a comma-separated string into a list of values.
    Automatically detects types (int, float, bool, None, str).

    Args:
        value_str: Comma-separated string (e.g., "100, 200, 500")

    Returns:
        List of parsed values with appropriate types
    """
    if not value_str or not value_str.strip():
        return []

    import ast

    # Try to parse as a Python literal first (e.g., "[1, 2, 3]")
    try:
        parsed = ast.literal_eval(value_str)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass

    # Parse as comma-separated values
    result = []
    for item in value_str.split(","):
        item = item.strip()
        if not item:
            continue

        # Try to convert to appropriate type
        if item.lower() == "none":
            result.append(None)
        elif item.lower() == "true":
            result.append(True)
        elif item.lower() == "false":
            result.append(False)
        else:
            # Try int, then float, then string
            try:
                result.append(int(item))
            except ValueError:
                try:
                    result.append(float(item))
                except ValueError:
                    result.append(item)

    return result
