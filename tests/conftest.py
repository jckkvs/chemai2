"""
Test Configuration & Fixtures - chemai2/tests/conftest.py
Shared pytest fixtures, hypothesis strategies, and async test utilities
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import pytest
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from unittest.mock import AsyncMock, patch

# Optional imports with graceful fallback
try:
    from fastapi.testclient import TestClient
    from backend.main import app
except ImportError:
    app = None
    TestClient = None

try:
    from backend.core.config import settings
except ImportError:
    settings = None

try:
    from backend.chem.plugins import DescriptorPluginRegistry
except ImportError:
    DescriptorPluginRegistry = None


# ========== Environment Override for Testing ==========
@pytest.fixture(autouse=True)
def test_env_override():
    """Override settings for test isolation"""
    # Save original environment
    orig_env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        os.environ["DATABASE_URL"] = "sqlite:///./test_chemai.db"
        os.environ["REDIS_URL"] = "redis://localhost:6379/15"
        os.environ["DATA_DIR"] = str(test_dir / "data")
        os.environ["EXPORT_DIR"] = str(test_dir / "exports")
        os.environ["CACHE_DIR"] = str(test_dir / ".cache")
        os.environ["DEBUG"] = "true"

        # Reload settings if available (using proper pydantic v2 pattern)
        if settings is not None:
            try:
                from importlib import reload
                import backend.core.config as config_module
                reload(config_module)
            except Exception:
                pass

        yield

        # Restore original environment
        os.environ.clear()
        os.environ.update(orig_env)


# ========== Data Fixtures ==========
@st.composite
def synthetic_chemical_data(draw):
    """Hypothesis strategy for generating valid synthetic chemical datasets"""
    n_rows = draw(st.integers(min_value=10, max_value=200))
    n_numeric = draw(st.integers(min_value=2, max_value=5))
    n_categorical = draw(st.integers(min_value=0, max_value=2))
    
    data = {}
    # Numeric features
    for i in range(n_numeric):
        data[f"numeric_{i}"] = draw(arrays(
            np.float64, shape=n_rows, elements=st.floats(min_value=-100, max_value=100, allow_nan=False)
        ))
    
    # Categorical features
    categories = ["catA", "catB", "catC"]
    for i in range(n_categorical):
        data[f"cat_{i}"] = draw(st.lists(st.sampled_from(categories), min_size=n_rows, max_size=n_rows))
    
    # Target variable
    target_vals = draw(arrays(np.float64, shape=n_rows, elements=st.floats(min_value=-10, max_value=10)))
    data["target"] = target_vals
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def mock_dataset():
    """Generate deterministic mock dataset for unit tests"""
    np.random.seed(42)
    df = pd.DataFrame({
        "smiles": ["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"] * 10,
        "mol_weight": np.random.normal(100, 20, 50),
        "logp": np.random.normal(2, 1, 50),
        "solubility": np.random.exponential(5, 50),
        "group": np.random.choice(["A", "B"], 50),
        "target": np.random.normal(50, 10, 50)
    })
    return df


@pytest.fixture
def client():
    """FastAPI TestClient with automatic lifespan handling"""
    if app is None or TestClient is None:
        pytest.skip("FastAPI backend not available")
    with TestClient(app) as c:
        yield c


# ========== Mock Plugins ==========
@pytest.fixture
def mock_descriptor_plugin(tmp_path):
    """Create a temporary plugin directory with a mock plugin"""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    
    plugin_code = '''
"""
# --- YAML Metadata ---
# name: "mock_plugin"
# category: "custom"
# param_schema:
#   scale_factor:
#     type: "float"
#     default: 1.0
#     description: "Scaling factor for output"
"""
from typing import List
import pandas as pd

def calculate_descriptors(smiles_list: List[str], scale_factor: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "mock_feat1": [len(s) * scale_factor for s in smiles_list],
        "mock_feat2": [s.count("C") * scale_factor for s in smiles_list]
    })
'''
    plugin_file = plugin_dir / "mock_descriptor.py"
    plugin_file.write_text(plugin_code)
    
    return plugin_dir


# ========== Async Utilities ==========
@pytest.fixture
def mock_websocket():
    """Mock WebSocket for async progress testing"""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=StopIteration)
    return ws
