# tests/test_llm_analyzer.py
import pytest
import pandas as pd
import json
from unittest.mock import MagicMock, AsyncMock
from backend.services.llm_data_analyzer import LLMDataAnalyzer
from backend.llm.base import LLMProvider

class MockLLMProvider(LLMProvider):
    def __init__(self):
        self.generate = AsyncMock(return_value=json.dumps({
            "data_overview": "Test data",
            "preprocessing": "Normalize",
            "feature_engineering": "SMILES",
            "model_candidates": ["RF", "LGBM"],
            "validation_strategy": "CV",
            "interpretation_plan": "SHAP",
            "cautions": "None"
        }))
        self.is_available = MagicMock(return_value=True)

    async def generate(self, prompt, **kwargs):
        return await self.generate(prompt, **kwargs)

    def is_available(self):
        return self.is_available()

@pytest.mark.asyncio
async def test_llm_analyzer_success():
    # Setup
    df = pd.DataFrame({
        "SMILES": ["C", "CC"],
        "Target": [1.0, 2.0]
    })
    analyzer = LLMDataAnalyzer()
    mock_provider = MockLLMProvider()
    analyzer.provider = mock_provider
    
    # Execute
    result = await analyzer.analyze(df)
    
    # Verify
    assert "data_overview" in result
    assert result["model_candidates"] == ["RF", "LGBM"]
    mock_provider.generate.assert_called_once()

@pytest.mark.asyncio
async def test_llm_analyzer_parse_error():
    # Setup
    analyzer = LLMDataAnalyzer()
    mock_provider = MockLLMProvider()
    mock_provider.generate = AsyncMock(return_value="Not a JSON string")
    analyzer.provider = mock_provider
    
    # Execute
    result = await analyzer.analyze(pd.DataFrame({"A": [1]}))
    
    # Verify
    assert "warning" in result
    assert "raw_output" in result
    assert result["raw_output"] == "Not a JSON string"

def test_prepare_dataframe_context():
    provider = MockLLMProvider()
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    context = provider.prepare_dataframe_context(df)
    
    assert context["shape"] == (3, 2)
    assert "A" in context["columns"]
    assert context["null_counts"]["A"] == 0
    assert len(context["sample_rows"]) == 3
