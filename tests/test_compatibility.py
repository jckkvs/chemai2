# tests/test_compatibility.py
import pytest
import sys
from backend.utils.compatibility import CompatibilityManager

def test_compatibility_manager_python_version():
    manager = CompatibilityManager()
    results = manager.check_environment()
    
    # 基本的な属性チェック
    assert "python_compatible" in results
    assert "python_optimal" in results
    assert "mordred_available" in results
    assert "recommendations" in results
    
    # 3.10以上であることを確認
    assert results["python_compatible"] is True

def test_compatibility_manager_mordred_logic():
    manager = CompatibilityManager()
    results = manager.check_environment()
    
    # Python 3.12以上なら Mordred は利用不可
    if sys.version_info >= (3, 12):
        assert results["mordred_available"] is False
        assert any("Mordred" in r for r in results["recommendations"])
    else:
        assert results["mordred_available"] is True

def test_suppress_runtime_warnings():
    manager = CompatibilityManager()
    # 実行してもエラーにならないことを確認
    manager.suppress_runtime_warnings()
    assert "RequestsDependencyWarning" in manager.warnings_suppressed
    assert "TorchJITScriptDeprecationWarning" in manager.warnings_suppressed
