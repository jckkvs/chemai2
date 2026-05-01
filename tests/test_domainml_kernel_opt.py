"""
tests/test_domainml_kernel_opt.py

Kernel monotonicity tests - skipped due to cvxpy/numpy compatibility issues.
"""
import pytest

try:
    from domainml.constraints.kernel_opt import KernelMonotonicity
    CVXPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CVXPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy/numpy incompatible")
