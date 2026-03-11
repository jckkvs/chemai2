"""backend/chem/__init__.py"""
from backend.chem.base import BaseChemAdapter, DescriptorResult


def _make_unavailable_adapter(name: str):
    """ライブラリ未インストール時のダミーアダプタークラスを動的生成する。"""
    class _UnavailableAdapter:
        _adapter_name = name
        def __init__(self, **kwargs):
            pass
        def is_available(self) -> bool:
            return False
        def compute(self, *args, **kwargs):
            raise RuntimeError(
                f"{self._adapter_name} は利用できません。"
                "必要なライブラリをインストールしてください。"
            )
    _UnavailableAdapter.__name__ = name
    _UnavailableAdapter.__qualname__ = name
    return _UnavailableAdapter


# ── 各アダプターを安全import ──────────────────────────────────

try:
    from backend.chem.rdkit_adapter import RDKitAdapter
except Exception:
    RDKitAdapter = _make_unavailable_adapter("RDKitAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.xtb_adapter import XTBAdapter
except Exception:
    XTBAdapter = _make_unavailable_adapter("XTBAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.cosmo_adapter import CosmoAdapter
except Exception:
    CosmoAdapter = _make_unavailable_adapter("CosmoAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.unipka_adapter import UniPkaAdapter
except Exception:
    UniPkaAdapter = _make_unavailable_adapter("UniPkaAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.group_contrib_adapter import GroupContribAdapter
except Exception:
    GroupContribAdapter = _make_unavailable_adapter("GroupContribAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.mordred_adapter import MordredAdapter
except Exception:
    MordredAdapter = _make_unavailable_adapter("MordredAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.molai_adapter import MolAIAdapter
except Exception:
    MolAIAdapter = _make_unavailable_adapter("MolAIAdapter")  # type: ignore[assignment,misc]

__all__ = [
    "BaseChemAdapter",
    "DescriptorResult",
    "RDKitAdapter",
    "XTBAdapter",
    "CosmoAdapter",
    "UniPkaAdapter",
    "GroupContribAdapter",
    "MordredAdapter",
    "MolAIAdapter",
]
