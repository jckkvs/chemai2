"""
tests/test_feature_integrity.py

必須アダプターがapp.pyから削除されていないことを検証する保護テスト。
UIリファクタリング等でコードが意図せず消えた場合に即時検出する。
"""
import ast
import pathlib
import pytest

APP_PY = pathlib.Path(__file__).parent.parent / "frontend_streamlit" / "app.py"

# 守るべき必須アダプター
REQUIRED_ADAPTERS = [
    "RDKitAdapter",
    "MordredAdapter",
    "XTBAdapter",
    "CosmoAdapter",
    "UniPkaAdapter",
    "GroupContribAdapter",
    "MolAIAdapter",
]

REQUIRED_IN_ALL_ADAPTERS_CALL = [
    "RDKitAdapter",
    "MordredAdapter",
    "XTBAdapter",
    "CosmoAdapter",
    "UniPkaAdapter",
    "GroupContribAdapter",
    "MolAIAdapter",
]


def _read_app() -> str:
    return APP_PY.read_text(encoding="utf-8")


class TestFeatureIntegrity:
    """app.py に必須アダプターが含まれることを検証する保護テスト群。"""

    def test_app_py_exists(self):
        """app.py が存在することを確認。"""
        assert APP_PY.exists(), f"app.py が見つかりません: {APP_PY}"

    @pytest.mark.parametrize("adapter", REQUIRED_ADAPTERS)
    def test_required_adapter_imported(self, adapter: str):
        """必須アダプターが app.py 内でインポートされていること。

        このテストが落ちた場合、UIリファクタリング等で
        重要なアダプタークラスが誤って削除された可能性があります。
        """
        content = _read_app()
        assert adapter in content, (
            f"【保護テスト失敗】{adapter} が app.py から消えています。\n"
            f"UIリファクタリング時に誰かが削除した可能性があります。\n"
            f"app.py の 'all_adapters' または import 文を確認してください。"
        )

    def test_all_adapters_list_present(self):
        """all_adapters 変数の初期化コードが app.py に存在すること。"""
        content = _read_app()
        assert "all_adapters" in content, (
            "【保護テスト失敗】'all_adapters' リストが app.py から消えています。"
        )

    @pytest.mark.parametrize("adapter", REQUIRED_IN_ALL_ADAPTERS_CALL)
    def test_adapter_in_all_adapters_section(self, adapter: str):
        """各アダプターが all_adapters 変数の近くに記載されていること。"""
        content = _read_app()
        # all_adapters 行を探してその周辺にアダプターがあるか確認
        lines = content.splitlines()
        all_adapters_lineno = next(
            (i for i, ln in enumerate(lines) if "all_adapters" in ln and "=" in ln),
            None,
        )
        assert all_adapters_lineno is not None, "'all_adapters =' 行が見つかりません。"

        # all_adapters 行の前後10行以内に adapter が含まれるか確認
        window = lines[max(0, all_adapters_lineno - 2): all_adapters_lineno + 5]
        window_text = "\n".join(window)
        assert adapter in window_text, (
            f"【保護テスト失敗】{adapter} が 'all_adapters' リスト内に含まれていません。\n"
            f"all_adapters 付近: {window_text[:300]}"
        )

    def test_smiles_expander_present(self):
        """SMILES記述子設定用のexpanderが存在すること。"""
        content = _read_app()
        assert "SMILES記述子設定" in content, (
            "【保護テスト失敗】SMILES記述子設定のUIブロックが消えています。"
        )
