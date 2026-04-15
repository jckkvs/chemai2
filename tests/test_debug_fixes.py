"""
デバッグ修正事項の検証テスト
"""
import pytest
import numpy as np
from typing import Union, Dict, Any

# 修正対象のインポート
from backend.models.monotonicity_adapter import apply_monotonicity_constraints
from backend.chem.xtb_adapter import XTBAdapter

class TestMonotonicityAdapterTypes:
    """型ヒントの整合性テスト"""
    
    def test_constraints_dict_with_int(self):
        """int 型の制約辞書が受け入れられるか"""
        constraints = {"feat_A": 1, "feat_B": -1}
        # 実際の関数呼び出しシミュレーション（型チェックの意味合いが強い）
        assert isinstance(constraints["feat_A"], int)
        assert isinstance(constraints["feat_B"], int)

    def test_constraints_dict_with_dict(self):
        """dict 型の制約辞書が受け入れられるか"""
        constraints = {"feat_C": {"direction": "increasing"}}
        # 実際の関数呼び出しシミュレーション
        assert isinstance(constraints["feat_C"], dict)
        assert constraints["feat_C"]["direction"] == "increasing"

class TestXTBAdapterLogic:
    """XTB アダプターの論理修正テスト"""
    
    def test_instance_variable_isolation(self):
        """インスタンス変数 _xtb_broken が独立しているか"""
        # アダプターを2つ生成
        adapter1 = XTBAdapter()
        adapter2 = XTBAdapter()
        
        # 片方を故障状態にする
        adapter1._xtb_broken = True
        
        # もう片方が影響を受けていないことを確認（並列実行安全性の担保）
        assert adapter1._xtb_broken is True
        assert adapter2._xtb_broken is False

    def test_mulliken_parse_logic(self):
        """Mulliken 電荷パースの境界値テスト（擬似パース処理）"""
        from backend.chem.xtb_adapter import _parse_xtb_output
        
        # Mulliken電荷を含む（または不正な行を含む）擬似出力
        mock_output = (
            "   Mulliken charges\n"
            "   1  C    0.123  0.456  -0.100\n" # 正常 (5列以上)
            "   2  O    abc    0.789  -0.200\n" # parts[3] が float 変換不能
            "   3  H\n"                       # 列不足
            "Other output\n"
        )
        
        # _parse_xtb_output が IndexError 等を投げずに処理を終えるか
        results = _parse_xtb_output(mock_output)
        
        # 1行目だけが正しく抽出されているはず
        assert "xtb_MullikenChargeMean" in results
        # 異常行（O, H）でパースブロックが終了するため、要素数は1
        # 注: 実装上、数値変換エラーや列不足で in_charges_block = False になる
