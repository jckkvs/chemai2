"""
backend/chem/cosmo_adapter.py

COSMO-RS 理論に基づく熱力学的記述子の計算アダプター。
外部ツール（cosmo_path）を直接実行する堅牢な実装。
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

class CosmoAdapter(BaseChemAdapter):
    """
    COSMO-RS 記述子アダプター。
    外部ツールを実行し、出力をパースして記述子を抽出する。
    """

    def __init__(self, cosmo_path: str = "cosmo_rs", timeout: int = 600) -> None:
        self.cosmo_path = cosmo_path
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "cosmo_rs"

    @property
    def description(self) -> str:
        return "COSMO-RS による熱力学的記述子の計算（外部ツール実行版）"

    def is_available(self) -> bool:
        return shutil.which(self.cosmo_path) is not None

    def _prepare_cosmo_input(self, smiles: str, charge: int) -> str:
        # 簡易的な入力ファイル生成ロジック（実際にはツールに合わせたフォーマットが必要）
        return f"SMILES {smiles}\nCHARGE {charge}\n"

    def _run_cosmo_calculation(self, smiles: str, charge: int) -> Dict[str, float]:
        """
        Execute COSMO-RS calculation via external tool
        
        【修正点1+2】安全なテンポラリファイル管理とリソース解放
        """
        import tempfile
        import subprocess
        import re
        import os
        from pathlib import Path
        
        # 【修正点1】NamedTemporaryFileで競合回避・自動削除
        # 【修正点2】delete=Falseで外部ツールがアクセス可能に、後で手動削除
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_file = tmpdir_path / "input.cosmo"
            output_file = tmpdir_path / "output.cosmo"
            
            try:
                # 入力ファイル作成（【修正点2】with文で確実なクローズ）
                with open(input_file, 'w', encoding='utf-8') as f:
                    f.write(self._prepare_cosmo_input(smiles, charge))
                
                # COSMO-RS実行（【修正点4】stderrもcaptureでエラー詳細取得）
                result = subprocess.run(
                    [self.cosmo_path, str(input_file), "-o", str(output_file)],
                    capture_output=True,
                    text=True,
                    timeout=getattr(self, 'timeout', 600),
                    cwd=str(tmpdir_path),
                    env={**os.environ, "COSMO_RS_THREADS": "1"}  # 【追加】並列抑制で安定性
                )
                
                # 【修正点5】エラー時の詳細ログ出力
                if result.returncode != 0:
                    logger.warning(
                        f"COSMO-RS failed for SMILES {smiles!r} (charge={charge}): "
                        f"returncode={result.returncode}, stderr={result.stderr[:300]}"
                    )
                    return {}
                
                # 【修正点4】大規模出力はストリーミングでパース（メモリ効率）
                properties = {}
                
                # 【修正点3】正規表現を厳密化: 行頭アンカー・単位指定・空白許容
                # 【修正点4】ファイルは行単位でストリーミング読み込み
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            line_stripped = line.strip()
                            if not line_stripped:
                                continue
                            
                            # Energy pattern
                            if line_stripped.startswith('Total energy:'):
                                match = re.match(
                                    r'^Total energy:\s*([-\d.]+)\s*(kcal/mol|kJ/mol|Eh)?',
                                    line_stripped,
                                    re.IGNORECASE
                                )
                                if match:
                                    try:
                                        val = float(match.group(1))
                                        unit = (match.group(2) or 'Eh').lower()
                                        if unit == 'kcal/mol': val *= 0.0015936
                                        elif unit == 'kJ/mol': val *= 0.0003809
                                        properties['cosmo_energy'] = val
                                    except ValueError:
                                        logger.debug(f"Failed to parse energy at line {line_num}")
                            
                            # Sigma moment patterns
                            elif 'sigma moment' in line_stripped.lower():
                                match = re.match(
                                    r'^.*sigma\s+moment\s*(\d+)?\s*:\s*([-\d.]+)',
                                    line_stripped,
                                    re.IGNORECASE
                                )
                                if match:
                                    try:
                                        order = match.group(1)
                                        val = float(match.group(2))
                                        key = f'cosmo_sigma_moment_{order}' if order else 'cosmo_sigma_moment'
                                        properties[key] = val
                                    except ValueError:
                                        logger.debug(f"Failed to parse sigma moment at line {line_num}")
                            
                            # Solvation energy
                            elif 'solvation energy' in line_stripped.lower():
                                match = re.search(
                                    r'solvation energy\s*[:=]\s*([-\d.]+)',
                                    line_stripped,
                                    re.IGNORECASE
                                )
                                if match:
                                    try:
                                        properties['cosmo_solvation_energy'] = float(match.group(1))
                                    except ValueError:
                                        logger.debug(f"Failed to parse solvation energy at line {line_num}")
                
                if not properties:
                    logger.warning(f"COSMO-RS produced no parseable output for SMILES {smiles!r}")
                    return {}
                
                return properties
                
            except subprocess.TimeoutExpired as e:
                logger.error(f"COSMO-RS timed out for SMILES {smiles!r}")
                return {}
            except Exception as e:
                logger.error(f"Unexpected error in COSMO-RS: {e}", exc_info=True)
                return {}

    def compute(self, smiles_list: List[str], **kwargs: Any) -> DescriptorResult:
        records = []
        failed_indices = []
        for i, smi in enumerate(smiles_list):
            charge = kwargs.get("charge", 0)
            res = self._run_cosmo_calculation(smi, charge)
            if not res:
                failed_indices.append(i)
                records.append({})
            else:
                records.append(res)
        
        df = pd.DataFrame(records)
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name
        )

    def get_descriptor_names(self) -> List[str]:
        return ["cosmo_energy", "cosmo_sigma_moment", "cosmo_solvation_energy"]
