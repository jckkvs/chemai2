"""
backend/chem/xtb_adapter.py

XTB (GFN2-xTB) による量子化学計算に基づく記述子生成アダプター。

Implements: §3.9 XTB量子化学記述子
引用: Bannwarth et al., J. Chem. Theory Comput. 2019, DOI: 10.1021/acs.jctc.8b01176

動作条件:
  - `xtb` バイナリが PATH に存在すること
  - RDKit がインストールされていること（SMILES → 3D 変換用）

インストール方法:
  conda install -c conda-forge xtb          # 推奨（Windows対応）
  または https://github.com/grimme-lab/xtb/releases からバイナリをダウンロード
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

_XTB_DESCRIPTORS = {
    "xtb_HomoLumoGap":         "HOMO-LUMOエネルギーギャップ（光吸収・反応性） [eV]",
    "xtb_HomoEnergy":           "HOMO エネルギー [eV]",
    "xtb_LumoEnergy":           "LUMO エネルギー [eV]",
    "xtb_TotalEnergy":          "全電子エネルギー [Hartree]",
    "xtb_DipoleMoment":         "双極子モーメント（極性の指標） [Debye]",
    "xtb_Polarizability":       "等方分極率 [Bohr³]",
    "xtb_IonizationPotential":  "イオン化ポテンシャル（推定） [eV]",
    "xtb_ElectronAffinity":     "電子親和力（推定） [eV]",
    "xtb_Electrophilicity":     "親電子性インデックス [eV]",
}


def _smiles_to_xyz(smiles: str) -> str | None:
    """SMILES → 3D座標 (XYZ 形式文字列)。RDKit を使用。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        positions = conf.GetPositions()

        lines = [str(len(atoms)), f"Generated from SMILES: {smiles}"]
        for sym, pos in zip(atoms, positions):
            lines.append(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
        return "\n".join(lines)
    except Exception as e:
        logger.debug("SMILES→XYZ 変換失敗: %s, err=%s", smiles[:30], e)
        return None


def _parse_xtb_output(output: str) -> dict[str, float]:
    """xtb 出力テキストから各種記述子を抽出する。

    Implements: §3.9 xtb出力パース
    """
    result: dict[str, float] = {}
    lines = output.splitlines()
    for line in lines:
        line_l = line.lower()
        try:
            if "homo-lumo gap" in line_l:
                result["xtb_HomoLumoGap"] = float(line.split()[-2])
            elif "| homo" in line_l and "eV" in line:
                result["xtb_HomoEnergy"] = float(line.split()[-2])
            elif "| lumo" in line_l and "eV" in line:
                result["xtb_LumoEnergy"] = float(line.split()[-2])
            elif "total energy" in line_l and "Eh" in line:
                result["xtb_TotalEnergy"] = float(line.split()[-2])
            elif "dipole moment" in line_l:
                # next non-empty line has values
                pass
            elif "| total" in line_l and "debye" in line_l.replace("| total", ""):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        result["xtb_DipoleMoment"] = float(parts[-1])
                    except ValueError:
                        pass
        except (ValueError, IndexError):
            pass

    # HOMO-LUMO 由来の推定値
    homo = result.get("xtb_HomoEnergy")
    lumo = result.get("xtb_LumoEnergy")
    if homo is not None and lumo is not None:
        ip = -homo
        ea = -lumo
        result["xtb_IonizationPotential"] = ip
        result["xtb_ElectronAffinity"] = ea
        mu = (ip + ea) / 2.0
        eta = (ip - ea) / 2.0
        if eta > 0:
            result["xtb_Electrophilicity"] = (mu ** 2) / (2.0 * eta)

    return result


class XTBAdapter(BaseChemAdapter):
    """
    XTB (GFN2-xTB) による量子化学計算記述子アダプター。

    SMILES → RDKit 3D構造生成 → xtb バイナリ (subprocess) → 記述子抽出

    Implements: §3.9 XTB量子化学記述子
    引用: Bannwarth et al., JCTC 2019, DOI: 10.1021/acs.jctc.8b01176
    API:
        gfn (int): GFN-xTB レベル（デフォルト 2）
    前提:
        - `xtb` バイナリが PATH に存在すること
        - conda install -c conda-forge xtb  でインストール可能
    """

    def __init__(self, gfn: int = 2):
        self.gfn = gfn

    @property
    def name(self) -> str:
        return "xtb"

    @property
    def description(self) -> str:
        return (
            f"XTB GFN{self.gfn}-xTB による量子化学的電子状態・エネルギー記述子。\n"
            "有効化: conda install -c conda-forge xtb"
        )

    def is_available(self) -> bool:
        """
        xtb バイナリが PATH に存在するか、または tools/ 配下に同梱バイナリがあるかを確認する。
        同梱バイナリが見つかった場合は自動的に PATH へ追加する。
        """
        import pathlib

        # まず PATH を検索
        if shutil.which("xtb") is not None:
            try:
                from rdkit import Chem  # noqa: F401
                return True
            except ImportError:
                return False

        # PATH にない場合は tools/ 配下の同梱バイナリを探す
        # このファイル: backend/chem/xtb_adapter.py → プロジェクトルート: ../../
        here = pathlib.Path(__file__).resolve().parent  # backend/chem/
        project_root = here.parent.parent               # chemai2/
        candidates = [
            project_root / "tools" / "xtb-6.7.1" / "bin",
            project_root / "tools" / "xtb" / "bin",
        ]
        for bin_dir in candidates:
            xtb_exe = bin_dir / ("xtb.exe" if os.name == "nt" else "xtb")
            if xtb_exe.exists():
                # PATH へ自動追加（このプロセス内）
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                logger.info("XTB バイナリを自動検出して PATH に追加しました: %s", bin_dir)
                try:
                    from rdkit import Chem  # noqa: F401
                    return True
                except ImportError:
                    return False

        return False


    def compute(
        self,
        smiles_list: list[str],
        selected_descriptors: list[str] | None = None,
        **kwargs: Any,
    ) -> DescriptorResult:
        """SMILES リストから XTB 量子化学記述子を計算する。

        Implements: §3.9 XTB計算フロー
        Args:
            smiles_list: 入力 SMILES のリスト
            selected_descriptors: 使用する記述子名（None = 全件）
        Returns:
            DescriptorResult: xtb_* 列からなる DataFrame
        """
        self._require_available()

        all_names = list(_XTB_DESCRIPTORS.keys())
        col_names = (
            [c for c in selected_descriptors if c in all_names]
            if selected_descriptors else all_names
        )

        rows: list[dict] = []
        failed_indices: list[int] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, smi in enumerate(smiles_list):
                row = {k: np.nan for k in col_names}
                try:
                    xyz = _smiles_to_xyz(smi)
                    if xyz is None:
                        raise ValueError(f"SMILES → XYZ 変換失敗: {smi[:30]}")

                    xyz_path = os.path.join(tmpdir, f"mol_{i}.xyz")
                    with open(xyz_path, "w") as f:
                        f.write(xyz)

                    # xtb single-point 計算
                    result = subprocess.run(
                        ["xtb", xyz_path, f"--gfn{self.gfn}", "--sp"],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    parsed = _parse_xtb_output(result.stdout)
                    for k in col_names:
                        if k in parsed:
                            row[k] = parsed[k]
                except subprocess.TimeoutExpired:
                    logger.warning("XTB タイムアウト: idx=%d smi=%s", i, smi[:30])
                    failed_indices.append(i)
                except Exception as e:
                    logger.warning("XTB 計算失敗: idx=%d err=%s", i, e)
                    failed_indices.append(i)
                rows.append(row)

        df = pd.DataFrame(rows, columns=col_names)
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={"gfn": self.gfn},
        )

    def get_descriptor_names(self) -> list[str]:
        return list(_XTB_DESCRIPTORS.keys())

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(
                name=name,
                meaning=meaning,
                is_count=False,
                is_binary=False,
                description="XTB GFN2-xTB 量子化学計算。Bannwarth et al. JCTC 2019",
            )
            for name, meaning in _XTB_DESCRIPTORS.items()
        ]
