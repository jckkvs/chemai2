import logging
import os
import shutil
import subprocess
import tempfile
import re
from typing import Any, List, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

# Windows での xtb クラッシュ時ポップアップ（アプリケーションエラー）を抑制する
if os.name == 'nt':
    try:
        import ctypes
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        )
    except Exception:
        pass

# クラッシュ検出時に2回目以降の実行をスキップするためのフラグ
_XTB_BROKEN_GLOBAL = False

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
    # Mulliken電荷統計（charge_config で xtb_mulliken 選択時）
    "xtb_MullikenChargeMax":    "原子Mulliken電荷の最大値（最も正電荷の原子）",
    "xtb_MullikenChargeMin":    "原子Mulliken電荷の最小値（最も負電荷の原子）",
    "xtb_MullikenChargeMean":   "原子Mulliken電荷の平均値",
    "xtb_MullikenChargeStd":    "原子Mulliken電荷の標準偏差",
}


def _parse_xtb_output(output: str) -> Dict[str, float]:
    """
    xtb 出力テキストから各種記述子を抽出する。

    Implements: §3.9 xtb出力パース
    Mulliken電荷の抽出も追加（qTOTAL 列を読む）
    """
    result: Dict[str, float] = {}
    lines = output.splitlines()
    mulliken_charges: List[float] = []
    in_charges_block = False

    for i, line in enumerate(lines):
        line_l = line.lower()
        parts = line.split()
        if not parts:
            continue

        try:
            if "homo-lumo gap" in line_l:
                if len(parts) >= 2:
                    try:
                        result["xtb_HomoLumoGap"] = float(parts[-2])
                    except ValueError:
                        pass
            elif "| homo" in line_l and "ev" in line_l:
                pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
                for pp in pipe_parts:
                    tokens = pp.split()
                    if tokens:
                        try:
                            result["xtb_HomoEnergy"] = float(tokens[0])
                        except ValueError:
                            pass
            elif "| lumo" in line_l and "ev" in line_l:
                pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
                for pp in pipe_parts:
                    tokens = pp.split()
                    if tokens:
                        try:
                            result["xtb_LumoEnergy"] = float(tokens[0])
                        except ValueError:
                            pass
            elif "total energy" in line_l and "eh" in line_l:
                if len(parts) >= 2:
                    try:
                        result["xtb_TotalEnergy"] = float(parts[-2])
                    except ValueError:
                        pass
            elif "| total" in line_l and "debye" in line_l:
                pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
                for pp in pipe_parts:
                    if "debye" in pp.lower():
                        tokens = pp.split()
                        for t in reversed(tokens):
                            try:
                                result["xtb_DipoleMoment"] = float(t)
                                break
                            except ValueError:
                                pass
                        break

            # Mulliken電荷ブロックの検出
            elif "mulliken" in line_l and "charge" in line_l:
                in_charges_block = True
                mulliken_charges = []
            elif in_charges_block:
                if len(parts) >= 5:
                    try:
                        q = float(parts[4])
                        mulliken_charges.append(q)
                    except ValueError:
                        if mulliken_charges:
                             in_charges_block = False
                elif len(parts) > 0:
                    if mulliken_charges:
                        in_charges_block = False

        except Exception as e:
            logger.debug(f"XTB line parse error at line {i}: {e}")
            continue

    # HOMO-LUMO 由来の推定値（Koopmans定理）
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

    # Mulliken電荷統計
    if mulliken_charges:
        charges_arr = np.array(mulliken_charges)
        result["xtb_MullikenChargeMax"]  = float(np.max(charges_arr))
        result["xtb_MullikenChargeMin"]  = float(np.min(charges_arr))
        result["xtb_MullikenChargeMean"] = float(np.mean(charges_arr))
        result["xtb_MullikenChargeStd"]  = float(np.std(charges_arr))

    return result


def _read_xyz_coords(xyz_path: str) -> Optional[Dict[str, Any]]:
    """
    XYZファイルから座標と原子番号を読み取る（ML派生特徴量用）。
    """
    _SYMBOL_TO_Z = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
        "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Ti": 22, "V": 23,
        "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
        "Sr": 38, "Y": 39, "Zr": 40, "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46,
        "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53,
        "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Pt": 78, "Au": 79, "Hg": 80,
        "Tl": 81, "Pb": 82, "Bi": 83,
    }
    try:
        with open(xyz_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None
        n_atoms = int(lines[0].strip())
        symbols: List[str] = []
        coords: List[List[float]] = []
        for line in lines[2: 2 + n_atoms]:
            parts = line.split()
            if len(parts) < 4:
                continue
            sym = parts[0].strip().capitalize()
            symbols.append(sym)
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(coords) != n_atoms:
            return None
        atomic_numbers = [_SYMBOL_TO_Z.get(s, 0) for s in symbols]
        return {
            "coords": np.array(coords),
            "atomic_numbers": atomic_numbers,
            "symbols": symbols,
        }
    except Exception:
        return None


class XTBAdapter(BaseChemAdapter):
    """
    XTB (GFN2-xTB) による量子化学計算記述子アダプター。
    """

    def __init__(
        self,
        gfn: int = 2,
        calc_type: str = "opt",
        convergence: str = "normal",
        solvent: str = "none",
        timeout: Optional[int] = None,
        max_retries: int = 3,
    ):
        from backend.utils.config import default_config
        self.gfn = gfn
        self.calc_type = calc_type
        self.convergence = convergence
        self.solvent = solvent
        self.timeout = timeout if timeout is not None else default_config.xtb_timeout_per_mol
        self.max_retries = max_retries
        self._xtb_broken = False

    @property
    def name(self) -> str:
        return "xtb"

    @property
    def description(self) -> str:
        return (
            f"XTB GFN{self.gfn}-xTB による量子化学的電子状態・エネルギー記述子。\n"
            f"計算タイプ: {self.calc_type} / 収束: {self.convergence} / "
            f"溶媒: {self.solvent}\n"
            "有効化: conda install -c conda-forge xtb"
        )

    def is_available(self) -> bool:
        import pathlib
        if shutil.which("xtb") is not None:
            try:
                from rdkit import Chem  # noqa: F401
                return True
            except ImportError:
                return False

        here = pathlib.Path(__file__).resolve().parent
        project_root = here.parent.parent
        candidates = [
            project_root / "tools" / "xtb-6.7.1" / "bin",
            project_root / "tools" / "xtb" / "bin",
        ]
        for bin_dir in candidates:
            xtb_exe = bin_dir / ("xtb.exe" if os.name == "nt" else "xtb")
            if xtb_exe.exists():
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                logger.info("XTB バイナリを自動検出して PATH に追加しました: %s", bin_dir)
                try:
                    from rdkit import Chem  # noqa: F401
                    return True
                except ImportError:
                    return False
        return False

    def _smiles_to_xyz(self, smiles: str, charge: int = 0, multiplicity: int = 1) -> Optional[str]:
        """
        SMILES → 3D座標 (XYZ 形式文字列)。RDKit を使用。
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result != 0:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                
            try:
                if charge != 0 or multiplicity > 1:
                    AllChem.UFFOptimizeMolecule(mol)
                else:
                    AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass

            conf = mol.GetConformer()
            atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
            positions = conf.GetPositions()

            lines = [str(len(atoms)), f"SMILES: {smiles} charge={charge} mult={multiplicity}"]
            for sym, pos in zip(atoms, positions):
                lines.append(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"SMILES→XYZ 変換失敗 ({smiles[:30]}): {e}")
            return None

    def _run_xtb_calculation(self, smiles: str, charge: int, multiplicity: int) -> Dict[str, float]:
        """
        Execute xTB calculation for a single SMILES
        """
        import tempfile
        import subprocess
        import re
        import os
        
        timeout = self.timeout
        
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, "input.xyz")
            
            try:
                xyz_content = self._smiles_to_xyz(smiles, charge, multiplicity)
                if not xyz_content:
                    return {}
                    
                with open(xyz_path, 'w') as f:
                    f.write(xyz_content)
                
                cmd = ["xtb", xyz_path, "--chrg", str(charge), "--uhf", str(multiplicity-1)]
                if self.gfn != 2:
                    cmd.append(f"--gfn{self.gfn}")
                if self.calc_type == "opt":
                    cmd.append("--opt")
                    if self.convergence != "normal":
                        cmd.append(self.convergence)
                
                if self.solvent and self.solvent.lower() not in ("none", "gas", "vacuum", ""):
                    cmd += ["--alpb", self.solvent]

                kwargs_sub = {}
                if os.name == "nt":
                    kwargs_sub["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                    **kwargs_sub
                )
                
                if result.returncode in (3221225794, -1073741502, 3221225749, -1073741515, 3221225477, -1073741819):
                    logger.critical("XTBがシステムエラーでクラッシュしました。")
                    self._xtb_broken = True
                    return {}

                if result.returncode != 0:
                    logger.warning(f"xTB failed for SMILES {smiles!r}: {result.stderr[:200]}")
                    return {}
                
                properties = _parse_xtb_output(result.stdout)
                
                # 最適化後座標の読み取り試行
                opt_xyz_path = os.path.join(tmpdir, "xtbopt.xyz")
                if os.path.exists(opt_xyz_path):
                    properties["_coord_info"] = _read_xyz_coords(opt_xyz_path)
                
                return properties
                
            except subprocess.TimeoutExpired:
                logger.error(f"xTB calculation timed out after {timeout}s for SMILES {smiles!r}")
                return {}
            except Exception as e:
                logger.error(f"Unexpected error in xTB calculation for SMILES {smiles!r}: {e}")
                return {}

    def compute(
        self,
        smiles_list: List[str],
        selected_descriptors: Optional[List[str]] = None,
        charge_config_store: Optional[Any] = None,
        **kwargs: Any,
    ) -> DescriptorResult:
        self._require_available()

        all_names = list(_XTB_DESCRIPTORS.keys())
        col_names = (
            [c for c in selected_descriptors if c in all_names]
            if selected_descriptors else all_names
        )

        rows: List[Dict[str, Any]] = []
        failed_indices: List[int] = []
        optimized_coords: List[Optional[Dict[str, Any]]] = []

        for i, smi in enumerate(smiles_list):
            row = {k: np.nan for k in col_names}
            
            if self._xtb_broken:
                failed_indices.append(i)
                optimized_coords.append(None)
                rows.append(row)
                continue

            try:
                if charge_config_store is not None:
                    charge = charge_config_store.resolve_charge(smi)
                    spin   = charge_config_store.resolve_spin(smi)
                    cfg    = charge_config_store.get_config(smi)
                    from backend.chem.protonation import apply_protonation
                    smi_for_xtb = apply_protonation(smi, cfg)
                    uhf = spin - 1
                    multiplicity = spin
                else:
                    from backend.chem.charge_config import _read_smiles_formal_charge
                    charge = _read_smiles_formal_charge(smi)
                    multiplicity = 1
                    smi_for_xtb = smi

                parsed = self._run_xtb_calculation(smi_for_xtb, charge, multiplicity)
                
                if not parsed:
                    failed_indices.append(i)
                    optimized_coords.append(None)
                else:
                    for k in col_names:
                        if k in parsed:
                            row[k] = parsed[k]
                    optimized_coords.append(parsed.get("_coord_info"))
            except Exception as e:
                logger.warning("XTB 計算失敗: idx=%d err=%s", i, e)
                failed_indices.append(i)
                optimized_coords.append(None)
            rows.append(row)

        df = pd.DataFrame(rows, columns=col_names)
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={
                "gfn": self.gfn,
                "calc_type": self.calc_type,
                "convergence": self.convergence,
                "solvent": self.solvent,
                "optimized_coords": optimized_coords,
            },
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

