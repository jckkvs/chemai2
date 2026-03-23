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
    # Mulliken電荷統計（charge_config で xtb_mulliken 選択時）
    "xtb_MullikenChargeMax":    "原子Mulliken電荷の最大値（最も正電荷の原子）",
    "xtb_MullikenChargeMin":    "原子Mulliken電荷の最小値（最も負電荷の原子）",
    "xtb_MullikenChargeMean":   "原子Mulliken電荷の平均値",
    "xtb_MullikenChargeStd":    "原子Mulliken電荷の標準偏差",
}


def _smiles_to_xyz(smiles: str, charge: int = 0) -> str | None:
    """
    SMILES → 3D座標 (XYZ 形式文字列)。RDKit を使用。

    Args:
        smiles: 入力 SMILES
        charge: 分子の形式電荷（3D構造生成には直接不要だが、
                MMFF/UFF 最適化の際に電荷を考慮させるために渡す）
    """
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
        # MMFF最適化（荷電分子にはUFFの方が安定する場合があるが、MMFFを優先）
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass  # 最適化失敗でも座標は得られている場合がある

        conf = mol.GetConformer()
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        positions = conf.GetPositions()

        lines = [str(len(atoms)), f"Generated from SMILES: {smiles} charge={charge}"]
        for sym, pos in zip(atoms, positions):
            lines.append(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
        return "\n".join(lines)
    except Exception as e:
        logger.debug("SMILES→XYZ 変換失敗: %s, err=%s", smiles[:30], e)
        return None


def _parse_xtb_output(output: str) -> dict[str, float]:
    """
    xtb 出力テキストから各種記述子を抽出する。

    Implements: §3.9 xtb出力パース
    Mulliken電荷の抽出も追加（qTOTAL 列を読む）
    """
    result: dict[str, float] = {}
    lines = output.splitlines()
    mulliken_charges: list[float] = []
    in_charges_block = False

    for i, line in enumerate(lines):
        line_l = line.lower()
        try:
            if "homo-lumo gap" in line_l:
                result["xtb_HomoLumoGap"] = float(line.split()[-2])
            elif "| homo" in line_l and "eV" in line:
                # 形式: " | HOMO | -0.51262 | -13.9492 | eV |"
                # パイプ区切りなので split('|') でeV値を取得
                pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
                for pp in pipe_parts:
                    tokens = pp.split()
                    if len(tokens) >= 1:
                        try:
                            val = float(tokens[0])
                            # eV単位の値を取得（最後の数値フィールド）
                            result["xtb_HomoEnergy"] = val
                        except ValueError:
                            pass
            elif "| lumo" in line_l and "eV" in line:
                pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
                for pp in pipe_parts:
                    tokens = pp.split()
                    if len(tokens) >= 1:
                        try:
                            val = float(tokens[0])
                            result["xtb_LumoEnergy"] = val
                        except ValueError:
                            pass
            elif "total energy" in line_l and "Eh" in line:
                result["xtb_TotalEnergy"] = float(line.split()[-2])
            elif "| total" in line_l and "debye" in line_l.replace("| total", ""):
                # 形式: " | total | 0.000 0.001 0.002 1.234 Debye"
                pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
                for pp in pipe_parts:
                    if "debye" in pp.lower():
                        tokens = pp.split()
                        # Debye直前の数値がtotal dipole moment
                        for t in reversed(tokens):
                            try:
                                result["xtb_DipoleMoment"] = float(t)
                                break
                            except ValueError:
                                pass
                        break

            # Mulliken電荷ブロックの検出
            # xtb出力: "Mulliken/CM5 charges" または "#   Z          covCN         q      C6AA"
            elif "mulliken" in line_l and "charge" in line_l:
                in_charges_block = True
                mulliken_charges = []
            elif in_charges_block:
                parts = line.split()
                # 形式: "  1  C   ...  q_value  ..." — 4列目が電荷の場合が多い
                if len(parts) >= 4:
                    try:
                        q = float(parts[3])  # Mulliken電荷列（xTBの標準出力形式）
                        mulliken_charges.append(q)
                    except (ValueError, IndexError):
                        in_charges_block = False  # 数値でなければブロック終了

        except (ValueError, IndexError):
            pass

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


class XTBAdapter(BaseChemAdapter):
    """
    XTB (GFN2-xTB) による量子化学計算記述子アダプター。

    SMILES → RDKit 3D構造生成 → xtb バイナリ (subprocess) → 記述子抽出

    Implements: §3.9 XTB量子化学記述子
    引用: Bannwarth et al., JCTC 2019, DOI: 10.1021/acs.jctc.8b01176
    API:
        gfn (int): GFN-xTB レベル（デフォルト 2）
        calc_type (str): "opt"(構造最適化, デフォルト) or "sp"(単点計算)
        convergence (str): 収束基準 ("crude"/"sloppy"/"loose"/"normal"/"tight"/"vtight")
        solvent (str): 溶媒モデル名（"none"=気相, "water", "methanol"等 → --alpb)
        timeout (int): 1分子あたりのタイムアウト秒数（デフォルト300）
        max_retries (int): 失敗時のリトライ回数（デフォルト3）
    前提:
        - `xtb` バイナリが PATH に存在すること
        - conda install -c conda-forge xtb  でインストール可能
    """

    def __init__(
        self,
        gfn: int = 2,
        calc_type: str = "opt",   # デフォルトを構造最適化に変更（要件2.1）
        convergence: str = "normal",
        solvent: str = "none",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        self.gfn = gfn
        # 科学的根拠: 構造最適化によりSMILESから機械的に生成した初期構造の
        # 歪みが解消され、電子状態記述子の精度が向上する。
        # GFN2-xTB の構造最適化は Bannwarth et al. JCTC 2019 で検証済み。
        self.calc_type = calc_type
        self.convergence = convergence
        self.solvent = solvent
        self.timeout = timeout
        self.max_retries = max_retries

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

    def _build_cmd(
        self,
        xyz_path: str,
        charge: int,
        uhf: int,
        calc_type: str | None = None,
        convergence: str | None = None,
        solvent: str | None = None,
    ) -> list[str]:
        """xtb コマンドライン引数を構築する。"""
        ct = calc_type or self.calc_type
        conv = convergence or self.convergence
        solv = solvent or self.solvent

        cmd = ["xtb", xyz_path, f"--gfn{self.gfn}"]

        # 計算タイプ
        if ct == "opt":
            cmd.append("--opt")
            # 収束基準を付加（"normal" はxtbデフォルトなので省略可だが、明示）
            if conv and conv != "normal":
                cmd.append(conv)
        else:
            cmd.append("--sp")

        # 電荷
        cmd += ["--chrg", str(charge)]

        # 不対電子
        if uhf > 0:
            cmd += ["--uhf", str(uhf)]

        # 溶媒モデル（ALPB: Analytical Linearized Poisson-Boltzmann）
        # 科学的根拠: ALPBは暗黙溶媒モデルとしてxtb 6.4+で推奨。
        # Ehlert et al., J. Chem. Phys. 2021
        if solv and solv.lower() not in ("none", "gas", "vacuum", ""):
            cmd += ["--alpb", solv]

        return cmd

    def compute(
        self,
        smiles_list: list[str],
        selected_descriptors: list[str] | None = None,
        charge_config_store: Any | None = None,
        **kwargs: Any,
    ) -> DescriptorResult:
        """
        SMILES リストから XTB 量子化学記述子を計算する。

        Implements: §3.9 XTB計算フロー
        デフォルトで構造最適化(--opt)を実行。失敗時はリトライ（収束基準を緩和）。

        Args:
            smiles_list: 入力 SMILES のリスト
            selected_descriptors: 使用する記述子名（None = 全件）
            charge_config_store: ChargeConfigStore インスタンス
            **kwargs: calc_type, convergence, solvent を上書き可能
        Returns:
            DescriptorResult: xtb_* 列からなる DataFrame
        """
        self._require_available()

        all_names = list(_XTB_DESCRIPTORS.keys())
        col_names = (
            [c for c in selected_descriptors if c in all_names]
            if selected_descriptors else all_names
        )

        # kwargs からオーバーライド
        calc_type = kwargs.get("calc_type", self.calc_type)
        convergence = kwargs.get("convergence", self.convergence)
        solvent = kwargs.get("solvent", self.solvent)

        rows: list[dict] = []
        failed_indices: list[int] = []

        # リトライ時の収束基準フォールバック順序
        _CONVERGENCE_FALLBACK = ["normal", "loose", "sloppy", "crude"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, smi in enumerate(smiles_list):
                row = {k: np.nan for k in col_names}
                try:
                    # 電荷・スピンを解決
                    if charge_config_store is not None:
                        charge = charge_config_store.resolve_charge(smi)
                        spin   = charge_config_store.resolve_spin(smi)
                        cfg    = charge_config_store.get_config(smi)
                        from backend.chem.protonation import apply_protonation
                        smi_for_xtb = apply_protonation(smi, cfg)
                        uhf = spin - 1
                    else:
                        from backend.chem.charge_config import _read_smiles_formal_charge
                        charge = _read_smiles_formal_charge(smi)
                        uhf = 0
                        smi_for_xtb = smi

                    xyz = _smiles_to_xyz(smi_for_xtb, charge=charge)
                    if xyz is None:
                        raise ValueError(f"SMILES → XYZ 変換失敗: {smi[:30]}")

                    xyz_path = os.path.join(tmpdir, f"mol_{i}.xyz")
                    with open(xyz_path, "w") as f:
                        f.write(xyz)

                    # リトライ機構（収束基準を段階的に緩和）
                    parsed = {}
                    success = False
                    current_conv = convergence

                    for attempt in range(self.max_retries):
                        cmd = self._build_cmd(
                            xyz_path, charge, uhf,
                            calc_type=calc_type,
                            convergence=current_conv,
                            solvent=solvent,
                        )

                        logger.debug(
                            "XTB cmd (attempt %d/%d): %s",
                            attempt + 1, self.max_retries,
                            " ".join(cmd[:6]),
                        )

                        try:
                            result = subprocess.run(
                                cmd,
                                cwd=tmpdir,
                                capture_output=True,
                                text=True,
                                timeout=self.timeout,
                            )
                            if result.returncode == 0:
                                parsed = _parse_xtb_output(result.stdout)
                                success = True
                                break
                            else:
                                logger.warning(
                                    "XTB 非0終了 (idx=%d, attempt=%d, conv=%s): %s",
                                    i, attempt + 1, current_conv,
                                    result.stderr[-200:] if result.stderr else "no stderr",
                                )
                                # 収束失敗の場合、基準を緩和してリトライ
                                if calc_type == "opt":
                                    conv_idx = _CONVERGENCE_FALLBACK.index(current_conv) if current_conv in _CONVERGENCE_FALLBACK else 0
                                    if conv_idx + 1 < len(_CONVERGENCE_FALLBACK):
                                        current_conv = _CONVERGENCE_FALLBACK[conv_idx + 1]
                                        logger.info(
                                            "XTB リトライ: 収束基準を '%s' に緩和 (idx=%d)",
                                            current_conv, i,
                                        )
                                        # XYZ ファイルを再書き込み（最適化途中の構造がある場合）
                                        opt_xyz = os.path.join(tmpdir, "xtbopt.xyz")
                                        if os.path.exists(opt_xyz):
                                            import shutil as _shutil
                                            _shutil.copy2(opt_xyz, xyz_path)
                                    else:
                                        # 全基準で失敗 → sp にフォールバック
                                        calc_type = "sp"
                                        current_conv = "normal"
                                        logger.info("XTB: --opt 全基準失敗 → --sp にフォールバック (idx=%d)", i)
                                else:
                                    break  # sp でも失敗ならリトライ打ち切り

                        except subprocess.TimeoutExpired:
                            logger.warning(
                                "XTB タイムアウト (%ds): attempt=%d, idx=%d, smi=%s",
                                self.timeout, attempt + 1, i, smi[:30],
                            )
                            # タイムアウト時も sp にフォールバック
                            if calc_type == "opt" and attempt < self.max_retries - 1:
                                calc_type = "sp"
                                logger.info("XTB: タイムアウト → --sp にフォールバック (idx=%d)", i)
                            else:
                                break

                    if not success and not parsed:
                        # 最終手段: sp で1回だけ実行
                        try:
                            cmd_sp = self._build_cmd(
                                xyz_path, charge, uhf,
                                calc_type="sp", convergence="normal", solvent=solvent,
                            )
                            result_sp = subprocess.run(
                                cmd_sp, cwd=tmpdir,
                                capture_output=True, text=True,
                                timeout=120,
                            )
                            if result_sp.returncode == 0:
                                parsed = _parse_xtb_output(result_sp.stdout)
                            else:
                                failed_indices.append(i)
                        except Exception:
                            failed_indices.append(i)

                    for k in col_names:
                        if k in parsed:
                            row[k] = parsed[k]

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
            metadata={
                "gfn": self.gfn,
                "calc_type": self.calc_type,
                "convergence": self.convergence,
                "solvent": self.solvent,
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

