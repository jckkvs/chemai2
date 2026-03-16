"""
backend/chem/rdkit_adapter.py

RDKit を使った化合物特徴量化アダプタ。
Descriptors.descList の全217記述子 + Gasteiger電荷 + フィンガープリントを計算。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorResult
from backend.utils.optional_import import safe_import

_rdkit = safe_import("rdkit", "rdkit")
logger = logging.getLogger(__name__)


# ─── 主要記述子の日本語説明・カテゴリ辞書 ──────────────────────
# format: {RDKit内部名: (日本語名, カテゴリ, 物理化学的意味)}
_DESCRIPTOR_JP_META: dict[str, tuple[str, str, str]] = {
    # ── 分子サイズ・形状 ──
    "MolWt":                ("分子量",          "分子サイズ",   "分子の質量（Da）。大きいほど沸点・粘度が上がる"),
    "HeavyAtomMolWt":      ("重原子分子量",     "分子サイズ",   "水素を除いた分子量"),
    "ExactMolWt":           ("精密分子量",       "分子サイズ",   "同位体を考慮した正確な分子量"),
    "HeavyAtomCount":       ("重原子数",         "分子サイズ",   "水素以外の原子の総数"),
    "NumAtoms":             ("全原子数",         "分子サイズ",   "水素含む全原子数"),
    "LabuteASA":            ("Labuteの近似表面積", "分子サイズ", "分子が溶媒に露出する表面積の近似値"),
    "PEOE_VSA1":            ("PEOE_VSA1",        "分子サイズ",   "部分電荷ベースの分子表面積（最も負電荷の領域）"),

    # ── 極性・電子状態 ──
    "MolLogP":              ("LogP",             "極性・溶解性", "脂溶性指標。正→疎水的、負→親水的"),
    "TPSA":                 ("極性表面積",       "極性・溶解性", "極性原子（O,N）由来の表面積。水溶性・膜透過性に相関"),
    "MolMR":                ("モル屈折",         "極性・溶解性", "分極率の指標。屈折率に直結"),
    "MaxPartialCharge":     ("最大部分電荷",     "電子状態",     "最も正に帯電した原子の電荷"),
    "MinPartialCharge":     ("最小部分電荷",     "電子状態",     "最も負に帯電した原子の電荷"),
    "MaxAbsPartialCharge":  ("最大|部分電荷|",   "電子状態",     "部分電荷の絶対値最大（反応性の指標）"),
    "MinAbsPartialCharge":  ("最小|部分電荷|",   "電子状態",     "部分電荷の絶対値最小"),

    # ── 水素結合 ──
    "NumHAcceptors":        ("水素結合受容体数", "水素結合",     "水素結合を受容できる原子数（O,N等）"),
    "NumHDonors":           ("水素結合供与体数", "水素結合",     "水素結合を供与できる-OH,-NH等の数"),

    # ── トポロジー・結合性 ──
    "NumRotatableBonds":    ("回転可能結合数",   "トポロジー",   "柔軟性の指標。多いほど分子が柔らかい"),
    "RingCount":            ("環の総数",         "トポロジー",   "脂環・芳香環を含むすべての環の数"),
    "NumAromaticRings":     ("芳香環数",         "トポロジー",   "π電子系の芳香環の数"),
    "NumAliphaticRings":    ("脂肪族環数",       "トポロジー",   "芳香族でない環の数"),
    "NumSaturatedRings":    ("飽和環数",         "トポロジー",   "不飽和結合を含まない環の数"),
    "NumHeterocycles":      ("ヘテロ環数",       "トポロジー",   "N,O,S等を含む環の数"),
    "NumAromaticHeterocycles": ("芳香族ヘテロ環数", "トポロジー", "芳香族かつヘテロ原子を含む環"),
    "FractionCSP3":         ("sp3炭素割合",      "トポロジー",   "sp3炭素の比率。高い→立体的、低い→平面的"),
    "NumValenceElectrons":  ("価電子数",         "トポロジー",   "価電子の総数"),

    # ── 位相的指標 ──
    "BalabanJ":             ("Balaban J指数",    "位相的指標",   "分子のグラフ構造を表す位相指標"),
    "BertzCT":              ("Bertz複雑度",      "位相的指標",   "分子の構造的複雑さ"),
    "HallKierAlpha":        ("Hall-Kier α",      "位相的指標",   "原子サイズに基づく分子形状指標"),
    "Kappa1":               ("κ1",               "位相的指標",   "1次の分子形状指数（直線性）"),
    "Kappa2":               ("κ2",               "位相的指標",   "2次の分子形状指数（分岐）"),
    "Kappa3":               ("κ3",               "位相的指標",   "3次の分子形状指数（空間的な広がり）"),
    "Chi0":                 ("χ0",               "位相的指標",   "0次の分子結合性指数"),
    "Chi0n":                ("χ0n",              "位相的指標",   "0次の結合性指数（原子価補正）"),
    "Chi1":                 ("χ1",               "位相的指標",   "1次の分子結合性指数"),
    "Chi1n":                ("χ1n",              "位相的指標",   "1次の結合性指数（原子価補正）"),
    "Ipc":                  ("情報含量",          "位相的指標",   "グラフの情報含量（複雑さの指標）"),
    "AvgIpc":               ("平均情報含量",      "位相的指標",   "原子あたりの平均情報含量"),

    # ── BCUT記述子 ──
    "BCUT2D_MWHI":          ("BCUT MW高",        "BCUT",         "分子量に基づくBurdenマトリクスの最大固有値"),
    "BCUT2D_MWLOW":         ("BCUT MW低",        "BCUT",         "分子量に基づくBurdenマトリクスの最小固有値"),
    "BCUT2D_CHGHI":         ("BCUT 電荷高",      "BCUT",         "部分電荷に基づく最大固有値"),
    "BCUT2D_CHGLO":         ("BCUT 電荷低",      "BCUT",         "部分電荷に基づく最小固有値"),
    "BCUT2D_LOGPHI":        ("BCUT LogP高",      "BCUT",         "LogPに基づく最大固有値"),
    "BCUT2D_LOGPLOW":       ("BCUT LogP低",      "BCUT",         "LogPに基づく最小固有値"),
    "BCUT2D_MRHI":          ("BCUT MR高",        "BCUT",         "モル屈折に基づく最大固有値"),
    "BCUT2D_MRLOW":         ("BCUT MR低",        "BCUT",         "モル屈折に基づく最小固有値"),

    # ── EState指標 ──
    "MaxAbsEStateIndex":    ("最大|EState|",     "EState",       "電気化学的状態指標の絶対値最大"),
    "MaxEStateIndex":       ("最大EState",        "EState",       "最も正のEState値"),
    "MinAbsEStateIndex":    ("最小|EState|",     "EState",       "EState絶対値最小"),
    "MinEStateIndex":       ("最小EState",        "EState",       "最も負のEState値（電子豊富な原子）"),

    # ── 薬品適性・QED ──
    "qed":                  ("QED(薬品適性)",     "薬品適性",     "薬品としての望ましさスコア（0-1）"),
    "SPS":                  ("SPS指数",           "薬品適性",     "合成容易性を考慮した薬品スコア"),
    "NHOHCount":            ("N-OH/N-H数",       "官能基",       "アルコール/アミンの-OH,-NHの数"),
    "NOCount":              ("NO数",              "官能基",       "窒素・酸素原子の合計数"),
    "NumRadicalElectrons":  ("ラジカル電子数",    "電子状態",     "不対電子の数（ラジカル種）"),

    # ── フィンガープリント密度 ──
    "FpDensityMorgan1":     ("FP密度Morgan1",    "FP密度",       "半径1のMorganFP密度"),
    "FpDensityMorgan2":     ("FP密度Morgan2",    "FP密度",       "半径2のMorganFP密度"),
    "FpDensityMorgan3":     ("FP密度Morgan3",    "FP密度",       "半径3のMorganFP密度"),
}

# fr_系（官能基フラグメントカウント）は自動で日本語名を付与
_FR_DESCRIPTORS_JP: dict[str, str] = {
    "fr_Al_COO": "脂肪族カルボン酸", "fr_Al_OH": "脂肪族ヒドロキシル",
    "fr_Al_OH_noTert": "脂肪族OH(3級以外)", "fr_ArN": "芳香族窒素",
    "fr_Ar_COO": "芳香族カルボン酸", "fr_Ar_N": "芳香族アミン",
    "fr_Ar_NH": "芳香族NH", "fr_Ar_OH": "フェノール性OH",
    "fr_COO": "カルボン酸(エステル含)", "fr_COO2": "カルボン酸",
    "fr_C_O": "カルボニル", "fr_C_O_noCOO": "カルボニル(COO除く)",
    "fr_C_S": "チオカルボニル", "fr_HOCCN": "β-ヒドロキシアミン",
    "fr_Imine": "イミン", "fr_NH0": "3級アミン",
    "fr_NH1": "2級アミン", "fr_NH2": "1級アミン",
    "fr_N_O": "N-酸化物", "fr_Ndealkylation1": "N-脱アルキル化部位1",
    "fr_Ndealkylation2": "N-脱アルキル化部位2",
    "fr_Nhpyrrole": "ピロール型NH", "fr_SH": "チオール",
    "fr_aldehyde": "アルデヒド", "fr_alkyl_carbamate": "アルキルカーバメート",
    "fr_alkyl_halide": "アルキルハライド", "fr_allylic_oxid": "アリル酸化部位",
    "fr_amide": "アミド", "fr_amidine": "アミジン",
    "fr_aniline": "アニリン", "fr_azide": "アジド",
    "fr_azo": "アゾ", "fr_barbitur": "バルビツール酸",
    "fr_benzene": "ベンゼン環", "fr_benzodiazepine": "ベンゾジアゼピン",
    "fr_bicyclic": "二環式", "fr_diazo": "ジアゾ",
    "fr_dihydropyridine": "ジヒドロピリジン", "fr_epoxide": "エポキシド",
    "fr_ester": "エステル", "fr_ether": "エーテル",
    "fr_furan": "フラン", "fr_guanido": "グアニジン",
    "fr_halogen": "ハロゲン", "fr_hdrzine": "ヒドラジン",
    "fr_hdrzone": "ヒドラゾン", "fr_imidazole": "イミダゾール",
    "fr_imide": "イミド", "fr_isocyan": "イソシアネート",
    "fr_isothiocyan": "イソチオシアネート", "fr_ketone": "ケトン",
    "fr_ketone_Topliss": "ケトン(Topliss)", "fr_lactam": "ラクタム",
    "fr_lactone": "ラクトン", "fr_methoxy": "メトキシ",
    "fr_morpholine": "モルホリン", "fr_nitrile": "ニトリル",
    "fr_nitro": "ニトロ", "fr_nitro_arom": "芳香族ニトロ",
    "fr_nitro_arom_nonortho": "芳香族ニトロ(非オルト)",
    "fr_nitroso": "ニトロソ", "fr_oxazole": "オキサゾール",
    "fr_oxime": "オキシム", "fr_para_hydroxylation": "パラ水酸化部位",
    "fr_phenol": "フェノール", "fr_phenol_noOrthoHbond": "フェノール(オルトHB無)",
    "fr_phos_acid": "リン酸", "fr_phos_ester": "リン酸エステル",
    "fr_piperdine": "ピペリジン", "fr_piperzine": "ピペラジン",
    "fr_priamide": "1級アミド", "fr_prisulfonamd": "1級スルホンアミド",
    "fr_pyridine": "ピリジン", "fr_quatN": "4級窒素",
    "fr_sulfide": "スルフィド", "fr_sulfonamd": "スルホンアミド",
    "fr_sulfone": "スルホン", "fr_term_acetylene": "末端アセチレン",
    "fr_tetrazole": "テトラゾール", "fr_thiazole": "チアゾール",
    "fr_thiocyan": "チオシアネート", "fr_thiophene": "チオフェン",
    "fr_unbrch_alkane": "非分岐アルカン", "fr_urea": "ウレア",
}


class RDKitAdapter(BaseChemAdapter):
    """
    RDKit による化合物記述子計算アダプタ。

    計算内容:
    - RDKit Descriptors.descList の全記述子（217個）
    - Gasteiger 部分電荷統計量（5個）
    - Morgan フィンガープリント（ECFP4: radius=2）
    - RDKit フィンガープリント（2048 bit）
    - MACCS Keys

    Args:
        compute_fp: フィンガープリントも計算するか
        morgan_radius: Morgan FPの半径（デフォルト2 = ECFP4）
        morgan_bits: Morgan FPのビット数
        rdkit_fp_bits: RDKit FPのビット数
        include_maccs: MACCS keysを含めるか
    """

    def __init__(
        self,
        compute_fp: bool = True,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        rdkit_fp_bits: int = 2048,
        include_maccs: bool = False,
        compute_gasteiger: bool = True,
    ) -> None:
        self.compute_fp = compute_fp
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.rdkit_fp_bits = rdkit_fp_bits
        self.include_maccs = include_maccs
        self.compute_gasteiger = compute_gasteiger

    @property
    def name(self) -> str:
        return "rdkit"

    @property
    def description(self) -> str:
        return "RDKit 全記述子（217個）+ 部分電荷 + フィンガープリント計算"

    def is_available(self) -> bool:
        return bool(_rdkit)

    def compute(
        self,
        smiles_list: list[str],
        charge_config_store: Any | None = None,
        **kwargs: Any,
    ) -> DescriptorResult:
        """
        SMILES リストから RDKit 全記述子を計算する。
        """
        self._require_available()

        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem, MACCSkeys
        from rdkit.Chem import rdPartialCharges

        # 全記述子リストを取得
        desc_list = Descriptors.descList  # list of (name, function)

        rows: list[dict[str, float]] = []
        failed: list[int] = []

        for idx, smi in enumerate(smiles_list):
            try:
                # プロトン化変換を適用
                if charge_config_store is not None:
                    cfg = charge_config_store.get_config(smi)
                    from backend.chem.protonation import apply_protonation
                    smi_to_use = apply_protonation(smi, cfg)
                else:
                    smi_to_use = smi

                mol = Chem.MolFromSmiles(smi_to_use)
                if mol is None:
                    mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"無効なSMILES: {smi!r}")

                row: dict[str, float] = {}

                # ── 全Descriptors.descList記述子を計算 ──
                for desc_name, desc_fn in desc_list:
                    try:
                        val = desc_fn(mol)
                        row[desc_name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
                    except Exception:
                        row[desc_name] = np.nan

                # ── Gasteiger 部分電荷統計量 ──
                if self.compute_gasteiger:
                    try:
                        mol_h = Chem.AddHs(mol)
                        rdPartialCharges.ComputeGasteigerCharges(mol_h)
                        charges = [
                            float(mol_h.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'))
                            for i in range(mol_h.GetNumAtoms())
                        ]
                        charges = [q for q in charges if np.isfinite(q)]
                        if charges:
                            ca = np.array(charges)
                            row["gasteiger_q_max"]      = float(np.max(ca))
                            row["gasteiger_q_min"]      = float(np.min(ca))
                            row["gasteiger_q_range"]    = float(np.max(ca) - np.min(ca))
                            row["gasteiger_q_std"]      = float(np.std(ca))
                            row["gasteiger_q_abs_mean"] = float(np.mean(np.abs(ca)))
                        else:
                            for k in ["gasteiger_q_max", "gasteiger_q_min",
                                      "gasteiger_q_range", "gasteiger_q_std", "gasteiger_q_abs_mean"]:
                                row[k] = np.nan
                    except Exception:
                        for k in ["gasteiger_q_max", "gasteiger_q_min",
                                  "gasteiger_q_range", "gasteiger_q_std", "gasteiger_q_abs_mean"]:
                            row[k] = np.nan

                # ── Morgan フィンガープリント (ECFP4) ──
                if self.compute_fp:
                    try:
                        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, self.morgan_radius, nBits=self.morgan_bits
                        )
                        for j, bit in enumerate(morgan_fp):
                            row[f"Morgan_r{self.morgan_radius}_{j}"] = float(bit)
                    except Exception:
                        for j in range(self.morgan_bits):
                            row[f"Morgan_r{self.morgan_radius}_{j}"] = 0.0

                    # RDKit トポロジカルフィンガープリント
                    try:
                        rdkit_fp = Chem.RDKFingerprint(mol, fpSize=self.rdkit_fp_bits)
                        for j, bit in enumerate(rdkit_fp):
                            row[f"RDKitFP_{j}"] = float(bit)
                    except Exception:
                        for j in range(self.rdkit_fp_bits):
                            row[f"RDKitFP_{j}"] = 0.0

                    # MACCS Keys (166 bit)
                    if self.include_maccs:
                        try:
                            maccs = MACCSkeys.GenMACCSKeys(mol)
                            for j, bit in enumerate(maccs):
                                row[f"MACCS_{j}"] = float(bit)
                        except Exception:
                            for j in range(167):
                                row[f"MACCS_{j}"] = 0.0

                rows.append(row)

            except Exception as e:
                logger.warning(f"RDKit: index={idx}, SMILES={smi!r}: {e}")
                failed.append(idx)
                rows.append({})

        df = pd.DataFrame(rows).fillna(0.0)
        logger.info(
            f"RDKit計算完了: {len(smiles_list)}件 / "
            f"失敗={len(failed)} / 記述子={df.shape[1]}"
        )
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed,
            adapter_name=self.name,
            metadata={
                "morgan_radius": self.morgan_radius,
                "morgan_bits": self.morgan_bits if self.compute_fp else 0,
                "total_descriptors": df.shape[1],
            },
        )

    def get_descriptors_metadata(self) -> list:
        """RDKit全記述子の詳細メタデータを返す。"""
        from backend.chem.base import DescriptorMetadata

        try:
            from rdkit.Chem import Descriptors
            desc_names = [name for name, _ in Descriptors.descList]
        except Exception:
            desc_names = list(_DESCRIPTOR_JP_META.keys())

        meta = []
        for name in desc_names:
            jp_info = _DESCRIPTOR_JP_META.get(name)
            if jp_info:
                jp_name, category, meaning = jp_info
                is_count = name.startswith("fr_") or "Count" in name or "Num" in name
                meta.append(DescriptorMetadata(name, f"{jp_name}：{meaning}", is_count=is_count))
            elif name.startswith("fr_"):
                jp_name = _FR_DESCRIPTORS_JP.get(name, name.replace("fr_", "").replace("_", " "))
                meta.append(DescriptorMetadata(name, f"官能基：{jp_name}", is_count=True))
            elif name.startswith("VSA_EState"):
                meta.append(DescriptorMetadata(name, f"EState表面積：{name}", is_count=False))
            elif name.startswith("PEOE_VSA"):
                meta.append(DescriptorMetadata(name, f"部分電荷表面積：{name}", is_count=False))
            elif name.startswith("SMR_VSA"):
                meta.append(DescriptorMetadata(name, f"MR表面積：{name}", is_count=False))
            elif name.startswith("SlogP_VSA"):
                meta.append(DescriptorMetadata(name, f"LogP表面積：{name}", is_count=False))
            elif name.startswith("EState_VSA"):
                meta.append(DescriptorMetadata(name, f"EState-VSA：{name}", is_count=False))
            elif name.startswith("Chi") or name.startswith("Kappa"):
                meta.append(DescriptorMetadata(name, f"位相的指標：{name}", is_count=False))
            else:
                meta.append(DescriptorMetadata(name, name, is_count=False))

        # Gasteiger
        for gn, gd in [
            ("gasteiger_q_max", "Gasteiger最大部分電荷"),
            ("gasteiger_q_min", "Gasteiger最小部分電荷"),
            ("gasteiger_q_range", "Gasteiger電荷レンジ"),
            ("gasteiger_q_std", "Gasteiger電荷標準偏差"),
            ("gasteiger_q_abs_mean", "Gasteiger|電荷|平均"),
        ]:
            meta.append(DescriptorMetadata(gn, gd, is_count=False))

        return meta

    def get_descriptor_names(self) -> list[str]:
        """
        計算可能な記述子名のリストを返す。
        """
        names = [m.name for m in self.get_descriptors_metadata()]
        if self.compute_fp:
            names += [f"Morgan_r{self.morgan_radius}_{j}" for j in range(self.morgan_bits)]
            names += [f"RDKitFP_{j}" for j in range(self.rdkit_fp_bits)]
            if self.include_maccs:
                names += [f"MACCS_{j}" for j in range(167)]
        return names

    @staticmethod
    def get_descriptor_jp_info(name: str) -> tuple[str, str, str] | None:
        """記述子名から (日本語名, カテゴリ, 意味) のタプルを返す。"""
        if name in _DESCRIPTOR_JP_META:
            return _DESCRIPTOR_JP_META[name]
        if name in _FR_DESCRIPTORS_JP:
            return (f"官能基：{_FR_DESCRIPTORS_JP[name]}", "官能基カウント", f"{_FR_DESCRIPTORS_JP[name]}の数")
        return None
