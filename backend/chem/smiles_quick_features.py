"""
backend/chem/smiles_quick_features.py
SMILES文字列からすぐ計算できる基本特徴量を返す（ホバー表示用）。

20260429.txtの要件：
  「SMILESはマウスオーバーすれば、その構造式やすぐ計算できる特徴量がわかるように」

対象特徴量（RDKitですぐ計算できるもの）：
  - 分子量 (MolWt)
  - LogP (MolLogP)
  - TPSA (TopologicalPolarSurfaceArea)
  - 水素結合ドナー数 (NumHDonors)
  - 水素結合アクセプター数 (NumHAcceptors)
  - 回転可能結合数 (NumRotatableBonds)
  - 芳香環数 (NumAromaticRings)
  - 飽和度 (FractionCSP3)
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# すぐ計算できる基本特徴量のリスト
QUICK_FEATURES = [
    ("MolWt", "分子量", "g/mol"),
    ("MolLogP", "LogP", ""),
    ("TPSA", "TPSA", "Ų"),
    ("NumHDonors", "Hドナー数", ""),
    ("NumHAcceptors", "Hアクセプター数", ""),
    ("NumRotatableBonds", "回転可能結合数", ""),
    ("NumAromaticRings", "芳香環数", ""),
    ("FractionCSP3", "飽和度", ""),
    ("NumHeavyAtoms", "重原子数", ""),
    ("RingCount", "環数", ""),
]


def compute_quick_features(smiles: str) -> Dict[str, Any]:
    """
    SMILESからすぐ計算できる基本特徴量を計算。

    Args:
        smiles: SMILES文字列

    Returns:
        特徴量名→値の辞書。計算失敗時は空dict。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "無効なSMILES"}

        result = {"SMILES": smiles}

        # 基本特徴量を計算
        result["MolWt"] = round(Descriptors.MolWt(mol), 2)
        result["MolLogP"] = round(Descriptors.MolLogP(mol), 2)
        result["TPSA"] = round(rdMolDescriptors.CalcTPSA(mol), 1)
        result["NumHDonors"] = Descriptors.NumHDonors(mol)
        result["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
        result["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
        result["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
        result["FractionCSP3"] = round(Descriptors.FractionCSP3(mol), 3)
        result["NumHeavyAtoms"] = Descriptors.HeavyAtomCount(mol)
        result["RingCount"] = Descriptors.RingCount(mol)

        return result

    except ImportError:
        logger.warning("RDKitがインストールされていません")
        return {"error": "RDKit未インストール"}
    except Exception as e:
        logger.warning(f"特徴量計算エラー: {e}")
        return {"error": str(e)}


def compute_quick_features_batch(smiles_list: List[str], max_count: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    複数SMILESの基本特徴量をまとめて計算（バッチ処理）。

    Args:
        smiles_list: SMILES文字列のリスト
        max_count: 最大計算件数

    Returns:
        {SMILES: {特徴量}} の辞書
    """
    result = {}
    for smi in smiles_list[:max_count]:
        if smi and isinstance(smi, str):
            feat = compute_quick_features(smi)
            result[smi] = feat
    return result


def format_features_for_hover(features: Dict[str, Any], max_items: int = 8) -> str:
    """
    特徴量辞書をホバー表示用のHTML文字列にフォーマット。

    Args:
        features: compute_quick_features()の結果
        max_items: 最大表示項目数

    Returns:
        HTML文字列
    """
    if not features:
        return "特徴量なし"

    if "error" in features:
        return f"<span style='color:#ef4444'>{features['error']}</span>"

    # 表示名のマッピング
    display_map = {key: (jp, unit) for key, jp, unit in QUICK_FEATURES}
    display_map["SMILES"] = ("SMILES", "")

    lines = []
    count = 0
    for key, val in features.items():
        if key == "SMILES":
            continue  # SMILESは別途表示
        if count >= max_items:
            lines.append(f"<i>...他 {len(features) - count - 1}件</i>")
            break
        if key in display_map:
            jp_name, unit = display_map[key]
            unit_str = f" {unit}" if unit else ""
            lines.append(f"<b>{jp_name}</b>: {val}{unit_str}")
            count += 1

    return "<br>".join(lines)


def get_feature_summary_html(smiles: str, img_uri: str = "", img_size: int = 200) -> str:
    """
    SMILES構造式＋特徴量を含む完全なホバーHTMLを生成。

    Args:
        smiles: SMILES文字列
        img_uri: 構造式画像URI（事前生成済み）
        img_size: 画像サイズ

    Returns:
        完全なHTML文字列
    """
    features = compute_quick_features(smiles)
    feature_html = format_features_for_hover(features)

    img_part = ""
    if img_uri:
        img_part = f'<img src="{img_uri}" width="{img_size}" height="{img_size}" style="background:white; border-radius:6px; margin-bottom:8px;">'

    smiles_display = smiles[:40] + ("..." if len(smiles) > 40 else "")

    html = f"""
    <div style="text-align:center; font-family:monospace; font-size:11px; color:#7fffd4; margin-bottom:4px;">
        {smiles_display}
    </div>
    {img_part}
    <div style="border-top:1px solid #4c9be8; padding-top:6px; margin-top:6px; font-size:11px; color:#e0e0f0;">
        {feature_html}
    </div>
    """
    return html
