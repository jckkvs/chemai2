"""
backend/chem/mixture_csv_template.py

混合物入力用CSVテンプレートの生成・パースモジュール。

テンプレートダウンロード → ユーザー記入 → アップロード → パース
のワークフローを支援する。

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# テンプレートのカラム定義
TEMPLATE_COLUMNS = [
    "session_id",
    "component_order",
    "smiles",
    "compound_name",
    "ratio_value",
    "ratio_unit",
    "other_ratio_unit",
]

# サンプルデータ（テンプレートに含める）
_SAMPLE_ROWS = [
    {
        "session_id": "MIX_001",
        "component_order": 1,
        "smiles": "CCO",
        "compound_name": "ethanol",
        "ratio_value": 70.0,
        "ratio_unit": "weight",
        "other_ratio_unit": "",
    },
    {
        "session_id": "MIX_001",
        "component_order": 2,
        "smiles": "CCCO",
        "compound_name": "1-propanol",
        "ratio_value": 30.0,
        "ratio_unit": "weight",
        "other_ratio_unit": "",
    },
    {
        "session_id": "MIX_002",
        "component_order": 1,
        "smiles": "c1ccccc1",
        "compound_name": "benzene",
        "ratio_value": 0.5,
        "ratio_unit": "mole",
        "other_ratio_unit": "",
    },
    {
        "session_id": "MIX_002",
        "component_order": 2,
        "smiles": "CC(C)O",
        "compound_name": "isopropanol",
        "ratio_value": 0.5,
        "ratio_unit": "mole",
        "other_ratio_unit": "",
    },
    {
        "session_id": "MIX_003",
        "component_order": 1,
        "smiles": "CC(=O)O",
        "compound_name": "acetic_acid",
        "ratio_value": 2.0,
        "ratio_unit": "other",
        "other_ratio_unit": "volume_fraction",
    },
    {
        "session_id": "MIX_003",
        "component_order": 2,
        "smiles": "O",
        "compound_name": "water",
        "ratio_value": 8.0,
        "ratio_unit": "other",
        "other_ratio_unit": "volume_fraction",
    },
]


@dataclass
class ParsedMixture:
    """パース済みの1混合物。"""
    session_id: str
    components: list[dict[str, Any]]
    ratio_unit: str
    other_ratio_unit: str = ""
    warnings: list[str] = field(default_factory=list)


def generate_template_csv(
    include_samples: bool = True,
    n_empty_rows: int = 4,
) -> bytes:
    """
    混合物入力用CSVテンプレートを生成する。

    Args:
        include_samples: True のときサンプルデータを含める。
        n_empty_rows: 空行の追加数。

    Returns:
        BOM付きUTF-8のCSVバイト列（Excel互換）。
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=TEMPLATE_COLUMNS)

    # ヘッダー行前のコメント
    buf.write("# ChemAI2 混合物入力テンプレート v1.0\n")
    buf.write("# 同一session_idの行が1混合物として処理されます\n")
    buf.write("# ratio_unit: weight(重量比) / mole(モル比) / other(その他)\n")
    buf.write("# other選択時はother_ratio_unitに単位名を記入\n")

    writer.writeheader()

    if include_samples:
        for row in _SAMPLE_ROWS:
            writer.writerow(row)

    # 空行
    for i in range(n_empty_rows):
        writer.writerow({c: "" for c in TEMPLATE_COLUMNS})

    csv_str = buf.getvalue()
    # BOM付きUTF-8 (Excel互換)
    return ("\ufeff" + csv_str).encode("utf-8")


def generate_template_dataframe(include_samples: bool = True) -> pd.DataFrame:
    """テンプレートをDataFrameとして返す。"""
    if include_samples:
        return pd.DataFrame(_SAMPLE_ROWS)
    return pd.DataFrame(columns=TEMPLATE_COLUMNS)


def parse_mixture_csv(
    csv_content: str | bytes | io.IOBase | Path,
) -> list[ParsedMixture]:
    """
    混合物CSVファイルをパースする。

    Args:
        csv_content: CSV内容（文字列、バイト列、ファイルオブジェクト、パス）。

    Returns:
        ParsedMixture のリスト（session_id単位でグループ化）。

    Raises:
        ValueError: 必須カラムが欠落している場合。
    """
    # 入力形式の統一
    if isinstance(csv_content, (str, bytes)):
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8-sig")
        # コメント行を除外
        lines = [
            line for line in csv_content.splitlines()
            if not line.strip().startswith("#") and line.strip()
        ]
        csv_str = "\n".join(lines)
        df = pd.read_csv(io.StringIO(csv_str))
    elif isinstance(csv_content, Path):
        text = csv_content.read_text(encoding="utf-8-sig")
        return parse_mixture_csv(text)
    else:
        df = pd.read_csv(csv_content, comment="#")

    # 必須カラムチェック
    required = {"session_id", "smiles", "ratio_value", "ratio_unit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必須カラムが不足: {missing}")

    # 空行を除去
    df = df.dropna(subset=["session_id", "smiles"]).copy()
    df["session_id"] = df["session_id"].astype(str).str.strip()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"].str.len() > 0]

    # session_id でグループ化
    mixtures: list[ParsedMixture] = []

    for sid, group in df.groupby("session_id", sort=False):
        warnings: list[str] = []
        components: list[dict[str, Any]] = []

        ratio_units = group["ratio_unit"].dropna().unique()
        if len(ratio_units) > 1:
            warnings.append(
                f"session '{sid}' 内で比率タイプが混在: {list(ratio_units)}"
            )
        ratio_unit = str(ratio_units[0]) if len(ratio_units) > 0 else "weight"

        for _, row in group.iterrows():
            comp = {
                "smiles": str(row["smiles"]).strip(),
                "ratio_value": float(row.get("ratio_value", 1.0)),
                "ratio_unit": ratio_unit,
                "compound_name": str(row.get("compound_name", "")).strip() or None,
                "component_order": int(row.get("component_order", 0)),
            }
            components.append(comp)

        if len(components) < 2:
            warnings.append(f"session '{sid}' の成分数が2未満: {len(components)}")

        # component_order でソート
        components.sort(key=lambda c: c.get("component_order", 0))

        other_unit = ""
        if ratio_unit == "other":
            ou = group["other_ratio_unit"].dropna()
            if len(ou) > 0:
                other_unit = str(ou.iloc[0]).strip()

        mixtures.append(ParsedMixture(
            session_id=str(sid),
            components=components,
            ratio_unit=ratio_unit,
            other_ratio_unit=other_unit,
            warnings=warnings,
        ))

    logger.info(
        "CSV パース完了: %d混合物, %d成分",
        len(mixtures),
        sum(len(m.components) for m in mixtures),
    )
    return mixtures
