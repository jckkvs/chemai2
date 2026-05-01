"""SMILES structure thumbnail display component for NiceGUI."""

from nicegui import ui
from pathlib import Path
import tempfile
import os

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def render_smiles_thumbnails(df, smiles_col: str, max_display: int = 20) -> None:
    """Render SMILES structure thumbnails in a grid.

    Args:
        df: DataFrame with SMILES data
        smiles_col: Column name containing SMILES strings
        max_display: Maximum number of thumbnails to display
    """
    if not RDKIT_AVAILABLE:
        ui.label("RDKitがインストールされていません。SMILES構造式を表示できません。").classes(
            "text-yellow-400 text-sm"
        )
        return

    if smiles_col not in df.columns:
        ui.label(f"列 '{smiles_col}' が見つかりません").classes("text-red-400 text-sm")
        return

    # Get unique SMILES (or first N rows)
    smiles_list = df[smiles_col].dropna().unique()[:max_display]

    if len(smiles_list) == 0:
        ui.label("有効なSMILESデータがありません").classes("text-gray-400 text-sm")
        return

    ui.label(f"SMILES構造式サムネイル（{len(smiles_list)}件表示）").classes(
        "text-sm text-gray-300 mb-2"
    )

    # Create a grid for thumbnails
    with ui.grid(columns=5).classes("gap-2 w-full") as grid:
        for idx, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    with ui.card().classes("w-24 h-24 bg-gray-800 p-1"):
                        ui.label("無効").classes("text-xs text-red-400")
                    continue

                # Generate image
                img = Draw.MolToImage(mol, size=(150, 150))

                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    img.save(tmp, format="PNG")
                    tmp_path = tmp.name

                # Display image with click handler for enlargement
                with ui.card().classes("w-24 h-24 p-1 bg-gray-800").style(
                    "border: 1px solid #374151;"
                ):
                    ui.image(src=f"/{tmp_path}").classes("w-full h-full object-cover")

                    # Show SMILES on hover (using tooltip)
                    if idx < 3:  # Only show for first few to save space
                        ui.label(smiles[:20] + "..." if len(smiles) > 20 else smiles).classes(
                            "text-xs text-gray-500 mt-1"
                        )

            except Exception as e:
                with ui.card().classes("w-24 h-24 bg-gray-800 p-1"):
                    ui.label("エラー").classes("text-xs text-red-400")


def show_smiles_preview(df, max_per_col: int = 10) -> None:
    """Show SMILES preview for all SMILES columns in the dataframe.

    Args:
        df: DataFrame
        max_per_col: Max thumbnails per SMILES column
    """
    if not RDKIT_AVAILABLE:
        return

    smiles_cols = [col for col in df.columns if "smiles" in col.lower()]

    if not smiles_cols:
        return

    ui.separator().classes("bg-gray-700 my-4")
    ui.label("🧬 SMILES構造式プレビュー").classes("text-lg text-blue-300 mb-4")

    for col in smiles_cols[:3]:  # Max 3 SMILES columns
        with ui.expansion(col, icon="science").classes("w-full bg-gray-800").style(
            "border: 1px solid #374151;"
        ):
            render_smiles_thumbnails(df, col, max_display=max_per_col)
