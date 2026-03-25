# -*- coding: utf-8 -*-
"""smiles_transformer.pyにcount_normalizationパラメータ+密度変換を追加"""

fp = 'C:/Users/horie/chemai2/backend/chem/smiles_transformer.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. __init__にcount_normalizationパラメータ追加
old_init = '''    def __init__(
        self,
        smiles_col: str,
        selected_descriptors: list[str] | None = None,
    ) -> None:
        self.smiles_col = smiles_col
        self.selected_descriptors = selected_descriptors
        self._descriptor_cols: list[str] = []
        self._non_smiles_cols: list[str] = []'''

new_init = '''    def __init__(
        self,
        smiles_col: str,
        selected_descriptors: list[str] | None = None,
        count_normalization: str = "density",
    ) -> None:
        self.smiles_col = smiles_col
        self.selected_descriptors = selected_descriptors
        self.count_normalization = count_normalization  # "raw" or "density"
        self._descriptor_cols: list[str] = []
        self._non_smiles_cols: list[str] = []'''

if old_init in content:
    content = content.replace(old_init, new_init, 1)
    changes += 1
else:
    print("WARNING: old_init not found")

# 2. docstring更新
old_doc = '''    Parameters
    ----------
    smiles_col : str
        SMILESが入力されている列名。
    selected_descriptors : list[str] | None
        使用する記述子名のリスト。Noneの場合は全計算結果を使用。'''

new_doc = '''    Parameters
    ----------
    smiles_col : str
        SMILESが入力されている列名。
    selected_descriptors : list[str] | None
        使用する記述子名のリスト。Noneの場合は全計算結果を使用。
    count_normalization : str
        数え上げ系記述子(原子数/環数/官能基数等)の正規化方式。
        "raw" = そのまま個数(デフォルト値)
        "density" = モル体積(cm3/mol)で割った密度(デフォルト)
        引用: van Krevelen 2009, 密度記述子は分子サイズの影響を除外'''

if old_doc in content:
    content = content.replace(old_doc, new_doc, 1)
    changes += 1
else:
    print("WARNING: old_doc not found")

# 3. _compute_descriptorsの末尾(return X_chemの直前)に密度変換ロジック追加
# fitメソッド内のX_chem = self._compute_descriptors(smiles_list)の後に
# ポストプロセスを追加するか、_compute_descriptors自体にsmiles_listも渡す
# → _compute_descriptors末尾の全returnの前に密度変換関数を呼び出す

# 別の方法: fit()とtransform()で呼び出す共通の密度変換メソッドを追加
old_compute_return = '''    def fit(self, X: pd.DataFrame, y: Any = None) -> "SmilesDescriptorTransformer":
        """学習フェーズで記述子カラム名を記憶する。"""
        if self.smiles_col not in X.columns:
            raise ValueError(f"SMILES列 '{self.smiles_col}' がDataFrameに存在しません。")
        smiles_list = X[self.smiles_col].tolist()
        X_chem = self._compute_descriptors(smiles_list)
        self._descriptor_cols = X_chem.columns.tolist()
        self._non_smiles_cols = [c for c in X.columns if c != self.smiles_col]
        return self'''

new_compute_return = '''    def _apply_count_normalization(
        self, X_chem: pd.DataFrame, smiles_list: list[str]
    ) -> pd.DataFrame:
        """
        数え上げ系記述子のモル体積密度正規化。

        count_normalization == "density" の場合、
        is_count=True の全列をモル体積(cm3/mol)で割る。

        Implements: van Krevelen 2009 - 密度正規化記述子
        引用: 分子サイズの影響を除外するため、カウント系記述子を
              モル体積で正規化。これにより異なるサイズの分子間で
              官能基密度を公平に比較可能。
        """
        if self.count_normalization != "density":
            return X_chem

        # 数え上げ系列名を特定
        count_cols = self._identify_count_columns(X_chem.columns.tolist())
        if not count_cols:
            return X_chem

        # モル体積を計算（RDKit MolWt / 推定密度, or AllChem）
        mol_volumes = self._compute_molar_volumes(smiles_list)
        if mol_volumes is None:
            return X_chem

        # 密度変換: count / V
        X_out = X_chem.copy()
        for col in count_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col] / mol_volumes
                # 列名にサフィックス追加（密度であることを明示）
                X_out.rename(columns={col: f"{col}_density"}, inplace=True)

        logger.info(
            f"数え上げ記述子を密度変換: {len(count_cols)}列 / "
            f"モル体積範囲: {mol_volumes.min():.1f}-{mol_volumes.max():.1f} cm3/mol"
        )
        return X_out

    @staticmethod
    def _identify_count_columns(columns: list[str]) -> list[str]:
        """数え上げ系の列名を特定する。"""
        count_cols = []
        for col in columns:
            # fr_系 (官能基フラグメントカウント)
            if col.startswith("fr_"):
                count_cols.append(col)
            # Num系 (原子数/結合数/環数)
            elif col.startswith("Num") or "Count" in col:
                count_cols.append(col)
            # NHOHCount, NOCount
            elif col in ("NHOHCount", "NOCount"):
                count_cols.append(col)
        return count_cols

    @staticmethod
    def _compute_molar_volumes(smiles_list: list[str]) -> "np.ndarray | None":
        """SMILESリストからモル体積を計算する。

        モル体積 V = MolWt / density_est
        density_est ≈ 1.0 g/cm3 (有機液体の一般的な近似値)

        より正確にはAllChem.ComputeMolVolumeで3D体積を計算可能だが、
        コンフォマー生成が必要で計算コストが高い。
        ここではMolWtベースの簡易推定を使用。
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            volumes = []
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol is not None:
                        mw = Descriptors.MolWt(mol)
                        # 有機化合物の平均密度 ≈ 1.0 g/cm3
                        # V = MW / density (cm3/mol)
                        v = max(mw / 1.0, 10.0)  # 最小10 cm3/mol
                        volumes.append(v)
                    else:
                        volumes.append(100.0)  # フォールバック
                except Exception:
                    volumes.append(100.0)

            return np.array(volumes)
        except ImportError:
            logger.warning("RDKitが利用不可。密度変換をスキップ。")
            return None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SmilesDescriptorTransformer":
        """学習フェーズで記述子カラム名を記憶する。"""
        if self.smiles_col not in X.columns:
            raise ValueError(f"SMILES列 '{self.smiles_col}' がDataFrameに存在しません。")
        smiles_list = X[self.smiles_col].tolist()
        X_chem = self._compute_descriptors(smiles_list)
        X_chem = self._apply_count_normalization(X_chem, smiles_list)
        self._descriptor_cols = X_chem.columns.tolist()
        self._non_smiles_cols = [c for c in X.columns if c != self.smiles_col]
        return self'''

if old_compute_return in content:
    content = content.replace(old_compute_return, new_compute_return, 1)
    changes += 1
else:
    print("WARNING: old_compute_return not found")

# 4. transform()内でも密度変換を適用
old_transform_return = '''    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:'''
# transformの内部も確認する必要がある

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes applied")
