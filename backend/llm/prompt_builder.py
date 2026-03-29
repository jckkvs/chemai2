"""
backend/llm/prompt_builder.py

外部LLM（ChatGPT, Copilot, Claude等）に渡すプロンプトを生成するモジュール。

役割:
  - ユーザーの「やりたいこと」をプラグイン作成プロンプトに変換
  - 外部AIが形式を間違えないよう、完全な仕様+例をプロンプトに含める
  - 生成されたコードをアプリに貼り付けて使えるよう検証もサポート
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DescriptorIntent:
    """ユーザーの記述子作成意図を表すデータクラス。"""
    library: str = ""          # 使用ライブラリ（例: rdkit, mordred, padelpy）
    what_to_calc: str = ""     # 計算したい物性（例: 分子量、LogP、HOMO-LUMOギャップ）
    output_type: str = "single"  # "single"=1値, "multi"=複数値(DataFrame)
    extra_notes: str = ""      # 追加の注意事項・制約

    @property
    def is_valid(self) -> bool:
        """最低限の情報が揃っているか。"""
        return bool(self.what_to_calc.strip())


def build_external_llm_prompt(intent: DescriptorIntent) -> str:
    """
    外部LLM（ChatGPT等）に渡すプロンプトを生成する。

    Args:
        intent: ユーザーの意図

    Returns:
        外部AIに貼り付けるための完全なプロンプト文字列
    """
    library_note = f"使用ライブラリ: **{intent.library}**" if intent.library else ""
    multi_note = "" if intent.output_type == "single" else _MULTI_OUTPUT_NOTE

    prompt = _PROMPT_TEMPLATE.format(
        what_to_calc=intent.what_to_calc.strip(),
        library_note=library_note,
        multi_note=multi_note,
        extra_notes=intent.extra_notes.strip() or "特になし",
    )
    return prompt.strip()


# ── プロンプトテンプレート ────────────────────────────────────────────────────

_PLUGIN_SPEC = '''\
## 📋 プラグインファイルの仕様

以下のモジュールレベル定数と `compute()` 関数を持つ Python ファイルを作成してください。

### 必須定数
```python
DESCRIPTOR_NAME = "記述子の英語識別名"     # 必須: 他と重複しない短い名前
DESCRIPTOR_CATEGORY = "カテゴリ名"          # 必須: 例 "物理化学", "電子状態", "トポロジー"
DESCRIPTOR_ENGINE = "エンジン名"            # 必須: 例 "RDKit", "PaDEL", "カスタム"
DESCRIPTOR_DESCRIPTION = "この記述子の説明" # 推奨: 日本語でOK
```

### compute() 関数（1つの値を返す場合）
```python
def compute(smiles_list: list[str]) -> list[float | None]:
    """記述子の計算。"""
    results = []
    for smi in smiles_list:
        try:
            # ← SMILESから値を計算するコードをここに記述
            value = ...
            results.append(float(value))
        except Exception:
            results.append(None)  # 失敗した分子はNoneを返す
    return results
```

### compute() 関数（複数の値を返す場合）
```python
MULTI_DESCRIPTOR = True  # 複数返す場合はこの定数を追加

def compute(smiles_list: list[str]) -> "pd.DataFrame":
    """複数記述子の計算。"""
    import pandas as pd
    rows = []
    for smi in smiles_list:
        try:
            # ← 複数値をdictで返す
            row = {"記述子1": ..., "記述子2": ...}
            rows.append(row)
        except Exception:
            rows.append({})  # 失敗した分子は空dict
    return pd.DataFrame(rows)
```

### ⚠️ 禁止事項
- `os.system()`, `subprocess`, `os.popen()` など外部プロセス実行**禁止**
- `eval()`, `exec()`, `compile()` **禁止**
- `open()` などファイルI/O **禁止**（ライブラリI/Oは除く）
- 型注釈は省略可だが、`compute()` の引数は必ずリストとして受け取ること
- エラー時は例外をraiseするのではなく `None` を返すこと

### ✅ 推奨事項
- 各SMILESのループ内で try/except を使い、1分子の失敗で全体が止まらないように
- RDKit を使う場合は `Chem.MolFromSmiles(smi)` でMolオブジェクトを作成し、`None` チェックを行う
- NumPy/pandas のインポートはループ外で行う（パフォーマンス向上）
'''

_EXAMPLE_RDKIT = '''\
## 💡 実装例（RDKit で分子量と LogP を計算する場合）

```python
DESCRIPTOR_NAME = "MW_LogP"
DESCRIPTOR_CATEGORY = "物理化学"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = "分子量(MW)とLogPを計算します"
MULTI_DESCRIPTOR = True

def compute(smiles_list: list[str]):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({"MW": None, "LogP": None})
                continue
            rows.append({
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
            })
        except Exception:
            rows.append({"MW": None, "LogP": None})
    return pd.DataFrame(rows)
```
'''

_MULTI_OUTPUT_NOTE = """\

### 📌 注意: 複数の記述子を返す場合
`MULTI_DESCRIPTOR = True` を追加し、`compute()` が `pd.DataFrame` を返すようにしてください。
"""

_PROMPT_TEMPLATE = """\
# ChemAI ML Studio 用 SMILES 記述子プラグイン作成依頼

あなたは化学情報処理の専門家として、ChemAI ML Studio のカスタム記述子プラグインを作成してください。

## 🎯 作成したい記述子
{what_to_calc}

{library_note}
{multi_note}

## 📝 追加の要件・制約
{extra_notes}

---

{plugin_spec}

---

{example}

---

## ✏️ 出力形式
- **Pythonコードのみ**を出力してください（説明文・コードブロック記号 ``` は不要）
- コードは上記仕様に従い、`DESCRIPTOR_NAME`, `DESCRIPTOR_CATEGORY`, `DESCRIPTOR_ENGINE`, `compute()` を必ず含めてください
- コードをそのまま `.py` ファイルとして保存すれば動作するよう、完全な実装にしてください
""".replace("{plugin_spec}", _PLUGIN_SPEC).replace("{example}", _EXAMPLE_RDKIT)
