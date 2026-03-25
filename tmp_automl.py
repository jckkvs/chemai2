# -*- coding: utf-8 -*-
"""automl.pyにcount_normalization伝搬 + analysis_runnerでstate連携"""

# 1. automl.py
fp = 'C:/Users/horie/chemai2/backend/models/automl.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# automl.pyの__init__にcount_normalizationパラメータ追加
old_init = '        self.selected_descriptors = selected_descriptors'
# automl.pyの__init__を確認
if 'self.count_normalization' not in content:
    # selected_descriptorsの直後にcount_normalization追加
    content = content.replace(
        old_init,
        old_init + '\n        self.count_normalization: str = "density"',
        1
    )
    changes += 1

# L245-248: 最初のSmilesDescriptorTransformer呼び出し
old1 = '''                    st_trans = SmilesDescriptorTransformer(
                        smiles_col=smiles_col,
                        selected_descriptors=self.selected_descriptors
                    )'''
new1 = '''                    st_trans = SmilesDescriptorTransformer(
                        smiles_col=smiles_col,
                        selected_descriptors=self.selected_descriptors,
                        count_normalization=self.count_normalization,
                    )'''
if old1 in content:
    content = content.replace(old1, new1, 1)
    changes += 1
else:
    print("WARNING: automl call 1 not found")

# L321-324: 2番目のSmilesDescriptorTransformer呼び出し
old2 = '''            st_trans = SmilesDescriptorTransformer(
                smiles_col=smiles_col,
                selected_descriptors=self.selected_descriptors
            )'''
new2 = '''            st_trans = SmilesDescriptorTransformer(
                smiles_col=smiles_col,
                selected_descriptors=self.selected_descriptors,
                count_normalization=self.count_normalization,
            )'''
if old2 in content:
    content = content.replace(old2, new2, 1)
    changes += 1
else:
    print("WARNING: automl call 2 not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"automl.py: {changes} changes")

# 2. analysis_runner.pyでstate["count_normalization"]をautomlに渡す
fp2 = 'C:/Users/horie/chemai2/frontend_nicegui/components/analysis_runner.py'
with open(fp2, 'r', encoding='utf-8') as f:
    content2 = f.read()

changes2 = 0

# AutoMLRunnerの構築箇所を探して count_normalization を渡す
if 'count_normalization' not in content2:
    # selected_descriptorsの渡し部分を探す
    old_runner = 'runner.selected_descriptors = set_descs'
    new_runner = '''runner.selected_descriptors = set_descs
    runner.count_normalization = state.get("count_normalization", "density")'''
    if old_runner in content2:
        content2 = content2.replace(old_runner, new_runner, 1)
        changes2 += 1
    else:
        print("WARNING: analysis_runner selected_descriptors not found")

with open(fp2, 'w', encoding='utf-8') as f:
    f.write(content2)

print(f"analysis_runner.py: {changes2} changes")
