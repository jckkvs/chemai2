# -*- coding: utf-8 -*-
"""run_analysis()の_run_engine_sync呼び出しにcount_normalization追加"""

fp = 'C:/Users/horie/chemai2/frontend_nicegui/components/analysis_runner.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# _run_engine_sync呼び出し時にcount_normalization追加
old = '''                    monotonic_constraints=monotonic_constraints,
                )'''
new = '''                    monotonic_constraints=monotonic_constraints,
                    count_normalization=state.get("count_normalization", "density"),
                )'''

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
else:
    print("WARNING: target not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes")
