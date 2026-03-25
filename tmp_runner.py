# -*- coding: utf-8 -*-
"""analysis_runner.pyにcount_normalization伝搬"""

fp = 'C:/Users/horie/chemai2/frontend_nicegui/components/analysis_runner.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. _run_engine_syncの引数にcount_normalization追加
old1 = '''    monotonic_constraints: dict[str, int] | None = None,
) -> Any:'''
new1 = '''    monotonic_constraints: dict[str, int] | None = None,
    count_normalization: str = "density",
) -> Any:'''
if old1 in content:
    content = content.replace(old1, new1, 1)
    changes += 1

# 2. AutoMLEngine構築でcount_normalizationを渡す
old2 = '''        monotonic_constraints_dict=monotonic_constraints,
    )'''
if old2 in content:
    content = content.replace(old2,
        '''        monotonic_constraints_dict=monotonic_constraints,
        count_normalization=count_normalization,
    )''', 1)
    changes += 1

# 3. run_analysis()からの呼び出しでcount_normalizationを渡す
old3 = '''            monotonic_constraints=state.get("monotonic_constraints"),'''
if old3 in content:
    content = content.replace(old3,
        '''            monotonic_constraints=state.get("monotonic_constraints"),
            count_normalization=state.get("count_normalization", "density"),''', 1)
    changes += 1

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"analysis_runner.py: {changes} changes")
