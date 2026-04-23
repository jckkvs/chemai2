import re
import os

files = [
    'frontend_nicegui/main.py',
    'frontend_nicegui/components/data_tab.py',
    'frontend_nicegui/components/ml_workflow.py',
    'frontend_nicegui/components/analysis_runner.py'
]

def fix_file(rel_path):
    abs_path = os.path.join(os.getcwd(), rel_path)
    if not os.path.exists(abs_path):
        print(f"File not found: {rel_path}")
        return

    # UTF-8で読み込み（BOMや変なエンコーディングを無視/修復）
    try:
        with open(abs_path, 'rb') as f:
            raw_content = f.read()
        
        # BOM除去とデコード
        if raw_content.startswith(b'\xef\xbb\xbf'):
            raw_content = raw_content[3:]
        
        content = raw_content.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Read error {rel_path}: {e}")
        return

    # 置換ルール
    replacements = [
        (r'from\s+future\s+import\s+annotations', 'from __future__ import annotations'),
        (r'logging\.getLogger\(name\)', 'logging.getLogger(__name__)'),
        (r'getLogger\(name\)', 'getLogger(__name__)'),
        (r'Path\(file\)', 'Path(__file__)'),
    ]

    new_content = content
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, new_content)

    if new_content != content:
        print(f"Fixed: {rel_path}")
    else:
        print(f"No changes needed for patterns in: {rel_path}")

    # クリーンなUTF-8 (BOMなし) で保存
    with open(abs_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)

for f in files:
    fix_file(f)

# frontend_nicegui内の全ファイルを一応チェック
for root, dirs, filenames in os.walk('frontend_nicegui'):
    for filename in filenames:
        if filename.endswith('.py'):
            fix_file(os.path.join(root, filename))
