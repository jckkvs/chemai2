import sys
sys.path.insert(0, 'C:/Users/horie/chemai2_cc')
print("=== TEST SCRIPT START ===", flush=True)
from frontend_nicegui.pages import automl_page
print(f"Module file: {automl_page.__file__}", flush=True)
import inspect
source = inspect.getsource(automl_page.AutoMLPage._execute_automl_pipeline)
for i, line in enumerate(source.split('\n'), 1):
    if 'cv_folds' in line:
        print(f'{i}: {line}', flush=True)
print("=== TEST SCRIPT END ===", flush=True)
