# scratch/check_imports.py
import sys
import os
sys.path.append(os.getcwd())

try:
    from backend.routers import health, data, ml, chem, export, ws
    print("SUCCESS: All routers imported successfully.")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except Exception as e:
    import traceback
    print(f"EXCEPTION: {e}")
    traceback.print_exc()
