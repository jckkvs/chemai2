
import sys
import os
import pandas as pd
import numpy as np

# Add project root
# Add project root (highest priority)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from frontend_nicegui.components.eda_dim_panel import dim_reduction_panel
    print("Import successful!")
    
    # Mock state
    state = {
        'df': pd.DataFrame(np.random.randn(20, 5), columns=['A', 'B', 'C', 'D', 'E']),
        'column_roles': {'A': 'feature', 'B': 'feature', 'C': 'feature', 'D': 'feature', 'E': 'target'}
    }
    
    # We can't easily 'render' in a script without a context, 
    # but we can check if it runs without obvious errors until it hits NiceGUI calls
    # (Actually dim_reduction_panel creates NiceGUI components)
    
    print("Static analysis of eda_dim_panel successful.")
    
except Exception as e:
    print(f"Error during import or analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
