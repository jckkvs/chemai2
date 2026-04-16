
import sys
import os
import numpy as np
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.preprocessing.group_scaler import GroupStandardScaler
from backend.ml.group_lasso import train_group_lasso

def test_group_scaler():
    print("Testing GroupStandardScaler...")
    data = {
        'Temp_A': [10, 20, 30],
        'Temp_B': [100, 200, 300],
        'MW': [50, 100, 150]
    }
    df = pd.DataFrame(data)
    
    feature_groups = {"temperature": ["Temp_A", "Temp_B"]}
    scaler = GroupStandardScaler(feature_groups=feature_groups)
    
    scaled = scaler.fit_transform(df)
    print("Scaled DataFrame:")
    print(scaled)
    
    # Temp_A and Temp_B should be scaled by the same factor (max std)
    # std of Temp_A is 8.16, std of Temp_B is 81.6
    # Group max std should be 81.6
    
    std_a = df['Temp_A'].std(ddof=0)
    std_b = df['Temp_B'].std(ddof=0)
    group_max_std = max(std_a, std_b)
    
    expected_a = (df['Temp_A'] - df['Temp_A'].mean()) / group_max_std
    print(f"Group Max Std: {group_max_std}")
    print(f"Expected Temp_A scaled: {expected_a.tolist()}")
    print(f"Actual Temp_A scaled: {scaled['Temp_A'].tolist()}")
    
    np.testing.assert_array_almost_equal(scaled['Temp_A'], expected_a)
    print("GroupStandardScaler test passed!\n")

def test_group_lasso():
    print("Testing GroupLasso...")
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=['A1', 'A2', 'B1', 'C1'])
    # y depends only on group A
    y = 3*X['A1'] + 2*X['A2'] + np.random.randn(100) * 0.1
    
    feature_groups = {"group_A": ["A1", "A2"]}
    
    # Train with high alpha to see group selection
    result = train_group_lasso(X, y, feature_groups=feature_groups, alpha=0.5)
    
    print("Metrics:")
    print(result['metrics'])
    
    # Check if grouped features are selected together
    selected = result['metrics']['selected_features']
    print(f"Selected features: {selected}")
    
    if 'A1' in selected:
        assert 'A2' in selected, "A2 should be selected if A1 is selected (same group)"
        
    print("GroupLasso test passed!\n")

if __name__ == "__main__":
    test_group_scaler()
    try:
        test_group_lasso()
    except Exception as e:
        print(f"GroupLasso test failed with error: {e}")
