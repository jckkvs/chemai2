"""
backend/api/session.py
Shared session state for ChemAI Nexus API
"""
from typing import Dict, Any, Optional

# In-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_all_sessions():
    """Return all active sessions"""
    return SESSIONS

def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve or initialize a session by ID"""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "df": None,
            "filename": None,
            "target_col": None,
            "task_type": "regression",
            "smiles_col": None,
            "config": {
                "num_scaler": "standard",
                "num_imputer": "median",
                "num_transform": "none",
                "cat_encoder": "onehot",
                "cat_imputer": "most_frequent",
                "do_polynomial": False,
                "poly_degree": 2,
                "poly_interaction_only": True,
                "feature_selector": "none",
                "n_features_to_select": 20,
                "selected_models": [],
                "monotonic_constraints": {},
                "do_eda": True,
                "do_prep": True,
                "do_eval": True,
                "do_pca": True,
                "do_shap": True
            },
            "metrics": {},
            "preview": [],
            "automl_result": None,
            "pipeline_result": None
        }
    return SESSIONS[session_id]

def clear_session(session_id: str):
    """Remove a session from the store"""
    SESSIONS.pop(session_id, None)
