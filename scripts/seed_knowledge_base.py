# scripts/seed_knowledge_base.py

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from backend.llm.vector_db import knowledge_base

INITIAL_PROTOCOLS = [
    {
        "id": "protocol_perovskite_regression",
        "content": """
        Perovskite Solar Cell Efficiency Prediction Protocol:
        1. Data Cleaning: Filter out samples with PCE < 0 or > 30.
        2. Feature Engineering: Use Magpie descriptors for inorganic components and RDKit descriptors for organic spacers.
        3. Model: XGBoost with Optuna hyperparameter optimization.
        4. Validation: 5-fold cross-validation, stratified by structural family if possible.
        """,
        "metadata": {"task": "regression", "domain": "perovskite", "topic": "efficiency"}
    },
    {
        "id": "protocol_catalyst_classification",
        "content": """
        Catalyst Activity Classification Protocol (Active/Inactive):
        1. Data Preprocessing: Handle missing values in experimental conditions.
        2. Features: Atomic properties from matminer (AtomicOrbital, Electronegativity).
        3. Model: Random Forest for robustness against small datasets.
        4. XAI: Use SHAP to identify critical atomic features for activity.
        """,
        "metadata": {"task": "classification", "domain": "catalyst", "topic": "activity"}
    },
    {
        "id": "guideline_data_leakage",
        "content": """
        Statistical Pitfall: Data Leakage Prevention.
        - Ensure SMILES used for training are not present in the test set.
        - Be careful with 'Group Shuffle Split' when dealing with multiple measurements of the same compound.
        - Feature scaling should only be fit on training data.
        """,
        "metadata": {"type": "guideline", "topic": "data_leakage"}
    }
]

def seed():
    print("Seeding MI Knowledge Base...")
    for p in INITIAL_PROTOCOLS:
        knowledge_base.add_protocol(p["id"], p["content"], p["metadata"])
    print("Successfully seeded initial protocols.")

if __name__ == "__main__":
    seed()
