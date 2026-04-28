import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import os

# --- CONFIGURATION ---
TRAIN_PATH = 'train_split.parquet'
BASELINE_MODEL_PATH = 'baseline_rf.joblib'
METRICS_LOG_PATH = 'baseline_metrics_v2.json'

def run_baseline_training():
    """
    Trains the static Random Forest baseline model using a stratified fair split
    and exports cross-validation metrics for dashboard comparison.
    """
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found. Run the data preparation logic first.")
        return

    print(f"Loading training data from {TRAIN_PATH}...")
    df = pd.read_parquet(TRAIN_PATH)
    X, y = df.drop(columns=['label']), df['label']
    
    print("Initializing Static Random Forest Baseline...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        n_jobs=-1,
        random_state=42
    )
    
    print("Executing 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # --- METRICS EXPORT ---
    metrics = {}
    for m in scoring:
        mean, std = np.mean(results[f'test_{m}']), np.std(results[f'test_{m}'])
        metrics[m] = {"mean": float(mean), "std": float(std)}
        print(f"{m.capitalize():<10}: {mean:.4f} (+/- {std:.4f})")
    
    with open(METRICS_LOG_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Training final model on full dataset and saving to {BASELINE_MODEL_PATH}...")
    model.fit(X, y)
    joblib.dump(model, BASELINE_MODEL_PATH)

if __name__ == "__main__":
    run_baseline_training()
