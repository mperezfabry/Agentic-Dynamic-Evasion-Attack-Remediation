import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import json
import os

# --- CONFIGURATION ---
TRAIN_PATH = 'private/train_split.parquet'
BASELINE_MODEL_PATH = 'baseline_rf.joblib'
METRICS_LOG_PATH = 'frontend/public/baseline_metrics_v2.json'
ROC_PLOT_PATH = 'frontend/public/baseline_roc.png'

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
        max_depth=3, 
        n_jobs=-1,
        random_state=42
    )
    
    # --- EVALUATE ON EXPLICIT SPLIT FOR ROC/AUC & SIZES ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Evaluating ROC AUC on test split...")
    eval_model = RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=-1, random_state=42)
    eval_model.fit(X_train, y_train)
    y_pred_proba = eval_model.predict_proba(X_test)[:, 1]
    y_pred = eval_model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    plt.figure()
    RocCurveDisplay.from_estimator(eval_model, X_test, y_test, name="Random Forest Baseline")
    plt.title("ROC Curve - Random Forest")
    plt.savefig(ROC_PLOT_PATH)
    plt.close()
    
    print("Executing 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # --- METRICS EXPORT ---
    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "roc_auc": float(auc),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1)
    }
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
