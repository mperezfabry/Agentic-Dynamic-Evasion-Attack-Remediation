import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import json
import os
import warnings

# Suppress sklearn future warnings for cleaner terminal output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
TRAIN_PATH = 'private/train_split.parquet'
LOGISTIC_MODEL_PATH = 'private/logistic_model.joblib'
METRICS_LOG_PATH = 'frontend/public/logistic_metrics.json'
ROC_PLOT_PATH = 'frontend/public/logistic_roc.png'

def run_logistic_baseline():
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    print(f"Loading training data from {TRAIN_PATH}...")
    df = pd.read_parquet(TRAIN_PATH)
    
    # LEAKAGE PREVENTION
    leakage_cols = ['Call_ptrace', 'Call_mprotect']
    X = df.drop(columns=['label'] + [c for c in leakage_cols if c in df.columns])
    y = df['label']
    
    print(f"Initial features for model training: {X.shape[1]}")
    
    print("Initializing Logistic Regression (Log1p + RobustScaler + ElasticNet)...")
    
    # 1. Log Transform: Compresses massive syscall spikes
    log_transformer = FunctionTransformer(np.log1p, validate=False)
    
    # 2. Pipeline Architecture
    pipeline = Pipeline([
        ('log', log_transformer),
        ('scaler', RobustScaler()), # Ignores massive outliers during scaling
        ('model', LogisticRegression(
            penalty='elasticnet', # Mix of L1 (feature selection) and L2 (grouping)
            solver='saga', 
            l1_ratio=0.5, # 50% L1, 50% L2
            C=0.1, 
            class_weight='balanced', 
            max_iter=2500, # Increased max_iter for ElasticNet convergence
            random_state=42
        ))
    ])
    
    # --- EVALUATE ON EXPLICIT SPLIT FOR ROC/AUC & SIZES ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Evaluating ROC AUC on test split...")
    eval_pipeline = Pipeline(steps=pipeline.steps)
    eval_pipeline.fit(X_train, y_train)
    y_pred_proba = eval_pipeline.predict_proba(X_test)[:, 1]
    y_pred = eval_pipeline.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    plt.figure()
    RocCurveDisplay.from_estimator(eval_pipeline, X_test, y_test, name="Logistic Baseline")
    plt.title("ROC Curve - Logistic Regression")
    plt.savefig(ROC_PLOT_PATH)
    plt.close()

    print("Executing 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
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
        print(f"Logistic {m.capitalize():<10}: {mean:.4f} (+/- {std:.4f})")
    
    with open(METRICS_LOG_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saving optimized Logistic baseline to {LOGISTIC_MODEL_PATH}...")
    pipeline.fit(X, y)
    
    # Extract the features that survived ElasticNet
    trained_model = pipeline.named_steps['model']
    surviving_features = X.columns[(trained_model.coef_ != 0)[0]].tolist()
    print(f"ElasticNet Selection complete. Kept {len(surviving_features)} out of {X.shape[1]} features.")
    
    model_data = {
        'pipeline': pipeline,
        'selected_features': surviving_features
    }
    joblib.dump(model_data, LOGISTIC_MODEL_PATH)

if __name__ == "__main__":
    run_logistic_baseline()