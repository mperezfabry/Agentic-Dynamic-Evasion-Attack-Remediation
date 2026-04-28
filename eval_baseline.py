import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
import json

# --- CONFIGURATION ---
DATA_PATH = 'strace.parquet'
BASELINE_MODEL_PATH = 'baseline_rf.joblib'
STATIC_LOGS_PATH = 'static_logs_v2.json'

# Evaluation boundaries (skipping initial data used for training/pool prep)
TRAIN_ROW_OFFSET = 500000

def log_event(log_type, text):
    """
    Appends evaluation events to the static log file for dashboard visualization.
    
    Args:
        log_type (str): Category of the log entry.
        text (str): Message content.
    """
    try:
        data = []
        if os.path.exists(STATIC_LOGS_PATH):
            with open(STATIC_LOGS_PATH, "r") as f: 
                data = json.load(f)
        data.append({"type": log_type, "text": text.strip()})
        with open(STATIC_LOGS_PATH, "w") as f: 
            json.dump(data, f, indent=2)
    except Exception: pass

def get_family_slice(pf, family_name, count, skip_first=0):
    """
    Extracts specific malware samples with a row offset to ensure evaluation on unseen data.
    """
    collected, needed, skipped = [], count, 0, 0
    for batch in pf.iter_batches(batch_size=100000):
        df = batch.to_pandas()
        if 'MalwareFamily' not in df.columns: continue
        matched = df[df['MalwareFamily'] == family_name]
        
        if skipped < skip_first:
            can_skip = min(len(matched), skip_first - skipped)
            matched = matched.iloc[can_skip:]
            skipped += can_skip
            
        if not matched.empty:
            take = min(len(matched), needed)
            collected.append(matched.iloc[:take])
            needed -= take
        if needed <= 0: break
    return pd.concat(collected) if collected else pd.DataFrame()

def stream_generator(attack_cache):
    """
    Simulates a 10% poisson-distributed attack stream using unseen data.
    """
    chunk_size = 500
    pf = pq.ParquetFile(DATA_PATH)
    batches = pf.iter_batches(batch_size=100000)
    
    # Skip training data boundary
    skipped = 0
    while skipped < TRAIN_ROW_OFFSET:
        try: skipped += len(next(batches))
        except StopIteration: break
    
    while True: 
        try:
            df = next(batches).to_pandas()
            if 'MalwareFamily' not in df.columns: continue
            df_benign = df[df['MalwareFamily'] == 'Benign']
            
            for i in range(0, len(df_benign) - chunk_size, chunk_size):
                chunk = df_benign.iloc[i:i+chunk_size].copy()
                chunk['label'], chunk['attack_type'] = 0, "Benign"
                num_attacks = np.random.poisson(chunk_size * 0.10)
                if num_attacks > 0:
                    attacks = attack_cache.sample(num_attacks, replace=True).copy()
                    chunk = pd.concat([chunk.iloc[:-num_attacks], attacks]).sample(frac=1).reset_index(drop=True)
                yield chunk
        except StopIteration: break

def run_static_evaluation():
    """
    Performs a million-row stream evaluation of the static baseline model.
    """
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: {BASELINE_MODEL_PATH} not found.")
        return

    if os.path.exists(STATIC_LOGS_PATH): os.remove(STATIC_LOGS_PATH)
    model = joblib.load(BASELINE_MODEL_PATH)
    pf = pq.ParquetFile(DATA_PATH)
    
    # Harvest evaluation attacks from the unseen section
    attack_dfs = []
    for fam in ['Gafgyt', 'Mirai', 'Tsunami', 'Agent', 'DarkNexus']:
        df = get_family_slice(pf, fam, 5000, skip_first=TRAIN_ROW_OFFSET)
        df['label'], df['attack_type'] = 1, fam
        attack_dfs.append(df)
    
    attack_cache = pd.concat(attack_dfs)
    total_tp, total_fp, total_tn, total_fn, total_rows = 0, 0, 0, 0, 0
    missed_counts = {}

    log_event("system", "Starting static model evaluation on unseen data...")
    for chunk in stream_generator(attack_cache):
        total_rows += len(chunk)
        X = chunk.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
        y_pred, y_true = model.predict(X), chunk['label'].values
        
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn
        
        m_counts = chunk[(y_pred == 0) & (y_true == 1)]['attack_type'].value_counts().to_dict() if fn > 0 else {}
        for s, count in m_counts.items(): missed_counts[s] = missed_counts.get(s, 0) + count
        
        log_event("soc", f"[{total_rows}] {'[WARN]' if fn > 0 else '[OK]'} | TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn} | Attacks: {m_counts} | Baseline Model")
        if total_rows >= 1000000: break

    # Export final cumulative results to metrics JSON
    try:
        if os.path.exists("baseline_metrics_v2.json"):
            with open("baseline_metrics_v2.json", "r") as f: metrics = json.load(f)
            metrics["missed_attacks"] = missed_counts
            with open("baseline_metrics_v2.json", "w") as f: json.dump(metrics, f, indent=4)
    except Exception: pass

    log_event("system", f"Evaluation complete. Total Missed: {total_fn}")

if __name__ == "__main__":
    run_static_evaluation()
