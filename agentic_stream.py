import os
import time
import re
import sys
import subprocess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from crewai import Agent, Task, Crew, LLM
import json
import joblib
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# --- CONFIGURATION ---
os.environ["CREWAI_ALLOW_CODE_EXECUTION_IN_ENV"] = "True"

CWD = os.path.abspath(os.getcwd())
MODELS_DIR = os.path.join(CWD, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
TRAIN_PATH = os.path.join(CWD, 'train_split.parquet')
DATA_PATH = os.path.join(CWD, 'strace.parquet')
ZERO_DAY_POOL = os.path.join(CWD, 'zero_day_pool.parquet')
KNOWN_ATTACK_POOL = os.path.join(CWD, 'known_attack_pool.parquet')

# Leakage Prevention boundaries
TRAIN_ROW_LIMIT = 500000
STREAM_ROW_START = 500001

def log_event(log_type, text):
    """
    Appends a timestamped event to logs.json for consumption by the Streamlit dashboard.
    
    Args:
        log_type (str): The category of the log (e.g., 'soc', 'agent', 'critical').
        text (str): The descriptive message to log.
    """
    try:
        logs_data = []
        if os.path.exists("logs.json"):
            with open("logs.json", "r") as f:
                logs_data = json.load(f)
        logs_data.append({"type": log_type, "text": text.strip()})
        with open("logs.json", "w") as f:
            json.dump(logs_data, f, indent=2)
    except Exception: pass

def get_family_slice(pf, family_name, count, skip_first=0, max_row=None):
    """
    Extracts a specific slice of syscall data for a given malware family.
    
    Args:
        pf (ParquetFile): The pyarrow ParquetFile handle.
        family_name (str): The MalwareFamily name to extract.
        count (int): Number of rows to collect.
        skip_first (int): Number of matching rows to skip before collecting.
        max_row (int): The global row index boundary to prevent data leakage.
    """
    collected, needed, skipped, current_row = [], count, 0, 0
    for batch in pf.iter_batches(batch_size=100000):
        df = batch.to_pandas()
        if 'MalwareFamily' not in df.columns: continue
        if max_row is not None and current_row > max_row: break
            
        matched = df[df['MalwareFamily'] == family_name]
        if skipped < skip_first:
            can_skip = min(len(matched), skip_first - skipped)
            matched = matched.iloc[can_skip:]
            skipped += can_skip
            
        if not matched.empty:
            take = min(len(matched), needed)
            collected.append(matched.iloc[:take])
            needed -= take
        current_row += len(df)
        if needed <= 0: break
    return pd.concat(collected) if collected else pd.DataFrame()

def prepare_data():
    """
    Prepares the experimental environment by creating strict splits for training
    and harvesting zero-day mimicry patterns from the unseen streaming section.
    """
    log_event("soc", "[SYSTEM] Initializing experimental splits (Rows 0-500k for training)...")
    pf = pq.ParquetFile(DATA_PATH)
    
    df_benign = get_family_slice(pf, 'Benign', 50000, max_row=TRAIN_ROW_LIMIT)
    df_benign['label'] = 0
    
    known_families = ['Gafgyt', 'Mirai', 'Tsunami', 'Agent']
    known_dfs = []
    for fam in known_families:
        df = get_family_slice(pf, fam, 5000, max_row=TRAIN_ROW_LIMIT)
        df['label'] = 1
        known_dfs.append(df)
        
    df_train = pd.concat([df_benign, pd.concat(known_dfs)]).sample(frac=1).reset_index(drop=True)
    df_train.select_dtypes(include=[np.number]).to_parquet(TRAIN_PATH)
    pd.concat(known_dfs).to_parquet(KNOWN_ATTACK_POOL)
    
    # Zero-Day Mimicry: Harvest DarkNexus from unseen data and mutate features
    df_dn = get_family_slice(pf, 'DarkNexus', 10000, skip_first=TRAIN_ROW_LIMIT)
    # Mutation: Zero out top identifying syscall columns to mimic benign activity
    for col in ['Call_open', 'Call_read']:
        if col in df_dn.columns: df_dn[col] = 0
    df_dn.to_parquet(ZERO_DAY_POOL)
    
    log_event("soc", "[SYSTEM] Experimental setup complete. Models have zero knowledge of mutated DarkNexus.")

class MalwareEnsemble:
    """Consensus-based voting ensemble that prioritizes recently trained specialists."""
    def __init__(self, models_dir):
        self.models_dir, self.models = models_dir, []
        self.load_models()

    def load_models(self):
        self.models = []
        if not os.path.exists(self.models_dir): return
        files = sorted([f for f in os.listdir(self.models_dir) if f.endswith('.joblib')], reverse=True)
        for f in files:
            try:
                self.models.append(joblib.load(os.path.join(self.models_dir, f)))
                if len(self.models) >= 5: break
            except Exception: continue

    def predict(self, X):
        if not self.models: return np.zeros(len(X))
        preds = [m.predict(X) for m in self.models]
        return (np.mean(preds, axis=0) >= 0.5).astype(int)

def stream_generator_v2(attack_cache):
    """Generator that yields syscall chunks starting after the training data boundary."""
    chunk_size = 500
    pf = pq.ParquetFile(DATA_PATH)
    batches = pf.iter_batches(batch_size=100000)
    
    skipped = 0
    while skipped < STREAM_ROW_START:
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

def run_continuous_stream(architect, lab_tech):
    """Main simulation loop orchestrating the SOC monitoring and self-healing logic."""
    attack_cache = pd.concat([pd.read_parquet(KNOWN_ATTACK_POOL), pd.read_parquet(ZERO_DAY_POOL)])
    attack_cache['label'], attack_cache['attack_type'] = 1, attack_cache['MalwareFamily']

    ensemble = MalwareEnsemble(MODELS_DIR)
    if not ensemble.models:
        trigger_agent_training(architect, lab_tech, "Initial Specialist Deployment")
        ensemble.load_models()

    rolling_fn, rolling_fp, missed_sigs = 0, 0, []
    total_rows, last_train_row = 0, 0
    if os.path.exists("logs.json"): os.remove("logs.json")
    
    for chunk in stream_generator_v2(attack_cache):
        total_rows += len(chunk)
        X = chunk.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
        y_pred, y_true = ensemble.predict(X), chunk['label'].values
        
        tp, fn, fp, tn = int(np.sum((y_pred == 1) & (y_true == 1))), int(np.sum((y_pred == 0) & (y_true == 1))), int(np.sum((y_pred == 1) & (y_true == 0))), int(np.sum((y_pred == 0) & (y_true == 0)))
        rolling_fn += fn; rolling_fp += fp
        
        m_counts = chunk[(y_pred == 0) & (y_true == 1)]['attack_type'].value_counts().to_dict() if fn > 0 else {}
        log_event("soc", f"[{total_rows}] {'[WARN]' if fn > 0 else '[OK]'} | TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn} | Attacks: {m_counts} | Ensemble: {len(ensemble.models)}")

        if fn > 0: missed_sigs.append(chunk[(y_pred == 0) & (y_true == 1)].select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore'))
            
        if (rolling_fn >= 50 or rolling_fp >= 300) and (total_rows - last_train_row > 10000):
            reason = "FN Evasion" if rolling_fn >= 50 else "FP Instability"
            log_event("critical", f"[{total_rows}] MIMICRY BREACH: {reason}. Orchestrating recovery...")
            
            df_err = pd.concat(missed_sigs) if rolling_fn >= 50 else chunk[(y_pred == 1) & (y_true == 0)]
            df_err_num = df_err.select_dtypes(include=[np.number])
            df_err_num['label'] = 1 if rolling_fn >= 50 else 0
            
            # Bootstrap existing knowledge with failure patterns
            cur_train = pd.read_parquet(TRAIN_PATH)
            new_train = pd.concat([cur_train, pd.concat([df_err_num]*20)]).sample(frac=1).reset_index(drop=True)
            new_train.to_parquet(TRAIN_PATH)
            
            if trigger_agent_training(architect, lab_tech, f"Recovering from {reason}"):
                ensemble.load_models()
            rolling_fn, rolling_fp, missed_sigs, last_train_row = 0, 0, [], total_rows
            
        if total_rows >= 1000000: break

def trigger_agent_training(architect, lab_tech, feedback):
    """Executes the agentic model search and deployment protocol."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = os.path.join(MODELS_DIR, f'specialist_{timestamp}.joblib')
    log_event("agent", f"[SYSTEM] Agentic Search Triggered: {feedback}")
    
    strategy_task = Task(description="Define hyperparameter grids for competition.", expected_output="Tuning strategy.", agent=architect)
    coding_task = Task(description=f"Write script to train best model on {TRAIN_PATH} and save to {new_model_path}. Prefix params with model__.", expected_output="Python code.", agent=lab_tech)
    
    crew = Crew(agents=[architect, lab_tech], tasks=[strategy_task, coding_task])
    result = str(crew.kickoff())
    match = re.search(r'```python\n(.*?)\n```', result, re.DOTALL)
    if match:
        code = match.group(1)
        with open("agent_patch.py", "w") as f: f.write(code)
        subprocess.run([sys.executable, "agent_patch.py"])
        return True
    return False

def main():
    prepare_data()
    # Initialize agents with local LLM connectivity
    local_llm = LLM(model="ollama/qwen2.5-coder:7b", base_url="http://localhost:11434", temperature=0.1)
    architect = Agent(role='Architect', goal='Define AutoML strategy.', backstory='Senior expert.', llm=local_llm)
    lab_tech = Agent(role='Engineer', goal='Implement search scripts.', backstory='Expert coder.', llm=local_llm)
    run_continuous_stream(architect, lab_tech)

if __name__ == "__main__":
    main()
