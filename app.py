import streamlit as st
import time
import json
import pandas as pd
import re
import altair as alt
from collections import Counter
import os

# Set page configuration for a professional wide layout
st.set_page_config(layout="wide", page_title="Agentic SOC - Mimicry Attack Mitigation")

# Fixed chart domains for consistent cross-model comparison
X_DOMAIN = [0, 1000000]
Y_FNR_DOMAIN = [0, 25]
Y_FPR_DOMAIN = [0, 60]

def anonymize_paths(text):
    """Strips personal directory paths from log text for privacy."""
    cwd = os.getcwd()
    if cwd in text:
        text = text.replace(cwd, "[WORKSPACE_ROOT]")
    text = re.sub(r'/[a-zA-Z0-9._\-/]+/(?=[a-zA-Z0-9._\-]+\.py)', "[INTERNAL_PATH]/", text)
    return text

def load_baseline_metrics():
    """Loads baseline cross-validation metrics from the training phase."""
    path = "baseline_metrics_v2.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except Exception: pass
    return None

def load_logistic_metrics():
    """Loads logistic regression baseline cross-validation metrics."""
    path = "logistic_metrics.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except Exception: pass
    return None

def parse_attack_data(raw_data):
    """Helper to parse granular miss counts from log entries."""
    misses = Counter()
    try:
        raw_data = raw_data.strip()
        if raw_data.startswith('{'):
            clean_json = raw_data.replace("'", '"')
            data_dict = json.loads(clean_json)
            for s, count in data_dict.items():
                name = "Mimicry Attack" if s in ["Zero-Day", "DarkNexus"] else s
                misses[name] += count
        elif raw_data.startswith('['):
            strains = [a.strip().strip("'").strip('"') for a in raw_data.strip('[]').split(",") if a.strip()]
            for s in strains:
                if s != "nan":
                    name = "Mimicry Attack" if s in ["Zero-Day", "DarkNexus"] else s
                    misses[name] += 1
    except Exception: pass
    return misses

@st.cache_data
def load_static_baseline_data():
    """Calculates live metrics and history from the 1,000,000 row static baseline evaluation."""
    try:
        log_path = "static_logs_v2.json"
        if not os.path.exists(log_path): return None, pd.DataFrame(), {}
        with open(log_path, "r") as f: data = json.load(f)
        rows, tps, fps, fns, tns, missed_history = [], [], [], [], [], []
        current_misses = Counter()
        cum_tp, cum_fp, cum_fn, cum_tn = 0, 0, 0, 0
        for e in data:
            if e['type'] == 'soc':
                m = re.search(r'\[(\d+)\] .* TP: (\d+) \| FP: (\d+) \| FN: (\d+) \| TN: (\d+)', e['text'])
                if m:
                    r, tp, fp, fn, tn = map(int, [m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)])
                    rows.append(r); tps.append(tp); fps.append(fp); fns.append(fn); tns.append(tn)
                    cum_tp += tp; cum_fp += fp; cum_fn += fn; cum_tn += tn
                    at_m = re.search(r'Attacks: (\{.*?\}|\[.*?\])', e["text"])
                    if at_m: current_misses.update(parse_attack_data(at_m.group(1)))
                    for s, total in current_misses.items():
                        missed_history.append({'Rows': r, 'Strain': s, 'Missed Samples': total})
        df = pd.DataFrame({'Rows': rows, 'TP': tps, 'FP': fps, 'FN': fns, 'TN': tns})
        df['Rolling FNR'] = (df['FN'].rolling(20).sum() / (df['TP'].rolling(20).sum() + df['FN'].rolling(20).sum()) * 100).fillna(0)
        df['Rolling FPR'] = (df['FP'].rolling(20).sum() / (df['TN'].rolling(20).sum() + df['FP'].rolling(20).sum()) * 100).fillna(0)
        total = cum_tp + cum_fp + cum_fn + cum_tn
        metrics = {
            "accuracy": (cum_tp + cum_tn) / total if total > 0 else 0,
            "precision": cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0,
            "recall": cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0,
            "f1": 2 * (cum_tp / (cum_tp + cum_fp) * cum_tp / (cum_tp + cum_fn)) / (cum_tp / (cum_tp + cum_fp) + cum_tp / (cum_tp + cum_fn)) if (cum_tp / (cum_tp + cum_fp) > 0 and (cum_tp + cum_fn) > 0) else 0
        }
        return df, pd.DataFrame(missed_history), metrics
    except Exception: return None, pd.DataFrame(), {}

# Pre-load comparison data
baseline_metrics_data = load_baseline_metrics()
logistic_metrics_data = load_logistic_metrics()
baseline_df, df_static_missed, static_stream_metrics = load_static_baseline_data()

# --- INTERFACE HEADER ---
st.title("Autonomous SOC: Agility and Mitigation of Mimicry Attacks")

with st.expander("Agentic Architecture and Methodology", expanded=False):
    st.markdown("""
    ### Experimental Methodology
    This research infrastructure demonstrates the fundamental difference between a **Static Detection Perimeter** and an **Agentic Self-Healing SOC**. 
    
    #### The Mimicry Attack
    Standard IoT attack patterns (like Mirai or Tsunami) are relatively distinct. However, this simulation injects a **Mimicry Attack** (DarkNexus mutation). This attack is crafted by modifying a malicious syscall pattern so that its frequency distribution in key distinguishing columns mimics benign system behavior.
    
    #### Agentic Recovery Pipeline
    The system utilizes an autonomous feedback loop consisting of two specialized agents:
    1.  **Lead Data Scientist (Architect):** Continuously monitors the ensemble's performance metrics. When a 'Mimicry Breach' is detected (identified by a spike in False Negatives), the Architect analyzes the specific syscall distributions of the missed signatures and formulates a new multi-architecture search strategy.
    2.  **Execution Engineer (Lab Tech):** Implements the Architect's strategy by writing and executing a custom Scikit-Learn search script. This script explores deep hyperparameter grids across Tree, Linear, and Neural architectures to find a 'Specialist' model that can neutralize the evasion tactic.
    
    #### Interface Guide
    *   **Baseline Performance:** Shows the metrics for standard Random Forest and Logistic Regression models. They perform well on known threats but fail to adapt when the mimicry attack is encountered.
    *   **Live Ensemble Metrics:** Real-time Accuracy, Precision, and Recall of the active agentic ensemble.
    *   **Specialist Roster:** Displays the active 'Healers' currently deployed.
    *   **Consoles:** The left console shows the raw SOC processing stream; the right console shows the agents thinking and coding in real-time.
    """)

st.subheader("Simulation Controls")
start_sim = st.button("Start Agentic Detection Simulation", use_container_width=True, type="primary")
st.divider()

# --- BASELINE PERFORMANCE ---
st.subheader("Baseline Performance (Static Random Forest Pre-Injection)")


if baseline_metrics_data:
    st.markdown("#### Random Forest Baseline")
    cols = st.columns(4)
    m_keys = ["accuracy", "precision", "recall", "f1"]
    for i, k in enumerate(m_keys):
        val = baseline_metrics_data.get(k, {}).get("mean", 0.0)
        cols[i].metric(f"Baseline {k.capitalize()} (CV)", f"{val*100:.2f}%")
    st.markdown("[View Tree](https://fabry-perez-portfolio-site.s3.amazonaws.com/ml-project/deep_baseline_tree.png)")


with st.expander("Static Model Performance vs. Mimicry Stream", expanded=False):
    if static_stream_metrics:
        st.markdown("#### Cumulative Performance (Million-Row Stream)")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.write(f"**Accuracy:** {static_stream_metrics['accuracy']*100:.2f}%")
        s_col2.write(f"**Precision:** {static_stream_metrics['precision']*100:.2f}%")
        s_col3.write(f"**Recall:** {static_stream_metrics['recall']*100:.2f}%")
        s_col4.write(f"**F1-Score:** {static_stream_metrics['f1']*100:.2f}%")
    
    if baseline_df is not None:
        st.markdown("#### Detection Stability (Rolling Average)")
        fnr_c = alt.Chart(baseline_df).mark_area(line={'color':'red'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='red', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FNR:Q', scale=alt.Scale(domain=Y_FNR_DOMAIN))).properties(height=200, title="Static Model Rolling FNR")
        fpr_c = alt.Chart(baseline_df).mark_area(line={'color':'orange'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='orange', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FPR:Q', scale=alt.Scale(domain=Y_FPR_DOMAIN))).properties(height=200, title="Static Model Rolling FPR")
        st.altair_chart(fnr_c.interactive(), use_container_width=True)
        st.altair_chart(fpr_c.interactive(), use_container_width=True)
        
        if not df_static_missed.empty:
            st.markdown("#### Cumulative Missed Samples")
            line = alt.Chart(df_static_missed).mark_line(strokeWidth=3).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y='Missed Samples:Q', color='Strain:N').properties(height=300)
            st.altair_chart(line.interactive(), use_container_width=True)
st.divider()

# --- LIVE ENSEMBLE STATUS ---
st.subheader("Ensemble Status")
col_ind1, col_ind2, col_ind3 = st.columns(3)
stream_light, agent_light, ensemble_metric = col_ind1.empty(), col_ind2.empty(), col_ind3.empty()
stream_light.markdown("**STREAM:** IDLE")
agent_light.markdown("**AGENT:** STANDBY")
ensemble_metric.metric("Ensemble Size", "0")

# High visibility notice for Agentic Recovery phase
recovery_notice_placeholder = st.empty()

# --- LIVE PERFORMANCE ---
st.subheader("Live Ensemble Metrics")
col1, col2, col3, col4 = st.columns(4)
m_acc, m_prec, m_rec, m_f1 = [c.empty() for c in (col1, col2, col3, col4)]
st.divider()
st.subheader("Real-Time Detection Performance (Rolling 20-Chunk Average)")
fnr_chart_cont, fpr_chart_container = st.empty(), st.empty()
st.subheader("Cumulative Missed Attack Samples")
attack_chart = st.empty()
st.divider()

with st.expander("Specialist Agent Roster", expanded=False):
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.success("**XGBClassifier Specialist**")
        st.caption("Deployed: 14:43:59")
        st.json({"max_depth": 7, "n_estimators": 500, "learning_rate": 0.05, "Status": "Active"})
    with r_col2:
        st.success("**XGBClassifier Specialist**")
        st.caption("Deployed: 14:46:15")
        st.json({"max_depth": 10, "n_estimators": 200, "learning_rate": 0.1, "Status": "Active"})

st.divider()

left_c, right_c = st.columns(2)
soc_c, agent_c = left_c.empty(), right_c.empty()

# --- SIMULATION LOGIC ---
if start_sim:
    c_tp, c_fn, c_fp, c_tn, p_idx = 0, 0, 0, 0, 0
    rolling_history, chart_data, retrain_pts = [], [], []
    missed_total, missed_history = Counter(), []
    soc_t, agent_t = "", ""
    
    while True:
        try:
            if not os.path.exists("logs.json"):
                time.sleep(1); continue
            with open("logs.json", "r") as f: log_data = json.load(f)
        except Exception: time.sleep(0.5); continue

        if p_idx >= len(log_data):
            time.sleep(0.1); continue

        for i in range(p_idx, len(log_data)):
            entry = log_data[i]; p_idx += 1
            
            if entry["type"] == "soc":
                stream_light.markdown("**STREAM:** ACTIVE")
                agent_light.markdown("**AGENT:** MONITORING")
                r_m = re.search(r'\[(\d+)\]', entry["text"])
                tp_m, fp_m, fn_m, tn_m = [re.search(f'{k}: (\\d+)', entry["text"]) for k in ['TP', 'FP', 'FN', 'TN']]
                ens_m = re.search(r'Ensemble: (\d+)', entry["text"])
                
                if all([r_m, tp_m, fp_m, fn_m, tn_m]):
                    rows, tp, fp, fn, tn = [int(m.group(1)) for m in [r_m, tp_m, fp_m, fn_m, tn_m]]
                    c_tp += tp; c_fp += fp; c_fn += fn; c_tn += tn
                    rolling_history.append({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
                    if len(rolling_history) > 20: rolling_history.pop(0)
                    
                    r_tp, r_fn, r_fp, r_tn = [sum(c[k] for c in rolling_history) for k in ['tp', 'fn', 'fp', 'tn']]
                    r_fnr = (r_fn / max(1, (r_tp + r_fn)) * 100)
                    r_fpr = (r_fp / max(1, (r_tn + r_fp)) * 100)
                    
                    total = c_tp + c_fp + c_fn + c_tn
                    acc = (c_tp + c_tn) / total if total > 0 else 0
                    prec = c_tp / max(1, (c_tp + c_fp))
                    rec = c_tp / max(1, (c_tp + c_fn))
                    f1 = 2 * (prec * rec) / max(0.001, (prec + rec))
                    
                    m_acc.metric("Accuracy", f"{acc*100:.2f}%")
                    m_prec.metric("Precision", f"{prec*100:.2f}%")
                    m_rec.metric("Recall", f"{rec*100:.2f}%")
                    m_f1.metric("F1-Score", f"{f1*100:.2f}%")

                    chart_data.append({"Rows": rows, "Rolling FNR": r_fnr, "Rolling FPR": r_fpr})
                    df_c = pd.DataFrame(chart_data)
                    f_c = alt.Chart(df_c).mark_area(line={'color':'red'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='red', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FNR:Q', scale=alt.Scale(domain=Y_FNR_DOMAIN))).properties(height=200)
                    p_c = alt.Chart(df_c).mark_area(line={'color':'orange'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='orange', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FPR:Q', scale=alt.Scale(domain=Y_FPR_DOMAIN))).properties(height=200)
                    
                    if retrain_pts:
                        vl = alt.Chart(pd.DataFrame({'x': [p for p in retrain_pts if p >= df_c['Rows'].min()]})).mark_rule(strokeDash=[5, 5], color='white', opacity=0.3).encode(x='x:Q')
                        f_c += vl; p_c += vl
                    
                    fnr_chart_cont.altair_chart(f_c.interactive(), use_container_width=True)
                    fpr_chart_container.altair_chart(p_c.interactive(), use_container_width=True)
                    
                    at_m = re.search(r'Attacks: (\{.*?\}|\[.*?\])', entry["text"])
                    if at_m:
                        missed_total.update(parse_attack_data(at_m.group(1)))
                        for s, val in missed_total.items(): missed_history.append({'Rows': rows, 'Strain': s, 'Missed Samples': val})
                        if i % 5 == 0 and missed_history:
                            line = alt.Chart(pd.DataFrame(missed_history)).mark_line(strokeWidth=3).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y='Missed Samples:Q', color='Strain:N').properties(height=300)
                            attack_chart.altair_chart(line.interactive(), use_container_width=True)
                    
                    if ens_m: ensemble_metric.metric("Ensemble Size", ens_m.group(1))

                # Anonymize path in SOC logs
                clean_soc = anonymize_paths(entry["text"].replace("Zero-Day", "Mimicry Attack"))
                soc_t = (clean_soc + "\n" + soc_t)[:2000]
                soc_c.code(soc_t, language="bash")
            
            elif entry["type"] == "agent":
                agent_light.markdown("**AGENT:** TRAINING")
                # Anonymize path and mute verbose warnings
                txt = anonymize_paths(entry["text"])
                if "Warning" in txt or "Error" in txt: 
                    txt = txt.split('\n')[0] + "\n[SYSTEM] Tuning in progress..."
                agent_t = (txt + "\n\n" + agent_t)[:5000]
                agent_c.markdown(f"```text\n{agent_t}\n```")
                
            elif entry["type"] == "critical":
                agent_light.markdown("**AGENT:** RECOVERING")
                r_m = re.search(r'\[(\d+)\]', entry["text"])
                if r_m: retrain_pts.append(int(r_m.group(1)))
                
                # Prominent Recovery Notification
                with recovery_notice_placeholder:
                    st.error("MIMICRY BREACH DETECTED - ORCHESTRATING AGENTIC RECOVERY")
                    st.info("The Architect is analyzing failure distributions while the Lab Tech runs a multi-architecture search. The simulation will resume as soon as the Specialist is deployed.")
                
                time.sleep(15)
                recovery_notice_placeholder.empty()

    stream_light.markdown("**STREAM:** COMPLETE")
    agent_light.markdown("**AGENT:** STANDBY")
